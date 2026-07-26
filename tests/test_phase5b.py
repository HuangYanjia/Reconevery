from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import struct
import sys
import types
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from recon2sim.adapters import REGISTRY
from recon2sim.adapters.completion_candidates import (
    SAM3D_CHECKPOINT_REVISION,
    SAM3D_COMMIT,
    TRELLIS2_CHECKPOINT_REVISION,
    TRELLIS2_COMMIT,
    TRELLIS2_RUNTIME_REVISIONS,
    Sam3DObjectsAdapterConfig,
    Trellis2ObjectsAdapterConfig,
    validate_native_candidate_asset,
    validate_worker_model_identity,
)
from recon2sim.adapters.completion_inputs import _crop_rgba
from recon2sim.artifacts import (
    CandidateEvaluationManifest,
    CandidateGenerationManifest,
    CandidateHeldoutEvaluation,
    CandidateSelectionArtifact,
    CompletionCropManifest,
    CompletionEligibilityStatus,
    CompletionEvidencePackage,
    CompletionEvidenceSplit,
    CompletionTrainingMeasuredGeometry,
    CompletionWorkerManifest,
    Phase5BConsistencyReport,
)
from recon2sim.cli import app
from recon2sim.completion import (
    candidate_id,
    completion_eligibility,
    positive_scale_sim3,
    select_diverse_anchors,
    split_object_evidence,
)
from recon2sim.completion_parity import (
    backend_layout_world_matrix,
    binary_mask_metrics,
    target_mask_metrics,
)
from recon2sim.config import load_config
from recon2sim.dense_mvs import read_dense_array
from recon2sim.ir import AssetType, SceneIR
from recon2sim.pipeline import PipelineRunner

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "examples/tabletop"
CONFIG = ROOT / "configs/phase5b_e2e_fake.yaml"


def _spacing_module() -> object:
    path = ROOT / "workers/measured_geometry/measured_geometry_worker/spacing.py"
    spec = importlib.util.spec_from_file_location("phase5b_spacing", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase5a_spacing_is_input_order_invariant() -> None:
    estimator = _spacing_module().estimate_spacing
    points = [(float(x), float(y), float(z)) for x in range(4) for y in range(3) for z in range(2)]
    expected = estimator(points, multiplier=1.5)
    random.Random(90210).shuffle(points)
    actual = estimator(points, multiplier=1.5)
    assert actual == expected
    assert actual["nearest_neighbor_median"] == 1.0
    assert actual["voxel_size"] == 1.5


def test_phase5a_fused_surfels_are_input_order_invariant() -> None:
    np = pytest.importorskip("numpy")
    worker_root = ROOT / "workers/measured_geometry"
    sys.path.insert(0, str(worker_root))
    try:
        from measured_geometry_worker.inference import fuse
    finally:
        sys.path.remove(str(worker_root))
    points = np.asarray(
        [(x / 10, y / 10, z / 10) for x in range(5) for y in range(4) for z in range(3)],
        dtype=np.float64,
    )
    normals = np.tile(np.asarray([[0.0, 0.0, 1.0]]), (len(points), 1))
    expected = fuse(points.copy(), normals.copy(), 1.5)
    order = np.random.default_rng(90210).permutation(len(points))
    actual = fuse(points[order], normals[order], 1.5)
    for expected_array, actual_array in zip(expected[:4], actual[:4], strict=True):
        assert expected_array.tobytes() == actual_array.tobytes()
    assert expected[4] == actual[4]


def test_completion_crop_is_deterministic_binary_and_exactly_invertible(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame.png"
    mask = tmp_path / "mask.png"
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (20, 12), (30, 90, 150)).save(frame)
    mask_image = Image.new("L", (20, 12))
    for x in range(0, 7):
        for y in range(2, 10):
            mask_image.putpixel((x, y), 255)
    mask_image.save(mask)
    crop_to_source, source_to_crop = _crop_rgba(
        frame,
        mask,
        first,
        bbox_xywh=(0, 2, 7, 8),
        margin_ratio=0.15,
        output_size=64,
    )
    repeated = _crop_rgba(
        frame,
        mask,
        second,
        bbox_xywh=(0, 2, 7, 8),
        margin_ratio=0.15,
        output_size=64,
    )
    assert first.read_bytes() == second.read_bytes()
    assert repeated == (crop_to_source, source_to_crop)
    rgba = Image.open(first)
    assert rgba.mode == "RGBA"
    assert set(rgba.getchannel("A").get_flattened_data()).issubset({0, 255})
    for row in range(3):
        for column in range(3):
            product = sum(
                crop_to_source[row * 3 + index] * source_to_crop[index * 3 + column]
                for index in range(3)
            )
            assert product == pytest.approx(1.0 if row == column else 0.0)


@pytest.mark.parametrize(
    ("label", "hint", "allow_unclassified", "expected"),
    [
        ("cup", AssetType.RIGID, False, CompletionEligibilityStatus.ELIGIBLE_RIGID),
        (
            "table",
            AssetType.STATIC_STRUCTURE,
            False,
            CompletionEligibilityStatus.ELIGIBLE_STATIC,
        ),
        (
            "cabinet",
            AssetType.RIGID,
            True,
            CompletionEligibilityStatus.DEFERRED_ARTICULATED,
        ),
        (
            "rope",
            AssetType.DEFORMABLE,
            True,
            CompletionEligibilityStatus.DEFERRED_DEFORMABLE,
        ),
        (
            "person",
            AssetType.RIGID,
            True,
            CompletionEligibilityStatus.DEFERRED_HUMAN,
        ),
        (
            "unknown thing",
            AssetType.UNCLASSIFIED,
            False,
            CompletionEligibilityStatus.DEFERRED_UNKNOWN,
        ),
        (
            "unknown thing",
            AssetType.UNCLASSIFIED,
            True,
            CompletionEligibilityStatus.ELIGIBLE_RIGID,
        ),
    ],
)
def test_typed_completion_eligibility(
    label: str,
    hint: AssetType,
    allow_unclassified: bool,
    expected: CompletionEligibilityStatus,
) -> None:
    assert (
        completion_eligibility(
            label,
            hint,
            allow_unclassified=allow_unclassified,
        )[0]
        is expected
    )


def test_explicit_override_is_auditable() -> None:
    status, reason, overridden = completion_eligibility(
        "cabinet",
        AssetType.ARTICULATED,
        allow_unclassified=False,
        override=CompletionEligibilityStatus.ELIGIBLE_RIGID,
    )
    assert status is CompletionEligibilityStatus.ELIGIBLE_RIGID
    assert overridden
    assert "explicit" in reason


def test_anchor_selection_prefers_score_then_diversity() -> None:
    selected = select_diverse_anchors(
        [
            ("front_best", 1.0, (0, 0, 1)),
            ("front_duplicate", 0.9, (0, 0.01, 1)),
            ("side", 0.8, (1, 0, 0)),
        ],
        maximum_count=2,
        minimum_angle_degrees=20,
    )
    assert selected == ["front_best", "side"]


def test_completion_evidence_split_is_disjoint_and_deterministic() -> None:
    first = split_object_evidence(
        "cup_0001",
        [f"frame_{index:03d}" for index in range(8)],
        ["frame_002", "frame_006"],
        minimum_heldout_frames=2,
        fitting_fraction=0.6,
    )
    second = split_object_evidence(
        "cup_0001",
        [f"frame_{index:03d}" for index in range(8)],
        ["frame_002", "frame_006"],
        minimum_heldout_frames=2,
        fitting_fraction=0.6,
    )
    assert first == second
    groups = [
        set(first.generation_anchor_frames),
        set(first.registration_fitting_frames),
        set(first.heldout_validation_frames),
    ]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]


def test_evidence_split_schema_rejects_leakage() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        CompletionEvidenceSplit.model_validate(
            {
                "frame_sequence_digest": "0" * 64,
                "objects": [
                    {
                        "object_id": "cup_0001",
                        "generation_anchor_frames": ["frame_1"],
                        "registration_fitting_frames": ["frame_2"],
                        "heldout_validation_frames": ["frame_1"],
                    }
                ],
                "seed": 42,
            }
        )


def test_training_geometry_sampling_cap_is_exactly_audited() -> None:
    payload = {
        "object_id": "cup_0001",
        "training_frame_ids": ["frame_000001", "frame_000002"],
        "heldout_frame_ids": ["frame_000003"],
        "raw_sample_count": 12,
        "boundary_rejected_count": 1,
        "invalid_geometry_rejected_count": 1,
        "sam_score_rejected_count": 0,
        "consistency_rejected_count": 0,
        "depth_discontinuity_rejected_count": 1,
        "multi_view_rejected_count": 1,
        "pre_cap_validated_point_count": 8,
        "validated_point_count": 5,
        "maximum_samples_per_object": 5,
        "sampling_cap_applied": True,
        "supporting_fitting_views": ["frame_000001", "frame_000002"],
        "point_cloud_path": ("reconstruction/completion/evidence/cup_0001/training_points.ply"),
        "point_cloud_sha256": "1" * 64,
        "normal_sha256": "2" * 64,
        "phase5a_all_view_validated_point_count": 10,
        "frame_records": [
            {
                "frame_id": "frame_000001",
                "raw_sample_count": 6,
                "backprojected_point_count": 5,
                "validated_point_count": 4,
                "maximum_supporting_views": 2,
                "median_relative_depth_residual": 0.01,
            },
            {
                "frame_id": "frame_000002",
                "raw_sample_count": 6,
                "backprojected_point_count": 5,
                "validated_point_count": 4,
                "maximum_supporting_views": 2,
                "median_relative_depth_residual": 0.02,
            },
        ],
        "backprojection_configuration": {"maximum_samples_per_object": 5},
        "consistency_configuration": {"minimum_supporting_views": 2},
    }
    artifact = CompletionTrainingMeasuredGeometry.model_validate(payload)
    assert artifact.pre_cap_validated_point_count == 8
    assert artifact.validated_point_count == 5
    assert artifact.sampling_cap_applied

    inconsistent = dict(payload)
    inconsistent["validated_point_count"] = 6
    with pytest.raises(ValueError, match="post-cap"):
        CompletionTrainingMeasuredGeometry.model_validate(inconsistent)

    wrong_status = dict(payload)
    wrong_status["sampling_cap_applied"] = False
    with pytest.raises(ValueError, match="sampling-cap status"):
        CompletionTrainingMeasuredGeometry.model_validate(wrong_status)


def test_candidate_id_is_backend_anchor_and_seed_stable() -> None:
    assert (
        candidate_id("cup_0001", "trellis2", "frame_000010", 42)
        == "cup_0001__trellis2__frame_000010__seed_42"
    )


@pytest.mark.parametrize(
    ("matrix", "valid"),
    [
        (
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            True,
        ),
        (
            [2, 0, 0, 1, 0, 2, 0, 2, 0, 0, 2, 3, 0, 0, 0, 1],
            True,
        ),
        (
            [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            False,
        ),
        (
            [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            False,
        ),
    ],
)
def test_candidate_transform_requires_proper_positive_scale_sim3(
    matrix: list[float],
    valid: bool,
) -> None:
    assert positive_scale_sim3(matrix) is valid


def test_asymmetric_registration_recovers_known_sim3_from_partial_measurements() -> None:
    np = pytest.importorskip("numpy")
    rotation_module = pytest.importorskip("scipy.spatial.transform")
    worker_root = ROOT / "workers/completion_evaluation"
    sys.path.insert(0, str(worker_root))
    try:
        from completion_evaluation_worker.sim3_registration import (
            register_asymmetric_sim3,
        )
    finally:
        sys.path.remove(str(worker_root))
    generator = np.random.default_rng(4)
    candidate = generator.normal(size=(500, 3)) * np.asarray([1.0, 0.6, 0.25])
    rotation = rotation_module.Rotation.from_euler(
        "xyz",
        [20, -15, 35],
        degrees=True,
    ).as_matrix()
    scale = 2.3
    translation = np.asarray([1.2, -0.7, 3.4])
    measured_partial = scale * (candidate[:300] @ rotation.T) + translation
    result = register_asymmetric_sim3(
        candidate,
        measured_partial,
        iterations=40,
        trimmed_fraction=0.9,
    )
    assert result.scale == pytest.approx(scale, abs=1e-8)
    assert result.matrix[:3, :3] == pytest.approx(scale * rotation, abs=1e-8)
    assert result.matrix[:3, 3] == pytest.approx(translation, abs=1e-8)
    assert result.median_residual < 1e-8


def test_official_candidate_backend_pins_are_exact() -> None:
    assert SAM3D_COMMIT == "f91db411c50efee93d8db7aeb323885650f6f722"
    assert SAM3D_CHECKPOINT_REVISION == "05929e2a63f234014031f9941f4aabefea5f382e"
    assert TRELLIS2_COMMIT == "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    assert TRELLIS2_CHECKPOINT_REVISION == "af44b45f2e35a493886929c6d786e563ec68364d"
    assert TRELLIS2_RUNTIME_REVISIONS == {
        "facebook/dinov3-vitl16-pretrain-lvd1689m": ("ea8dc2863c51be0a264bab82070e3e8836b02d51"),
        "microsoft/TRELLIS-image-large": "25e0d31ffbebe4b5a97464dd851910efc3002d96",
    }


def test_sam3d_backend_layout_converts_pytorch3d_camera_axes_to_opencv() -> None:
    result = backend_layout_world_matrix(
        {
            "scale": [[2.0, 2.0, 2.0]],
            "rotation": [[1.0, 0.0, 0.0, 0.0]],
            "translation": [[1.0, 2.0, 3.0]],
        },
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    assert result == [
        [-2.0, 0.0, 0.0, -1.0],
        [0.0, -2.0, 0.0, -2.0],
        [0.0, 0.0, 2.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    quarter_turn = backend_layout_world_matrix(
        {
            "scale": [[1.0, 1.0, 1.0]],
            "rotation": [[2**-0.5, 0.0, 0.0, 2**-0.5]],
            "translation": [[0.0, 0.0, 1.0]],
        },
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    assert [value for row in quarter_turn for value in row] == pytest.approx(
        [
            value
            for row in [
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            for value in row
        ],
        abs=1e-12,
    )


def test_real_candidate_configs_require_checkpoint_and_runtime_hashes() -> None:
    with pytest.raises(ValueError, match="checkpoint file hashes"):
        Sam3DObjectsAdapterConfig.model_validate(
            {
                "execution_mode": "docker",
                "worker_module": "sam3d_objects_worker",
            }
        )
    with pytest.raises(ValueError, match="checkpoint file hashes"):
        Trellis2ObjectsAdapterConfig.model_validate(
            {
                "execution_mode": "docker",
                "worker_module": "trellis2_objects_worker",
            }
        )


def test_core_rejects_malformed_native_candidate_assets(tmp_path: Path) -> None:
    broken_glb = tmp_path / "broken.glb"
    broken_glb.write_bytes(b"glTF\x02\x00\x00\x00\xff\x00\x00\x00")
    with pytest.raises(ValueError, match="declared size"):
        validate_native_candidate_asset(broken_glb, "pbr_glb")
    empty_ply = tmp_path / "empty.ply"
    empty_ply.write_bytes(
        b"ply\nformat ascii 1.0\nelement vertex 0\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    with pytest.raises(ValueError, match="no declared vertices"):
        validate_native_candidate_asset(empty_ply, "gaussian_splat_ply")


def test_official_sam_native_export_preserves_gaussian_and_glb(tmp_path: Path) -> None:
    worker_root = ROOT / "workers/sam3d_objects"
    sys.path.insert(0, str(worker_root))
    try:
        from sam3d_objects_worker.native_export import export_native
    finally:
        sys.path.remove(str(worker_root))

    class Gaussian:
        def save_ply(self, path: str) -> None:
            Path(path).write_bytes(b"official-gaussian")

    class Glb:
        def export(self, path: str) -> None:
            Path(path).write_bytes(b"official-glb")

    assets = export_native(
        {"gaussian": [Gaussian()], "glb": Glb()},
        tmp_path / "native",
    )
    assert [(item[1], item[2]) for item in assets] == [
        ("gaussian_splat_ply", "official_gaussian_splat"),
        ("pbr_glb", "official_optional_visual_glb"),
    ]
    assert all(path.is_file() for path, _, _ in assets)


def test_official_sam_api_loader_is_not_shadowed_by_jupyter_notebook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "official"
    (checkout / "notebook").mkdir(parents=True)
    (checkout / "notebook/inference.py").write_text(
        "class Inference:\n    identity = 'official-exact-checkout'\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(sys.modules, "notebook", types.ModuleType("notebook"))
    worker_root = ROOT / "workers/sam3d_objects"
    sys.path.insert(0, str(worker_root))
    try:
        from sam3d_objects_worker.official_api import load_inference_class
    finally:
        sys.path.remove(str(worker_root))
    assert load_inference_class(checkout).identity == "official-exact-checkout"


def test_trellis2_rgba_loader_skips_unused_background_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForbiddenBackgroundRemoval:
        def __init__(self, **_: object) -> None:
            raise AssertionError("canonical RGBA input must not initialize background removal")

    rembg = types.SimpleNamespace(BiRefNet=ForbiddenBackgroundRemoval)

    class Pipeline:
        def __init__(self) -> None:
            self.rembg_model = rembg.BiRefNet(model_name="BiRefNet")

        @classmethod
        def from_pretrained(cls, _: str) -> Pipeline:
            return cls()

    pipelines = types.ModuleType("trellis2.pipelines")
    pipelines.Trellis2ImageTo3DPipeline = Pipeline
    pipelines.rembg = rembg
    trellis2 = types.ModuleType("trellis2")
    monkeypatch.setitem(sys.modules, "trellis2", trellis2)
    monkeypatch.setitem(sys.modules, "trellis2.pipelines", pipelines)
    worker_root = ROOT / "workers/trellis2_objects"
    sys.path.insert(0, str(worker_root))
    try:
        from trellis2_objects_worker.official_api import load_rgba_pipeline

        pipeline = load_rgba_pipeline(tmp_path)
    finally:
        sys.path.remove(str(worker_root))
    assert pipeline.rembg_model is None
    assert rembg.BiRefNet is ForbiddenBackgroundRemoval


def test_dense_depth_occlusion_and_negative_space_are_distinct() -> None:
    np = pytest.importorskip("numpy")
    worker_root = ROOT / "workers/completion_evaluation"
    sys.path.insert(0, str(worker_root))
    try:
        from completion_evaluation_worker.negative_space import classify_candidate_pixels
    finally:
        sys.path.remove(str(worker_root))
    candidate = np.asarray([[2.0, 0.5, 1.0, np.nan]], dtype=np.float32)
    scene = np.asarray([[1.0, 1.0, 2.0, 1.0]], dtype=np.float32)
    mask = np.asarray([[False, False, True, False]])
    result = classify_candidate_pixels(candidate, scene, mask, relative_tolerance=0.05)
    assert result["occluded"].tolist() == [[True, False, False, False]]
    assert result["front"].tolist() == [[False, True, False, False]]
    assert not result["negative"][0, 0]
    assert result["negative"][0, 1]


@pytest.mark.parametrize("channels", [1, 3])
def test_phase5b_dense_reader_is_byte_equivalent_to_phase5a(
    tmp_path: Path,
    channels: int,
) -> None:
    np = pytest.importorskip("numpy")
    worker_root = ROOT / "workers/completion_evaluation"
    sys.path.insert(0, str(worker_root))
    try:
        from completion_evaluation_worker.dense_io import read_array
    finally:
        sys.path.remove(str(worker_root))
    width, height = 3, 2
    values = [float(index) + 0.25 for index in range(width * height * channels)]
    path = tmp_path / f"known-{channels}.bin"
    path.write_bytes(
        f"{width}&{height}&{channels}&".encode() + struct.pack(f"<{len(values)}f", *values)
    )
    phase5a = read_dense_array(path, expected_channels=channels)
    phase5b = read_array(path, channels, require_finite=True)
    expected = np.asarray(
        [
            [
                [phase5a.value(column, row, channel) for channel in range(channels)]
                for column in range(width)
            ]
            for row in range(height)
        ],
        dtype=np.float32,
    )
    actual = phase5b[:, :, None] if channels == 1 else phase5b
    assert actual.tobytes() == expected.tobytes()


def test_phase5b_dense_reader_rejects_channels_truncation_and_nonfinite_normals(
    tmp_path: Path,
) -> None:
    pytest.importorskip("numpy")
    worker_root = ROOT / "workers/completion_evaluation"
    sys.path.insert(0, str(worker_root))
    try:
        from completion_evaluation_worker.dense_io import read_array
    finally:
        sys.path.remove(str(worker_root))
    wrong_channels = tmp_path / "wrong.bin"
    wrong_channels.write_bytes(b"1&1&1&" + struct.pack("<f", 1.0))
    with pytest.raises(ValueError, match="expected 3"):
        read_array(wrong_channels, 3)
    truncated = tmp_path / "truncated.bin"
    truncated.write_bytes(b"2&1&3&" + struct.pack("<3f", 1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="expected 6"):
        read_array(truncated, 3)
    nonfinite = tmp_path / "nonfinite.bin"
    nonfinite.write_bytes(b"1&1&3&" + struct.pack("<3f", 0.0, float("nan"), 1.0))
    with pytest.raises(ValueError, match="non-finite"):
        read_array(nonfinite, 3, require_finite=True)


def test_evaluation_schema_rejects_implicit_representation_transfer() -> None:
    common = {
        "candidate_id": "cup__sam3d",
        "object_id": "cup_0001",
        "backend": "sam3d_objects",
        "registration_asset_id": "visual_glb",
        "registration_asset_path": "candidate/visual.glb",
        "evaluation_asset_id": "visual_glb",
        "evaluation_asset_path": "candidate/visual.glb",
        "selection_asset_id": "native_gaussian",
        "selection_asset_path": "candidate/gaussians.ply",
        "transform_sha256": "0" * 64,
        "anchor_sanity": {
            "frame_ids": ["frame_1"],
            "transform_source": "backend_predicted_layout",
            "mask_precision": 0.5,
            "mask_recall": 0.5,
            "mask_iou": 0.5,
            "negative_space_violation_ratio": 0,
            "front_of_scene_violation_ratio": 0,
            "valid_candidate_pixel_count": 1,
            "per_frame": [],
        },
        "fitting_metrics": {
            "frame_ids": ["frame_2"],
            "transform_source": "frozen_registration",
            "mask_precision": 0.5,
            "mask_recall": 0.5,
            "mask_iou": 0.5,
            "negative_space_violation_ratio": 0,
            "front_of_scene_violation_ratio": 0,
            "valid_candidate_pixel_count": 1,
            "per_frame": [],
        },
        "heldout_frame_ids": ["frame_3"],
        "metrics": {
            "mask_precision": 0.5,
            "mask_recall": 0.5,
            "mask_iou": 0.5,
            "per_frame_iou": {"frame_3": 0.5},
            "dense_depth_relative_residual": 0.01,
            "depth_inlier_fraction": 0.9,
            "negative_space_violation_ratio": 0,
            "front_of_scene_violation_ratio": 0,
            "measured_point_to_candidate_median": 0.01,
            "measured_point_to_candidate_p90": 0.02,
            "normal_agreement": 0.9,
            "candidate_visible_coverage": 0.5,
            "validation_view_count": 1,
            "visible_candidate_area": 1,
            "occluded_candidate_area": 0,
            "negative_space_violation_area": 0,
            "front_of_scene_violation_area": 0,
        },
        "measured_baseline_metrics": {
            "mask_precision": 0.5,
            "mask_recall": 0.5,
            "mask_iou": 0.5,
            "per_frame_iou": {"frame_3": 0.5},
            "dense_depth_relative_residual": 0.01,
            "depth_inlier_fraction": 0.9,
            "negative_space_violation_ratio": 0,
            "front_of_scene_violation_ratio": 0,
            "measured_point_to_candidate_median": 0.01,
            "measured_point_to_candidate_p90": 0.02,
            "normal_agreement": 0.9,
            "candidate_visible_coverage": 0.5,
            "validation_view_count": 1,
            "visible_candidate_area": 1,
            "occluded_candidate_area": 0,
            "negative_space_violation_area": 0,
            "front_of_scene_violation_area": 0,
        },
        "completion_gain": {
            "recall_gain_vs_measured_baseline": 0,
            "iou_gain_vs_measured_baseline": 0,
            "precision_change_vs_measured_baseline": 0,
            "depth_residual_change": 0,
            "visible_coverage_gain": 0,
            "negative_space_change": 0,
        },
        "passed_hard_gates": True,
        "failed_gates": [],
        "evaluation_runtime_seconds": 0,
        "license_record": {
            "backend": "sam3d_objects",
            "code_license": "SAM License",
            "checkpoint_license": "SAM License",
            "dependency_licenses": {},
            "asset_license": "SAM License",
            "access_conditions": [],
            "commercial_use_review_status": "not_reviewed",
            "research_evaluation_allowed": True,
            "production_selectable": False,
        },
        "failure_classification": "passed",
    }
    with pytest.raises(ValueError, match="without accepted parity"):
        CandidateHeldoutEvaluation.model_validate(common)


def test_representation_parity_detects_equivalence_and_mismatch() -> None:
    equivalent = [0, 255, 255, 0]
    metrics = binary_mask_metrics(equivalent, equivalent, 2, 2)
    assert metrics[:2] == (1.0, 1.0)
    assert metrics[2] == 0.0
    mismatch = binary_mask_metrics(
        [255, 0, 0, 0],
        [0, 0, 0, 255],
        2,
        2,
    )
    assert mismatch[0] == 0.0
    assert mismatch[1] == 0.0
    assert mismatch[2] is not None and mismatch[2] > 0
    assert target_mask_metrics([255, 255, 0, 0], [255, 0, 255, 0]) == (
        0.5,
        0.5,
        1 / 3,
    )


def test_measured_renderer_control_uses_mesh_rasterizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    worker_root = ROOT / "workers/completion_evaluation"
    sys.path.insert(0, str(worker_root))
    try:
        import completion_evaluation_worker.inference as worker_inference
    finally:
        sys.path.remove(str(worker_root))
    expected = np.asarray([[1.0]], dtype=np.float32)
    calls = []

    def fake_render(path: Path, transform: object, camera: object) -> object:
        calls.append((path, transform, camera))
        return type("Rendered", (), {"depth": expected})()

    monkeypatch.setattr(worker_inference, "render_mesh_candidate", fake_render)
    result = worker_inference._candidate_depth(  # noqa: SLF001
        input_root=tmp_path,
        asset={
            "relative_path": "control.ply",
            "format": "mesh_ply",
            "role": "fitting_only_open_surface_renderer_control",
        },
        points=np.asarray([[0.0, 0.0, 1.0]]),
        transform=np.eye(4),
        camera_from_world=np.eye(4),
        undistortion={
            "dense_dimensions": [1, 1],
            "dense_intrinsics": [1.0, 1.0, 0.0, 0.0],
        },
        measured_baseline=True,
    )
    assert result is expected
    assert len(calls) == 1


def test_candidate_glb_node_transforms_are_applied_during_surface_loading(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    trimesh = pytest.importorskip("trimesh")
    worker_root = ROOT / "workers/completion_evaluation"
    sys.path.insert(0, str(worker_root))
    try:
        from completion_evaluation_worker.candidate_io import load_candidate_surface
    finally:
        sys.path.remove(str(worker_root))
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    scene = trimesh.Scene()
    transform = np.eye(4)
    transform[:3, 3] = (4.0, -2.0, 3.0)
    scene.add_geometry(mesh, node_name="translated", transform=transform)
    path = tmp_path / "transformed.glb"
    scene.export(path)
    points = load_candidate_surface(path, maximum_samples=20_000, seed=42)
    assert np.median(points, axis=0) == pytest.approx((4.0, -2.0, 3.0), abs=0.05)


@pytest.mark.parametrize(
    ("anchor_iou", "fitting_iou", "heldout_iou", "expected"),
    [
        (0.0, 0.0, 0.0, "registration_failed"),
        (0.5, 0.0, 0.0, "fitting_view_inconsistent"),
        (0.5, 0.4, 0.0, "fitting_overfit_heldout_failure"),
        (0.5, 0.4, 0.3, "heldout_shape_inconsistent"),
    ],
)
def test_zero_iou_failure_classification_is_stage_specific(
    anchor_iou: float,
    fitting_iou: float,
    heldout_iou: float,
    expected: str,
) -> None:
    pytest.importorskip("numpy")
    worker_root = ROOT / "workers/completion_evaluation"
    sys.path.insert(0, str(worker_root))
    try:
        from completion_evaluation_worker.inference import _failure_classification
    finally:
        sys.path.remove(str(worker_root))
    candidate = {"backend": "sam3d_objects"}
    anchor = {"valid_candidate_pixel_count": 1, "mask_iou": anchor_iou}
    fitting = {"mask_iou": fitting_iou}
    heldout = {"mask_iou": heldout_iou}
    assert (
        _failure_classification(
            candidate,
            anchor,
            fitting,
            heldout,
            ["minimum_mask_iou"],
        )
        == expected
    )


def test_fake_phase5b_pipeline_and_resume(tmp_path: Path) -> None:
    run_dir = tmp_path / "phase5b"
    runner = PipelineRunner(load_config(CONFIG), INPUT, run_dir)
    first = runner.run()
    assert all(stage["status"] == "succeeded" for stage in first["stages"].values())
    sam = CandidateGenerationManifest.model_validate_json(
        (run_dir / "reconstruction/completion/sam3d_generation_manifest.json").read_text()
    )
    trellis = CandidateGenerationManifest.model_validate_json(
        (run_dir / "reconstruction/completion/trellis2_generation_manifest.json").read_text()
    )
    assert sam.candidates and trellis.candidates
    assert {
        asset.format.value for candidate in sam.candidates for asset in candidate.native_assets
    } >= {"gaussian_splat_ply"}
    assert {
        asset.format.value for candidate in trellis.candidates for asset in candidate.native_assets
    } == {"pbr_glb"}
    for candidate in (*sam.candidates, *trellis.candidates):
        declared = {asset.asset_id: asset.relative_path for asset in candidate.native_assets}
        assert declared[candidate.registration_asset_id] == candidate.registration_asset_path
        assert declared[candidate.evaluation_asset_id] == candidate.evaluation_asset_path
        assert declared[candidate.selection_asset_id] == candidate.selection_asset_path
    evaluation = CandidateEvaluationManifest.model_validate_json(
        (run_dir / "reconstruction/completion/evaluation_manifest.json").read_text()
    )
    assert evaluation.transforms_frozen_before_heldout_evaluation
    assert all(item.heldout_frame_ids for item in evaluation.evaluations)
    selection = CandidateSelectionArtifact.model_validate_json(
        (run_dir / "reconstruction/completion/selection.json").read_text()
    )
    assert any(item.best_research_candidate is not None for item in selection.objects)
    assert all(item.best_production_eligible_candidate is None for item in selection.objects)
    report = Phase5BConsistencyReport.model_validate_json(
        (run_dir / "validation/phase5b_rigid_completion.json").read_text()
    )
    assert report.passed
    scene = SceneIR.model_validate_json((run_dir / "scene_ir/phase5b_scene.json").read_text())
    assert any(asset.geometry_status == "partial_measured" for asset in scene.geometry_assets)
    assert any(
        asset.geometry_status == "complete_visual_candidate" for asset in scene.geometry_assets
    )
    second = runner.run(resume=True)
    assert all(stage["last_execution"] == "cache_hit" for stage in second["stages"].values())


def test_fake_phase5b_keeps_same_label_instances_and_heldout_evidence_separate(
    phase5b_run: Path,
) -> None:
    split = CompletionEvidenceSplit.model_validate_json(
        (phase5b_run / "reconstruction/completion/evidence_split.json").read_text()
    )
    request = json.loads(
        (phase5b_run / "reconstruction/completion/evidence_request.json").read_text()
    )
    split_by_id = {item.object_id: item for item in split.objects}
    for object_id, inputs in request["object_inputs"].items():
        heldout = set(split_by_id[object_id].heldout_validation_frames)
        assert not heldout & set(inputs["training_masks"])
        assert not heldout & set(inputs["training_dense_maps"])
        training = json.loads(
            (
                phase5b_run / f"reconstruction/completion/evidence/{object_id}/"
                "training_measured_geometry.json"
            ).read_text()
        )
        assert training["validated_point_count"] > 0
        assert training["renderer_control_face_count"] > 0
        assert len(training["point_cloud_sha256"]) == 64
        assert len(training["normal_sha256"]) == 64
        assert not heldout & set(training["supporting_fitting_views"])
    generation = CandidateGenerationManifest.model_validate_json(
        (phase5b_run / "reconstruction/completion/trellis2_generation_manifest.json").read_text()
    )
    same_label = [item for item in generation.candidates if item.semantic_label == "table"]
    assert len({item.object_id for item in same_label}) >= 2
    assert len({item.candidate_id for item in same_label}) == len(same_label)
    measured = CandidateGenerationManifest.model_validate_json(
        (phase5b_run / "reconstruction/completion/measured_generation_manifest.json").read_text()
    )
    for candidate in measured.candidates:
        assert (
            candidate.anchor_frame_id
            == split_by_id[candidate.object_id].generation_anchor_frames[0]
        )


def test_sam_gaussian_first_does_not_override_explicit_visual_glb(
    phase5b_run: Path,
) -> None:
    generation = CandidateGenerationManifest.model_validate_json(
        (phase5b_run / "reconstruction/completion/sam3d_generation_manifest.json").read_text()
    )
    evaluation = CandidateEvaluationManifest.model_validate_json(
        (phase5b_run / "reconstruction/completion/evaluation_manifest.json").read_text()
    )
    evaluated = {item.candidate_id: item for item in evaluation.evaluations}
    sam_candidates = [
        candidate
        for candidate in generation.candidates
        if candidate.native_assets[0].asset_id == "native_gaussian"
    ]
    assert sam_candidates
    for candidate in sam_candidates:
        assert candidate.evaluation_asset_id == "official_visual_glb"
        assert candidate.selection_asset_id == "official_visual_glb"
        item = evaluated[candidate.candidate_id]
        assert item.evaluation_asset_id == "official_visual_glb"
        assert item.selection_asset_id == "official_visual_glb"
        assert item.evaluation_asset_path == item.selection_asset_path


def test_candidate_signature_uses_only_declared_generation_inputs(
    phase5b_run: Path,
) -> None:
    runner = PipelineRunner(load_config(CONFIG), INPUT, phase5b_run)
    manifest = runner.load_manifest()
    adapter = REGISTRY["sam3d_object_candidates"]()
    before, inputs_before = runner._stage_signature(  # noqa: SLF001
        "sam3d_object_candidates",
        adapter,
        manifest,
    )

    evidence = CompletionEvidencePackage.model_validate_json(
        (phase5b_run / "reconstruction/completion/evidence/evidence_package.json").read_text()
    )
    training_points_path = evidence.objects[0].training_points_path
    assert training_points_path is not None
    training_points = phase5b_run / training_points_path
    original_training = training_points.read_bytes()
    training_points.write_bytes(original_training + b"\n")
    try:
        after_training_change, _ = runner._stage_signature(  # noqa: SLF001
            "sam3d_object_candidates",
            adapter,
            manifest,
        )
    finally:
        training_points.write_bytes(original_training)
    assert after_training_change == before
    assert "declared_inputs" in inputs_before
    assert "upstream_execution_signatures" not in inputs_before

    crops = CompletionCropManifest.model_validate_json(
        (phase5b_run / "reconstruction/completion/crop_manifest.json").read_text()
    )
    crop = phase5b_run / crops.anchors[0].crop_path
    original_crop = crop.read_bytes()
    crop.write_bytes(original_crop + b"\n")
    changed_manifest = json.loads(json.dumps(manifest))
    crop_record = next(
        record
        for record in changed_manifest["stages"]["completion_evidence_package"]["artifacts"]
        if record["relative_path"] == crops.anchors[0].crop_path
    )
    crop_record["sha256"] = hashlib.sha256(crop.read_bytes()).hexdigest()
    crop_record["size_bytes"] = crop.stat().st_size
    try:
        after_crop_change, _ = runner._stage_signature(  # noqa: SLF001
            "sam3d_object_candidates",
            adapter,
            changed_manifest,
        )
    finally:
        crop.write_bytes(original_crop)
    assert after_crop_change != before


def test_core_candidate_request_matches_both_official_worker_schemas(
    phase5b_run: Path,
) -> None:
    generation = CandidateGenerationManifest.model_validate_json(
        (phase5b_run / "reconstruction/completion/sam3d_generation_manifest.json").read_text()
    )
    payload = generation.requests[0].model_dump_json()
    worker_roots = [
        ROOT / "workers/sam3d_objects",
        ROOT / "workers/trellis2_objects",
    ]
    sys.path[:0] = [str(path) for path in worker_roots]
    try:
        from sam3d_objects_worker.schema import CandidateRequest as SamRequest
        from trellis2_objects_worker.schema import CandidateRequest as TrellisRequest
    finally:
        del sys.path[: len(worker_roots)]
    assert SamRequest.model_validate_json(payload).run_id
    assert TrellisRequest.model_validate_json(payload).run_id
    request = generation.requests[0]
    worker = CompletionWorkerManifest(
        worker_name="identity-test",
        worker_version="0.1.0",
        action="generate",
        backend="fake",
        request_sha256="0" * 64,
        official_repository=request.official_repository,
        official_code_commit="0" * 40,
        checkpoint_repository=request.checkpoint_repository,
        checkpoint_revision=request.checkpoint_revision,
        checkpoint_hashes=request.checkpoint_hashes,
        runtime_model_revisions=request.runtime_model_revisions,
        runtime_model_hashes=request.runtime_model_hashes,
        runtime_seconds=0,
    )
    with pytest.raises(RuntimeError, match="model identity mismatch"):
        validate_worker_model_identity(worker, request)


def test_completion_cli_inspection_export_and_validation(
    tmp_path: Path,
    phase5b_run: Path,
) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["completion", "inspect", str(phase5b_run)])
    assert result.exit_code == 0
    assert "selected_research_count" in result.stdout
    selection = CandidateSelectionArtifact.model_validate_json(
        (phase5b_run / "reconstruction/completion/selection.json").read_text()
    )
    selected = next(item for item in selection.objects if item.selected_candidate is not None)
    result = runner.invoke(
        app,
        [
            "completion",
            "explain-selection",
            str(phase5b_run),
            selected.object_id,
        ],
    )
    assert result.exit_code == 0
    output = tmp_path / "candidate.asset"
    result = runner.invoke(
        app,
        [
            "completion",
            "export-selected",
            str(phase5b_run),
            selected.object_id,
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0
    assert output.is_file()
    result = runner.invoke(app, ["validation", "verify-phase5b", str(phase5b_run)])
    assert result.exit_code == 0


@pytest.fixture
def phase5b_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("phase5b-cli")
    PipelineRunner(load_config(CONFIG), INPUT, run_dir).run()
    return run_dir
