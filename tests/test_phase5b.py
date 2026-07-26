from __future__ import annotations

import hashlib
import importlib.util
import json
import random
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
    CandidateSelectionArtifact,
    CompletionCropManifest,
    CompletionEligibilityStatus,
    CompletionEvidencePackage,
    CompletionEvidenceSplit,
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
from recon2sim.config import load_config
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


def test_measured_baseline_uses_point_rendering_even_with_ply_asset() -> None:
    pytest.importorskip("numpy")
    worker_root = ROOT / "workers/completion_evaluation"
    sys.path.insert(0, str(worker_root))
    try:
        from completion_evaluation_worker.inference import _mesh_render_asset
    finally:
        sys.path.remove(str(worker_root))
    measured = {
        "backend": "measured_partial_baseline",
        "native_assets": [{"format": "mesh_ply", "relative_path": "measured_points.ply"}],
    }
    generated = {
        "backend": "sam3d_objects",
        "native_assets": [{"format": "pbr_glb", "relative_path": "visual_asset.glb"}],
    }
    assert _mesh_render_asset(measured) is None
    assert _mesh_render_asset(generated) == generated["native_assets"][0]


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
    assert {candidate.native_assets[0].format.value for candidate in sam.candidates} == {
        "gaussian_splat_ply"
    }
    assert {candidate.native_assets[0].format.value for candidate in trellis.candidates} == {
        "pbr_glb"
    }
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
    generation = CandidateGenerationManifest.model_validate_json(
        (phase5b_run / "reconstruction/completion/trellis2_generation_manifest.json").read_text()
    )
    same_label = [item for item in generation.candidates if item.semantic_label == "table"]
    assert len({item.object_id for item in same_label}) >= 2
    assert len({item.candidate_id for item in same_label}) == len(same_label)


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
