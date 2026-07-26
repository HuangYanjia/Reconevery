from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import recon2sim.adapters.object_lifting as object_lifting_adapter_module
from recon2sim.adapters.base import StageContext
from recon2sim.adapters.object_lifting import (
    ObjectLiftingAdapterConfig,
    ObjectSurfaceLiftingAdapter,
)
from recon2sim.artifacts import (
    CameraMeshAlignmentArtifact,
    ObjectSurfaceEvidenceArtifact,
    ObjectSurfaceLiftingRequest,
    ObjectSurfaceMethodComparison,
    Phase4ConsistencyReport,
)
from recon2sim.cli import app
from recon2sim.config import PipelineConfig, load_config
from recon2sim.ir import GeometrySourceType, SceneIR
from recon2sim.object_lifting import (
    coordinate_metadata_is_raw_colmap,
    read_compact_face_ids,
    render_summary_previews,
    write_compact_face_ids,
)
from recon2sim.pipeline import PipelineRunner

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "examples" / "tabletop"
FAKE_CONFIG = ROOT / "configs" / "phase4_e2e_fake.yaml"
WORKER_ROOT = ROOT / "workers" / "object_lifting"
sys.path.insert(0, str(WORKER_ROOT))

from object_lifting_worker.camera_projection import (  # noqa: E402
    camera_from_world,
    homogeneous_clip_coordinates,
    ndc_to_pixel,
    pixel_to_ndc,
    project_pinhole,
    transform_world_point_to_camera,
)
from object_lifting_worker.distortion import (  # noqa: E402
    distort_normalized,
    distortion_coefficients,
)
from object_lifting_worker.face_evidence import FaceStatistics  # noqa: E402
from object_lifting_worker.inference import _resolve_overlaps  # noqa: E402
from object_lifting_worker.rasterization import (  # noqa: E402
    cpu_rasterize_face_ids,
    triangle_outside_clip,
)
from object_lifting_worker.schema import WorkerRequest  # noqa: E402
from object_lifting_worker.surface_extraction import (  # noqa: E402
    connected_face_components,
    filter_components,
)


def _config(mode: str = "success", timeout_s: float = 30) -> PipelineConfig:
    config = load_config(FAKE_CONFIG).model_copy(deep=True)
    stage = config.stages["object_surface_lifting"]
    stage.adapter.config["fake_mode"] = mode
    stage.adapter.timeout_s = timeout_s
    return config


def _run(
    tmp_path: Path,
    *,
    mode: str = "success",
    timeout_s: float = 30,
) -> tuple[Path, dict[str, object]]:
    run_dir = tmp_path / f"run-{mode}"
    manifest = PipelineRunner(_config(mode, timeout_s), INPUT, run_dir).run()
    return run_dir, manifest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fake_phase4_dag_and_consistency_report(tmp_path: Path) -> None:
    run_dir, manifest = _run(tmp_path)
    assert all(
        stage["status"] == "succeeded"
        for stage in manifest["stages"].values()  # type: ignore[union-attr]
    )
    evidence = ObjectSurfaceEvidenceArtifact.model_validate_json(
        (run_dir / "reconstruction/object_surfaces/evidence_manifest.json").read_text()
    )
    comparison = ObjectSurfaceMethodComparison.model_validate_json(
        (run_dir / "reconstruction/object_surfaces/method_comparison.json").read_text()
    )
    alignment = CameraMeshAlignmentArtifact.model_validate_json(
        (run_dir / "reconstruction/object_surfaces/camera_mesh_alignment.json").read_text()
    )
    report = Phase4ConsistencyReport.model_validate_json(
        (run_dir / "validation/phase4_object_surface_consistency.json").read_text()
    )
    assert report.passed
    assert report.real_2d_tracks_lifted_to_global_3d
    assert not report.hidden_surface_completion_implemented
    assert not report.sim_ready_scene_implemented
    assert evidence.hypotheses
    assert comparison.selected_method == "surface_sample_fusion_v2"
    assert {item.method for item in comparison.metrics} == {
        "exact_face_vote_v1",
        "surface_sample_fusion_v2",
    }
    assert alignment.frame_sequence_digest == evidence.frame_sequence_digest
    assert all(
        item.geometry_status == "partial_observation_supported" for item in evidence.hypotheses
    )
    scene = SceneIR.model_validate_json((run_dir / "scene_ir/phase4_scene.json").read_text())
    partial_assets = [
        asset
        for asset in scene.geometry_assets
        if asset.geometry_status == "partial_observation_supported"
    ]
    assert partial_assets
    assert all(asset.source is GeometrySourceType.FUSED for asset in partial_assets)
    assert all(asset.sim_ready is False for asset in partial_assets)
    assert not scene.collision_assets


def test_fake_phase4_resume_hits_all_stages(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    resumed = PipelineRunner(_config(), INPUT, run_dir).run(resume=True)
    assert {
        stage["last_execution"]
        for stage in resumed["stages"].values()  # type: ignore[union-attr]
    } == {"cache_hit"}


def test_prompt_change_invalidates_sam_and_lifting_not_genrecon(
    tmp_path: Path,
) -> None:
    config = _config()
    prompt = tmp_path / "prompts.yaml"
    shutil.copy2(ROOT / "configs/prompts/tabletop.yaml", prompt)
    config.stages["segmentation_tracking"].adapter.config["prompt_manifest"] = str(prompt)
    run_dir = tmp_path / "prompt-run"
    PipelineRunner(config, INPUT, run_dir).run()
    prompt.write_text(prompt.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    resumed = PipelineRunner(config, INPUT, run_dir).run(resume=True)
    assert resumed["stages"]["segmentation_tracking"]["last_execution"] == "executed"
    assert resumed["stages"]["global_reconstruction"]["last_execution"] == "cache_hit"
    assert resumed["stages"]["object_surface_lifting"]["last_execution"] == "executed"
    assert resumed["stages"]["phase4_consistency_validation"]["last_execution"] == "executed"


def test_selective_materialization_excludes_raw_workspaces(tmp_path: Path) -> None:
    run_dir = tmp_path / "selective"
    PipelineRunner(_config(), INPUT, run_dir).run(until_stage="global_reconstruction")
    mesh_path = run_dir / "reconstruction/global/mesh.ply"
    mesh_hash_before = _sha(mesh_path)
    manifest = PipelineRunner(_config(), INPUT, run_dir).run(from_stage="object_surface_lifting")
    attempts = manifest["stages"]["object_surface_lifting"]["attempts"]  # type: ignore[index]
    materialized = {item["relative_path"] for item in attempts[-1]["materialized_inputs"]}
    assert "inputs/manifest.json" in materialized
    assert "camera/reconstruction.json" in materialized
    assert "observations/object_tracks.json" in materialized
    assert "reconstruction/global/metadata.json" in materialized
    assert "reconstruction/global/mesh.ply" in materialized
    assert "camera/genrecon_package/package_manifest.json" in materialized
    assert "camera/genrecon_package/images.txt" in materialized
    assert "camera/genrecon_package/points3D.txt" in materialized
    assert "camera/genrecon_package/registered_frames.json" in materialized
    assert any(path.startswith("observations/masks/") for path in materialized)
    assert "camera/colmap/database.db" not in materialized
    assert not any(path.startswith("camera/colmap/") for path in materialized)
    assert not any(path.startswith("observations/raw/") for path in materialized)
    assert not any(path.startswith("reconstruction/global/raw/") for path in materialized)
    attempt_root = (
        run_dir / "work" / "object_surface_lifting" / f"attempt_{attempts[-1]['attempt']}"
    )
    attempt_mesh = attempt_root / "reconstruction/global/mesh.ply"
    assert attempt_mesh.is_file()
    assert _sha(attempt_mesh) == mesh_hash_before
    assert not (attempt_root / "reconstruction/global/scene.glb").exists()
    for forbidden in (
        "camera/colmap/database.db",
        "camera/colmap/sparse",
        "camera/colmap/logs",
        "observations/raw",
        "reconstruction/global/raw",
    ):
        assert not (attempt_root / forbidden).exists()
    mesh_record = next(
        item
        for item in attempts[-1]["materialized_inputs"]
        if item["relative_path"] == "reconstruction/global/mesh.ply"
    )
    assert mesh_record["materialization_mode"] == "reflink_or_copy"
    assert _sha(mesh_path) == mesh_hash_before


def test_faulty_worker_cannot_modify_canonical_mesh(tmp_path: Path) -> None:
    run_dir = tmp_path / "upstream-immutability"
    PipelineRunner(_config(), INPUT, run_dir).run(until_stage="global_reconstruction")
    mesh_path = run_dir / "reconstruction/global/mesh.ply"
    before = mesh_path.read_bytes()
    PipelineRunner(_config("modify_upstream"), INPUT, run_dir).run(
        from_stage="object_surface_lifting"
    )
    assert mesh_path.read_bytes() == before


def test_local_and_docker_workers_receive_only_attempt_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = tmp_path / "work/object_surface_lifting/attempt_1"
    canonical = tmp_path / "canonical"
    attempt.mkdir(parents=True)
    canonical.mkdir()
    stage_config = _config().stages["object_surface_lifting"]
    context = StageContext(
        stage_name="object_surface_lifting",
        input_dir=INPUT,
        run_dir=attempt,
        canonical_run_dir=canonical,
        config=stage_config,
        seed=42,
    )
    adapter = ObjectSurfaceLiftingAdapter()
    local = adapter._inference_command(
        context,
        ObjectLiftingAdapterConfig.model_validate(stage_config.adapter.config),
    )
    assert local[local.index("--input-root") + 1] == str(attempt.resolve())
    assert str(canonical.resolve()) not in local

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        object_lifting_adapter_module,
        "resolve_executable",
        lambda _name: "/usr/bin/docker",
    )
    docker = adapter._inference_command(
        context,
        ObjectLiftingAdapterConfig.model_validate(
            {
                "execution_mode": "docker",
                "docker_image": "reconevery/object-lifting:test",
                "device": "cuda",
            }
        ),
    )
    mounts = [docker[index + 1] for index, value in enumerate(docker) if value == "-v"]
    assert mounts == [f"{attempt.resolve()}:/workspace:rw"]
    assert str(canonical.resolve()) not in docker


def test_compact_face_ids_round_trip_and_bounds(tmp_path: Path) -> None:
    path = tmp_path / "faces.bin"
    manifest = write_compact_face_ids(
        path,
        [1, 7, 42],
        global_mesh_sha256="a" * 64,
        relative_path="faces.bin",
    )
    assert manifest.dtype == "uint32"
    assert read_compact_face_ids(tmp_path, manifest, global_face_count=43) == (1, 7, 42)
    with pytest.raises(ValueError, match="exceeds"):
        read_compact_face_ids(tmp_path, manifest, global_face_count=42)


def test_compact_face_ids_uses_uint64_when_needed(tmp_path: Path) -> None:
    manifest = write_compact_face_ids(
        tmp_path / "faces.bin",
        [2**32 + 1],
        global_mesh_sha256="b" * 64,
        relative_path="faces.bin",
    )
    assert manifest.dtype == "uint64"
    assert read_compact_face_ids(tmp_path, manifest) == (2**32 + 1,)


def test_compact_face_ids_rejects_unsorted_and_corruption(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sorted and unique"):
        write_compact_face_ids(
            tmp_path / "bad.bin",
            [2, 1],
            global_mesh_sha256="c" * 64,
            relative_path="bad.bin",
        )
    manifest = write_compact_face_ids(
        tmp_path / "faces.bin",
        [1, 2],
        global_mesh_sha256="d" * 64,
        relative_path="faces.bin",
    )
    (tmp_path / "faces.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_compact_face_ids(tmp_path, manifest)


def test_request_rejects_bad_lineage_and_unsafe_paths(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    payload = json.loads((run_dir / "reconstruction/object_surfaces/request.json").read_text())
    ObjectSurfaceLiftingRequest.model_validate(payload)
    payload["registered_frame_ids"] = payload["registered_frame_ids"][:-1]
    with pytest.raises(ValueError, match="cover master frames"):
        ObjectSurfaceLiftingRequest.model_validate(payload)
    payload = json.loads((run_dir / "reconstruction/object_surfaces/request.json").read_text())
    payload["normalized_frame_paths"][payload["master_frame_order"][0]] = "../escape.png"
    with pytest.raises(ValueError, match="relative"):
        ObjectSurfaceLiftingRequest.model_validate(payload)


def test_raw_colmap_coordinate_contract(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    request = ObjectSurfaceLiftingRequest.model_validate_json(
        (run_dir / "reconstruction/object_surfaces/request.json").read_text()
    )
    assert coordinate_metadata_is_raw_colmap(request.coordinate_convention)
    assert request.coordinate_convention.linear_units.value == "arbitrary_units"
    assert request.coordinate_convention.scale_status.value == "scale_ambiguous"


def test_camera_inversion_identity_and_translation() -> None:
    rotation, translation = camera_from_world((0, 0, 0), (0, 0, 0, 1))
    assert rotation == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert translation == (0.0, 0.0, 0.0)
    assert transform_world_point_to_camera(
        (1, 0, 2),
        (1, 0, 0),
        (0, 0, 0, 1),
    ) == pytest.approx((0, 0, 2))


def test_camera_known_90_degree_rotation_and_arbitrary_world() -> None:
    sine = math.sqrt(0.5)
    point = transform_world_point_to_camera(
        (0, 1, 2),
        (0, 0, 0),
        (0, 0, sine, sine),
    )
    assert point == pytest.approx((1, 0, 2), abs=1e-9)
    rotated = transform_world_point_to_camera(
        (4, -2, 8),
        (4, -2, 3),
        (sine, 0, 0, sine),
    )
    assert rotated == pytest.approx((0, 5, 0), abs=1e-9)


def test_projection_roundtrip_offsets_non_square_and_behind() -> None:
    point = project_pinhole((1, 2, 4), fx=200, fy=300, cx=23, cy=17)
    assert point == pytest.approx((73, 167))
    ndc = pixel_to_ndc(point[0], point[1], 640, 360)
    assert ndc_to_pixel(*ndc, 640, 360) == pytest.approx(point)
    assert project_pinhole((0, 0, -1), fx=1, fy=1, cx=0, cy=0) is None
    assert project_pinhole((0, 0, 0), fx=1, fy=1, cx=0, cy=0) is None


def test_homogeneous_projection_exact_pixel_contract() -> None:
    clip = homogeneous_clip_coordinates(
        (1.0, 2.0, 4.0),
        fx=200.0,
        fy=300.0,
        cx=23.0,
        cy=17.0,
        width=640,
        height=360,
        near=0.1,
        far=100.0,
    )
    pixel = ndc_to_pixel(clip[0] / clip[3], clip[1] / clip[3], 640, 360)
    assert pixel == pytest.approx((73.0, 167.0))
    assert clip[3] == 4.0


@pytest.mark.parametrize(
    "triangle,expected",
    [
        ([(-0.5, -0.5, 0.0, 1.0), (0.5, -0.5, 0.0, 1.0), (0.0, 0.5, 0.0, 1.0)], False),
        ([(-0.5, -0.5, 0.0, -1.0), (0.5, -0.5, 0.0, -1.0), (0.0, 0.5, 0.0, -1.0)], True),
        ([(-2.0, 0.0, 0.0, 1.0), (-3.0, 0.5, 0.0, 1.0), (-4.0, -0.5, 0.0, 1.0)], True),
        ([(2.0, 0.0, 0.0, 1.0), (3.0, 0.5, 0.0, 1.0), (4.0, -0.5, 0.0, 1.0)], True),
        ([(0.0, 2.0, 0.0, 1.0), (0.5, 3.0, 0.0, 1.0), (-0.5, 4.0, 0.0, 1.0)], True),
        ([(0.0, -2.0, 0.0, 1.0), (0.5, -3.0, 0.0, 1.0), (-0.5, -4.0, 0.0, 1.0)], True),
        ([(-2.0, 0.0, 0.0, 1.0), (0.0, 0.5, 0.0, 1.0), (0.0, -0.5, 0.0, 1.0)], False),
        ([(-4.0, -4.0, 0.0, 1.0), (4.0, -4.0, 0.0, 1.0), (0.0, 4.0, 0.0, 1.0)], False),
    ],
)
def test_conservative_clip_culling(
    triangle: list[tuple[float, float, float, float]],
    expected: bool,
) -> None:
    assert triangle_outside_clip(triangle) is expected


def test_reference_rasterizer_clips_near_and_camera_planes() -> None:
    intrinsics = {
        "width": 40,
        "height": 24,
        "fx": 20,
        "fy": 20,
        "cx": 19.5,
        "cy": 11.5,
    }
    near_crossing = cpu_rasterize_face_ids(
        [(-0.5, -0.5, 0.05), (0.5, -0.5, 1.0), (0.0, 0.5, 1.0)],
        [(0, 1, 2)],
        translation_world_from_camera=(0, 0, 0),
        rotation_xyzw_world_from_camera=(0, 0, 0, 1),
        intrinsics=intrinsics,
        near_plane=0.1,
        far_plane=10.0,
    )
    assert 0 in {value for row in near_crossing for value in row}
    camera_crossing = cpu_rasterize_face_ids(
        [(-0.2, -0.2, -0.1), (0.5, -0.5, 1.0), (0.0, 0.5, 1.0)],
        [(0, 1, 2)],
        translation_world_from_camera=(0, 0, 0),
        rotation_xyzw_world_from_camera=(0, 0, 0, 1),
        intrinsics=intrinsics,
        near_plane=0.1,
        far_plane=10.0,
    )
    assert 0 in {value for row in camera_crossing for value in row}
    behind = cpu_rasterize_face_ids(
        [(-0.5, -0.5, -1.0), (0.5, -0.5, -1.0), (0.0, 0.5, -1.0)],
        [(0, 1, 2)],
        translation_world_from_camera=(0, 0, 0),
        rotation_xyzw_world_from_camera=(0, 0, 0, 1),
        intrinsics=intrinsics,
        near_plane=0.1,
        far_plane=10.0,
    )
    assert 0 not in {value for row in behind for value in row}


def test_distortion_camera_models() -> None:
    assert distortion_coefficients("PINHOLE", [1.0]) == (0, 0, 0, 0, 0)
    assert distortion_coefficients("SIMPLE_RADIAL", [0.1]) == (0.1, 0, 0, 0, 0)
    assert distortion_coefficients("RADIAL", [0.1, -0.01]) == (
        0.1,
        -0.01,
        0,
        0,
        0,
    )
    assert distort_normalized(1.0, 0.0, "SIMPLE_RADIAL", [0.1]) == pytest.approx((1.1, 0.0))
    assert distort_normalized(1.0, 1.0, "OPENCV", [0, 0, 0.01, 0.02]) == pytest.approx((1.10, 1.08))
    with pytest.raises(ValueError, match="unsupported camera model"):
        distortion_coefficients("FOV", [])


def test_reference_rasterizer_exact_face_and_occlusion() -> None:
    vertices = [
        (-1.0, -1.0, 4.0),
        (1.0, -1.0, 4.0),
        (0.0, 1.0, 4.0),
        (-1.0, -1.0, 2.0),
        (1.0, -1.0, 2.0),
        (0.0, 1.0, 2.0),
    ]
    faces = [(0, 1, 2), (3, 4, 5)]
    raster = cpu_rasterize_face_ids(
        vertices,
        faces,
        translation_world_from_camera=(0, 0, 0),
        rotation_xyzw_world_from_camera=(0, 0, 0, 1),
        intrinsics={"width": 32, "height": 24, "fx": 16, "fy": 16, "cx": 16, "cy": 12},
    )
    assert raster[12][16] == 1
    assert 0 not in {value for row in raster for value in row}


def test_reference_rasterizer_preserves_face_ids_under_world_rotation() -> None:
    sine = math.sqrt(0.5)
    raster = cpu_rasterize_face_ids(
        [(1.0, -1.0, 4.0), (1.0, 1.0, 4.0), (-1.0, 0.0, 4.0)],
        [(0, 1, 2)],
        translation_world_from_camera=(0, 0, 0),
        rotation_xyzw_world_from_camera=(0, 0, sine, sine),
        intrinsics={"width": 32, "height": 32, "fx": 16, "fy": 16, "cx": 16, "cy": 16},
    )
    assert 0 in {value for row in raster for value in row}


def test_connected_components_and_tiny_component_filtering() -> None:
    vertices = [
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (2.0, 2.0, 0.0),
        (10.0, 0.0, 0.0),
        (10.1, 0.0, 0.0),
        (10.0, 0.1, 0.0),
    ]
    faces = [(0, 1, 2), (2, 1, 3), (4, 5, 6)]
    assert connected_face_components(faces, [0, 1, 2]) == [[0, 1], [2]]
    retained, diagnostics = filter_components(
        vertices,
        faces,
        [0, 1, 2],
        min_faces=2,
        min_relative_area=0.5,
    )
    assert retained == [0, 1]
    assert diagnostics[0]["retained"]
    assert not diagnostics[1]["retained"]
    assert diagnostics[0]["relative_face_ratio"] == pytest.approx(2 / 3)
    assert diagnostics[0]["relative_surface_area"] > 0.99


def test_same_label_exclusivity_and_cross_label_overlap() -> None:
    tracks = [
        SimpleNamespace(object_id="box_0001", semantic_label="box"),
        SimpleNamespace(object_id="box_0002", semantic_label="box"),
        SimpleNamespace(object_id="drawer_0001", semantic_label="drawer"),
    ]
    request = SimpleNamespace(
        object_tracks=tracks,
        face_evidence_configuration={"instance_score_margin": 0.05},
    )
    stats = {
        "box_0001": {5: FaceStatistics(support_score=0.9)},
        "box_0002": {5: FaceStatistics(support_score=0.7)},
        "drawer_0001": {5: FaceStatistics(support_score=0.8)},
    }
    accepted = {"box_0001": [5], "box_0002": [5], "drawer_0001": [5]}
    ambiguous = {"box_0001": [], "box_0002": [], "drawer_0001": []}
    conflicts, same, different = _resolve_overlaps(
        request,  # type: ignore[arg-type]
        stats,
        accepted,
        ambiguous,
    )
    assert accepted["box_0001"] == [5]
    assert accepted["box_0002"] == []
    assert accepted["drawer_0001"] == [5]
    assert same == {5}
    assert different == {5}
    assert {item["conflict_type"] for item in conflicts} == {
        "same_class_instance",
        "different_semantic_label",
    }


@pytest.mark.parametrize(
    "mode,pattern",
    [
        ("wrong_mesh_hash", "global_mesh_sha256"),
        ("wrong_segmentation_hash", "segmentation_tracking_sha256"),
        ("wrong_camera_hash", "camera_reconstruction_sha256"),
        ("coordinate_mismatch", "coordinate semantics"),
        ("corrupt_face_array", "byte count mismatch"),
        ("nonfinite_surface", "non-finite"),
        ("malformed_manifest", "malformed"),
        ("path_escape", "relative to the run directory"),
        ("nonzero_exit", "return code"),
        ("oom", "out of GPU memory"),
        ("rasterizer_failure", "rasterizer failed"),
        ("unsupported_camera", "unsupported camera model"),
    ],
)
def test_fake_worker_failure_modes(
    tmp_path: Path,
    mode: str,
    pattern: str,
) -> None:
    with pytest.raises((RuntimeError, ValueError), match=pattern):
        PipelineRunner(_config(mode), INPUT, tmp_path / mode).run()


def test_fake_worker_timeout(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="timed out"):
        PipelineRunner(_config("timeout", 0.05), INPUT, tmp_path / "timeout").run()


def test_unresolved_objects_are_valid_success(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path, mode="unresolved")
    evidence = ObjectSurfaceEvidenceArtifact.model_validate_json(
        (run_dir / "reconstruction/object_surfaces/evidence_manifest.json").read_text()
    )
    assert evidence.hypotheses
    assert all(item.status == "unresolved" for item in evidence.hypotheses)
    assert all(item.accepted_global_face_ids.count == 0 for item in evidence.hypotheses)


def test_failed_attempt_preserves_previous_canonical_outputs(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    evidence_path = run_dir / "reconstruction/object_surfaces/evidence_manifest.json"
    before = evidence_path.read_bytes()
    with pytest.raises(RuntimeError):
        PipelineRunner(_config("nonzero_exit"), INPUT, run_dir).run()
    assert evidence_path.read_bytes() == before


def test_fake_healthcheck_and_real_backend_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    context = StageContext(
        stage_name="object_surface_lifting",
        input_dir=INPUT,
        run_dir=tmp_path,
        canonical_run_dir=tmp_path,
        config=config.stages["object_surface_lifting"],
        seed=42,
    )
    result = ObjectSurfaceLiftingAdapter().healthcheck(context)
    assert result.ok
    assert "fake" in result.message
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        ObjectLiftingAdapterConfig.model_validate(
            {
                "execution_mode": "local_worker",
                "worker_python": sys.executable,
                "device": "cuda",
            }
        )


def test_worker_schema_rejects_coordinate_mismatch(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    payload = json.loads((run_dir / "reconstruction/object_surfaces/request.json").read_text())
    WorkerRequest.model_validate(payload)
    payload["coordinate_convention"]["linear_units"] = "meters"
    with pytest.raises(ValueError, match="raw COLMAP"):
        WorkerRequest.model_validate(payload)


def test_cli_inspect_export_and_verify(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    runner = CliRunner()
    summary = runner.invoke(app, ["objects", "inspect-surfaces", str(run_dir)])
    assert summary.exit_code == 0
    assert '"sim_ready": false' in summary.stdout
    detail = runner.invoke(
        app,
        ["objects", "inspect-surface", str(run_dir), "table_0001"],
    )
    assert detail.exit_code == 0
    output_mesh = tmp_path / "surface.ply"
    exported = runner.invoke(
        app,
        [
            "objects",
            "export-surface",
            str(run_dir),
            "table_0001",
            "--output",
            str(output_mesh),
        ],
    )
    assert exported.exit_code == 0
    assert output_mesh.is_file()
    output_ids = tmp_path / "faces.bin"
    exported_ids = runner.invoke(
        app,
        [
            "objects",
            "export-face-ids",
            str(run_dir),
            "table_0001",
            "--output",
            str(output_ids),
        ],
    )
    assert exported_ids.exit_code == 0
    assert output_ids.is_file()
    verified = runner.invoke(app, ["validation", "verify-phase4", str(run_dir)])
    assert verified.exit_code == 0


def test_preview_regeneration_is_deterministic(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    artifact = ObjectSurfaceEvidenceArtifact.model_validate_json(
        (run_dir / "reconstruction/object_surfaces/evidence_manifest.json").read_text()
    )
    render_summary_previews(run_dir, artifact)
    first = {
        path.name: path.read_bytes()
        for path in (run_dir / "reconstruction/object_surfaces/previews").glob("*.png")
    }
    render_summary_previews(run_dir, artifact)
    second = {
        path.name: path.read_bytes()
        for path in (run_dir / "reconstruction/object_surfaces/previews").glob("*.png")
    }
    assert first == second
