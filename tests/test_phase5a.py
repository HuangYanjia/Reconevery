from __future__ import annotations

import json
import shutil
import struct
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from typer.testing import CliRunner

from recon2sim.adapters.base import StageContext
from recon2sim.adapters.dense_mvs import DenseMVSAdapter, DenseMVSAdapterConfig
from recon2sim.adapters.measured_geometry import (
    MeasuredGeometryAdapterConfig,
    MeasuredObjectGeometryAdapter,
)
from recon2sim.artifacts import (
    DenseDepthManifest,
    DenseFusionArtifact,
    DenseMVSRequest,
    MeasuredObjectGeometryArtifact,
    Phase5AConsistencyReport,
)
from recon2sim.cli import app
from recon2sim.config import AdapterConfig, StageConfig, load_config
from recon2sim.dense_mvs import (
    DenseMapFormatError,
    iter_consistency_graph,
    ply_counts,
    read_dense_array,
)
from recon2sim.ir import GeometrySourceType, SceneIR
from recon2sim.measured_geometry import (
    backproject_pixel_world,
    observed_triangle_is_local,
    project_world_pixel,
    relative_depth_agrees,
)
from recon2sim.pipeline import PipelineRunner

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "examples" / "tabletop"
CONFIG = ROOT / "configs" / "phase5a_e2e_fake.yaml"


def _patchmatch_writer() -> object:
    path = ROOT / "workers/dense_mvs/dense_mvs_worker/patchmatch.py"
    spec = spec_from_file_location("phase5a_patchmatch", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.write_patch_match_config


def _run(tmp_path: Path, *, measured_mode: str = "success") -> tuple[Path, dict[str, object]]:
    config = load_config(CONFIG).model_copy(deep=True)
    config.stages["measured_object_geometry"].adapter.config["fake_mode"] = measured_mode
    run_dir = tmp_path / f"phase5a-{measured_mode}"
    return run_dir, PipelineRunner(config, INPUT, run_dir).run()


def test_colmap_dense_array_parser_preserves_column_major_layout(tmp_path: Path) -> None:
    path = tmp_path / "depth.bin"
    path.write_bytes(b"2&2&1&" + struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))
    dense = read_dense_array(path, expected_channels=1)
    assert (dense.width, dense.height, dense.channels) == (2, 2, 1)
    assert dense.value(0, 0) == 1.0
    assert dense.value(1, 0) == 2.0
    assert dense.value(0, 1) == 3.0
    assert dense.value(1, 1) == 4.0


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"2&2&1&" + b"\0" * 4, "expected 16"),
        (b"2&2&3&" + b"\0" * 48, "expected 1"),
        (b"2&x&1&", "invalid"),
    ],
)
def test_colmap_dense_array_rejects_malformed_data(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    path = tmp_path / "bad.bin"
    path.write_bytes(payload)
    with pytest.raises(DenseMapFormatError, match=message):
        read_dense_array(path, expected_channels=1)


def test_colmap_consistency_graph_parser_and_index_validation(tmp_path: Path) -> None:
    graph = tmp_path / "graph.bin"
    graph.write_bytes(b"4&3&1&" + struct.pack("<6i", 3, 2, 3, 0, 1, 2))
    entry = list(iter_consistency_graph(graph, image_count=3))[0]
    assert (entry.row, entry.column) == (2, 3)
    assert entry.source_image_indices == (
        0,
        1,
        2,
    )
    graph.write_bytes(b"4&3&1&" + struct.pack("<4i", 1, 2, 1, 3))
    with pytest.raises(DenseMapFormatError, match="invalid image indices"):
        list(iter_consistency_graph(graph, image_count=3))


@pytest.mark.parametrize("vertex", ["nan 0 0", "0 0"])
def test_ply_validation_rejects_nonfinite_or_truncated_vertices(
    tmp_path: Path, vertex: str
) -> None:
    path = tmp_path / "bad.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 0\nproperty list uchar int vertex_indices\n"
        f"end_header\n{vertex}\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="non-finite|truncated"):
        ply_counts(path)


def test_patchmatch_config_preserves_manifest_order_and_source_modes(tmp_path: Path) -> None:
    writer = _patchmatch_writer()
    frame_ids = ["frame_b", "frame_a", "frame_c"]
    names = {frame_id: f"{frame_id}.png" for frame_id in frame_ids}
    path = tmp_path / "patch-match.cfg"
    writer(  # type: ignore[operator]
        path,
        ordered_frame_ids=frame_ids,
        filename_by_frame=names,
        mode="sequential_neighbors",
        explicit_source_ids={},
        neighbor_count=1,
    )
    assert path.read_text().splitlines() == [
        "frame_b.png",
        "frame_a.png",
        "frame_a.png",
        "frame_b.png",
        "frame_c.png",
        "frame_a.png",
    ]
    writer(  # type: ignore[operator]
        path,
        ordered_frame_ids=frame_ids,
        filename_by_frame=names,
        mode="explicit",
        explicit_source_ids={
            "frame_b": ["frame_c"],
            "frame_a": ["frame_b"],
            "frame_c": ["frame_a"],
        },
        neighbor_count=1,
    )
    assert path.read_text().splitlines()[1::2] == [
        "frame_c.png",
        "frame_b.png",
        "frame_a.png",
    ]


@pytest.mark.parametrize(
    ("adapter", "name", "image", "extra"),
    [
        (
            DenseMVSAdapter(),
            "dense_mvs",
            "reconevery/dense-mvs:test",
            {"use_gpu": True, "executable": "colmap"},
        ),
        (
            MeasuredObjectGeometryAdapter(),
            "measured_object_geometry",
            "reconevery/measured-geometry:test",
            {},
        ),
    ],
)
def test_phase5a_docker_healthcheck_runs_inside_configured_image(
    tmp_path: Path,
    adapter: object,
    name: str,
    image: str,
    extra: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        'case "$1 $2" in\n'
        '  "version ") echo "Docker 27" ;;\n'
        '  "image inspect") echo "sha256:phase5a-image" ;;\n'
        '  "run --rm") echo \'{"available": true, "worker_version": "0.1.0"}\' ;;\n'
        "  *) exit 2 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    stage = StageConfig(
        adapter=AdapterConfig(
            name=name,
            config={
                "execution_mode": "docker",
                "docker_executable": str(docker),
                "docker_image": image,
                **extra,
            },
        )
    )
    context = StageContext(
        stage_name=name,
        input_dir=tmp_path,
        run_dir=tmp_path / "attempt",
        canonical_run_dir=tmp_path / "canonical",
        config=stage,
        seed=42,
    )
    result = adapter.healthcheck(context)  # type: ignore[attr-defined]
    assert result.ok, result.message
    assert "sha256:phase5a-image" in result.message


def test_phase5a_docker_inference_uses_image_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    docker.chmod(0o755)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    context = StageContext(
        stage_name="phase5a",
        input_dir=tmp_path,
        run_dir=tmp_path / "attempt",
        canonical_run_dir=tmp_path / "canonical",
        config=StageConfig(adapter=AdapterConfig(name="dense_mvs", config={})),
        seed=42,
    )
    dense_config = DenseMVSAdapterConfig(
        execution_mode="docker",
        docker_executable=str(docker),
        docker_image="reconevery/dense-mvs:test",
        use_gpu=True,
    )
    measured_config = MeasuredGeometryAdapterConfig(
        execution_mode="docker",
        docker_executable=str(docker),
        docker_image="reconevery/measured-geometry:test",
    )
    dense = DenseMVSAdapter()._inference_command(context, dense_config)
    measured = MeasuredObjectGeometryAdapter()._inference_command(context, measured_config)
    assert dense[dense.index("reconevery/dense-mvs:test") + 1] == "infer"
    assert measured[measured.index("reconevery/measured-geometry:test") + 1] == "infer"


def test_fake_phase5a_dag_produces_measured_geometry(tmp_path: Path) -> None:
    run_dir, manifest = _run(tmp_path)
    assert all(
        stage["status"] == "succeeded"
        for stage in manifest["stages"].values()  # type: ignore[union-attr]
    )
    request = DenseMVSRequest.model_validate_json(
        (run_dir / "reconstruction/dense/request.json").read_text()
    )
    depth = DenseDepthManifest.model_validate_json(
        (run_dir / "reconstruction/dense/depth_manifest.json").read_text()
    )
    fusion = DenseFusionArtifact.model_validate_json(
        (run_dir / "reconstruction/dense/fusion.json").read_text()
    )
    measured = MeasuredObjectGeometryArtifact.model_validate_json(
        (run_dir / "reconstruction/measured_objects/geometry_manifest.json").read_text()
    )
    report = Phase5AConsistencyReport.model_validate_json(
        (run_dir / "validation/phase5a_measured_geometry.json").read_text()
    )
    assert request.registered_frame_ids == [
        frame_id
        for frame_id in request.master_frame_order
        if frame_id not in request.unregistered_frame_ids
    ]
    assert len(depth.records) == len(request.registered_frame_ids)
    assert fusion.point_count > 0
    assert report.passed
    assert report.measured_dense_geometry_available
    assert report.measured_object_geometry_available
    assert not report.generated_geometry_used_as_source
    assert not report.hidden_surface_completion_implemented
    assert any(item.status != "unresolved" for item in measured.hypotheses)
    assert all(item.completeness_confidence == 0 for item in measured.hypotheses)


def test_fake_phase5a_resume_hits_every_stage(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    resumed = PipelineRunner(load_config(CONFIG), INPUT, run_dir).run(resume=True)
    assert {
        stage["last_execution"]
        for stage in resumed["stages"].values()  # type: ignore[union-attr]
    } == {"cache_hit"}


def test_dense_and_measured_attempts_materialize_only_declared_inputs(
    tmp_path: Path,
) -> None:
    run_dir, manifest = _run(tmp_path)
    dense_attempt = manifest["stages"]["dense_mvs"]["attempts"][-1]  # type: ignore[index]
    dense_paths = {item["relative_path"] for item in dense_attempt["materialized_inputs"]}
    assert "camera/colmap/database.db" not in dense_paths
    assert not any(path.startswith("observations/") for path in dense_paths)
    assert not any(path.startswith("reconstruction/global/") for path in dense_paths)
    assert sum(path.endswith("cameras.bin") for path in dense_paths) == 1
    measured_attempt = manifest["stages"]["measured_object_geometry"]["attempts"][-1]  # type: ignore[index]
    measured_paths = {item["relative_path"] for item in measured_attempt["materialized_inputs"]}
    assert "observations/object_tracks.json" in measured_paths
    assert any(path.startswith("observations/masks/") for path in measured_paths)
    assert not any(path.startswith("observations/raw/") for path in measured_paths)
    assert not any(path.startswith("camera/colmap/") for path in measured_paths)
    assert not any(path.startswith("reconstruction/global/") for path in measured_paths)


def test_unresolved_measured_object_is_a_successful_result(tmp_path: Path) -> None:
    run_dir, manifest = _run(tmp_path, measured_mode="unresolved")
    assert manifest["stages"]["measured_object_geometry"]["status"] == "succeeded"  # type: ignore[index]
    measured = MeasuredObjectGeometryArtifact.model_validate_json(
        (run_dir / "reconstruction/measured_objects/geometry_manifest.json").read_text()
    )
    assert measured.hypotheses
    assert {item.status for item in measured.hypotheses} == {"unresolved"}
    assert all(item.point_cloud is None for item in measured.hypotheses)


def test_scene_ir_keeps_measured_partial_assets_separate(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    scene = SceneIR.model_validate_json((run_dir / "scene_ir/phase5a_scene.json").read_text())
    measured = [
        asset for asset in scene.geometry_assets if asset.source is GeometrySourceType.MEASURED
    ]
    assert measured
    assert all(asset.geometry_status == "partial_measured" for asset in measured)
    assert all(asset.sim_ready is False for asset in measured)
    assert not scene.collision_assets


def test_phase5a_cli_inspection_and_export(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    runner = CliRunner()
    dense = runner.invoke(app, ["dense", "inspect", str(run_dir)])
    assert dense.exit_code == 0
    assert '"fused_points": 4' in dense.stdout
    measured = runner.invoke(app, ["objects", "inspect-measured", str(run_dir)])
    assert measured.exit_code == 0
    artifact = MeasuredObjectGeometryArtifact.model_validate_json(
        (run_dir / "reconstruction/measured_objects/geometry_manifest.json").read_text()
    )
    object_id = next(item.object_id for item in artifact.hypotheses if item.point_cloud)
    output = tmp_path / "exported.ply"
    export = runner.invoke(
        app,
        [
            "objects",
            "export-measured-points",
            str(run_dir),
            object_id,
            "--output",
            str(output),
        ],
    )
    assert export.exit_code == 0
    assert output.is_file()
    verify = runner.invoke(app, ["validation", "verify-phase5a", str(run_dir)])
    assert verify.exit_code == 0


def test_prompt_change_reruns_sam_and_measured_not_dense(tmp_path: Path) -> None:
    config = load_config(CONFIG).model_copy(deep=True)
    prompt = tmp_path / "prompts.yaml"
    shutil.copy2(ROOT / "configs/prompts/tabletop.yaml", prompt)
    config.stages["segmentation_tracking"].adapter.config["prompt_manifest"] = str(prompt)
    run_dir = tmp_path / "prompt-run"
    PipelineRunner(config, INPUT, run_dir).run()
    prompt.write_text(prompt.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    resumed = PipelineRunner(config, INPUT, run_dir).run(resume=True)
    assert resumed["stages"]["segmentation_tracking"]["last_execution"] == "executed"
    assert resumed["stages"]["dense_mvs"]["last_execution"] == "cache_hit"
    assert resumed["stages"]["measured_object_geometry"]["last_execution"] == "executed"
    assert resumed["stages"]["phase5a_consistency_validation"]["last_execution"] == "executed"


def test_dense_parser_rejects_non_finite_values(tmp_path: Path) -> None:
    path = tmp_path / "nan.bin"
    path.write_bytes(b"1&1&1&" + struct.pack("<f", float("nan")))
    with pytest.raises(DenseMapFormatError, match="non-finite"):
        read_dense_array(path, expected_channels=1, reject_non_finite=True)


def test_fake_dense_manifest_is_deterministic(tmp_path: Path) -> None:
    run_a, _ = _run(tmp_path / "a")
    run_b, _ = _run(tmp_path / "b")
    payload_a = json.loads((run_a / "reconstruction/dense/depth_manifest.json").read_text())
    payload_b = json.loads((run_b / "reconstruction/dense/depth_manifest.json").read_text())
    for payload in (payload_a, payload_b):
        for record in payload["records"]:
            record["depth_path"] = Path(record["depth_path"]).name
            record["normal_path"] = Path(record["normal_path"]).name
            record["consistency_graph_path"] = Path(record["consistency_graph_path"]).name
    assert payload_a == payload_b


def test_measured_projection_round_trip_with_arbitrary_world_pose() -> None:
    intrinsics = (420.0, 390.0, 301.0, 199.0)
    translation = (3.0, -2.0, 0.5)
    rotation = (0.0, 0.0, 2**-0.5, 2**-0.5)
    point = backproject_pixel_world(
        pixel_xy=(353.5, 238.0),
        depth=2.4,
        intrinsics=intrinsics,
        translation_world_from_camera=translation,
        rotation_world_from_camera_xyzw=rotation,
    )
    pixel, depth = project_world_pixel(
        point_world=point,
        intrinsics=intrinsics,
        translation_world_from_camera=translation,
        rotation_world_from_camera_xyzw=rotation,
    )
    assert pixel == pytest.approx((353.5, 238.0))
    assert depth == pytest.approx(2.4)


def test_measured_projection_rejects_points_behind_camera() -> None:
    with pytest.raises(ValueError, match="behind"):
        project_world_pixel(
            point_world=(0.0, 0.0, -1.0),
            intrinsics=(100.0, 100.0, 50.0, 50.0),
            translation_world_from_camera=(0.0, 0.0, 0.0),
            rotation_world_from_camera_xyzw=(0.0, 0.0, 0.0, 1.0),
        )


def test_depth_consistency_and_observed_mesh_do_not_bridge_discontinuity() -> None:
    assert relative_depth_agrees(2.0, 2.04, 0.03)
    assert not relative_depth_agrees(2.0, 2.2, 0.03)
    assert observed_triangle_is_local((2.0, 2.01, 2.03), 0.03)
    assert not observed_triangle_is_local((2.0, 2.01, 3.0), 0.03)
