from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
import tomllib
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from recon2sim.adapters.base import StageContext
from recon2sim.adapters.genrecon import (
    GenReconAdapterConfig,
    GenReconGlobalReconstructionAdapter,
    Phase3EndToEndConsistencyAdapter,
    _transform_roundtrip_error,
)
from recon2sim.artifacts import (
    CameraReconstruction,
    EndToEndConsistencyReport,
    GenReconCameraPackageManifest,
    GenReconCheckpointManifest,
    GenReconInferenceRequest,
    GenReconWorkerManifest,
    GlobalSceneDiagnostics,
    GlobalSceneReconstructionArtifact,
    IngestManifest,
)
from recon2sim.cli import app
from recon2sim.colmap import ColmapModel, read_model
from recon2sim.config import PipelineConfig, load_config
from recon2sim.genrecon import (
    OFFICIAL_GENRECON_COMMIT,
    OFFICIAL_GENRECON_SUBMODULES,
    export_colmap_text_package,
    inspect_glb,
    inspect_global_mesh,
    sha256_file,
    validate_camera_package,
)
from recon2sim.ir import SceneIR
from recon2sim.lineage import frame_sequence_digest
from recon2sim.pipeline import PipelineRunner

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "examples" / "tabletop"
FAKE_CONFIG = ROOT / "configs" / "phase3_e2e_fake.yaml"
CHECKPOINTS = ROOT / "tests" / "fixtures" / "genrecon_checkpoints"


def _config(mode: str = "success") -> PipelineConfig:
    config = load_config(FAKE_CONFIG).model_copy(deep=True)
    config.stages["global_reconstruction"].adapter.config["fake_mode"] = mode
    return config


def _run(
    tmp_path: Path,
    *,
    mode: str = "success",
    until_stage: str | None = None,
) -> tuple[Path, dict[str, object]]:
    run_dir = tmp_path / f"run-{mode}"
    manifest = PipelineRunner(_config(mode), INPUT, run_dir).run(
        until_stage=until_stage,
    )
    return run_dir, manifest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frame_sequence_digest_is_order_sensitive(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path, until_stage="ingest")
    manifest = IngestManifest.model_validate_json(
        (run_dir / "inputs/manifest.json").read_text(encoding="utf-8")
    )
    assert frame_sequence_digest(manifest.frames) == manifest.frame_sequence_digest
    assert frame_sequence_digest(reversed(manifest.frames)) != manifest.frame_sequence_digest


def test_fake_full_phase3_dag_and_consistency_report(tmp_path: Path) -> None:
    run_dir, manifest = _run(tmp_path)
    assert all(
        entry["status"] == "succeeded"
        for entry in manifest["stages"].values()  # type: ignore[union-attr]
    )
    report = EndToEndConsistencyReport.model_validate_json(
        (run_dir / "validation/phase3_e2e_consistency.json").read_text(encoding="utf-8")
    )
    assert report.passed
    assert report.real_modules_share_consistent_inputs
    assert not report.object_level_2d_3d_fusion_implemented
    assert not report.sim_ready_scene_implemented


def test_fake_full_phase3_resume_hits_every_stage(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    resumed = PipelineRunner(_config(), INPUT, run_dir).run(resume=True)
    assert {
        entry["last_execution"]
        for entry in resumed["stages"].values()  # type: ignore[union-attr]
    } == {"cache_hit"}


def test_camera_package_is_deterministic(tmp_path: Path) -> None:
    first, _ = _run(tmp_path / "first", until_stage="genrecon_camera_package")
    second, _ = _run(tmp_path / "second", until_stage="genrecon_camera_package")
    for filename in ("cameras.txt", "images.txt", "points3D.txt", "registered_frames.json"):
        assert (first / "camera/genrecon_package" / filename).read_bytes() == (
            second / "camera/genrecon_package" / filename
        ).read_bytes()
    preview = first / "camera/genrecon_package/previews/camera_trajectory_and_sparse_points.png"
    assert preview.is_file()
    assert (
        preview.read_bytes()
        == (
            second / "camera/genrecon_package/previews/camera_trajectory_and_sparse_points.png"
        ).read_bytes()
    )


def test_camera_package_uses_manifest_order_and_deterministic_ids(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path, until_stage="genrecon_camera_package")
    package = GenReconCameraPackageManifest.model_validate_json(
        (run_dir / "camera/genrecon_package/package_manifest.json").read_text(encoding="utf-8")
    )
    assert package.eligible_frame_ids == package.master_frame_ids
    assert [record.package_image_id for record in package.registered_frames] == [1, 2, 3]
    image_headers = [
        line
        for line in (run_dir / package.images_path).read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and len(line.split(maxsplit=9)) == 10
    ]
    assert [int(line.split()[0]) for line in image_headers] == [1, 2, 3]


def test_camera_package_pose_round_trip(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path, until_stage="genrecon_camera_package")
    raw = read_model(run_dir / "camera/colmap/sparse/0")
    headers = [
        line.split(maxsplit=9)
        for line in (run_dir / "camera/genrecon_package/images.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line and not line.startswith("#") and len(line.split(maxsplit=9)) == 10
    ]
    raw_order = sorted(raw.images.values(), key=lambda image: image.name)
    assert [tuple(map(float, header[1:5])) for header in headers] == [
        image.qvec_wxyz for image in raw_order
    ]
    assert [tuple(map(float, header[5:8])) for header in headers] == [
        image.tvec for image in raw_order
    ]


def test_camera_package_contains_only_selected_model_inputs(tmp_path: Path) -> None:
    run_dir, manifest = _run(tmp_path, until_stage="genrecon_camera_package")
    attempt = manifest["stages"]["genrecon_camera_package"]["attempts"][-1]  # type: ignore[index]
    materialized = {entry["relative_path"] for entry in attempt["materialized_inputs"]}
    assert "camera/colmap/sparse/0/cameras.bin" in materialized
    assert "camera/colmap/database.db" not in materialized
    assert not any(path.startswith("camera/colmap/logs/") for path in materialized)
    assert (run_dir / "camera/colmap/database.db").is_file()


def test_validate_camera_package_detects_changed_text(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path, until_stage="genrecon_camera_package")
    package = GenReconCameraPackageManifest.model_validate_json(
        (run_dir / "camera/genrecon_package/package_manifest.json").read_text(encoding="utf-8")
    )
    (run_dir / package.cameras_path).write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content hash"):
        validate_camera_package(run_dir, package)


def test_registered_frame_filtering_excludes_unregistered(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path, until_stage="genrecon_camera_package")
    manifest = IngestManifest.model_validate_json(
        (run_dir / "inputs/manifest.json").read_text(encoding="utf-8")
    )
    camera = CameraReconstruction.model_validate_json(
        (run_dir / "camera/reconstruction.json").read_text(encoding="utf-8")
    )
    raw = read_model(run_dir / "camera/colmap/sparse/0")
    kept_frame = manifest.frames[0]
    kept_image = next(
        image for image in raw.images.values() if image.name == Path(kept_frame.relative_path).name
    )
    filtered = ColmapModel(
        cameras=raw.cameras,
        images={kept_image.image_id: kept_image},
        points3d=raw.points3d,
    )
    filtered_camera = camera.model_copy(
        update={
            "poses": camera.poses[:1],
            "registered_frame_ids": [kept_frame.frame_id],
            "unregistered_frame_ids": [frame.frame_id for frame in manifest.frames[1:]],
        }
    )
    package = export_colmap_text_package(
        model=filtered,
        manifest=manifest,
        camera=filtered_camera,
        output_dir=tmp_path / "filtered",
        selected_model_id="0",
        source_model_hashes={
            name: sha256_file(run_dir / "camera/colmap/sparse/0" / name)
            for name in ("cameras.bin", "images.bin", "points3D.bin")
        },
        manifest_sha256=sha256_file(run_dir / "inputs/manifest.json"),
        camera_reconstruction_sha256=sha256_file(run_dir / "camera/reconstruction.json"),
    )
    assert package.eligible_frame_ids == [kept_frame.frame_id]


def test_global_attempt_materializes_no_sam_or_raw_colmap(tmp_path: Path) -> None:
    _, manifest = _run(tmp_path)
    attempt = manifest["stages"]["global_reconstruction"]["attempts"][-1]  # type: ignore[index]
    paths = {entry["relative_path"] for entry in attempt["materialized_inputs"]}
    assert not any(path.startswith("observations/") for path in paths)
    assert not any(path.startswith("camera/colmap/") for path in paths)
    assert {path for path in paths if path.startswith("frames/")} == {
        "frames/frame_000000.png",
        "frames/frame_000001.png",
        "frames/frame_000002.png",
    }
    references = [
        entry
        for entry in attempt["materialized_inputs"]
        if entry["artifact_type"] == "genrecon_checkpoint"
    ]
    assert all(entry["materialization_mode"] == "reference_only" for entry in references)


def test_reference_only_checkpoints_are_not_copied(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    attempt = next((run_dir / "work/global_reconstruction").glob("attempt_*"))
    assert not (attempt / "reconstruction/global/checkpoint_refs").exists()


def test_global_scene_metadata_and_scene_ir_preserve_coordinates(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    metadata = GlobalSceneReconstructionArtifact.model_validate_json(
        (run_dir / "reconstruction/global/metadata.json").read_text(encoding="utf-8")
    )
    scene = SceneIR.model_validate_json(
        (run_dir / "scene_ir/scene.json").read_text(encoding="utf-8")
    )
    assert metadata.coordinate_convention.world_frame == "colmap_arbitrary"
    assert metadata.coordinate_convention.linear_units == "arbitrary_units"
    assert metadata.scale_status == "scale_ambiguous"
    assert {asset.asset_id for asset in scene.geometry_assets} == {
        "global_scene_pbr",
        "global_scene_mesh",
    }
    assert not scene.collision_assets


def test_mesh_and_glb_inspection(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    statistics = inspect_global_mesh(
        run_dir / "reconstruction/global/mesh.ply",
        run_dir / "reconstruction/global/scene.glb",
    )
    assert statistics.vertex_count == 8
    assert statistics.face_count == 12
    assert statistics.material_count == 1
    assert statistics.texture_count == 1


def test_invalid_glb_inspection_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.glb"
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="too short"):
        inspect_glb(path)


@pytest.mark.parametrize(
    "mode",
    [
        "invalid_glb",
        "missing_scene",
        "missing_intermediate",
        "wrong_commit",
        "wrong_checkpoint",
        "frame_order_mismatch",
        "registered_mismatch",
        "nonfinite_mesh",
        "zero_mesh",
        "path_escape",
        "nonzero_exit",
        "oom",
        "checkpoint_missing",
        "cuda_extension_failure",
        "dinov3_unauthorized",
        "empty_chunks",
        "malformed_manifest",
        "bad_transform",
        "interruption",
    ],
)
def test_fake_worker_failure_modes_are_rejected(tmp_path: Path, mode: str) -> None:
    with pytest.raises((RuntimeError, ValueError)):
        _run(tmp_path, mode=mode, until_stage="global_reconstruction")


def test_fake_worker_timeout_is_terminated(tmp_path: Path) -> None:
    config = _config("timeout")
    config.stages["global_reconstruction"].adapter.timeout_s = 0.1
    with pytest.raises(RuntimeError, match="timed out"):
        PipelineRunner(config, INPUT, tmp_path / "run").run(until_stage="global_reconstruction")


def test_failed_global_attempt_preserves_previous_outputs(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    metadata_path = run_dir / "reconstruction/global/metadata.json"
    before = metadata_path.read_bytes()
    with pytest.raises(RuntimeError):
        PipelineRunner(_config("wrong_commit"), INPUT, run_dir).run(
            resume=True,
            from_stage="global_reconstruction",
            until_stage="global_reconstruction",
        )
    assert metadata_path.read_bytes() == before
    GlobalSceneReconstructionArtifact.model_validate_json(before)


def test_checkpoint_manifest_is_typed_and_complete(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    manifest = GenReconCheckpointManifest.model_validate_json(
        (run_dir / "reconstruction/global/checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    assert {record.checkpoint_id for record in manifest.checkpoints} == {
        "sparse_structure",
        "shape_slat",
        "texture_slat",
    }
    assert all(len(record.sha256) == 64 for record in manifest.checkpoints)


def test_worker_manifest_pins_official_identity(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    worker = GenReconWorkerManifest.model_validate_json(
        (run_dir / "reconstruction/global/worker_manifest.json").read_text(encoding="utf-8")
    )
    assert worker.official_code_commit == OFFICIAL_GENRECON_COMMIT
    assert worker.submodule_commits == OFFICIAL_GENRECON_SUBMODULES
    assert _transform_roundtrip_error(worker) == 0


def test_prompt_change_does_not_invalidate_genrecon(tmp_path: Path) -> None:
    config = _config()
    prompt = tmp_path / "prompts.yaml"
    shutil.copy2(ROOT / "configs/prompts/tabletop.yaml", prompt)
    config.stages["segmentation_tracking"].adapter.config["prompt_manifest"] = str(prompt)
    run_dir = tmp_path / "run"
    PipelineRunner(config, INPUT, run_dir).run()
    prompt.write_text(prompt.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    resumed = PipelineRunner(config, INPUT, run_dir).run(resume=True)
    assert resumed["stages"]["segmentation_tracking"]["last_execution"] == "executed"
    assert resumed["stages"]["global_reconstruction"]["last_execution"] == "cache_hit"
    assert resumed["stages"]["end_to_end_consistency_validation"]["last_execution"] == "executed"


def test_camera_input_change_invalidates_sam_and_genrecon(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    shutil.copytree(INPUT, input_dir)
    config = _config()
    run_dir = tmp_path / "run"
    PipelineRunner(config, input_dir, run_dir).run()
    frame_path = input_dir / "frames/frame_000.png"
    Image.new("RGB", (32, 24), (170, 40, 90)).save(frame_path)
    resumed = PipelineRunner(config, input_dir, run_dir).run(resume=True)
    for stage in (
        "ingest",
        "camera_recovery",
        "segmentation_tracking",
        "genrecon_camera_package",
        "global_reconstruction",
        "end_to_end_consistency_validation",
    ):
        assert resumed["stages"][stage]["last_execution"] == "executed"


def test_previews_are_deterministic(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    previews = sorted((run_dir / "reconstruction/global/previews").glob("*.png"))
    before = {path.name: _sha(path) for path in previews}
    result = CliRunner().invoke(
        app,
        ["reconstruction", "render-global-preview", str(run_dir)],
    )
    assert result.exit_code == 0, result.output
    assert {path.name: _sha(path) for path in previews} == before


def test_reconstruction_cli_inspect_and_export(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    runner = CliRunner()
    inspect_result = runner.invoke(
        app,
        ["reconstruction", "inspect-global", str(run_dir)],
    )
    assert inspect_result.exit_code == 0
    assert OFFICIAL_GENRECON_COMMIT in inspect_result.output
    output = tmp_path / "exported.ply"
    export_result = runner.invoke(
        app,
        [
            "reconstruction",
            "export-global-mesh",
            str(run_dir),
            "--output",
            str(output),
        ],
    )
    assert export_result.exit_code == 0
    assert output.read_bytes() == (run_dir / "reconstruction/global/mesh.ply").read_bytes()


def test_validation_cli_inspect_and_verify(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    runner = CliRunner()
    inspect_result = runner.invoke(
        app,
        ["validation", "inspect-phase3-e2e", str(run_dir)],
    )
    verify_result = runner.invoke(
        app,
        ["validation", "verify-phase3-e2e", str(run_dir)],
    )
    assert inspect_result.exit_code == 0
    assert '"object_level_2d_3d_fusion_implemented": false' in inspect_result.output
    assert verify_result.exit_code == 0


def test_fake_worker_healthcheck_is_configuration_aware(tmp_path: Path) -> None:
    config = _config()
    stage = config.stages["global_reconstruction"]
    context = StageContext(
        stage_name="global_reconstruction",
        input_dir=INPUT,
        run_dir=tmp_path,
        canonical_run_dir=tmp_path,
        config=stage,
        seed=config.seed,
    )
    result = GenReconGlobalReconstructionAdapter().healthcheck(context)
    assert result.ok
    assert "fake_worker" in result.message


def test_request_schema_rejects_missing_checkpoint(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    payload = json.loads(
        (run_dir / "reconstruction/global/request.json").read_text(encoding="utf-8")
    )
    del payload["checkpoint_paths"]["texture_slat"]
    with pytest.raises(ValueError, match="three checkpoint paths"):
        GenReconInferenceRequest.model_validate(payload)


def test_e2e_report_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    attempt = run_dir / "work/end_to_end_consistency_validation/attempt_1"
    request_path = attempt / "reconstruction/global/request.json"
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    payload["manifest_sha256"] = "0" * 64
    request_path.write_text(json.dumps(payload), encoding="utf-8")
    config = _config()
    context = StageContext(
        stage_name="end_to_end_consistency_validation",
        input_dir=INPUT,
        run_dir=attempt,
        canonical_run_dir=run_dir,
        config=config.stages["end_to_end_consistency_validation"],
        seed=config.seed,
    )
    report = Phase3EndToEndConsistencyAdapter().build_report(context)
    assert not report.passed
    assert not next(check for check in report.checks if check.check_id == "manifest_sha").passed


def test_e2e_report_rejects_coordinate_mismatch(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    attempt = run_dir / "work/end_to_end_consistency_validation/attempt_1"
    metadata_path = attempt / "reconstruction/global/metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["coordinate_convention"]["world_frame"] = "canonical_x_forward_y_left_z_up"
    payload["coordinate_convention"]["alignment_status"] = "canonical"
    payload["coordinate_convention"]["linear_units"] = "meters"
    payload["coordinate_convention"]["scale_status"] = "metric_scale_known"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    config = _config()
    context = StageContext(
        stage_name="end_to_end_consistency_validation",
        input_dir=INPUT,
        run_dir=attempt,
        canonical_run_dir=run_dir,
        config=config.stages["end_to_end_consistency_validation"],
        seed=config.seed,
    )
    report = Phase3EndToEndConsistencyAdapter().build_report(context)
    assert not next(
        check for check in report.checks if check.check_id == "coordinate_semantics"
    ).passed


def test_checked_in_phase3_schemas_are_current() -> None:
    models = {
        "genrecon_camera_package.schema.json": GenReconCameraPackageManifest,
        "genrecon_checkpoints.schema.json": GenReconCheckpointManifest,
        "genrecon_inference_request.schema.json": GenReconInferenceRequest,
        "genrecon_worker_manifest.schema.json": GenReconWorkerManifest,
        "global_scene_reconstruction.schema.json": GlobalSceneReconstructionArtifact,
        "global_scene_diagnostics.schema.json": GlobalSceneDiagnostics,
        "phase3_e2e_consistency.schema.json": EndToEndConsistencyReport,
    }
    for filename, model in models.items():
        checked_in = json.loads((ROOT / "schemas" / filename).read_text(encoding="utf-8"))
        assert checked_in == model.model_json_schema()


def test_genrecon_worker_package_declares_import_package() -> None:
    payload = tomllib.loads((ROOT / "workers/genrecon/pyproject.toml").read_text(encoding="utf-8"))
    assert payload["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["genrecon_worker"]
    checkpoint_loader = (ROOT / "workers/genrecon/genrecon_worker/checkpoint_loader.py").read_text(
        encoding="utf-8"
    )
    assert "from datetime import UTC" not in checkpoint_loader


def test_local_worker_accepts_isolated_venv_python_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = tmp_path / "genrecon-env"
    (environment / "bin").mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text("home = isolated\n", encoding="utf-8")
    (environment / "bin/python").symlink_to(sys.executable)
    checkout = tmp_path / "GenRecon"
    checkout.mkdir()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    config = GenReconAdapterConfig(
        execution_mode="local_worker",
        worker_python=str(environment / "bin/python"),
        official_checkout_path=str(checkout),
        sparse_structure_checkpoint=str(CHECKPOINTS / "sparse_structure.pt"),
        shape_checkpoint=str(CHECKPOINTS / "shape_slat.pt"),
        texture_checkpoint=str(CHECKPOINTS / "texture_slat.pt"),
    )
    assert config.worker_python == str(environment / "bin/python")


def test_core_request_matches_real_worker_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, _ = _run(tmp_path)
    monkeypatch.syspath_prepend(str(ROOT / "workers" / "genrecon"))
    worker_schema = importlib.import_module("genrecon_worker.schema")
    request = worker_schema.InferenceRequest.model_validate_json(
        (run_dir / "reconstruction/global/request.json").read_text(encoding="utf-8")
    )
    assert request.official_checkout_path


def test_docker_huggingface_cache_mount_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    config = GenReconAdapterConfig(
        execution_mode="docker",
        hf_cache_path=str(cache),
        sparse_structure_checkpoint=str(CHECKPOINTS / "sparse_structure.pt"),
        shape_checkpoint=str(CHECKPOINTS / "shape_slat.pt"),
        texture_checkpoint=str(CHECKPOINTS / "texture_slat.pt"),
    )
    assert GenReconGlobalReconstructionAdapter._docker_hf_cache_arguments(config) == [
        "-v",
        f"{cache}:/hf-cache:rw",
        "-e",
        "HF_HOME=/hf-cache",
    ]
