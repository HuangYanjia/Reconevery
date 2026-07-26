from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from recon2sim.adapters.alignment import (
    CameraMeshAlignmentAdapter,
    CameraMeshAlignmentAdapterConfig,
)
from recon2sim.adapters.base import StageContext
from recon2sim.alignment import transform_point, validate_similarity_transform
from recon2sim.artifacts import (
    AlignmentTransform,
    CameraMeshAlignmentDiagnostics,
    CameraMeshAlignmentResult,
    ObjectLiftingAlignmentComparison,
    ObjectSurfaceEvidenceArtifact,
    Phase4_2ConsistencyReport,
    TransformChainAudit,
)
from recon2sim.cli import app
from recon2sim.config import PipelineConfig, load_config
from recon2sim.pipeline import PipelineRunner

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "examples" / "tabletop"
FAKE_CONFIG = ROOT / "configs" / "phase4_2_e2e_fake.yaml"


def _config(mode: str = "success_full_sim3") -> PipelineConfig:
    config = load_config(FAKE_CONFIG).model_copy(deep=True)
    config.stages["camera_mesh_alignment"].adapter.config["fake_mode"] = mode
    return config


def _run(
    tmp_path: Path,
    *,
    mode: str = "success_full_sim3",
) -> tuple[Path, dict[str, object]]:
    run_dir = tmp_path / f"run-{mode}"
    manifest = PipelineRunner(_config(mode), INPUT, run_dir).run()
    return run_dir, manifest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fake_phase4_2_dag_and_consistency(tmp_path: Path) -> None:
    run_dir, manifest = _run(tmp_path)
    assert all(
        stage["status"] == "succeeded"
        for stage in manifest["stages"].values()  # type: ignore[union-attr]
    )
    alignment = CameraMeshAlignmentResult.model_validate_json(
        (run_dir / "reconstruction/alignment/alignment.json").read_text()
    )
    audit = TransformChainAudit.model_validate_json(
        (run_dir / "reconstruction/alignment/transform_chain_audit.json").read_text()
    )
    report = Phase4_2ConsistencyReport.model_validate_json(
        (run_dir / "validation/phase4_2_camera_mesh_alignment.json").read_text()
    )
    comparison = ObjectLiftingAlignmentComparison.model_validate_json(
        (run_dir / "reconstruction/alignment/object_lifting_comparison.json").read_text()
    )
    evidence = ObjectSurfaceEvidenceArtifact.model_validate_json(
        (run_dir / "reconstruction/object_surfaces/evidence_manifest.json").read_text()
    )
    assert alignment.status == "accepted_global_sim3"
    assert alignment.accepted
    assert audit.status == "consistent"
    assert report.passed
    assert report.global_similarity_accepted
    assert not report.camera_poses_modified
    assert not report.mesh_topology_modified
    assert not report.metric_scale_known
    assert not report.hidden_surface_completion_implemented
    assert comparison.alignment_accepted
    assert comparison.objects
    assert evidence.alignment_sha256 == _sha(run_dir / "reconstruction/alignment/alignment.json")


def test_fake_phase4_2_resume_hits_every_stage(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    resumed = PipelineRunner(_config(), INPUT, run_dir).run(resume=True)
    assert {
        stage["last_execution"]
        for stage in resumed["stages"].values()  # type: ignore[union-attr]
    } == {"cache_hit"}


def test_rejected_alignment_is_valid_and_not_applied(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path, mode="no_improvement")
    alignment = CameraMeshAlignmentResult.model_validate_json(
        (run_dir / "reconstruction/alignment/alignment.json").read_text()
    )
    evidence = ObjectSurfaceEvidenceArtifact.model_validate_json(
        (run_dir / "reconstruction/object_surfaces/evidence_manifest.json").read_text()
    )
    comparison = ObjectLiftingAlignmentComparison.model_validate_json(
        (run_dir / "reconstruction/alignment/object_lifting_comparison.json").read_text()
    )
    assert alignment.status == "rejected_no_validation_improvement"
    assert not alignment.accepted
    assert not evidence.alignment_accepted
    assert comparison.baseline_scene_metrics == comparison.aligned_scene_metrics


def test_symmetric_alignment_ambiguity_is_reported(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path, mode="symmetric_ambiguity")
    alignment = CameraMeshAlignmentResult.model_validate_json(
        (run_dir / "reconstruction/alignment/alignment.json").read_text()
    )
    diagnostics = CameraMeshAlignmentDiagnostics.model_validate_json(
        (run_dir / "reconstruction/alignment/diagnostics.json").read_text()
    )

    assert alignment.status == "global_sim3_insufficient"
    assert not alignment.accepted
    assert diagnostics.candidate_solution_ambiguous
    assert diagnostics.competing_candidate_ids == ["candidate_01_symmetric"]


def test_alignment_attempt_materializes_only_declared_inputs(tmp_path: Path) -> None:
    run_dir, manifest = _run(tmp_path)
    attempts = manifest["stages"]["camera_mesh_alignment"]["attempts"]  # type: ignore[index]
    materialized = {item["relative_path"] for item in attempts[-1]["materialized_inputs"]}
    assert "inputs/manifest.json" in materialized
    assert "camera/reconstruction.json" in materialized
    assert "camera/genrecon_package/cameras.txt" in materialized
    assert "camera/genrecon_package/images.txt" in materialized
    assert "camera/genrecon_package/points3D.txt" in materialized
    assert "reconstruction/global/mesh.ply" in materialized
    assert "reconstruction/global/raw/working_transform.json" in materialized
    assert "reconstruction/global/raw/chunk_transforms.json" in materialized
    assert "reconstruction/global/raw/cameras.json" in materialized
    assert "camera/colmap/database.db" not in materialized
    assert not any(path.startswith("camera/colmap/") for path in materialized)
    assert not any(path.startswith("observations/") for path in materialized)
    assert "reconstruction/global/raw/to_glb_inputs.pt" not in materialized
    assert "reconstruction/global/raw/chunk_inputs.pt" not in materialized
    attempt = run_dir / "work" / "camera_mesh_alignment" / f"attempt_{attempts[-1]['attempt']}"
    assert not (attempt / "camera/colmap").exists()
    assert not (attempt / "observations").exists()


def test_prompt_change_does_not_invalidate_alignment_or_genrecon(
    tmp_path: Path,
) -> None:
    config = _config()
    prompt = tmp_path / "prompts.yaml"
    shutil.copy2(ROOT / "configs/prompts/tabletop.yaml", prompt)
    config.stages["segmentation_tracking"].adapter.config["prompt_manifest"] = str(prompt)
    run_dir = tmp_path / "prompt-run"
    PipelineRunner(config, INPUT, run_dir).run()
    prompt.write_text(
        prompt.read_text(encoding="utf-8") + "\n# cache invalidation\n",
        encoding="utf-8",
    )
    resumed = PipelineRunner(config, INPUT, run_dir).run(resume=True)
    assert resumed["stages"]["segmentation_tracking"]["last_execution"] == "executed"
    assert resumed["stages"]["global_reconstruction"]["last_execution"] == "cache_hit"
    assert resumed["stages"]["camera_mesh_alignment"]["last_execution"] == "cache_hit"
    assert resumed["stages"]["object_surface_lifting"]["last_execution"] == "executed"
    assert resumed["stages"]["phase4_2_consistency_validation"]["last_execution"] == "executed"


def test_failed_alignment_preserves_previous_canonical_output(
    tmp_path: Path,
) -> None:
    run_dir, _ = _run(tmp_path)
    alignment_path = run_dir / "reconstruction/alignment/alignment.json"
    before = alignment_path.read_bytes()
    broken = _config("wrong_mesh_hash")
    with pytest.raises(RuntimeError, match="global_mesh_sha256"):
        PipelineRunner(broken, INPUT, run_dir).run(
            resume=True,
            from_stage="camera_mesh_alignment",
            until_stage="camera_mesh_alignment",
        )
    assert alignment_path.read_bytes() == before


def test_similarity_transform_validation_and_application() -> None:
    transform = AlignmentTransform(
        matrix_original_mesh_to_aligned_colmap=[
            [0.0, -2.0, 0.0, 3.0],
            [2.0, 0.0, 0.0, -4.0],
            [0.0, 0.0, 2.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        inverse_matrix=[
            [0.0, 0.5, 0.0, 2.0],
            [-0.5, 0.0, 0.0, 1.5],
            [0.0, 0.0, 0.5, -0.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        scale=2.0,
        rotation_matrix=[
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        rotation_axis_angle=(0.0, 0.0, 1.5707963267948966),
        rotation_degrees=90.0,
        translation=(3.0, -4.0, 1.0),
        translation_scene_diagonal_ratio=0.5,
        determinant=8.0,
        roundtrip_error=0.0,
    )
    validate_similarity_transform(transform)
    assert transform_point((1.0, 2.0, 3.0), transform.matrix_original_mesh_to_aligned_colmap) == (
        -1.0,
        -2.0,
        7.0,
    )


def test_local_worker_accepts_symlinked_venv_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv = tmp_path / "alignment-env"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    (venv / "bin/python").symlink_to(sys.executable)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    config = CameraMeshAlignmentAdapterConfig(
        execution_mode="local_worker",
        worker_python=str(venv / "bin/python"),
    )

    assert config.worker_python == str(venv / "bin/python")


def test_docker_healthcheck_uses_declared_python_and_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = tmp_path / "docker"
    docker.write_text(
        """#!/bin/sh
case "$1 $2" in
  "version --format") echo "27.0.0" ;;
  "image inspect") echo "sha256:alignment-image" ;;
  "run --rm")
    printf '%s\\n' "$@" | grep -qx 'python3.10'
    printf '%s\\n' "$@" | grep -qx 'reconevery/alignment:test'
    echo '{"available": true, "device_name": "fake H100"}'
    ;;
  *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    pipeline = _config()
    stage = pipeline.stages["camera_mesh_alignment"].model_copy(deep=True)
    stage.adapter.config = {
        "execution_mode": "docker",
        "docker_executable": str(docker),
        "docker_image": "reconevery/alignment:test",
        "device": "cuda",
    }
    context = StageContext(
        stage_name="camera_mesh_alignment",
        input_dir=INPUT,
        run_dir=tmp_path / "attempt",
        canonical_run_dir=tmp_path / "canonical",
        config=stage,
        seed=42,
    )

    result = CameraMeshAlignmentAdapter().healthcheck(context)

    assert result.ok, result.message
    assert "fake H100" in result.message


def test_alignment_cli_commands(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    runner = CliRunner()
    for arguments in (
        ["alignment", "inspect", str(run_dir)],
        ["alignment", "inspect-transform-chain", str(run_dir)],
        ["alignment", "inspect-camera", str(run_dir), "frame_000000"],
        ["alignment", "render-previews", str(run_dir)],
        ["alignment", "compare-object-lifting", str(run_dir)],
        ["validation", "inspect-phase4-2", str(run_dir)],
        ["validation", "verify-phase4-2", str(run_dir)],
    ):
        result = runner.invoke(app, arguments)
        assert result.exit_code == 0, result.output
    transform_output = tmp_path / "alignment.json"
    result = runner.invoke(
        app,
        [
            "alignment",
            "export-transform",
            str(run_dir),
            "--output",
            str(transform_output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert transform_output.is_file()
    mesh_output = tmp_path / "aligned_mesh.ply"
    result = runner.invoke(
        app,
        [
            "alignment",
            "export-aligned-mesh",
            str(run_dir),
            "--output",
            str(mesh_output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert mesh_output.is_file()
