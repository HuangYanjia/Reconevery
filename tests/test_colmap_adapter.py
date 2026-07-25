from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

from recon2sim.adapters.base import StageContext
from recon2sim.adapters.colmap import (
    ColmapAdapterConfig,
    ColmapCameraRecoveryAdapter,
    SparseModelCandidate,
    rank_sparse_models,
)
from recon2sim.artifacts import (
    CameraDiagnostics,
    CameraReconstruction,
    ColmapWorkspaceManifest,
)
from recon2sim.colmap.model import ColmapModel
from recon2sim.config import AdapterConfig, PipelineConfig, StageConfig, load_config
from recon2sim.ir import (
    AlignmentStatus,
    CameraAxes,
    LinearUnits,
    ScaleStatus,
    SceneIR,
    WorldFrame,
)
from recon2sim.pipeline import PipelineRunner


def _write_fake_colmap(path: Path, mode: str = "success") -> Path:
    source = f"""#!{sys.executable}
import struct
import sys
import time
from pathlib import Path

MODE = {mode!r}
args = sys.argv[1:]
if args == ["-h"]:
    print("COLMAP 3.11.1 fake")
    raise SystemExit(0)
command = args[0]
def option(name):
    return args[args.index(name) + 1]
if MODE == "timeout" and command == "feature_extractor":
    time.sleep(5)
if MODE == "nonzero" and command == "sequential_matcher":
    print("simulated matcher failure", file=sys.stderr)
    raise SystemExit(9)
if command == "feature_extractor":
    database = Path(option("--database_path"))
    database.parent.mkdir(parents=True, exist_ok=True)
    database.write_bytes(b"fake sqlite")
elif command == "mapper":
    if MODE == "no_model":
        raise SystemExit(0)
    output = Path(option("--output_path"))
    frames = sorted(Path(option("--image_path")).glob("*.png"))
    def write_model(model_id, registered, points):
        root = output / str(model_id)
        root.mkdir(parents=True, exist_ok=True)
        camera_data = struct.pack(
            "<QiiQQdddd",
            2 if MODE == "multi_camera" else 1,
            1, 1, 32, 20, 30.0, 31.0, 16.0, 10.0
        )
        if MODE == "multi_camera":
            camera_data += struct.pack(
                "<iiQQdddd", 2, 1, 32, 20, 30.0, 31.0, 16.0, 10.0
            )
        root.joinpath("cameras.bin").write_bytes(camera_data)
        image_data = struct.pack("<Q", len(registered))
        for index, frame in enumerate(registered):
            image_data += struct.pack(
                "<idddddddi", index + 1, 1.0, 0.0, 0.0, 0.0,
                float(index), 0.0, 0.0, 1
            )
            image_data += frame.name.encode() + b"\\0" + struct.pack("<Q", 0)
        root.joinpath("images.bin").write_bytes(image_data)
        point_data = struct.pack("<Q", points)
        for index in range(points):
            point_data += struct.pack(
                "<QdddBBBdQ", index + 1, float(index), 0.0, 0.0,
                20, 30, 40, 0.5 + model_id * 0.1, 0
            )
        root.joinpath("points3D.bin").write_bytes(point_data)
    if MODE == "malformed":
        broken = output / "0"
        broken.mkdir(parents=True, exist_ok=True)
        broken.joinpath("cameras.bin").write_bytes(b"broken")
        broken.joinpath("images.bin").write_bytes(b"broken")
        broken.joinpath("points3D.bin").write_bytes(b"broken")
    else:
        write_model(0, frames[:-1], 9)
        write_model(1, frames, 3)
"""
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path


def _write_fake_docker(path: Path, calls_path: Path, *, image_available: bool = True) -> Path:
    source = f"""#!{sys.executable}
import json
import sys
from pathlib import Path
calls_path = Path({str(calls_path)!r})
calls = json.loads(calls_path.read_text()) if calls_path.exists() else []
calls.append(sys.argv[1:])
calls_path.write_text(json.dumps(calls))
if sys.argv[1:3] == ["image", "inspect"] and not {image_available!r}:
    raise SystemExit(1)
print("Docker fake 1.0")
"""
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path


def _input_images(root: Path) -> None:
    for index, color in enumerate(((30, 80, 130), (80, 130, 30), (130, 30, 80))):
        destination = root / "images" / f"{index:03d}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (48, 30), color).save(destination)


def _pipeline(
    executable: Path,
    *,
    timeout_s: float = 5,
    min_registered_frames: int = 2,
) -> PipelineConfig:
    return PipelineConfig(
        stages={
            "ingest": StageConfig(
                adapter=AdapterConfig(
                    name="ffmpeg_ingest",
                    config={
                        "input_mode": "image_directory",
                        "resize_max_edge": 32,
                        "min_brightness": 0,
                        "max_brightness": 255,
                    },
                )
            ),
            "camera_recovery": StageConfig(
                adapter=AdapterConfig(
                    name="colmap_camera_recovery",
                    timeout_s=timeout_s,
                    env=["PATH"],
                    config={
                        "executable": str(executable),
                        "matcher": "sequential",
                        "use_gpu": False,
                        "min_registered_frames": min_registered_frames,
                        "min_registration_ratio": 0.5,
                    },
                ),
                depends_on=["ingest"],
            ),
        }
    )


def test_fake_colmap_full_adapter_workflow_and_multiple_model_selection(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    _input_images(input_dir)
    fake = _write_fake_colmap(tmp_path / "fake_colmap")
    run_dir = tmp_path / "run"

    result = PipelineRunner(_pipeline(fake), input_dir, run_dir).run()
    reconstruction = CameraReconstruction.model_validate_json(
        (run_dir / "camera" / "reconstruction.json").read_text(encoding="utf-8")
    )
    diagnostics = CameraDiagnostics.model_validate_json(
        (run_dir / "camera" / "diagnostics.json").read_text(encoding="utf-8")
    )
    workspace = ColmapWorkspaceManifest.model_validate_json(
        (run_dir / "camera" / "colmap" / "workspace_manifest.json").read_text(encoding="utf-8")
    )

    assert diagnostics.selected_model == "1"
    assert [model.model_id for model in diagnostics.models] == ["0", "1"]
    assert reconstruction.registered_frame_ids == [
        "frame_000000",
        "frame_000001",
        "frame_000002",
    ]
    assert reconstruction.unregistered_frame_ids == []
    assert reconstruction.scale_status is ScaleStatus.SCALE_AMBIGUOUS
    assert reconstruction.coordinate_convention.world_frame is WorldFrame.COLMAP_ARBITRARY
    assert reconstruction.coordinate_convention.alignment_status is AlignmentStatus.UNORIENTED
    assert reconstruction.coordinate_convention.camera_axes is CameraAxes.X_RIGHT_Y_DOWN_Z_FORWARD
    assert reconstruction.coordinate_convention.linear_units is LinearUnits.ARBITRARY_UNITS
    assert reconstruction.coordinate_convention.scale_status is ScaleStatus.SCALE_AMBIGUOUS
    assert "translation_m" not in reconstruction.poses[0].model_dump()
    assert reconstruction.intrinsics.distortion == []
    assert (run_dir / "camera" / "colmap" / "database.db").is_file()
    assert (run_dir / "camera" / "colmap" / "sparse" / "0" / "points3D.bin").is_file()
    assert (run_dir / "camera" / "colmap" / "sparse" / "1" / "points3D.bin").is_file()
    feature_command = workspace.commands[0].command
    matcher_command = workspace.commands[1].command
    assert "--SiftExtraction.use_gpu" in feature_command
    assert feature_command[feature_command.index("--SiftExtraction.use_gpu") + 1] == "0"
    assert matcher_command[1] == "sequential_matcher"
    assert result["stages"]["camera_recovery"]["metrics"]["selected_model"] == "1"


def test_real_ingest_and_colmap_artifacts_feed_complete_mock_downstream(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    _input_images(input_dir)
    fake = _write_fake_colmap(tmp_path / "fake_colmap")
    config = load_config(Path("configs/colmap_with_mock_downstream.yaml"))
    config.stages["ingest"].adapter.config.update(
        {
            "input_mode": "image_directory",
            "resize_max_edge": 32,
            "min_brightness": 0,
            "max_brightness": 255,
        }
    )
    config.stages["camera_recovery"].adapter.config.update(
        {
            "executable": str(fake),
            "min_registered_frames": 2,
            "min_registration_ratio": 0.5,
        }
    )
    run_dir = tmp_path / "run"
    manifest = PipelineRunner(config, input_dir, run_dir).run()
    scene = SceneIR.model_validate_json(
        (run_dir / "scene_ir" / "scene.json").read_text(encoding="utf-8")
    )

    assert all(entry["status"] == "succeeded" for entry in manifest["stages"].values())
    assert [frame.frame_id for frame in scene.frames] == [
        "frame_000000",
        "frame_000001",
        "frame_000002",
    ]
    assert scene.cameras[0].scale_status is ScaleStatus.SCALE_AMBIGUOUS
    assert scene.metadata.coordinate_convention.world_frame is WorldFrame.COLMAP_ARBITRARY
    assert scene.cameras[0].coordinate_convention.world_frame is WorldFrame.COLMAP_ARBITRARY


def test_production_colmap_configs_stop_after_camera_recovery() -> None:
    for path in (
        Path("configs/colmap.yaml"),
        Path("configs/colmap_cpu.yaml"),
        Path("configs/colmap_docker.example.yaml"),
    ):
        assert list(load_config(path).stages) == ["ingest", "camera_recovery"]

    assert (
        "segmentation_tracking"
        in load_config(Path("configs/colmap_with_mock_downstream.yaml")).stages
    )


def test_rank_sparse_models_uses_documented_deterministic_order(tmp_path: Path) -> None:
    model = ColmapModel(cameras={}, images={}, points3d={})
    candidates = [
        SparseModelCandidate("10", tmp_path / "10", model, 5, 0.5, 100, 0.5),
        SparseModelCandidate("2", tmp_path / "2", model, 5, 0.5, 100, 0.5),
        SparseModelCandidate("1", tmp_path / "1", model, 5, 0.5, 99, 0.1),
    ]
    selected, diagnostics = rank_sparse_models(
        candidates,
        min_registered_frames=2,
        min_registration_ratio=0.4,
    )
    assert selected.model_id == "2"
    assert next(item for item in diagnostics if item.model_id == "2").selected is True

    with pytest.raises(ValueError, match="no COLMAP sparse model meets registration thresholds"):
        rank_sparse_models(
            candidates,
            min_registered_frames=6,
            min_registration_ratio=0.4,
        )


@pytest.mark.parametrize(
    ("mode", "error_match", "failed_subcommand"),
    [
        ("nonzero", "return code 9", "sequential_matcher"),
        ("no_model", "produced no sparse model", "model_selection"),
        ("malformed", "malformed COLMAP binary", "model_selection"),
        ("multi_camera", "requires exactly one COLMAP camera", "model_selection"),
    ],
)
def test_fake_colmap_failures_preserve_partial_workspace(
    tmp_path: Path,
    mode: str,
    error_match: str,
    failed_subcommand: str,
) -> None:
    input_dir = tmp_path / "input"
    _input_images(input_dir)
    fake = _write_fake_colmap(tmp_path / "fake_colmap", mode)
    run_dir = tmp_path / "run"
    with pytest.raises(RuntimeError, match=error_match):
        PipelineRunner(_pipeline(fake), input_dir, run_dir).run()

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    attempt = manifest["stages"]["camera_recovery"]["attempts"][0]
    assert attempt["details"]["failed_subcommand"] == failed_subcommand
    workspace = run_dir / attempt["workspace"]
    partial_manifest = ColmapWorkspaceManifest.model_validate_json(
        (workspace / "camera" / "colmap" / "workspace_manifest.json").read_text(encoding="utf-8")
    )
    assert partial_manifest.failed_subcommand == failed_subcommand
    assert (workspace / "camera" / "colmap" / "database.db").is_file()
    assert not (run_dir / "camera" / "reconstruction.json").exists()


def test_fake_colmap_timeout_terminates_process_and_records_failure(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    _input_images(input_dir)
    fake = _write_fake_colmap(tmp_path / "fake_colmap", "timeout")
    run_dir = tmp_path / "run"
    with pytest.raises(RuntimeError, match="timed out"):
        PipelineRunner(_pipeline(fake, timeout_s=0.1), input_dir, run_dir).run()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    attempt = manifest["stages"]["camera_recovery"]["attempts"][0]
    assert attempt["details"]["failed_subcommand"] == "feature_extractor"


def test_registration_threshold_failure_retains_candidate_diagnostics(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    _input_images(input_dir)
    fake = _write_fake_colmap(tmp_path / "fake_colmap")
    run_dir = tmp_path / "run"
    with pytest.raises(RuntimeError, match="min_registered_frames=4"):
        PipelineRunner(
            _pipeline(fake, min_registered_frames=4),
            input_dir,
            run_dir,
        ).run()
    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    workspace = run_dir / run_manifest["stages"]["camera_recovery"]["attempts"][0]["workspace"]
    diagnostics = CameraDiagnostics.model_validate_json(
        (workspace / "camera" / "diagnostics.json").read_text(encoding="utf-8")
    )
    assert [model.model_id for model in diagnostics.models] == ["0", "1"]
    assert all(model.rejection_reason == "registered_frames<4" for model in diagnostics.models)


def test_docker_healthcheck_checks_daemon_and_configured_image(tmp_path: Path) -> None:
    calls_path = tmp_path / "docker_calls.json"
    docker = _write_fake_docker(tmp_path / "docker", calls_path)
    stage = StageConfig(
        adapter=AdapterConfig(
            name="colmap_camera_recovery",
            config={
                "execution_mode": "docker",
                "docker_executable": str(docker),
                "docker_image": "reconevery/colmap:test",
                "use_gpu": False,
            },
        )
    )
    context = StageContext(
        stage_name="camera_recovery",
        input_dir=tmp_path,
        run_dir=tmp_path / "run",
        canonical_run_dir=tmp_path / "run",
        config=stage,
        seed=7,
    )
    result = ColmapCameraRecoveryAdapter().healthcheck(context)
    assert result.ok is True
    assert json.loads(calls_path.read_text(encoding="utf-8")) == [
        ["version"],
        [
            "image",
            "inspect",
            "reconevery/colmap:test",
            "--format",
            "{{.Id}}",
        ],
        [
            "run",
            "--rm",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "reconevery/colmap:test",
            "colmap",
            "-h",
        ],
    ]


def test_docker_execution_requires_cpu_mode() -> None:
    with pytest.raises(ValueError, match="requires use_gpu=false"):
        ColmapAdapterConfig(execution_mode="docker", use_gpu=True)


def test_command_environment_is_allowlisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setenv("RECON2SIM_ALLOWED_TEST", "yes")
    monkeypatch.setenv("RECON2SIM_SECRET_TEST", "no")
    output_script = (
        "import json, os; "
        "open('environment.json', 'w').write(json.dumps(dict(os.environ), sort_keys=True))"
    )
    config = PipelineConfig(
        stages={
            "command": StageConfig(
                adapter=AdapterConfig(
                    name="command",
                    command=[sys.executable, "-c", output_script],
                    env=["RECON2SIM_ALLOWED_TEST"],
                    expected_outputs=[],
                )
            )
        }
    )
    PipelineRunner(config, input_dir, tmp_path / "run").run()
    environment = json.loads(
        (tmp_path / "run" / "work" / "command" / "attempt_1" / "environment.json").read_text(
            encoding="utf-8"
        )
    )
    assert environment["RECON2SIM_ALLOWED_TEST"] == "yes"
    assert "RECON2SIM_SECRET_TEST" not in environment
    assert environment["RECON2SIM_ATTEMPT"] == "1"
    assert os.environ["RECON2SIM_SECRET_TEST"] == "no"
