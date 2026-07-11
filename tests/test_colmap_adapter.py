from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from recon2sim.artifacts import CameraDiagnostics, CameraReconstruction, ColmapWorkspaceManifest
from recon2sim.cli import app
from recon2sim.config import AdapterConfig, PipelineConfig, StageConfig, load_config
from recon2sim.ir import ScaleStatus, SceneIR, WorldFrameStatus
from recon2sim.pipeline import PipelineRunner


def _write_images(path: Path, count: int = 4) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.new(
            "RGB",
            (32, 24),
            (40 + index * 40, 60 + index * 30, 80 + index * 20),
        ).save(path / f"source_{index:02d}.png")


def _write_fake_colmap(path: Path, mode: str = "success") -> Path:
    script = f"""#!{sys.executable}
import struct
import sys
import time
from pathlib import Path

MODE = {mode!r}
args = sys.argv[1:]
if args == ['-h']:
    print('COLMAP fake-0.1.0')
    raise SystemExit(0)
command = args[0]
if '-h' in args:
    if command == 'feature_extractor':
        print('--FeatureExtraction.use_gpu')
    elif command.endswith('_matcher'):
        print('--FeatureMatching.use_gpu')
    raise SystemExit(0)

def option(name):
    return args[args.index(name) + 1]

def write_model(root, names, point_count):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'cameras.bin').open('wb') as file:
        camera_count = 2 if MODE == 'multi_camera' else 1
        camera_model = 5 if MODE == 'unsupported_camera' else 4
        file.write(struct.pack('<Q', camera_count))
        file.write(struct.pack('<iiQQ', 1, camera_model, 32, 24))
        file.write(struct.pack('<dddddddd', 30.0, 31.0, 16.0, 12.0, 0.01, -0.001, 0.0, 0.0))
        if camera_count == 2:
            file.write(struct.pack('<iiQQ', 2, camera_model, 32, 24))
            file.write(struct.pack('<dddddddd', 30.0, 31.0, 16.0, 12.0,
                                   0.01, -0.001, 0.0, 0.0))
    with (root / 'images.bin').open('wb') as file:
        file.write(struct.pack('<Q', len(names)))
        for index, name in enumerate(names, start=1):
            camera_id = 1 if MODE != 'multi_camera' or index % 2 else 2
            file.write(struct.pack('<idddddddi', index, 1.0, 0.0, 0.0, 0.0,
                                   float(index - 1), 0.0, 0.0, camera_id))
            file.write(name.encode() + b'\\0')
            file.write(struct.pack('<Q', 1 if point_count else 0))
            if point_count:
                file.write(struct.pack('<ddq', 10.0, 10.0, 1))
    with (root / 'points3D.bin').open('wb') as file:
        file.write(struct.pack('<Q', point_count))
        for point_id in range(1, point_count + 1):
            file.write(struct.pack('<QdddBBBd', point_id, float(point_id), 0.0, 1.0,
                                   100, 120, 140, 0.5))
            track = list(range(1, min(len(names), 2) + 1)) if point_id == 1 else []
            file.write(struct.pack('<Q', len(track)))
            for image_id in track:
                file.write(struct.pack('<ii', image_id, 0))

if MODE == 'timeout' and command == 'feature_extractor':
    time.sleep(5)
if MODE == 'retry_once' and command == 'feature_extractor':
    counter = Path(__file__).with_suffix('.counter')
    attempt = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(attempt))
    if attempt == 1:
        print('simulated transient failure', file=sys.stderr)
        raise SystemExit(8)
if MODE == 'failure' and command == 'feature_extractor':
    print('simulated feature extraction failure', file=sys.stderr)
    raise SystemExit(9)
if MODE == 'stale':
    raise SystemExit(0)
if command == 'feature_extractor':
    database = Path(option('--database_path'))
    database.parent.mkdir(parents=True, exist_ok=True)
    database.write_bytes(b'bad database' if MODE == 'bad_database' else b'SQLite format 3\\0fake')
elif command.endswith('_matcher'):
    pass
elif command == 'mapper':
    output = Path(option('--output_path'))
    names = sorted(path.name for path in Path(option('--image_path')).glob('*.png'))
    if MODE == 'malformed':
        model = output / '0'
        model.mkdir(parents=True, exist_ok=True)
        for name in ('cameras.bin', 'images.bin', 'points3D.bin'):
            (model / name).write_bytes(b'bad')
    elif MODE == 'multiple':
        write_model(output / '0', names[:2], 20)
        write_model(output / '1', names, 10)
    elif MODE == 'low_registration':
        write_model(output / '0', names[:1], 3)
    elif MODE == 'wrong_name':
        write_model(output / '0', ['not_in_manifest.png', *names[1:]], 3)
    else:
        write_model(output / '0', names, 10)
else:
    print(f'unexpected fake command: {{command}}', file=sys.stderr)
    raise SystemExit(2)
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


def _phase_one_config(executable: Path) -> PipelineConfig:
    return PipelineConfig(
        stages={
            "ingest": StageConfig(
                adapter=AdapterConfig(
                    name="ffmpeg_ingest",
                    config={
                        "input_mode": "image_directory",
                        "duplicate_threshold": None,
                    },
                )
            ),
            "camera_recovery": StageConfig(
                adapter=AdapterConfig(
                    name="colmap_camera_recovery",
                    timeout_s=5,
                    config={
                        "execution_mode": "local",
                        "executable": str(executable),
                        "matcher": "sequential",
                        "camera_model": "OPENCV",
                        "single_camera": True,
                        "use_gpu": False,
                        "min_registered_frames": 2,
                        "min_registration_ratio": 0.5,
                        "mapper": {"multiple_models": True},
                        "sequential_matcher": {"overlap": 3, "loop_detection": False},
                    },
                ),
                depends_on=["ingest"],
            ),
        }
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fake_colmap_exercises_full_adapter_and_preserves_workspace(tmp_path: Path) -> None:
    input_dir = tmp_path / "images"
    _write_images(input_dir)
    executable = _write_fake_colmap(tmp_path / "colmap")
    run_dir = tmp_path / "run"
    config = _phase_one_config(executable)
    PipelineRunner(config, input_dir, run_dir).run()

    reconstruction = CameraReconstruction.model_validate_json(
        (run_dir / "camera" / "reconstruction.json").read_text(encoding="utf-8")
    )
    diagnostics = CameraDiagnostics.model_validate_json(
        (run_dir / "camera" / "diagnostics.json").read_text(encoding="utf-8")
    )
    workspace = ColmapWorkspaceManifest.model_validate_json(
        (run_dir / "camera" / "colmap" / "workspace_manifest.json").read_text(encoding="utf-8")
    )
    assert reconstruction.model == "OPENCV"
    assert reconstruction.scale_status is ScaleStatus.SCALE_AMBIGUOUS
    assert reconstruction.world_frame_status is WorldFrameStatus.COLMAP_UNALIGNED
    assert reconstruction.coordinate_convention.units == "arbitrary_scale"
    assert reconstruction.coordinate_convention.world_axes == "colmap_arbitrary"
    assert len(reconstruction.poses) == 4
    assert diagnostics.registration_ratio == 1.0
    assert diagnostics.sparse_points == 10
    assert workspace.selected_model == "0"
    assert [command.name for command in workspace.commands] == [
        "version",
        "feature_extractor_help",
        "sequential_matcher_help",
        "feature_extractor",
        "sequential_matcher",
        "mapper",
    ]
    feature_command = next(
        command for command in workspace.commands if command.name == "feature_extractor"
    )
    matcher_command = next(
        command for command in workspace.commands if command.name == "sequential_matcher"
    )
    assert "--FeatureExtraction.use_gpu" in feature_command.arguments
    assert "--FeatureMatching.use_gpu" in matcher_command.arguments
    assert (run_dir / "camera" / "colmap" / "database.db").is_file()
    assert (run_dir / "camera" / "colmap" / "sparse" / "0" / "cameras.bin").is_file()

    cli = CliRunner()
    config_path = tmp_path / "phase_one_config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    health = cli.invoke(
        app,
        [
            "adapters",
            "healthcheck",
            "--config",
            str(config_path),
            "--input",
            str(input_dir),
        ],
    )
    assert health.exit_code == 0, health.output
    assert "ingest/ffmpeg_ingest: available" in health.output
    assert "camera_recovery/colmap_camera_recovery: available" in health.output
    ingest_inspect = cli.invoke(app, ["ingest", "inspect", str(run_dir)])
    assert ingest_inspect.exit_code == 0, ingest_inspect.output
    assert '"selected_frames": 4' in ingest_inspect.output
    camera_inspect = cli.invoke(app, ["camera", "inspect", str(run_dir)])
    assert camera_inspect.exit_code == 0, camera_inspect.output
    assert '"scale_status": "scale_ambiguous"' in camera_inspect.output
    stats = cli.invoke(app, ["camera", "colmap-stats", str(run_dir)])
    assert stats.exit_code == 0, stats.output
    assert '"selected_model": "0"' in stats.output
    trajectory = tmp_path / "trajectory.json"
    exported = cli.invoke(
        app,
        ["camera", "export-trajectory", str(run_dir), "--output", str(trajectory)],
    )
    assert exported.exit_code == 0, exported.output
    assert len(json.loads(trajectory.read_text(encoding="utf-8"))["poses"]) == 4


def test_multiple_sparse_models_are_ranked_and_reported(tmp_path: Path) -> None:
    input_dir = tmp_path / "images"
    _write_images(input_dir)
    executable = _write_fake_colmap(tmp_path / "colmap", "multiple")
    run_dir = tmp_path / "run"
    PipelineRunner(_phase_one_config(executable), input_dir, run_dir).run()
    diagnostics = CameraDiagnostics.model_validate_json(
        (run_dir / "camera" / "diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics.selected_model == "1"
    assert [model.model_id for model in diagnostics.models] == ["1", "0"]
    assert diagnostics.models[0].selected is True


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("failure", "COLMAP feature_extractor failed"),
        ("bad_database", "without a valid SQLite header"),
        ("malformed", "malformed COLMAP sparse model"),
        ("stale", "did not create database.db"),
        ("multi_camera", "Phase 1 rejects multi-camera"),
        ("unsupported_camera", "unsupported COLMAP camera model"),
        ("low_registration", "no COLMAP sparse model met registration thresholds"),
        ("wrong_name", "registered inconsistent frame name"),
    ],
)
def test_fake_colmap_actionable_failure_modes(tmp_path: Path, mode: str, message: str) -> None:
    input_dir = tmp_path / "images"
    _write_images(input_dir)
    executable = _write_fake_colmap(tmp_path / "colmap", mode)
    run_dir = tmp_path / "run"
    with pytest.raises((RuntimeError, ValueError), match=message):
        PipelineRunner(_phase_one_config(executable), input_dir, run_dir).run()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["camera_recovery"]["status"] == "failed"
    assert (run_dir / "work" / "camera_recovery" / "attempt_1" / "camera").is_dir()
    if mode == "failure":
        attempt = manifest["stages"]["camera_recovery"]["attempts"][0]
        assert attempt["details"]["failed_subcommand"] == "feature_extractor"


def test_fake_colmap_timeout_preserves_attempt_logs(tmp_path: Path) -> None:
    input_dir = tmp_path / "images"
    _write_images(input_dir)
    executable = _write_fake_colmap(tmp_path / "colmap", "timeout")
    config = _phase_one_config(executable)
    config.stages["camera_recovery"].adapter.timeout_s = 0.1
    run_dir = tmp_path / "run"
    with pytest.raises(RuntimeError, match="timed out"):
        PipelineRunner(config, input_dir, run_dir).run()
    log = (
        run_dir
        / "work"
        / "camera_recovery"
        / "attempt_1"
        / "camera"
        / "colmap"
        / "logs"
        / "feature_extractor.stderr.log"
    )
    assert log.is_file()


def test_fake_colmap_retry_uses_fresh_attempt_workspace(tmp_path: Path) -> None:
    input_dir = tmp_path / "images"
    _write_images(input_dir)
    executable = _write_fake_colmap(tmp_path / "colmap", "retry_once")
    config = _phase_one_config(executable)
    config.stages["camera_recovery"].adapter.retries = 1
    run_dir = tmp_path / "run"
    manifest = PipelineRunner(config, input_dir, run_dir).run()
    attempts = manifest["stages"]["camera_recovery"]["attempts"]
    assert [attempt["status"] for attempt in attempts] == ["failed", "succeeded"]
    assert (run_dir / "work" / "camera_recovery" / "attempt_1").is_dir()
    assert (run_dir / "work" / "camera_recovery" / "attempt_2").is_dir()
    assert (run_dir / "camera" / "reconstruction.json").is_file()


def test_failed_colmap_attempt_preserves_previous_success(tmp_path: Path) -> None:
    input_dir = tmp_path / "images"
    _write_images(input_dir)
    executable = _write_fake_colmap(tmp_path / "colmap")
    config = _phase_one_config(executable)
    run_dir = tmp_path / "run"
    PipelineRunner(config, input_dir, run_dir).run()
    reconstruction_path = run_dir / "camera" / "reconstruction.json"
    previous_hash = _sha256(reconstruction_path)
    previous_database_hash = _sha256(run_dir / "camera" / "colmap" / "database.db")

    _write_fake_colmap(executable, "failure")
    with pytest.raises(RuntimeError, match="feature_extractor failed"):
        PipelineRunner(config, input_dir, run_dir).run(from_stage="camera_recovery")

    assert _sha256(reconstruction_path) == previous_hash
    assert _sha256(run_dir / "camera" / "colmap" / "database.db") == previous_database_hash
    assert (
        run_dir
        / "work"
        / "camera_recovery"
        / "attempt_2"
        / "camera"
        / "colmap"
        / "logs"
        / "feature_extractor.stderr.log"
    ).is_file()


def test_existing_mock_downstream_consumes_real_camera_artifact(tmp_path: Path) -> None:
    input_dir = tmp_path / "images"
    _write_images(input_dir)
    executable = _write_fake_colmap(tmp_path / "colmap")
    config = load_config(Path("configs/mock.yaml"))
    phase_one = _phase_one_config(executable)
    config.stages["ingest"] = phase_one.stages["ingest"]
    config.stages["camera_recovery"] = phase_one.stages["camera_recovery"]
    run_dir = tmp_path / "run"
    PipelineRunner(config, input_dir, run_dir).run()
    scene = SceneIR.model_validate_json(
        (run_dir / "scene_ir" / "scene.json").read_text(encoding="utf-8")
    )
    assert scene.metadata.scale_status is ScaleStatus.SCALE_AMBIGUOUS
    assert scene.metadata.world_frame_status is WorldFrameStatus.COLMAP_UNALIGNED
    assert scene.cameras[0].model == "OPENCV"
    assert len(scene.cameras[0].poses) == 4
