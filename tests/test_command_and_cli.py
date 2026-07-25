from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from recon2sim.adapters.process import terminate_process_group
from recon2sim.cli import app
from recon2sim.config import AdapterConfig, OutputConfig, PipelineConfig, StageConfig
from recon2sim.pipeline import OutputValidationError, PipelineRunner


def _command_config(
    command: list[str],
    *,
    expected_output: OutputConfig | None = None,
    retries: int = 0,
    timeout_s: float = 5,
) -> PipelineConfig:
    return PipelineConfig(
        stages={
            "command_stage": StageConfig(
                adapter=AdapterConfig(
                    name="command",
                    command=command,
                    timeout_s=timeout_s,
                    retries=retries,
                    expected_outputs=[] if expected_output is None else [expected_output],
                )
            )
        }
    )


def test_command_adapter_retries_and_records_execution(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    script = """
import os
from pathlib import Path
import sys
attempt = int(os.environ["RECON2SIM_ATTEMPT"])
if attempt < 3:
    sys.exit(7)
Path("result.json").write_text('{"ok": true}')
"""
    config = _command_config(
        [sys.executable, "-c", script],
        expected_output=OutputConfig(
            path="result.json",
            artifact_type="test_result",
            media_type="application/json",
            validation="json",
        ),
        retries=2,
    )
    manifest = PipelineRunner(config, input_dir, tmp_path / "run").run()
    entry = manifest["stages"]["command_stage"]
    assert entry["status"] == "succeeded"
    assert [attempt["status"] for attempt in entry["attempts"]] == [
        "failed",
        "failed",
        "succeeded",
    ]
    assert (tmp_path / "run" / "logs" / "command_stage.attempt_1.stderr.log").exists()
    assert entry["metrics"]["return_code"] == 0
    assert entry["attempts"][0]["workspace"] == "work/command_stage/attempt_1"


def test_zero_exit_with_missing_output_fails_stage(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    config = _command_config(
        [sys.executable, "-c", "pass"],
        expected_output=OutputConfig(path="missing.json", validation="json"),
    )
    runner = PipelineRunner(config, input_dir, tmp_path / "run")
    with pytest.raises(OutputValidationError, match="did not produce required output"):
        runner.run()
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["command_stage"]["status"] == "failed"


def test_invalid_adapter_json_fails_validation(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    script = "from pathlib import Path; Path('result.json').write_text('{not-json')"
    config = _command_config(
        [sys.executable, "-c", script],
        expected_output=OutputConfig(path="result.json", validation="json"),
    )
    with pytest.raises(OutputValidationError, match="produced invalid output 'result.json'"):
        PipelineRunner(config, input_dir, tmp_path / "run").run()


def test_stale_canonical_output_cannot_satisfy_new_attempt(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    run_dir = tmp_path / "run"
    output = OutputConfig(path="result.json", validation="json")
    first = _command_config(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('result.json').write_text('{\"run\": 1}')",
        ],
        expected_output=output,
    )
    PipelineRunner(first, input_dir, run_dir).run()

    stale = _command_config([sys.executable, "-c", "pass"], expected_output=output)
    with pytest.raises(OutputValidationError, match="did not produce required output"):
        PipelineRunner(stale, input_dir, run_dir).run()

    assert json.loads((run_dir / "result.json").read_text(encoding="utf-8")) == {"run": 1}
    assert not (run_dir / "work" / "command_stage" / "attempt_2" / "result.json").exists()


def test_failed_attempt_preserves_previous_successful_output(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    run_dir = tmp_path / "run"
    output = OutputConfig(path="result.json", validation="json")
    good = _command_config(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('result.json').write_text('{\"stable\": true}')",
        ],
        expected_output=output,
    )
    PipelineRunner(good, input_dir, run_dir).run()
    previous = (run_dir / "result.json").read_bytes()

    invalid = _command_config(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('result.json').write_text('{invalid')",
        ],
        expected_output=output,
    )
    with pytest.raises(OutputValidationError, match="produced invalid output"):
        PipelineRunner(invalid, input_dir, run_dir).run()

    assert (run_dir / "result.json").read_bytes() == previous
    failed = run_dir / "work" / "command_stage" / "attempt_2" / "result.json"
    assert failed.read_text(encoding="utf-8") == "{invalid"


def test_subprocess_timeout_terminates_and_preserves_logs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    config = _command_config(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        timeout_s=0.1,
    )
    with pytest.raises(RuntimeError, match="timed out"):
        PipelineRunner(config, input_dir, tmp_path / "run").run()
    result_path = tmp_path / "run" / "logs" / "command_stage.attempt_1.command.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["timed_out"] is True
    assert (tmp_path / "run" / result["stdout_path"]).exists()
    assert (tmp_path / "run" / result["stderr_path"]).exists()


def test_process_group_termination_escalates_to_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        pid = 4321
        calls = 0

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(["fake"], timeout)
            return ("stdout", "stderr")

    signals: list[signal.Signals] = []
    monkeypatch.setattr(
        "recon2sim.adapters.process.os.killpg",
        lambda pid, sent_signal: signals.append(sent_signal),
    )

    stdout, stderr = terminate_process_group(Process())  # type: ignore[arg-type]

    assert (stdout, stderr) == ("stdout", "stderr")
    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_cli_help_uses_real_typer() -> None:
    runner = CliRunner()
    for arguments in (
        ["--help"],
        ["run", "--help"],
        ["adapters", "--help"],
        ["ingest", "--help"],
        ["camera", "--help"],
    ):
        result = runner.invoke(app, arguments)
        assert result.exit_code == 0, result.output
        assert "Usage:" in result.output


def test_cli_healthcheck_uses_configured_executable(tmp_path: Path) -> None:
    executable = tmp_path / "configured_colmap"
    executable.write_text(
        f"#!{sys.executable}\nprint('COLMAP configured fake')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    config = tmp_path / "health.yaml"
    config.write_text(
        f"""
stages:
  ingest:
    adapter:
      name: ffmpeg_ingest
      config:
        input_mode: image_directory
  camera_recovery:
    adapter:
      name: colmap_camera_recovery
      config:
        executable: {executable}
        use_gpu: false
    depends_on: [ingest]
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["adapters", "healthcheck", "--config", str(config)],
    )

    assert result.exit_code == 0, result.output
    assert "ingest (ffmpeg_ingest): ok" in result.output
    assert f"colmap={executable}" in result.output


def test_cli_reports_invalid_paths_and_stages(tmp_path: Path) -> None:
    runner = CliRunner()
    missing = runner.invoke(
        app,
        [
            "run",
            "--input",
            str(tmp_path / "missing"),
            "--config",
            "configs/mock.yaml",
            "--run-dir",
            str(tmp_path / "run"),
        ],
    )
    assert missing.exit_code != 0
    assert "does not exist" in missing.output

    unknown_stage = runner.invoke(
        app,
        [
            "run",
            "--input",
            "examples/tabletop",
            "--config",
            "configs/mock.yaml",
            "--run-dir",
            str(tmp_path / "run"),
            "--from-stage",
            "unknown",
        ],
    )
    assert unknown_stage.exit_code != 0
    assert "from-stage 'unknown' does not exist" in unknown_stage.output


def test_cli_clean_end_to_end(tmp_path: Path) -> None:
    runner = CliRunner()
    run_dir = tmp_path / "run"
    result = runner.invoke(
        app,
        [
            "run",
            "--input",
            "examples/tabletop",
            "--config",
            "configs/mock.yaml",
            "--run-dir",
            str(run_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    validation = runner.invoke(app, ["validate-ir", str(run_dir / "scene_ir" / "scene.json")])
    assert validation.exit_code == 0, validation.output
    assert "valid Scene IR: tabletop_demo" in validation.output


def test_ingest_and_camera_inspection_and_trajectory_export(completed_run: Path) -> None:
    runner = CliRunner()
    ingest = runner.invoke(app, ["ingest", "inspect", str(completed_run)])
    assert ingest.exit_code == 0, ingest.output
    assert '"selected_frames": 3' in ingest.output

    camera = runner.invoke(app, ["camera", "inspect", str(completed_run)])
    assert camera.exit_code == 0, camera.output
    assert '"registered_frames": 3' in camera.output
    assert '"camera_model": "pinhole"' in camera.output

    output = completed_run / "trajectory.json"
    exported = runner.invoke(
        app,
        [
            "camera",
            "export-trajectory",
            str(completed_run),
            "--output",
            str(output),
        ],
    )
    assert exported.exit_code == 0, exported.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["poses"]) == 3
    assert payload["scale_status"] == "metric_scale_known"
    assert payload["coordinate_convention"]["world_frame"] == ("canonical_x_forward_y_left_z_up")
    assert payload["coordinate_convention"]["linear_units"] == "meters"
    assert "translation" in payload["poses"][0]["transform_world_from_camera"]
    assert "translation_m" not in payload["poses"][0]["transform_world_from_camera"]
