from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from recon2sim.cli import app
from recon2sim.config import AdapterConfig, OutputConfig, PipelineConfig, StageConfig
from recon2sim.pipeline import OutputValidationError, PipelineRunner
from typer.testing import CliRunner


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
from pathlib import Path
import sys
counter = Path("counter.txt")
attempt = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(attempt))
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


def test_cli_help_uses_real_typer() -> None:
    runner = CliRunner()
    for arguments in (["--help"], ["run", "--help"], ["adapters", "--help"]):
        result = runner.invoke(app, arguments)
        assert result.exit_code == 0, result.output
        assert "Usage:" in result.output


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
