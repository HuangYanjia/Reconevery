from __future__ import annotations

import json
from pathlib import Path

import pytest

from recon2sim.adapters.command import CommandAdapter
from recon2sim.cli import app
from recon2sim.config import AdapterConfig, PipelineConfig, StageConfig, load_config
from recon2sim.ir import SceneIR
from recon2sim.pipeline import PipelineRunner
from recon2sim.storage import atomic_write_json
from typer.testing import CliRunner


def test_scene_ir_model_roundtrip() -> None:
    cfg = load_config(Path("configs/mock.yaml"))
    run = Path("/tmp/recon2sim_test_model")
    PipelineRunner(cfg, Path("examples/tabletop"), run).run()
    scene = SceneIR.model_validate_json((run / "scene_ir/scene.json").read_text())
    assert {o.object_id for o in scene.objects} >= {"floor", "table", "cup", "cabinet"}
    assert scene.metadata.source == "mock"


def test_json_schema_roundtrip() -> None:
    schema = SceneIR.model_json_schema()
    assert schema["title"] == "SceneIR"
    atomic_write_json(Path("/tmp/schema.json"), schema)
    assert json.loads(Path("/tmp/schema.json").read_text())["title"] == "SceneIR"


def test_malformed_scene_ir_negative() -> None:
    with pytest.raises(Exception):
        SceneIR.model_validate(
            {"metadata": {"scene_id": "x"}, "objects": [{"object_id": "a"}, {"object_id": "a"}]}
        )


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    run = tmp_path / "run"
    manifest = PipelineRunner(
        load_config(Path("configs/mock.yaml")), Path("examples/tabletop"), run
    ).run()
    assert manifest["stages"]["export"]["status"] == "succeeded"
    assert (run / "validation/report.json").exists()
    report = json.loads((run / "validation/report.json").read_text())
    assert any(i["severity"] == "warning" for i in report["issues"])


def test_resume_idempotency(tmp_path: Path) -> None:
    run = tmp_path / "run"
    cfg = load_config(Path("configs/mock.yaml"))
    PipelineRunner(cfg, Path("examples/tabletop"), run).run()
    manifest = PipelineRunner(cfg, Path("examples/tabletop"), run).run(resume=True)
    assert manifest["stages"]["ingest"]["status"] == "skipped"
    assert manifest["stages"]["ingest"]["skipped_reason"].startswith("already")


def test_partial_run_recovery(tmp_path: Path) -> None:
    run = tmp_path / "run"
    cfg = load_config(Path("configs/mock.yaml"))
    PipelineRunner(cfg, Path("examples/tabletop"), run).run(until_stage="camera_recovery")
    manifest = PipelineRunner(cfg, Path("examples/tabletop"), run).run(
        resume=True, from_stage="segmentation_tracking"
    )
    assert manifest["stages"]["export"]["status"] == "succeeded"


def test_adapter_failure_marks_failed(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        stages={
            "bad": StageConfig(
                adapter=AdapterConfig(
                    name="command", command=["python", "-c", "import sys; sys.exit(2)"]
                )
            )
        }
    )
    with pytest.raises(RuntimeError):
        PipelineRunner(cfg, Path("examples/tabletop"), tmp_path / "run").run()
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["stages"]["bad"]["status"] == "failed"


def test_command_timeout(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        stages={
            "slow": StageConfig(
                adapter=AdapterConfig(
                    name="command",
                    command=["python", "-c", "import time; time.sleep(2)"],
                    timeout_s=0.1,
                )
            )
        }
    )
    with pytest.raises(Exception):
        PipelineRunner(cfg, Path("examples/tabletop"), tmp_path / "run").run()


def test_atomic_write(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    atomic_write_json(p, {"a": 1})
    assert json.loads(p.read_text()) == {"a": 1}
    assert not list(tmp_path.glob(".x.json.*"))


def test_cli_smoke(tmp_path: Path) -> None:
    runner = CliRunner()
    run = tmp_path / "run"
    res = runner.invoke(
        app,
        [
            "run",
            "--input",
            "examples/tabletop",
            "--config",
            "configs/mock.yaml",
            "--run-dir",
            str(run),
        ],
    )
    assert res.exit_code == 0, res.output
    res = runner.invoke(app, ["validate-ir", str(run / "scene_ir/scene.json")])
    assert res.exit_code == 0


def test_command_healthcheck() -> None:
    assert CommandAdapter().healthcheck().ok
