from __future__ import annotations

import json
from pathlib import Path

import pytest

from recon2sim.adapters import (
    REGISTRY,
    HealthcheckResult,
    OutputSpec,
    StageContext,
    StageResult,
)
from recon2sim.config import AdapterConfig, PipelineConfig, StageConfig, load_config
from recon2sim.images import write_solid_png
from recon2sim.pipeline import PipelineConfigurationError, PipelineRunner


class DeterministicFilesAdapter:
    name = "deterministic_files_test"
    version = "0.1.0"

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return HealthcheckResult(True, "test adapter available")

    def prepare(self, context: StageContext) -> None:
        context.run_dir.mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(path, "test", "text/plain", "test")
            for path in context.config.adapter.config["paths"]
        ]

    def run(self, context: StageContext) -> StageResult:
        content = str(context.config.adapter.config["content"])
        for path in context.config.adapter.config["paths"]:
            destination = context.path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(f"{path}:{content}", encoding="utf-8")
        return StageResult()


class InterruptingAdapter:
    name = "interrupting_test"
    version = "0.1.0"

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return HealthcheckResult(True, "test adapter available")

    def prepare(self, context: StageContext) -> None:
        context.path("prepared").write_text("preserved", encoding="utf-8")

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [OutputSpec("never.txt", "test", "text/plain", "test")]

    def run(self, context: StageContext) -> StageResult:
        if context.config.adapter.config["exception"] == "keyboard":
            raise KeyboardInterrupt("stop")
        raise SystemExit("stop")


def _simple_config(stages: dict[str, list[str]]) -> PipelineConfig:
    return PipelineConfig(
        stages={
            name: StageConfig(
                adapter=AdapterConfig(name="mock_export"),
                depends_on=dependencies,
            )
            for name, dependencies in stages.items()
        }
    )


def test_dag_rejects_unknown_dependency(tmp_path: Path) -> None:
    config = _simple_config({"assemble": ["missing_camera"]})
    runner = PipelineRunner(config, tmp_path, tmp_path / "run")
    with pytest.raises(PipelineConfigurationError, match="unknown dependencies.*missing_camera"):
        runner.order()


def test_dag_rejects_cycle(tmp_path: Path) -> None:
    config = _simple_config({"camera": ["tracking"], "tracking": ["camera"]})
    runner = PipelineRunner(config, tmp_path, tmp_path / "run")
    with pytest.raises(PipelineConfigurationError, match="camera -> tracking -> camera"):
        runner.order()


def test_stage_range_validation(tmp_path: Path) -> None:
    config = _simple_config({"first": [], "second": ["first"]})
    runner = PipelineRunner(config, tmp_path, tmp_path / "run")
    with pytest.raises(PipelineConfigurationError, match="from-stage 'unknown' does not exist"):
        runner.selected("unknown", None)
    with pytest.raises(PipelineConfigurationError, match="occurs after"):
        runner.selected("second", "first")


def test_enabled_stage_rejects_disabled_dependency(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    config = PipelineConfig(
        stages={
            "disabled": StageConfig(
                enabled=False,
                adapter=AdapterConfig(name="mock_export"),
            ),
            "consumer": StageConfig(
                adapter=AdapterConfig(name="mock_export"),
                depends_on=["disabled"],
            ),
        }
    )
    with pytest.raises(PipelineConfigurationError, match="depends on disabled stage 'disabled'"):
        PipelineRunner(config, input_dir, tmp_path / "run").run()


def test_repeated_resume_keeps_success_status(input_dir: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config = load_config(Path("configs/mock.yaml"))
    first = PipelineRunner(config, input_dir, run_dir).run()
    first_signatures = {
        name: entry["execution_signature"] for name, entry in first["stages"].items()
    }
    second = PipelineRunner(config, input_dir, run_dir).run(resume=True)
    third = PipelineRunner(config, input_dir, run_dir).run(resume=True)

    for name, entry in second["stages"].items():
        assert entry["status"] == "succeeded", name
        assert entry["last_execution"] == "cache_hit", name
        assert entry["execution_signature"] == first_signatures[name]
    assert all(entry["last_execution"] == "cache_hit" for entry in third["stages"].values())


def test_forced_identical_upstream_rerun_does_not_invalidate_downstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(REGISTRY, DeterministicFilesAdapter.name, DeterministicFilesAdapter)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    run_dir = tmp_path / "run"
    config = PipelineConfig(
        stages={
            "upstream": StageConfig(
                adapter=AdapterConfig(
                    name=DeterministicFilesAdapter.name,
                    config={"paths": ["data/upstream.txt"], "content": "stable"},
                )
            ),
            "downstream": StageConfig(
                adapter=AdapterConfig(
                    name=DeterministicFilesAdapter.name,
                    config={"paths": ["data/downstream.txt"], "content": "stable"},
                ),
                depends_on=["upstream"],
            ),
        }
    )
    first = PipelineRunner(config, input_dir, run_dir).run()
    downstream_signature = first["stages"]["downstream"]["execution_signature"]

    forced = PipelineRunner(config, input_dir, run_dir).run(
        from_stage="upstream",
        until_stage="upstream",
    )
    assert forced["stages"]["upstream"]["execution_count"] == 2
    assert (
        forced["stages"]["upstream"]["execution_signature"]
        == first["stages"]["upstream"]["execution_signature"]
    )

    resumed = PipelineRunner(config, input_dir, run_dir).run(resume=True)
    assert resumed["stages"]["downstream"]["last_execution"] == "cache_hit"
    assert resumed["stages"]["downstream"]["execution_count"] == 1
    assert resumed["stages"]["downstream"]["execution_signature"] == downstream_signature


@pytest.mark.parametrize(
    ("exception_name", "exception_type"),
    [("keyboard", KeyboardInterrupt), ("system_exit", SystemExit)],
)
def test_interruptions_are_not_retried_and_preserve_attempt_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_name: str,
    exception_type: type[BaseException],
) -> None:
    monkeypatch.setitem(REGISTRY, InterruptingAdapter.name, InterruptingAdapter)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    run_dir = tmp_path / "run"
    config = PipelineConfig(
        stages={
            "interrupt": StageConfig(
                adapter=AdapterConfig(
                    name=InterruptingAdapter.name,
                    retries=3,
                    config={"exception": exception_name},
                )
            )
        }
    )

    with pytest.raises(exception_type):
        PipelineRunner(config, input_dir, run_dir).run()

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["stages"]["interrupt"]
    assert entry["status"] == "interrupted"
    assert entry["last_execution"] == "interrupted"
    assert [attempt["status"] for attempt in entry["attempts"]] == ["interrupted"]
    workspace = run_dir / entry["attempts"][0]["workspace"]
    assert (workspace / "prepared").read_text(encoding="utf-8") == "preserved"
    assert not (run_dir / "work" / "interrupt" / "attempt_2").exists()


def test_multi_file_promotion_failure_rolls_back_complete_previous_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(REGISTRY, DeterministicFilesAdapter.name, DeterministicFilesAdapter)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    run_dir = tmp_path / "run"

    def config(content: str) -> PipelineConfig:
        return PipelineConfig(
            stages={
                "producer": StageConfig(
                    adapter=AdapterConfig(
                        name=DeterministicFilesAdapter.name,
                        config={
                            "paths": ["canonical/one.txt", "canonical/two.txt"],
                            "content": content,
                        },
                    )
                )
            }
        )

    first = PipelineRunner(config("old"), input_dir, run_dir).run()
    previous = {
        path: (run_dir / path).read_bytes() for path in ("canonical/one.txt", "canonical/two.txt")
    }
    runner = PipelineRunner(config("new"), input_dir, run_dir)
    original_replace = runner._replace_promoted_output
    calls = 0

    def fail_second_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected promotion failure")
        original_replace(source, destination)

    monkeypatch.setattr(runner, "_replace_promoted_output", fail_second_replace)
    with pytest.raises(OSError, match="injected promotion failure"):
        runner.run()

    assert {
        path: (run_dir / path).read_bytes() for path in ("canonical/one.txt", "canonical/two.txt")
    } == previous
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["producer"]["artifacts"] == first["stages"]["producer"]["artifacts"]


def test_input_byte_change_invalidates_all_downstream_stages(
    input_dir: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    config = load_config(Path("configs/mock.yaml"))
    PipelineRunner(config, input_dir, run_dir).run()
    write_solid_png(input_dir / "frames" / "frame_001.png", 32, 24, (5, 10, 15))
    manifest = PipelineRunner(config, input_dir, run_dir).run(resume=True)
    assert all(entry["last_execution"] == "executed" for entry in manifest["stages"].values())


def test_config_change_selectively_invalidates_dependents(input_dir: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config = load_config(Path("configs/mock.yaml"))
    PipelineRunner(config, input_dir, run_dir).run()
    config.stages["segmentation_tracking"].adapter.config["variant"] = "changed"
    manifest = PipelineRunner(config, input_dir, run_dir).run(resume=True)

    assert manifest["stages"]["ingest"]["last_execution"] == "cache_hit"
    assert manifest["stages"]["camera_recovery"]["last_execution"] == "cache_hit"
    assert manifest["stages"]["segmentation_tracking"]["last_execution"] == "executed"
    assert manifest["stages"]["global_reconstruction"]["last_execution"] == "cache_hit"
    for stage in [
        "object_reconstruction",
        "scene_ir_assembly",
        "scene_compilation",
        "validation",
        "export",
    ]:
        assert manifest["stages"][stage]["last_execution"] == "executed"


def test_upstream_output_modification_invalidates_dependents(
    input_dir: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    config = load_config(Path("configs/mock.yaml"))
    PipelineRunner(config, input_dir, run_dir).run()
    camera_path = run_dir / "camera" / "reconstruction.json"
    camera_path.write_text(camera_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    manifest = PipelineRunner(config, input_dir, run_dir).run(resume=True)

    assert manifest["stages"]["ingest"]["last_execution"] == "cache_hit"
    for stage in [
        "camera_recovery",
        "segmentation_tracking",
        "global_reconstruction",
        "object_reconstruction",
        "scene_ir_assembly",
        "scene_compilation",
        "validation",
        "export",
    ]:
        assert manifest["stages"][stage]["last_execution"] == "executed"


def test_selective_downstream_rerun(input_dir: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config = load_config(Path("configs/mock.yaml"))
    first = PipelineRunner(config, input_dir, run_dir).run()
    previous_count = first["stages"]["object_reconstruction"]["execution_count"]
    rerun = PipelineRunner(config, input_dir, run_dir).run(from_stage="object_reconstruction")

    assert rerun["stages"]["global_reconstruction"]["last_execution"] == "not_selected"
    assert rerun["stages"]["object_reconstruction"]["execution_count"] == previous_count + 1
    for stage in [
        "object_reconstruction",
        "scene_ir_assembly",
        "scene_compilation",
        "validation",
        "export",
    ]:
        assert rerun["stages"][stage]["last_execution"] == "executed"


def test_manifest_contains_expanded_artifact_records(completed_run: Path) -> None:
    manifest = json.loads((completed_run / "manifest.json").read_text(encoding="utf-8"))
    record = manifest["stages"]["scene_ir_assembly"]["artifacts"][0]
    assert set(record) == {
        "relative_path",
        "artifact_type",
        "media_type",
        "sha256",
        "size_bytes",
        "producer_stage",
        "producer_adapter",
        "source_type",
        "schema_identifier",
    }
    assert not Path(record["relative_path"]).is_absolute()
    assert len(record["sha256"]) == 64
