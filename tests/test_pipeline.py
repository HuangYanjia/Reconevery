from __future__ import annotations

import json
from pathlib import Path

import pytest

from recon2sim.config import AdapterConfig, PipelineConfig, StageConfig, load_config
from recon2sim.images import write_solid_png
from recon2sim.pipeline import PipelineConfigurationError, PipelineRunner


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
