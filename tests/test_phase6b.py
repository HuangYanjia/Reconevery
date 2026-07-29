from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from recon2sim.adapters.ingest import ProcessExecutionError
from recon2sim.artifacts import (
    Phase6BConsistencyReport,
    SceneAssemblyBundle,
    SceneAssemblyInputManifest,
    SceneAssemblyOverlapReport,
    SceneAssemblyPlan,
)
from recon2sim.assembly import multiply_matrix4, transform_point
from recon2sim.calibration import sha256_file
from recon2sim.cli import app
from recon2sim.config import PipelineConfig, load_config
from recon2sim.ir import SceneIR
from recon2sim.pipeline import PipelineRunner

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "examples/tabletop"
CONFIG = ROOT / "configs/phase6b_e2e_fake.yaml"


def _config(
    mode: str,
    *,
    preview_mode: str = "success",
) -> PipelineConfig:
    payload = load_config(CONFIG).model_dump(mode="json")
    payload["stages"]["scene_assembly_inputs"]["adapter"]["config"]["fake_mode"] = mode
    payload["stages"]["assembly_previews"]["adapter"]["config"]["fake_mode"] = preview_mode
    return PipelineConfig.model_validate(payload)


def _run(tmp_path: Path, mode: str, *, preview_mode: str = "success") -> Path:
    run_dir = tmp_path / mode
    PipelineRunner(_config(mode, preview_mode=preview_mode), INPUT, run_dir).run()
    return run_dir


@pytest.mark.parametrize(
    ("mode", "world_mode", "metric", "gravity"),
    [
        ("source_arbitrary_measured_only", "source_arbitrary", False, False),
        ("full_canonical_scene", "canonical_metric", True, True),
        ("metric_only_scene", "metric_unoriented", True, False),
        (
            "gravity_only_scene",
            "gravity_aligned_arbitrary_scale",
            False,
            True,
        ),
    ],
)
def test_calibration_optional_world_modes(
    tmp_path: Path,
    mode: str,
    world_mode: str,
    metric: bool,
    gravity: bool,
) -> None:
    plan = SceneAssemblyPlan.model_validate_json(
        (_run(tmp_path, mode) / "assembly/assembly_plan.json").read_text(encoding="utf-8")
    )
    assert plan.world.world_mode.value == world_mode
    assert plan.world.metric_scale_known is metric
    assert plan.world.gravity_alignment_known is gravity


def test_fake_phase6b_dag_dual_bundle_resume_and_cli(tmp_path: Path) -> None:
    run_dir = tmp_path / "phase6b"
    runner = PipelineRunner(_config("deployment_bundle_excluding_research_asset"), INPUT, run_dir)
    first = runner.run()
    assert all(item["last_execution"] == "executed" for item in first["stages"].values())
    plan_sha = sha256_file(run_dir / "assembly/assembly_plan.json")
    research = SceneAssemblyBundle.model_validate_json(
        (run_dir / "assembly/research_visual_bundle.json").read_text(encoding="utf-8")
    )
    deployment = SceneAssemblyBundle.model_validate_json(
        (run_dir / "assembly/deployment_eligible_visual_bundle.json").read_text(encoding="utf-8")
    )
    assert "cup_candidate_visual" in research.asset_ids
    assert "cup_candidate_visual" not in deployment.asset_ids
    assert "cup_measured" in deployment.asset_ids
    assert "global_context" not in deployment.asset_ids
    deployment_decision = next(
        item for item in deployment.object_decisions if item.object_id == "cup_0001"
    )
    assert deployment_decision.status.value == "measured_only"
    assert deployment_decision.selected_candidate_id is None
    assert deployment_decision.selected_visual_asset_ids == []
    report = Phase6BConsistencyReport.model_validate_json(
        (run_dir / "validation/phase6b_layered_scene_assembly.json").read_text(encoding="utf-8")
    )
    assert report.passed
    assert len(report.checks) == 25
    assert report.visual_scene_assembled
    assert not report.collision_generation_implemented
    assert not report.physics_identification_implemented
    resumed = runner.run(resume=True)
    assert all(item["last_execution"] == "cache_hit" for item in resumed["stages"].values())
    assert sha256_file(run_dir / "assembly/assembly_plan.json") == plan_sha
    cli = CliRunner()
    verify = cli.invoke(app, ["validation", "verify-phase6b", str(run_dir)])
    assert verify.exit_code == 0, verify.output
    inspect = cli.invoke(app, ["assembly", "inspect-object", str(run_dir), "cup_0001"])
    assert inspect.exit_code == 0, inspect.output
    assert json.loads(inspect.output)["decision"]["selected_candidate_id"] == "cup_candidate"
    previews = cli.invoke(app, ["assembly", "render-previews", str(run_dir)])
    assert previews.exit_code == 0, previews.output


def test_scene_ir_has_exact_phase6b_references(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "source_arbitrary_measured_only")
    scene = SceneIR.model_validate_json(
        (run_dir / "scene_ir/phase6b_layered_scene.json").read_text(encoding="utf-8")
    )
    reference = scene.metadata.scene_assembly
    assert reference is not None
    assert reference.assembly_plan_sha256 == sha256_file(run_dir / reference.assembly_plan_path)
    assert reference.research_bundle_sha256 == sha256_file(run_dir / reference.research_bundle_path)
    assert reference.deployment_bundle_sha256 == sha256_file(
        run_dir / reference.deployment_bundle_path
    )
    assert reference.compiler_manifest_sha256 == sha256_file(
        run_dir / reference.compiler_manifest_path
    )
    assert not reference.collision_ready
    assert not reference.sim_ready


def test_measured_anchor_retained_when_articulated_candidate_rejected(
    tmp_path: Path,
) -> None:
    run_dir = _run(tmp_path, "rejected_articulated_candidate")
    plan = SceneAssemblyPlan.model_validate_json(
        (run_dir / "assembly/assembly_plan.json").read_text(encoding="utf-8")
    )
    decision = plan.decisions[0]
    assert decision.status.value == "deferred_articulated_unresolved"
    assert decision.measured_motion is not None
    assert decision.selected_visual_asset_ids == []
    for name in ("research_visual_bundle.json", "deployment_eligible_visual_bundle.json"):
        bundle = SceneAssemblyBundle.model_validate_json(
            (run_dir / "assembly" / name).read_text(encoding="utf-8")
        )
        assert "cup_measured" in bundle.asset_ids
        assert "cup_candidate_visual" not in bundle.asset_ids


def test_production_candidate_appears_in_both_bundles(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "accepted_rigid_candidate")
    research = SceneAssemblyBundle.model_validate_json(
        (run_dir / "assembly/research_visual_bundle.json").read_text(encoding="utf-8")
    )
    deployment = SceneAssemblyBundle.model_validate_json(
        (run_dir / "assembly/deployment_eligible_visual_bundle.json").read_text(encoding="utf-8")
    )
    assert "cup_candidate_visual" in research.asset_ids
    assert "cup_candidate_visual" in deployment.asset_ids


def test_candidate_base_transform_composition(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "accepted_rigid_candidate")
    plan = SceneAssemblyPlan.model_validate_json(
        (run_dir / "assembly/assembly_plan.json").read_text(encoding="utf-8")
    )
    candidate = next(item for item in plan.assets if item.asset.asset_id == "cup_candidate_visual")
    assert transform_point(candidate.asset_to_assembly_world, (0.0, 0.0, 0.0)) == (
        2.0,
        3.0,
        4.0,
    )


def test_link_local_transform_and_articulation_are_preserved(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "accepted_articulated_candidate")
    plan = SceneAssemblyPlan.model_validate_json(
        (run_dir / "assembly/assembly_plan.json").read_text(encoding="utf-8")
    )
    candidate = next(item for item in plan.assets if item.asset.asset_id == "cup_candidate_visual")
    expected = multiply_matrix4(
        candidate.asset.object_to_source_world,
        candidate.asset.asset_to_object,
    )
    assert candidate.asset_to_assembly_world == expected
    assert transform_point(expected, (0.0, 0.0, 0.0)) == (2.0, 3.25, 4.0)
    decision = plan.decisions[0]
    assert decision.articulated_kinematic_bundle is not None
    kinematic_path = run_dir / decision.articulated_kinematic_bundle.path
    assert sha256_file(kinematic_path) == decision.articulated_kinematic_bundle.sha256
    kinematic = json.loads(kinematic_path.read_text(encoding="utf-8"))
    assert kinematic["joints"] == [{"joint_id": "drawer_joint", "joint_type": "prismatic"}]
    assert kinematic["prismatic_position_scale_to_m"] == 2.0


def test_reference_world_asset_receives_full_wrapper_once(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "full_canonical_scene")
    plan = SceneAssemblyPlan.model_validate_json(
        (run_dir / "assembly/assembly_plan.json").read_text(encoding="utf-8")
    )
    measured = next(item for item in plan.assets if item.asset.asset_id == "cup_measured")
    assert measured.asset_to_assembly_world == plan.world.source_world_to_assembly_world
    assert transform_point(measured.asset_to_assembly_world, (1.0, 1.0, 1.0)) == (
        3.0,
        0.0,
        2.5,
    )


def test_license_blocked_candidate_is_not_inserted(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "license_blocked_rigid_candidate")
    plan = SceneAssemblyPlan.model_validate_json(
        (run_dir / "assembly/assembly_plan.json").read_text(encoding="utf-8")
    )
    assert plan.decisions[0].status.value == "deferred_license_blocked"
    assert not plan.decisions[0].selected_visual_asset_ids


def test_overlap_diagnostics_are_non_destructive(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "deployment_bundle_excluding_research_asset")
    overlap = SceneAssemblyOverlapReport.model_validate_json(
        (run_dir / "assembly/overlap_diagnostics.json").read_text(encoding="utf-8")
    )
    diagnostic = overlap.diagnostics[0]
    assert diagnostic.candidate_measured_overlap_ratio is not None
    assert diagnostic.candidate_measured_overlap_ratio > 0
    assert diagnostic.potential_duplicate_geometry_ratio is not None
    assert not overlap.source_geometry_modified


def test_plan_digest_is_deterministic_across_runs(tmp_path: Path) -> None:
    first = SceneAssemblyPlan.model_validate_json(
        (
            _run(tmp_path / "first", "source_arbitrary_measured_only")
            / "assembly/assembly_plan.json"
        ).read_text(encoding="utf-8")
    )
    second = SceneAssemblyPlan.model_validate_json(
        (
            _run(tmp_path / "second", "source_arbitrary_measured_only")
            / "assembly/assembly_plan.json"
        ).read_text(encoding="utf-8")
    )
    assert first.deterministic_plan_digest == second.deterministic_plan_digest


@pytest.mark.parametrize("mode", ["empty_global_context", "no_selected_candidates"])
def test_empty_or_unselected_inputs_remain_measured_only(tmp_path: Path, mode: str) -> None:
    run_dir = _run(tmp_path, mode)
    plan = SceneAssemblyPlan.model_validate_json(
        (run_dir / "assembly/assembly_plan.json").read_text(encoding="utf-8")
    )
    assert plan.decisions[0].status.value == "measured_only"
    assert all(layer.role.value != "global_context" for layer in plan.layers)


def test_accepted_state_alignment_connects_lineages(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "accepted_state_alignment_lineage")
    plan = SceneAssemblyPlan.model_validate_json(
        (run_dir / "assembly/assembly_plan.json").read_text(encoding="utf-8")
    )
    assert plan.decisions[0].status.value == "measured_only"


def test_cross_lineage_assets_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unconnected reconstruction lineages"):
        PipelineRunner(
            _config("cross_lineage_asset_rejection"),
            INPUT,
            tmp_path / "cross_lineage",
        ).run()


def test_reference_world_asset_cannot_receive_object_transform(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="world-space evidence"):
        PipelineRunner(
            _config("double_world_transform"),
            INPUT,
            tmp_path / "double_world",
        ).run()


def test_articulated_candidate_cannot_masquerade_as_reference_world(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="candidate visual assets"):
        PipelineRunner(
            _config("double_articulated_transform"),
            INPUT,
            tmp_path / "double_articulated",
        ).run()


def test_missing_asset_hash_fails_before_planning(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="asset hash mismatch"):
        PipelineRunner(
            _config("missing_asset_hash"),
            INPUT,
            tmp_path / "bad_hash",
        ).run()


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be relative"):
        PipelineRunner(
            _config("path_escape"),
            INPUT,
            tmp_path / "path_escape",
        ).run()


def test_preview_worker_cannot_modify_upstream_plan(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="modified immutable upstream input"):
        PipelineRunner(
            _config(
                "source_arbitrary_measured_only",
                preview_mode="worker_modifying_upstream_assets",
            ),
            INPUT,
            tmp_path / "worker_modified",
        ).run()


def test_preview_glbs_are_explicit_diagnostics(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "accepted_rigid_candidate")
    for name in ("research_scene.glb", "deployment_scene.glb"):
        payload = (run_dir / "assembly/preview_assets" / name).read_bytes()
        assert payload[:4] == b"glTF"
    manifest = json.loads((run_dir / "assembly/preview_manifest.json").read_text(encoding="utf-8"))
    assert manifest["diagnostic_only"]
    assert not manifest["source_geometry_modified"]


def test_preview_material_loss_is_reported(tmp_path: Path) -> None:
    run_dir = _run(
        tmp_path,
        "preview_material_loss",
        preview_mode="preview_material_loss",
    )
    manifest = json.loads((run_dir / "assembly/preview_manifest.json").read_text(encoding="utf-8"))
    assert manifest["material_count_after"] < manifest["material_count_before"]
    assert manifest["representation_warnings"]


def test_preview_worker_timeout_is_fail_closed(tmp_path: Path) -> None:
    config = _config("source_arbitrary_measured_only", preview_mode="timeout")
    config.stages["assembly_previews"].adapter.timeout_s = 0.1
    with pytest.raises(ProcessExecutionError, match="timed out"):
        PipelineRunner(config, INPUT, tmp_path / "timeout").run()


def test_input_manifest_rejects_unapproved_production_license() -> None:
    payload = {
        "schema_version": "0.1.0",
        "assembly_id": "invalid",
        "primary_lineage_id": "lineage",
        "lineages": [
            {
                "lineage_id": "lineage",
                "frame_sequence_digest": "0" * 64,
                "camera_reconstruction": {"path": "camera.json", "sha256": "0" * 64},
                "source_scene_ir": {"path": "scene.json", "sha256": "1" * 64},
                "world_frame": "colmap_arbitrary",
            }
        ],
        "source_scene_ir": {"path": "scene.json", "sha256": "1" * 64},
        "assets": [
            {
                "asset_id": "measured",
                "object_id": "object",
                "lineage_id": "lineage",
                "role": "measured_anchor",
                "source": "measured",
                "asset_path": "measured.ply",
                "asset_sha256": "2" * 64,
                "format": "ply",
                "asset_native_space": "reference_world",
                "asset_to_object": [
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                ],
                "object_to_source_world": [
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                ],
                "license": {
                    "license_id": "unreviewed",
                    "license_name": "Unknown",
                    "research_evaluation_allowed": True,
                    "production_selectable": True,
                    "commercial_review_status": "not_reviewed",
                },
            }
        ],
        "objects": [
            {
                "object_id": "object",
                "lineage_id": "lineage",
                "asset_type": "rigid",
                "measured_anchor_asset_ids": ["measured"],
                "upstream_status": "measured",
            }
        ],
    }
    with pytest.raises(ValidationError, match="require license approval"):
        SceneAssemblyInputManifest.model_validate(payload)
