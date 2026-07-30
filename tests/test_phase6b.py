from __future__ import annotations

import json
from collections.abc import Callable
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
from recon2sim.assembly import (
    IDENTITY_MATRIX4,
    multiply_matrix4,
    resolve_world,
    transform_point,
)
from recon2sim.assembly_sources import normalize_assembly_manifest
from recon2sim.calibration import sha256_file
from recon2sim.cli import app
from recon2sim.config import PipelineConfig, load_config
from recon2sim.ir import SceneIR
from recon2sim.pipeline import PipelineRunner

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "examples/tabletop"
CONFIG = ROOT / "configs/phase6b_e2e_fake.yaml"
PHASE5B_CONFIG = ROOT / "configs/phase5b_e2e_fake.yaml"
PHASE5C_CONFIG = ROOT / "configs/phase5c_e2e_fake.yaml"
PHASE6A_CONFIG = ROOT / "configs/phase6a_e2e_fake.yaml"
PHASE3_CONFIG = ROOT / "configs/phase3_e2e_fake.yaml"


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


@pytest.fixture(scope="module")
def phase5b_source_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("phase6b-source-binding")
    PipelineRunner(load_config(PHASE5B_CONFIG), INPUT, run_dir).run()
    return run_dir


@pytest.fixture(scope="module")
def phase3_source_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("phase6b-global-context-binding")
    PipelineRunner(load_config(PHASE3_CONFIG), INPUT, run_dir).run()
    return run_dir


@pytest.fixture(scope="module")
def phase6a_source_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("phase6b-calibration-binding")
    PipelineRunner(load_config(PHASE6A_CONFIG), INPUT, run_dir).run()
    return run_dir


@pytest.fixture(scope="module")
def phase6a_gravity_only_source_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    config = load_config(PHASE6A_CONFIG)
    world_stage = config.stages["world_calibration"]
    world_adapter = world_stage.adapter.model_copy(
        update={"config": {**world_stage.adapter.config, "fake_mode": "gravity_only"}}
    )
    config = config.model_copy(
        update={
            "stages": {
                key: (
                    value.model_copy(update={"adapter": world_adapter})
                    if key == "world_calibration"
                    else value
                )
                for key, value in config.stages.items()
                if key != "phase6a_consistency_validation"
            }
        }
    )
    run_dir = tmp_path_factory.mktemp("phase6b-gravity-only-binding")
    PipelineRunner(config, INPUT, run_dir).run()
    return run_dir


@pytest.fixture(scope="module")
def phase6a_nonidentity_source_run(
    tmp_path_factory: pytest.TempPathFactory,
    phase6a_source_run: Path,
) -> Path:
    source_run = tmp_path_factory.mktemp("phase6b-nonidentity-scene-source")
    PipelineRunner(
        _config("source_arbitrary_measured_only"),
        INPUT,
        source_run,
    ).run()
    root = tmp_path_factory.mktemp("phase6b-full-canonical-nonidentity-binding")
    for relative, source in (
        (
            "calibration/source/camera_reconstruction.json",
            source_run / "assembly/source/camera_reconstruction.json",
        ),
        (
            "calibration/source/scene_ir.json",
            source_run / "assembly/source/scene_ir.json",
        ),
        (
            "calibration/world_calibration.json",
            phase6a_source_run / "calibration/world_calibration.json",
        ),
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    wrapper = json.loads(
        (phase6a_source_run / "calibration/canonical_scene_wrapper.json").read_text(
            encoding="utf-8"
        )
    )
    wrapper["source_scene_ir_sha256"] = sha256_file(root / "calibration/source/scene_ir.json")
    wrapper["source_camera_reconstruction_sha256"] = sha256_file(
        root / "calibration/source/camera_reconstruction.json"
    )
    wrapper["asset_mappings"] = []
    destination = root / "calibration/canonical_scene_wrapper.json"
    destination.write_text(
        json.dumps(wrapper, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture(scope="module")
def phase5c_source_run(
    tmp_path_factory: pytest.TempPathFactory,
    phase5b_source_run: Path,
) -> Path:
    run_dir = tmp_path_factory.mktemp("phase6b-articulated-source-binding")
    PipelineRunner(load_config(PHASE5C_CONFIG), INPUT, run_dir).run()
    _materialize_phase5c_camera(run_dir, phase5b_source_run)
    return run_dir


@pytest.fixture(scope="module")
def phase5c_rejected_source_run(
    tmp_path_factory: pytest.TempPathFactory,
    phase5b_source_run: Path,
) -> Path:
    config = load_config(PHASE5C_CONFIG)
    stage = config.stages["articulation_evaluation"]
    adapter = stage.adapter.model_copy(
        update={"config": {**stage.adapter.config, "fake_mode": "missing_heldout_views"}}
    )
    config = config.model_copy(
        update={
            "stages": {
                **config.stages,
                "articulation_evaluation": stage.model_copy(update={"adapter": adapter}),
            }
        }
    )
    run_dir = tmp_path_factory.mktemp("phase6b-articulated-rejected-binding")
    PipelineRunner(config, INPUT, run_dir).run()
    _materialize_phase5c_camera(run_dir, phase5b_source_run)
    return run_dir


def _source_ref(
    root: Path,
    path: str,
    artifact_type: str,
) -> dict[str, str]:
    return {
        "path": path,
        "sha256": sha256_file(root / path),
        "artifact_type": artifact_type,
    }


def _materialize_phase5c_camera(root: Path, template_root: Path) -> None:
    capture = json.loads(
        (root / "reconstruction/articulation/capture_manifest.json").read_text(encoding="utf-8")
    )
    template = json.loads(
        (template_root / "camera/reconstruction.json").read_text(encoding="utf-8")
    )
    for state in capture["states"]:
        camera = json.loads(json.dumps(template, sort_keys=True))
        frame_ids = state["registered_frame_ids"]
        template_pose = camera["poses"][0]
        camera["poses"] = [json.loads(json.dumps(template_pose, sort_keys=True)) for _ in frame_ids]
        for pose, frame_id in zip(camera["poses"], frame_ids, strict=True):
            pose["frame_id"] = frame_id
        camera["registered_frame_ids"] = frame_ids
        camera["unregistered_frame_ids"] = []
        camera["frame_sequence_digest"] = state["frame_sequence_digest"]
        path = root / f"assembly_binding/{state['state_id']}_camera_reconstruction.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(camera, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        state["camera_reconstruction_sha256"] = sha256_file(path)
    binding_capture_path = root / "assembly_binding/capture_manifest.json"
    binding_capture_path.write_text(
        json.dumps(capture, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    alignment = json.loads(
        (root / "reconstruction/articulation/state_alignment.json").read_text(encoding="utf-8")
    )
    alignment["capture_manifest_sha256"] = sha256_file(binding_capture_path)
    (root / "assembly_binding/state_alignment.json").write_text(
        json.dumps(alignment, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _rigid_source_manifest(root: Path) -> dict[str, object]:
    selection_path = "reconstruction/completion/selection.json"
    evaluation_path = "reconstruction/completion/evaluation_manifest.json"
    registration_path = "reconstruction/completion/registration_manifest.json"
    generation_paths = [
        "reconstruction/completion/sam3d_generation_manifest.json",
        "reconstruction/completion/trellis2_generation_manifest.json",
        "reconstruction/completion/measured_generation_manifest.json",
    ]
    selection = json.loads((root / selection_path).read_text(encoding="utf-8"))
    selected = next(
        item for item in selection["objects"] if item["best_research_candidate"] is not None
    )
    candidate_id = selected["best_research_candidate"]
    evaluation = json.loads((root / evaluation_path).read_text(encoding="utf-8"))
    evaluated = next(
        item for item in evaluation["evaluations"] if item["candidate_id"] == candidate_id
    )
    generation_path = next(
        path
        for path in generation_paths
        if any(
            item["candidate_id"] == candidate_id
            for item in json.loads((root / path).read_text(encoding="utf-8"))["candidates"]
        )
    )
    generation = json.loads((root / generation_path).read_text(encoding="utf-8"))
    candidate = next(
        item for item in generation["candidates"] if item["candidate_id"] == candidate_id
    )
    native = next(
        item
        for item in candidate["native_assets"]
        if item["asset_id"] == evaluated["selection_asset_id"]
    )
    measured_path = "reconstruction/measured_objects/geometry_manifest.json"
    measured = json.loads((root / measured_path).read_text(encoding="utf-8"))
    hypothesis = next(
        item for item in measured["hypotheses"] if item["object_id"] == selected["object_id"]
    )
    point_cloud = hypothesis["point_cloud"]
    assert point_cloud is not None
    identity = list(
        (
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    )
    camera_ref = _source_ref(root, "camera/reconstruction.json", "camera_reconstruction")
    scene_ref = _source_ref(root, "scene_ir/phase5b_scene.json", "source_scene_ir")
    selection_ref = _source_ref(root, selection_path, "rigid_selection")
    evaluation_ref = _source_ref(root, evaluation_path, "rigid_evaluation")
    registration_ref = _source_ref(root, registration_path, "rigid_registration")
    generation_refs = [_source_ref(root, path, "rigid_generation") for path in generation_paths]
    return {
        "schema_version": "0.2.0",
        "assembly_id": "source_binding",
        "primary_lineage_id": "lineage",
        "lineages": [
            {
                "lineage_id": "lineage",
                "camera_reconstruction": camera_ref,
                "source_scene_ir": scene_ref,
            }
        ],
        "source_scene_ir": scene_ref,
        "assets": [
            {
                "asset_id": "measured",
                "object_id": selected["object_id"],
                "lineage_id": "lineage",
                "role": "measured_anchor",
                "asset_path": point_cloud["relative_path"],
                "source_native_asset_path": point_cloud["relative_path"],
                "asset_sha256": point_cloud["sha256"],
                "format": "ply",
                "asset_to_object": identity,
                "object_to_source_world": identity,
                "measured_geometry": _source_ref(
                    root,
                    measured_path,
                    "measured_geometry",
                ),
            },
            {
                "asset_id": "candidate",
                "object_id": selected["object_id"],
                "lineage_id": "lineage",
                "role": "visual_completion",
                "asset_path": native["relative_path"],
                "asset_sha256": native["sha256"],
                "format": "glb" if native["relative_path"].endswith(".glb") else "ply",
                "candidate_id": candidate_id,
            },
        ],
        "objects": [
            {
                "object_id": selected["object_id"],
                "lineage_id": "lineage",
                "asset_type": "rigid",
                "measured_anchor_asset_ids": ["measured"],
                "candidate_asset_ids": ["candidate"],
                "rigid_selection_artifact": selection_ref,
                "rigid_evaluation_artifact": evaluation_ref,
                "rigid_registration_artifact": registration_ref,
                "rigid_generation_artifacts": generation_refs,
            }
        ],
    }


def _articulated_source_manifest(root: Path) -> dict[str, object]:
    articulation_root = Path("reconstruction/articulation")
    selection_path = str(articulation_root / "selection.json")
    candidate_path = str(articulation_root / "candidate_manifest.json")
    evaluation_path = str(articulation_root / "evaluation_manifest.json")
    fitting_path = str(articulation_root / "fitting_manifest.json")
    assignment_path = str(articulation_root / "link_assignments.json")
    measured_path = str(articulation_root / "measured_states/manifest.json")
    motion_path = str(articulation_root / "measured_motion.json")
    selection = json.loads((root / selection_path).read_text(encoding="utf-8"))
    selected = selection["objects"][0]
    candidate_manifest = json.loads((root / candidate_path).read_text(encoding="utf-8"))
    candidate_id = selected["best_research_articulated_candidate"]
    if candidate_id is None:
        candidate_id = next(
            item["candidate_id"]
            for item in candidate_manifest["candidates"]
            if item["source_family"] == "artvip"
        )
    candidate = next(
        item for item in candidate_manifest["candidates"] if item["candidate_id"] == candidate_id
    )
    measured_manifest = json.loads((root / measured_path).read_text(encoding="utf-8"))
    measured = next(
        item
        for item in measured_manifest["geometries"]
        if item["articulated_object_id"] == selected["articulated_object_id"]
        and item["state_id"] == "state_000"
        and item["part_id"] == "cabinet_body"
    )
    camera_path = "assembly_binding/state_000_camera_reconstruction.json"
    camera_ref = _source_ref(root, camera_path, "camera_reconstruction")
    scene_ref = _source_ref(root, "scene_ir/phase5c_scene.json", "source_scene_ir")
    assets: list[dict[str, object]] = [
        {
            "asset_id": "cabinet_body_measured",
            "object_id": selected["articulated_object_id"],
            "part_id": "cabinet_body",
            "lineage_id": "lineage",
            "role": "measured_anchor",
            "asset_path": measured["measured_point_cloud_path"],
            "source_native_asset_path": measured["measured_point_cloud_path"],
            "asset_sha256": measured["measured_point_cloud_sha256"],
            "format": "ply",
            "asset_to_object": list(IDENTITY_MATRIX4),
            "object_to_source_world": list(IDENTITY_MATRIX4),
            "measured_geometry": _source_ref(root, measured_path, "measured_geometry"),
        }
    ]
    candidate_asset_ids: list[str] = []
    for link in candidate["links"]:
        for index, path in enumerate(link["visual_asset_paths"]):
            asset_id = f"candidate_{link['link_id']}_{index}"
            candidate_asset_ids.append(asset_id)
            assets.append(
                {
                    "asset_id": asset_id,
                    "object_id": selected["articulated_object_id"],
                    "lineage_id": "lineage",
                    "role": "articulated_visual",
                    "asset_path": path,
                    "asset_sha256": link["visual_asset_hashes"][path],
                    "format": "glb" if path.endswith(".glb") else "ply",
                    "candidate_id": candidate_id,
                    "link_id": link["link_id"],
                }
            )
    object_input: dict[str, object] = {
        "object_id": selected["articulated_object_id"],
        "lineage_id": "lineage",
        "asset_type": "articulated",
        "measured_anchor_asset_ids": ["cabinet_body_measured"],
        "candidate_asset_ids": candidate_asset_ids,
        "articulated_selection_artifact": _source_ref(
            root, selection_path, "articulated_selection"
        ),
        "articulated_candidate_manifest": _source_ref(
            root, candidate_path, "articulated_candidate_manifest"
        ),
        "articulated_evaluation_artifact": _source_ref(
            root, evaluation_path, "articulated_evaluation"
        ),
        "articulated_fitting_artifact": _source_ref(root, fitting_path, "articulated_fitting"),
        "articulated_link_assignment_artifact": _source_ref(
            root, assignment_path, "articulated_link_assignment"
        ),
        "measured_motion": _source_ref(root, motion_path, "measured_motion"),
    }
    if selected["selected_candidate_id"] is not None:
        object_input["selected_identity_manifest"] = _source_ref(
            root,
            selected["selected_identity_manifest_path"],
            "selected_identity_manifest",
        )
        object_input["kinematic_bundle"] = _source_ref(
            root,
            selected["kinematic_bundle_path"],
            "kinematic_bundle",
        )
    return {
        "schema_version": "0.2.0",
        "assembly_id": "articulated_source_binding",
        "primary_lineage_id": "lineage",
        "lineages": [
            {
                "lineage_id": "lineage",
                "camera_reconstruction": camera_ref,
                "source_scene_ir": scene_ref,
            }
        ],
        "source_scene_ir": scene_ref,
        "assets": assets,
        "objects": [object_input],
    }


def _global_context_source_manifest(
    root: Path,
    *,
    asset_format: str = "glb",
) -> dict[str, object]:
    metadata = json.loads(
        (root / "reconstruction/global/metadata.json").read_text(encoding="utf-8")
    )
    asset_path = (
        metadata["scene_asset_path"] if asset_format == "glb" else metadata["mesh_asset_path"]
    )
    camera_ref = _source_ref(root, "camera/reconstruction.json", "camera_reconstruction")
    scene_ref = _source_ref(root, "scene_ir/scene.json", "source_scene_ir")
    return {
        "schema_version": "0.3.0",
        "assembly_id": "global_context_source_binding",
        "primary_lineage_id": "lineage",
        "lineages": [
            {
                "lineage_id": "lineage",
                "camera_reconstruction": camera_ref,
                "source_scene_ir": scene_ref,
            }
        ],
        "source_scene_ir": scene_ref,
        "assets": [
            {
                "asset_id": "global_context",
                "object_id": None,
                "lineage_id": "lineage",
                "role": "global_context",
                "source": "generated",
                "asset_path": asset_path,
                "asset_sha256": sha256_file(root / asset_path),
                "source_native_asset_path": asset_path,
                "format": asset_format,
                "asset_native_space": "global_context",
                "asset_to_object": list(IDENTITY_MATRIX4),
                "object_to_source_world": list(IDENTITY_MATRIX4),
                "global_scene_reconstruction": _source_ref(
                    root,
                    "reconstruction/global/metadata.json",
                    "phase3_global_reconstruction",
                ),
                "license_source_record": _source_ref(
                    root,
                    "reconstruction/global/worker_manifest.json",
                    "global_context_manifest",
                ),
            }
        ],
        "objects": [],
    }


def _calibration_source_manifest(root: Path) -> dict[str, object]:
    camera_ref = _source_ref(
        root,
        "calibration/source/camera_reconstruction.json",
        "camera_reconstruction",
    )
    scene_ref = _source_ref(
        root,
        "calibration/source/scene_ir.json",
        "source_scene_ir",
    )
    return {
        "schema_version": "0.2.0",
        "assembly_id": "calibration_binding",
        "primary_lineage_id": "lineage",
        "lineages": [
            {
                "lineage_id": "lineage",
                "camera_reconstruction": camera_ref,
                "source_scene_ir": scene_ref,
            }
        ],
        "source_scene_ir": scene_ref,
        "calibration_artifact": _source_ref(
            root,
            "calibration/world_calibration.json",
            "world_calibration",
        ),
        "canonical_wrapper": _source_ref(
            root,
            "calibration/canonical_scene_wrapper.json",
            "canonical_wrapper",
        ),
        "assets": [],
        "objects": [],
    }


def _run_source_bound_calibration_assembly(
    root: Path,
    raw: dict[str, object],
    *,
    run_name: str,
) -> Path:
    payload = json.loads(json.dumps(raw))

    def add_local_paths(value: object) -> None:
        if isinstance(value, dict):
            path = value.get("path")
            if path is not None and value.get("artifact_type") is not None:
                value["local_path"] = str(root / path)
            for child in value.values():
                add_local_paths(child)
        elif isinstance(value, list):
            for child in value:
                add_local_paths(child)

    add_local_paths(payload)
    manifest_path = root / f"assembly_binding/{run_name}.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    config = _config("source_arbitrary_measured_only")
    stage = config.stages["scene_assembly_inputs"]
    adapter = stage.adapter.model_copy(
        update={
            "config": {
                **stage.adapter.config,
                "fake_mode": None,
                "manifest_path": str(manifest_path),
            }
        }
    )
    config = config.model_copy(
        update={
            "stages": {
                **config.stages,
                "scene_assembly_inputs": stage.model_copy(update={"adapter": adapter}),
            }
        }
    )
    run_dir = root / run_name
    PipelineRunner(config, INPUT, run_dir).run()
    return run_dir


def _connected_articulated_source_manifest(
    root: Path,
    *,
    child_camera_path: str = "assembly_binding/state_001_camera_reconstruction.json",
    capture_path: str = "assembly_binding/capture_manifest.json",
    alignment_path: str = "assembly_binding/state_alignment.json",
    alignment_state_id: str = "state_001",
) -> dict[str, object]:
    raw = _articulated_source_manifest(root)
    raw["lineages"].append(  # type: ignore[union-attr]
        {
            "lineage_id": "state_001_lineage",
            "camera_reconstruction": _source_ref(
                root,
                child_camera_path,
                "camera_reconstruction",
            ),
            "source_scene_ir": raw["source_scene_ir"],
            "connected_to_lineage_id": "lineage",
            "accepted_alignment": _source_ref(
                root,
                alignment_path,
                "state_alignment",
            ),
            "alignment_capture_manifest": _source_ref(
                root,
                capture_path,
                "articulation_capture_manifest",
            ),
            "alignment_state_id": alignment_state_id,
        }
    )
    return raw


def _write_capture_alignment_revision(
    root: Path,
    name: str,
    mutate_capture: Callable[[dict[str, object]], None],
) -> tuple[str, str]:
    capture = json.loads(
        (root / "assembly_binding/capture_manifest.json").read_text(encoding="utf-8")
    )
    mutate_capture(capture)
    capture_relative = f"assembly_binding/{name}_capture.json"
    (root / capture_relative).write_text(
        json.dumps(capture, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    alignment = json.loads(
        (root / "assembly_binding/state_alignment.json").read_text(encoding="utf-8")
    )
    alignment["capture_manifest_sha256"] = sha256_file(root / capture_relative)
    alignment_relative = f"assembly_binding/{name}_alignment.json"
    (root / alignment_relative).write_text(
        json.dumps(alignment, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return capture_relative, alignment_relative


@pytest.mark.parametrize(
    ("mode", "world_mode", "metric", "gravity"),
    [
        ("source_arbitrary_measured_only", "source_arbitrary", False, False),
        ("full_canonical_scene", "canonical_metric", True, True),
        ("metric_only_scene", "metric_unoriented", True, False),
        (
            "gravity_only_scene",
            "source_arbitrary",
            False,
            False,
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
    if mode == "gravity_only_scene":
        assert plan.world.source_world_to_assembly_world == IDENTITY_MATRIX4
        assert plan.world.warnings == [
            "gravity_evidence_available_but_no_typed_orientation_transform"
        ]


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
    assert deployment_decision.decision.status.value == "measured_only"
    assert deployment_decision.decision.selected_candidate_id is None
    assert deployment_decision.decision.selected_visual_asset_ids == []
    report = Phase6BConsistencyReport.model_validate_json(
        (run_dir / "validation/phase6b_layered_scene_assembly.json").read_text(encoding="utf-8")
    )
    assert report.passed
    assert len(report.checks) == 36
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
    inspected = json.loads(inspect.output)["decision"]
    assert inspected["research_decision"]["selected_candidate_id"] == "cup_candidate"
    assert inspected["deployment_decision"]["selected_candidate_id"] is None
    previews = cli.invoke(app, ["assembly", "render-previews", str(run_dir)])
    assert previews.exit_code == 0, previews.output


def test_scene_ir_has_exact_phase6b_references(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "source_arbitrary_measured_only")
    scene = SceneIR.model_validate_json(
        (run_dir / "scene_ir/phase6b_layered_scene.json").read_text(encoding="utf-8")
    )
    reference = scene.metadata.scene_assembly
    assert reference is not None
    assert reference.scene_data_space == "source_world"
    assert reference.source_scene_ir_sha256 == sha256_file(run_dir / reference.source_scene_ir_path)
    assert reference.source_coordinate_convention == scene.metadata.coordinate_convention
    assert reference.assembly_world_mode == "source_arbitrary"
    assert reference.source_world_to_assembly_world == IDENTITY_MATRIX4
    assert not reference.geometry_requires_assembly_transform
    assert not reference.camera_poses_require_assembly_transform
    assert not reference.object_roots_require_assembly_transform
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


@pytest.mark.parametrize(
    ("mode", "requires_transform", "assembly_units", "assembly_alignment"),
    [
        ("full_canonical_scene", True, "meters", "canonical"),
        ("metric_only_scene", True, "meters", "unoriented"),
        ("gravity_only_scene", False, "arbitrary_units", "unoriented"),
        ("source_arbitrary_measured_only", False, "arbitrary_units", "unoriented"),
    ],
)
def test_layered_scene_ir_keeps_nonidentity_source_numeric_space(
    tmp_path: Path,
    mode: str,
    requires_transform: bool,
    assembly_units: str,
    assembly_alignment: str,
) -> None:
    run_dir = _run(tmp_path, mode)
    source = SceneIR.model_validate_json(
        (run_dir / "assembly/source/scene_ir.json").read_text(encoding="utf-8")
    )
    layered = SceneIR.model_validate_json(
        (run_dir / "scene_ir/phase6b_layered_scene.json").read_text(encoding="utf-8")
    )
    reference = layered.metadata.scene_assembly
    assert reference is not None
    assert layered.metadata.coordinate_convention == source.metadata.coordinate_convention
    assert layered.cameras == source.cameras
    assert layered.frames == source.frames
    assert layered.objects == source.objects
    assert layered.geometry_assets == source.geometry_assets
    assert source.cameras[0].poses[0].transform_world_from_camera.translation == (
        1.0,
        2.0,
        3.0,
    )
    assert source.objects[0].transform.translation == (4.0, 5.0, 6.0)
    assert source.objects[1].transform.translation == (-2.0, 1.0, 0.5)
    assert reference.scene_data_space == "source_world"
    assert reference.assembly_linear_units == assembly_units
    assert reference.assembly_alignment_status == assembly_alignment
    assert reference.geometry_requires_assembly_transform is requires_transform
    assert reference.camera_poses_require_assembly_transform is requires_transform
    assert reference.object_roots_require_assembly_transform is requires_transform
    compiler = json.loads(
        (run_dir / "assembly/compiler_input_manifest.json").read_text(encoding="utf-8")
    )
    contract = compiler["coordinate_contract"]
    assert contract["source_scene_ir"]["sha256"] == sha256_file(
        run_dir / contract["source_scene_ir"]["path"]
    )
    assert contract[
        "source_coordinate_convention"
    ] == source.metadata.coordinate_convention.model_dump(mode="json")
    assert contract["source_world_to_assembly_world"] == list(
        reference.source_world_to_assembly_world
    )
    assert contract["reference_world_assets_are_source_space"]
    assert contract["apply_world_transform_at_compile_time"] is requires_transform


def test_measured_anchor_retained_when_articulated_candidate_rejected(
    tmp_path: Path,
) -> None:
    run_dir = _run(tmp_path, "rejected_articulated_candidate")
    plan = SceneAssemblyPlan.model_validate_json(
        (run_dir / "assembly/assembly_plan.json").read_text(encoding="utf-8")
    )
    decision = plan.decisions[0]
    assert decision.research_decision.status.value == "deferred_articulated_unresolved"
    assert decision.measured_motion is not None
    assert decision.research_decision.selected_visual_asset_ids == []
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
    source = decision.research_decision.articulated_model_source
    assert source is not None
    kinematic_path = run_dir / source.path
    assert sha256_file(kinematic_path) == source.sha256
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
    assert plan.decisions[0].research_decision.status.value == "deferred_license_blocked"
    assert not plan.decisions[0].research_decision.selected_visual_asset_ids


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
    assert plan.decisions[0].research_decision.status.value == "measured_only"
    assert all(layer.role.value != "global_context" for layer in plan.layers)


def test_accepted_state_alignment_connects_lineages(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "accepted_state_alignment_lineage")
    plan = SceneAssemblyPlan.model_validate_json(
        (run_dir / "assembly/assembly_plan.json").read_text(encoding="utf-8")
    )
    assert plan.decisions[0].research_decision.status.value == "measured_only"


def test_research_and_deployment_choose_different_candidates(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "different_bundle_candidates")
    plan = SceneAssemblyPlan.model_validate_json(
        (run_dir / "assembly/assembly_plan.json").read_text(encoding="utf-8")
    )
    decision = plan.decisions[0]
    assert decision.research_decision.selected_candidate_id == "research_candidate_A"
    assert decision.research_decision.selected_visual_asset_ids == ["research_candidate_A_visual"]
    assert decision.deployment_decision.selected_candidate_id == "production_candidate_B"
    assert decision.deployment_decision.selected_visual_asset_ids == [
        "production_candidate_B_visual"
    ]
    compiler = json.loads(
        (run_dir / "assembly/compiler_input_manifest.json").read_text(encoding="utf-8")
    )
    assert (
        compiler["research_object_instances"][0]["decision"]["selected_candidate_id"]
        == "research_candidate_A"
    )
    assert (
        compiler["deployment_object_instances"][0]["decision"]["selected_candidate_id"]
        == "production_candidate_B"
    )
    overlap = SceneAssemblyOverlapReport.model_validate_json(
        (run_dir / "assembly/overlap_diagnostics.json").read_text(encoding="utf-8")
    ).diagnostics[0]
    assert overlap.measured_anchor_asset_ids == ["cup_measured", "cup_measured_part2"]
    assert overlap.candidate_asset_ids == [
        "production_candidate_B_visual",
        "research_candidate_A_visual",
    ]
    assert set(overlap.per_asset_overlap) == set(overlap.candidate_asset_ids)


def test_rigid_selection_evaluation_representation_and_license_are_source_bound(
    phase5b_source_run: Path,
) -> None:
    raw = _rigid_source_manifest(phase5b_source_run)
    manifest = normalize_assembly_manifest(raw, phase5b_source_run)
    object_input = manifest.objects[0]
    candidate = next(item for item in manifest.assets if item.asset_id == "candidate")
    assert object_input.preferred_research_candidate_id == candidate.candidate_id
    assert object_input.preferred_deployment_candidate_id is None
    assert candidate.selected_upstream
    assert candidate.observation_validation_passed
    assert candidate.representation_id in {"official_pbr_glb", "official_visual_glb"}
    assert candidate.source_native_asset_path is not None
    assert candidate.candidate_selection is not None
    assert candidate.candidate_evaluation is not None
    assert candidate.candidate_generation is not None
    assert candidate.license.research_evaluation_allowed
    assert not candidate.license.production_selectable
    assert candidate.object_to_source_world != IDENTITY_MATRIX4


def test_local_candidate_assertions_cannot_override_upstream_sources(
    phase5b_source_run: Path,
) -> None:
    raw = _rigid_source_manifest(phase5b_source_run)
    candidate = raw["assets"][1]  # type: ignore[index]
    candidate["representation_id"] = "stale_representation"  # type: ignore[index]
    candidate["production_selectable"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="rigid representation"):
        normalize_assembly_manifest(raw, phase5b_source_run)


def test_upstream_preferred_candidate_cannot_be_omitted_from_assembly(
    phase5b_source_run: Path,
) -> None:
    raw = _rigid_source_manifest(phase5b_source_run)
    raw["assets"] = raw["assets"][:1]  # type: ignore[index]
    raw["objects"][0]["candidate_asset_ids"] = []  # type: ignore[index]
    with pytest.raises(ValueError, match="omits an upstream preferred rigid candidate"):
        normalize_assembly_manifest(raw, phase5b_source_run)


def test_candidate_object_and_evaluation_identity_mismatch_fails_closed(
    phase5b_source_run: Path,
) -> None:
    raw = _rigid_source_manifest(phase5b_source_run)
    raw["assets"][1]["object_id"] = "wrong_object"  # type: ignore[index]
    with pytest.raises(ValueError, match="belongs to"):
        normalize_assembly_manifest(raw, phase5b_source_run)


def test_camera_frame_digest_is_derived_and_mismatch_rejected(
    phase5b_source_run: Path,
) -> None:
    raw = _rigid_source_manifest(phase5b_source_run)
    raw["lineages"][0]["frame_sequence_digest"] = "f" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="frame-sequence digest"):
        normalize_assembly_manifest(raw, phase5b_source_run)


def test_research_only_license_cannot_be_promoted_locally(
    phase5b_source_run: Path,
) -> None:
    raw = _rigid_source_manifest(phase5b_source_run)
    raw["assets"][1]["license"] = {  # type: ignore[index]
        "license_id": "locally_promoted",
        "license_name": "Local override",
        "research_evaluation_allowed": True,
        "production_selectable": True,
        "commercial_review_status": "approved",
        "restrictions": [],
    }
    with pytest.raises(ValueError, match="rigid candidate license"):
        normalize_assembly_manifest(raw, phase5b_source_run)


@pytest.mark.parametrize("asset_format", ["glb", "ply"])
def test_global_context_geometry_is_bound_to_phase3_output(
    phase3_source_run: Path,
    asset_format: str,
) -> None:
    manifest = normalize_assembly_manifest(
        _global_context_source_manifest(phase3_source_run, asset_format=asset_format),
        phase3_source_run,
    )
    asset = manifest.assets[0]
    metadata = json.loads(
        (phase3_source_run / "reconstruction/global/metadata.json").read_text(encoding="utf-8")
    )
    expected_path = (
        metadata["scene_asset_path"] if asset_format == "glb" else metadata["mesh_asset_path"]
    )
    assert asset.source_native_asset_path == expected_path
    assert asset.asset_path == expected_path
    assert asset.global_scene_reconstruction is not None
    assert asset.global_context_source is not None
    assert asset.license.source_record == asset.license_source_record
    assert asset.license.research_evaluation_allowed
    assert not asset.license.production_selectable
    assert asset.license.commercial_review_status == "not_reviewed"


def test_global_context_rejects_representation_path_mismatch(
    phase3_source_run: Path,
) -> None:
    raw = _global_context_source_manifest(phase3_source_run)
    raw["assets"][0]["format"] = "ply"  # type: ignore[index]
    with pytest.raises(ValueError, match="native representation|promoted Phase 3"):
        normalize_assembly_manifest(raw, phase3_source_run)


def test_global_context_rejects_correct_license_with_wrong_geometry_hash(
    phase3_source_run: Path,
) -> None:
    raw = _global_context_source_manifest(phase3_source_run)
    raw["assets"][0]["asset_sha256"] = sha256_file(  # type: ignore[index]
        phase3_source_run / "reconstruction/global/mesh.ply"
    )
    with pytest.raises(ValueError, match="staged asset bytes"):
        normalize_assembly_manifest(raw, phase3_source_run)


def test_global_context_rejects_foreign_lineage(
    phase3_source_run: Path,
) -> None:
    raw = _global_context_source_manifest(phase3_source_run)
    raw["assets"][0]["lineage_id"] = "foreign_lineage"  # type: ignore[index]
    with pytest.raises(ValueError, match="undeclared lineage"):
        normalize_assembly_manifest(raw, phase3_source_run)


def test_articulated_selection_fit_evaluation_and_license_are_source_bound(
    phase5c_source_run: Path,
) -> None:
    manifest = normalize_assembly_manifest(
        _articulated_source_manifest(phase5c_source_run),
        phase5c_source_run,
    )
    object_input = manifest.objects[0]
    candidates = [item for item in manifest.assets if item.role.value == "articulated_visual"]
    assert object_input.preferred_research_candidate_id == candidates[0].candidate_id
    assert object_input.preferred_deployment_candidate_id == (
        "cabinet_0001__measured_motion__baseline"
    )
    assert all(item.selected_upstream for item in candidates)
    assert all(item.observation_validation_passed for item in candidates)
    assert all(item.source.value == "retrieved" for item in candidates)
    assert all(item.candidate_selection is not None for item in candidates)
    assert all(item.candidate_evaluation is not None for item in candidates)
    assert all(item.candidate_generation is not None for item in candidates)
    assert all(item.kinematic_bundle is not None for item in candidates)
    assert all(item.license.research_evaluation_allowed for item in candidates)
    assert all(not item.license.production_selectable for item in candidates)


def test_rejected_articulated_evaluation_cannot_be_promoted_locally(
    phase5c_rejected_source_run: Path,
) -> None:
    raw = _articulated_source_manifest(phase5c_rejected_source_run)
    candidate = raw["assets"][1]  # type: ignore[index]
    candidate["selected_upstream"] = True  # type: ignore[index]
    candidate["observation_validation_passed"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="articulated candidate selected upstream"):
        normalize_assembly_manifest(raw, phase5c_rejected_source_run)


def test_selected_articulated_identity_reference_mismatch_fails_closed(
    phase5c_source_run: Path,
) -> None:
    raw = _articulated_source_manifest(phase5c_source_run)
    object_input = raw["objects"][0]  # type: ignore[index]
    bundle = object_input["kinematic_bundle"]  # type: ignore[index]
    object_input["selected_identity_manifest"] = {  # type: ignore[index]
        **bundle,
        "artifact_type": "selected_identity_manifest",
    }
    with pytest.raises(ValueError, match="identity reference"):
        normalize_assembly_manifest(raw, phase5c_source_run)


def test_articulated_candidate_link_and_object_identity_mismatch_fails_closed(
    phase5c_source_run: Path,
) -> None:
    raw = _articulated_source_manifest(phase5c_source_run)
    raw["assets"][1]["link_id"] = "stale_link"  # type: ignore[index]
    with pytest.raises(ValueError, match="link identity"):
        normalize_assembly_manifest(raw, phase5c_source_run)
    raw = _articulated_source_manifest(phase5c_source_run)
    raw["assets"][1]["object_id"] = "foreign_object"  # type: ignore[index]
    with pytest.raises(ValueError, match="belongs to"):
        normalize_assembly_manifest(raw, phase5c_source_run)


def test_lineage_connection_requires_an_accepted_typed_alignment(
    phase5c_source_run: Path,
) -> None:
    raw = _connected_articulated_source_manifest(phase5c_source_run)
    normalized = normalize_assembly_manifest(raw, phase5c_source_run)
    assert normalized.lineages[1].transform_connected_from_lineage == IDENTITY_MATRIX4
    assert normalized.lineages[0].source_state_id == "state_000"
    assert normalized.lineages[1].source_state_id == "state_001"
    raw["assets"][0]["lineage_id"] = "state_001_lineage"  # type: ignore[index]
    connected_asset = normalize_assembly_manifest(raw, phase5c_source_run)
    assert connected_asset.assets[0].lineage_id == "state_001_lineage"
    raw["assets"][0]["lineage_id"] = "lineage"  # type: ignore[index]
    raw["lineages"][1]["source_state_id"] = "state_002"  # type: ignore[index]
    with pytest.raises(ValueError, match="alignment source state"):
        normalize_assembly_manifest(raw, phase5c_source_run)
    raw["lineages"][1]["source_state_id"] = "state_001"  # type: ignore[index]

    rejected = json.loads(
        (phase5c_source_run / "assembly_binding/state_alignment.json").read_text(encoding="utf-8")
    )
    transform = next(item for item in rejected["transforms"] if item["state_id"] == "state_001")
    transform["accepted"] = False
    transform["failure_reason"] = "synthetic rejected alignment"
    rejected["accepted_alignment_state_ids"].remove("state_001")
    rejected["aligned_state_count"] -= 1
    rejected_path = phase5c_source_run / "assembly_binding/rejected_alignment.json"
    rejected_path.write_text(
        json.dumps(rejected, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    raw["lineages"][1]["accepted_alignment"] = _source_ref(  # type: ignore[index]
        phase5c_source_run,
        "assembly_binding/rejected_alignment.json",
        "state_alignment",
    )
    with pytest.raises(ValueError, match="accepted alignment"):
        normalize_assembly_manifest(raw, phase5c_source_run)


def test_state_alignment_requires_capture_manifest_reference(
    phase5c_source_run: Path,
) -> None:
    raw = _connected_articulated_source_manifest(phase5c_source_run)
    del raw["lineages"][1]["alignment_capture_manifest"]  # type: ignore[index]
    with pytest.raises(ValueError, match="articulation_capture_manifest"):
        normalize_assembly_manifest(raw, phase5c_source_run)


def test_state_alignment_rejects_camera_from_another_capture_state(
    phase5c_source_run: Path,
) -> None:
    raw = _connected_articulated_source_manifest(
        phase5c_source_run,
        child_camera_path="assembly_binding/state_002_camera_reconstruction.json",
    )
    with pytest.raises(ValueError, match="child lineage camera hash/digest"):
        normalize_assembly_manifest(raw, phase5c_source_run)


def test_state_alignment_rejects_correct_camera_hash_with_wrong_frame_digest(
    phase5c_source_run: Path,
) -> None:
    def wrong_digest(capture: dict[str, object]) -> None:
        states = capture["states"]
        assert isinstance(states, list)
        state = next(item for item in states if item["state_id"] == "state_001")
        state["frame_sequence_digest"] = "f" * 64

    capture_path, alignment_path = _write_capture_alignment_revision(
        phase5c_source_run,
        "wrong_digest",
        wrong_digest,
    )
    raw = _connected_articulated_source_manifest(
        phase5c_source_run,
        capture_path=capture_path,
        alignment_path=alignment_path,
    )
    with pytest.raises(ValueError, match="child lineage camera hash/digest"):
        normalize_assembly_manifest(raw, phase5c_source_run)


def test_state_alignment_rejects_capture_sha_mismatch(
    phase5c_source_run: Path,
) -> None:
    raw = _connected_articulated_source_manifest(
        phase5c_source_run,
        capture_path="reconstruction/articulation/capture_manifest.json",
    )
    with pytest.raises(ValueError, match="not bound to the referenced capture"):
        normalize_assembly_manifest(raw, phase5c_source_run)


def test_state_alignment_rejects_swapped_child_and_reference_states(
    phase5c_source_run: Path,
) -> None:
    raw = _connected_articulated_source_manifest(
        phase5c_source_run,
        alignment_state_id="state_000",
    )
    with pytest.raises(ValueError, match="child lineage camera hash/digest"):
        normalize_assembly_manifest(raw, phase5c_source_run)


def test_state_alignment_rejects_wrong_reference_camera(
    phase5c_source_run: Path,
) -> None:
    raw = _connected_articulated_source_manifest(phase5c_source_run)
    raw["lineages"][0]["camera_reconstruction"] = _source_ref(  # type: ignore[index]
        phase5c_source_run,
        "assembly_binding/state_002_camera_reconstruction.json",
        "camera_reconstruction",
    )
    with pytest.raises(ValueError, match="reference lineage camera hash/digest"):
        normalize_assembly_manifest(raw, phase5c_source_run)


def test_calibration_status_and_transform_are_derived_from_phase6a(
    phase6a_source_run: Path,
) -> None:
    manifest = normalize_assembly_manifest(
        _calibration_source_manifest(phase6a_source_run),
        phase6a_source_run,
    )
    artifact = json.loads(
        (phase6a_source_run / "calibration/world_calibration.json").read_text(encoding="utf-8")
    )
    assert manifest.calibration_status.value == artifact["status"]  # type: ignore[union-attr]
    assert (
        list(manifest.source_world_to_assembly_world or ())
        == artifact["accepted_transform"]["matrix_canonical_from_colmap"]
    )


def test_source_bound_gravity_only_calibration_preserves_source_world(
    phase6a_gravity_only_source_run: Path,
) -> None:
    raw = _calibration_source_manifest(phase6a_gravity_only_source_run)
    manifest = normalize_assembly_manifest(raw, phase6a_gravity_only_source_run)
    assert manifest.calibration_status is not None
    assert manifest.calibration_status.value == "accepted_gravity_only"
    assert manifest.source_world_to_assembly_world is None
    require_payload = manifest.model_dump(mode="json")
    require_payload["calibration_policy"] = "require_full_canonical"
    with pytest.raises(ValueError, match="requires an accepted full-canonical"):
        resolve_world(SceneAssemblyInputManifest.model_validate(require_payload))
    preserve_payload = manifest.model_dump(mode="json")
    preserve_payload["calibration_policy"] = "preserve_source_world"
    preserved = resolve_world(SceneAssemblyInputManifest.model_validate(preserve_payload))
    assert preserved.world_mode.value == "source_arbitrary"
    assert preserved.source_world_to_assembly_world == IDENTITY_MATRIX4
    assert not preserved.gravity_alignment_known

    run_dir = _run_source_bound_calibration_assembly(
        phase6a_gravity_only_source_run,
        raw,
        run_name="phase6b",
    )
    plan = SceneAssemblyPlan.model_validate_json(
        (run_dir / "assembly/assembly_plan.json").read_text(encoding="utf-8")
    )
    assert plan.world.world_mode.value == "source_arbitrary"
    assert plan.world.source_world_to_assembly_world == IDENTITY_MATRIX4
    assert not plan.world.gravity_alignment_known
    assert plan.world.warnings == ["gravity_evidence_available_but_no_typed_orientation_transform"]


def test_source_bound_full_canonical_keeps_nonidentity_scene_ir_in_source_space(
    phase6a_nonidentity_source_run: Path,
) -> None:
    raw = _calibration_source_manifest(phase6a_nonidentity_source_run)
    run_dir = _run_source_bound_calibration_assembly(
        phase6a_nonidentity_source_run,
        raw,
        run_name="phase6b",
    )
    source = SceneIR.model_validate_json(
        (run_dir / "calibration/source/scene_ir.json").read_text(encoding="utf-8")
    )
    layered = SceneIR.model_validate_json(
        (run_dir / "scene_ir/phase6b_layered_scene.json").read_text(encoding="utf-8")
    )
    reference = layered.metadata.scene_assembly
    assert reference is not None
    assert reference.assembly_world_mode == "canonical_metric"
    assert reference.assembly_linear_units == "meters"
    assert reference.assembly_alignment_status == "canonical"
    assert reference.source_world_to_assembly_world != IDENTITY_MATRIX4
    assert reference.camera_poses_require_assembly_transform
    assert reference.object_roots_require_assembly_transform
    assert layered.metadata.coordinate_convention == source.metadata.coordinate_convention
    assert layered.cameras == source.cameras
    assert layered.objects == source.objects
    assert source.cameras[0].poses[0].transform_world_from_camera.translation == (
        1.0,
        2.0,
        3.0,
    )
    assert source.objects[0].transform.translation == (4.0, 5.0, 6.0)
    compiler = json.loads(
        (run_dir / "assembly/compiler_input_manifest.json").read_text(encoding="utf-8")
    )
    assert compiler["coordinate_contract"]["source_world_to_assembly_world"] == list(
        reference.source_world_to_assembly_world
    )
    assert compiler["coordinate_contract"]["apply_world_transform_at_compile_time"]


def test_local_calibration_status_or_transform_mismatch_fails_closed(
    phase6a_source_run: Path,
) -> None:
    raw = _calibration_source_manifest(phase6a_source_run)
    raw["calibration_status"] = "insufficient_evidence"
    with pytest.raises(ValueError, match="calibration status"):
        normalize_assembly_manifest(raw, phase6a_source_run)
    raw = _calibration_source_manifest(phase6a_source_run)
    raw["source_world_to_assembly_world"] = list(IDENTITY_MATRIX4)
    with pytest.raises(ValueError, match="calibration transform"):
        normalize_assembly_manifest(raw, phase6a_source_run)


def test_canonical_wrapper_source_scene_mismatch_fails_closed(
    phase6a_source_run: Path,
    tmp_path: Path,
) -> None:
    for relative in (
        "calibration/source/camera_reconstruction.json",
        "calibration/source/scene_ir.json",
        "calibration/world_calibration.json",
        "calibration/canonical_scene_wrapper.json",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((phase6a_source_run / relative).read_bytes())
    wrapper_path = tmp_path / "calibration/canonical_scene_wrapper.json"
    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    wrapper["source_scene_ir_sha256"] = "f" * 64
    wrapper_path.write_text(
        json.dumps(wrapper, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    raw = _calibration_source_manifest(tmp_path)
    with pytest.raises(ValueError, match="primary lineage"):
        normalize_assembly_manifest(raw, tmp_path)


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
