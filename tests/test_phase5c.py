from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from recon2sim.articulation import (
    estimate_analytic_joint,
    evidence_level,
    invert_sim3,
    proper_positive_sim3,
    select_articulated_candidate,
    sha256_file,
    split_articulation_evidence,
)
from recon2sim.artifacts import (
    ArticulatedCandidate,
    ArticulatedCandidateEvaluation,
    ArticulatedCandidateManifest,
    ArticulatedCandidateSelection,
    ArticulatedCandidateStatus,
    ArticulatedEvaluationManifest,
    ArticulatedJoint,
    ArticulatedJointType,
    ArticulatedLicenseMode,
    ArticulatedRetrievalResult,
    ArticulatedSourceFamily,
    ArticulationCaptureManifest,
    ArticulationEvidenceLevel,
    ArticulationStateEvaluation,
    ArticulationStateTransform,
    MeasuredPartMotionArtifact,
    Phase5CConsistencyReport,
)
from recon2sim.cli import app
from recon2sim.config import load_config
from recon2sim.ir import SceneIR
from recon2sim.pipeline import PipelineRunner

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "examples/tabletop"
CONFIG = ROOT / "configs/phase5c_e2e_fake.yaml"


def _worker_module(name: str) -> object:
    worker_root = ROOT / "workers/articulation_alignment"
    sys.path.insert(0, str(worker_root))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(worker_root))


def _evaluation_worker_module(name: str) -> object:
    worker_root = ROOT / "workers/articulation_evaluation"
    sys.path.insert(0, str(worker_root))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(worker_root))


def _retrieval_worker_module(name: str) -> object:
    worker_root = ROOT / "workers/articulated_retrieval"
    sys.path.insert(0, str(worker_root))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(worker_root))


def _state_evaluation(candidate_id: str, iou: float) -> ArticulatedCandidateEvaluation:
    state = ArticulationStateEvaluation(
        state_id="state_heldout",
        heldout=True,
        base_mask_iou=iou,
        movable_part_mask_iou=iou,
        whole_object_mask_iou=iou,
        per_link_depth_residual={"base": 0.01, "part": 0.02},
        depth_inlier_fraction=iou,
        negative_space_violation_ratio=0.02,
        front_of_scene_violation_ratio=0.01,
        base_motion_scene_diagonals=0.001,
        joint_constraint_residual=0.005,
        axis_error_degrees=2.0,
        pivot_residual_part_diagonals=0.01,
        inferred_joint_positions={"joint": 0.5},
        joint_position_source="measured_geometry",
    )
    return ArticulatedCandidateEvaluation(
        candidate_id=candidate_id,
        status=ArticulatedCandidateStatus.MULTI_STATE,
        fitting_sha256="0" * 64,
        state_evaluations=[state],
        passed_hard_gates=True,
        failed_gates=[],
        heldout_state_validation_used=True,
        link_assignment_confidence=0.9,
        runtime_seconds=0,
    )


def test_articulation_evidence_levels_are_not_overstated() -> None:
    assert evidence_level(1) is ArticulationEvidenceLevel.SINGLE_STATE_PRIOR_ONLY
    assert evidence_level(2) is ArticulationEvidenceLevel.TWO_STATE_MOTION_SUPPORTED
    assert evidence_level(3) is ArticulationEvidenceLevel.MULTI_STATE_HELDOUT_VALIDATED
    with pytest.raises(ValueError, match="at least one"):
        evidence_level(0)


def test_articulation_split_is_disjoint_and_heldout_is_last_state() -> None:
    split = split_articulation_evidence(
        "cabinet_0001",
        ["closed", "half", "open"],
        {
            "closed": ["c0", "c1"],
            "half": ["h0", "h1"],
            "open": ["o0", "o1", "o2", "o3"],
        },
        seed=42,
    )
    assert split.candidate_generation_states == ["closed"]
    assert split.kinematic_fitting_states == ["half"]
    assert split.heldout_validation_states == ["open"]
    assert split.heldout_views_by_state == {"open": ["o0", "o2"]}
    assert not (set(split.candidate_generation_states) & set(split.heldout_validation_states))


def test_prismatic_motion_recovery() -> None:
    transforms = []
    for position in (0.0, 0.35, 0.8):
        transforms.append(
            (
                1.0,
                0.0,
                0.0,
                position,
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
    estimate = estimate_analytic_joint(
        transforms,
    )
    assert estimate.joint_type == "prismatic"
    assert estimate.axis is not None
    assert abs(estimate.axis[0]) == pytest.approx(1.0)
    assert estimate.positions[-1] == pytest.approx(0.8)


def test_revolute_motion_recovery() -> None:
    transforms = []
    for angle in (0.0, math.radians(25.0), math.radians(55.0)):
        cosine, sine = math.cos(angle), math.sin(angle)
        transforms.append(
            (
                cosine,
                -sine,
                0.0,
                1.0 - cosine,
                sine,
                cosine,
                0.0,
                -sine,
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
    estimate = estimate_analytic_joint(
        transforms,
    )
    assert estimate.joint_type == "revolute"
    assert estimate.axis is not None
    assert abs(estimate.axis[2]) == pytest.approx(1.0)
    assert estimate.pivot is not None
    assert estimate.pivot[0] == pytest.approx(1.0, abs=1e-6)
    assert estimate.pivot[1] == pytest.approx(0.0, abs=1e-6)


def test_two_state_revolute_motion_is_partially_supported() -> None:
    angle = math.radians(35.0)
    cosine, sine = math.cos(angle), math.sin(angle)
    estimate = estimate_analytic_joint(
        [
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
            ),
            (
                cosine,
                -sine,
                0.0,
                1.0 - cosine,
                sine,
                cosine,
                0.0,
                -sine,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ),
        ]
    )
    assert estimate.joint_type == "revolute"
    assert estimate.axis is not None
    assert abs(estimate.axis[2]) == pytest.approx(1.0)
    assert estimate.pivot is not None
    assert estimate.pivot[0] == pytest.approx(1.0, abs=1e-6)


def test_state_alignment_sim3_recovers_known_transform() -> None:
    np = pytest.importorskip("numpy")
    sim3 = _worker_module("articulation_alignment_worker.sim3")
    generator = np.random.default_rng(42)
    source = generator.normal(size=(1000, 3))
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = 1.7 * (source @ rotation.T) + np.array([0.4, -0.2, 0.8])
    fit = sim3.robust_icp_sim3(source, target, iterations=50)
    recovered = sim3.apply_transform(source, fit.matrix)
    assert np.median(np.linalg.norm(recovered - target, axis=1)) < 1e-6
    assert fit.scale == pytest.approx(1.7, rel=1e-6)


def test_heldout_joint_position_fits_only_q(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    pytest.importorskip("trimesh")
    evaluator = _evaluation_worker_module("articulation_evaluation_worker.cli")
    generator = np.random.default_rng(7)
    candidate_points = generator.normal(size=(500, 3))
    measured_points = candidate_points + np.asarray([0.7, 0.0, 0.0])

    def write_points(path: Path, points: object) -> None:
        values = np.asarray(points)
        rows = "\n".join(" ".join(f"{value:.12g}" for value in row) for row in values)
        path.write_text(
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {len(values)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
            f"{rows}\n",
            encoding="ascii",
        )

    candidate_path = tmp_path / "candidate.ply"
    measured_path = tmp_path / "measured.ply"
    write_points(candidate_path, candidate_points)
    write_points(measured_path, measured_points)
    position = evaluator._heldout_joint_position(
        input_root=tmp_path,
        link={"visual_asset_paths": ["candidate.ply"]},
        joint={
            "joint_id": "drawer_joint",
            "joint_type": "prismatic",
            "axis": [1.0, 0.0, 0.0],
            "pivot": None,
            "candidate_limit_lower": 0.0,
            "candidate_limit_upper": 1.0,
            "limit_source": "candidate_prior",
        },
        measured_joint=None,
        measured_part={"measured_point_cloud_path": "measured.ply"},
        base_matrix=np.eye(4),
        reference_from_state=np.eye(4),
    )
    assert position == pytest.approx(0.7, abs=2e-3)


def test_improper_or_singular_state_transform_is_rejected() -> None:
    identity = [
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
    ]
    reflected = identity.copy()
    reflected[0] = -1.0
    assert proper_positive_sim3(identity, identity)
    assert not proper_positive_sim3(reflected, reflected)
    singular = identity.copy()
    singular[10] = 0.0
    assert not proper_positive_sim3(singular, identity)
    scaled = identity.copy()
    scaled[0] = scaled[5] = scaled[10] = 2.0
    scaled[3], scaled[7], scaled[11] = 0.4, -0.6, 1.2
    inverse = invert_sim3(scaled)
    assert inverse is not None
    assert proper_positive_sim3(scaled, inverse)
    with pytest.raises(ValueError, match="scale is inconsistent"):
        ArticulationStateTransform(
            state_id="bad-scale",
            matrix_reference_from_state=scaled,
            inverse_matrix=inverse,
            scale=1.0,
            rotation_determinant=1.0,
            translation=(0.4, -0.6, 1.2),
            fitting_median_residual_scene_diagonal=0.0,
            fitting_p90_residual_scene_diagonal=0.0,
            heldout_static_depth_inlier_fraction=1.0,
            static_correspondence_count=1000,
            excluded_movable_part_ids=["drawer"],
            accepted=False,
            failure_reason="invalid scale",
        )


def test_articulated_joint_axis_and_graph_contract() -> None:
    with pytest.raises(ValueError, match="normalized"):
        ArticulatedJoint(
            joint_id="bad",
            parent_link_id="base",
            child_link_id="door",
            joint_type=ArticulatedJointType.REVOLUTE,
            axis=(2.0, 0.0, 0.0),
            pivot=(0.0, 0.0, 0.0),
            limit_source="unknown",
        )
    run_dir = Path("/tmp")
    candidate_payload = {
        "candidate_id": "cycle",
        "articulated_object_id": "cabinet",
        "source_family": "measured_motion_analytic",
        "source_asset_id": "measured",
        "links": [
            {
                "link_id": item,
                "name": item,
                "visual_asset_paths": [],
                "visual_asset_hashes": {},
                "native_bounds_min": [0, 0, 0],
                "native_bounds_max": [1, 1, 1],
            }
            for item in ("a", "b")
        ],
        "joints": [
            {
                "joint_id": "ab",
                "parent_link_id": "a",
                "child_link_id": "b",
                "joint_type": "fixed",
                "axis": [1, 0, 0],
                "limit_source": "unknown",
            },
            {
                "joint_id": "ba",
                "parent_link_id": "b",
                "child_link_id": "a",
                "joint_type": "fixed",
                "axis": [1, 0, 0],
                "limit_source": "unknown",
            },
        ],
        "states": [],
        "native_coordinate_convention": "arbitrary",
        "native_units": "arbitrary_units",
        "license_record": {
            "source_family": "measured_motion_analytic",
            "code_license": "project",
            "checkpoint_license": "not_applicable",
            "asset_license": "source observation",
            "commercial_review_status": "approved_by_project_policy",
            "research_evaluation_allowed": True,
            "production_selectable": True,
        },
        "production_selectable": True,
        "provenance": {
            "adapter_name": "test",
            "adapter_version": "0.1.0",
            "input_artifact_paths": [],
            "output_artifact_paths": [],
            "confidence": {"score": 0.5, "method": "test"},
            "source": "fused",
        },
    }
    del run_dir
    with pytest.raises(ValueError, match="cycle"):
        ArticulatedCandidate.model_validate(candidate_payload)


def test_license_aware_articulated_selection() -> None:
    research = _state_evaluation("partnet", 0.9)
    production = _state_evaluation("approved", 0.8)
    result = select_articulated_candidate(
        [production, research],
        production_selectable={"partnet": False, "approved": True},
        mode=ArticulatedLicenseMode.PRODUCTION_CANDIDATE,
    )
    assert result == ("partnet", "approved", "approved")


def test_offline_retrieval_normalizes_selected_candidate_assets(tmp_path: Path) -> None:
    worker = _retrieval_worker_module("articulated_retrieval_worker.cli")
    root = tmp_path
    articulation = root / "reconstruction/articulation"
    catalog = articulation / "catalogs/artvip/selected/asset_001"
    catalog.mkdir(parents=True)
    source_bundle = catalog / "source_candidate.json"
    source_base = catalog / "visuals/000.ply"
    source_drawer = catalog / "visuals/001.ply"
    source_base.parent.mkdir(parents=True)
    source_base.write_text("base visual\n", encoding="ascii")
    source_drawer.write_text("drawer visual\n", encoding="ascii")
    license_record = {
        "source_family": "artvip",
        "code_license": "catalog code",
        "checkpoint_license": "not_applicable",
        "dependency_licenses": {},
        "asset_license": "research only",
        "training_data_notes": [],
        "commercial_review_status": "research_only",
        "research_evaluation_allowed": True,
        "production_selectable": False,
    }
    source_bundle.write_text(
        json.dumps(
            {
                "candidate_id": "source",
                "articulated_object_id": "source",
                "source_family": "artvip",
                "source_asset_id": "asset_001",
                "links": [
                    {
                        "link_id": "cabinet_body",
                        "name": "cabinet body",
                        "visual_asset_paths": ["assets/base.ply"],
                        "visual_asset_hashes": {"assets/base.ply": "0" * 64},
                        "native_bounds_min": [0, 0, 0],
                        "native_bounds_max": [1, 1, 1],
                    },
                    {
                        "link_id": "drawer_0001",
                        "name": "drawer",
                        "visual_asset_paths": ["assets/drawer.ply"],
                        "visual_asset_hashes": {"assets/drawer.ply": "0" * 64},
                        "native_bounds_min": [0, 0, 0],
                        "native_bounds_max": [1, 1, 1],
                    },
                ],
                "joints": [
                    {
                        "joint_id": "drawer_joint",
                        "parent_link_id": "cabinet_body",
                        "child_link_id": "drawer_0001",
                        "joint_type": "prismatic",
                        "axis": [1, 0, 0],
                        "limit_source": "candidate_prior",
                    }
                ],
                "states": [],
                "native_coordinate_convention": "+Z up",
                "native_units": "asset_units",
                "license_record": license_record,
                "production_selectable": False,
                "provenance": {
                    "adapter_name": "source",
                    "adapter_version": "0.1.0",
                    "input_artifact_paths": [],
                    "output_artifact_paths": [],
                    "confidence": {"score": 0.5, "method": "source"},
                    "source": "retrieved",
                },
            }
        ),
        encoding="utf-8",
    )
    measured_path = articulation / "measured_motion.json"
    prompts_path = articulation / "part_prompt_manifest.json"
    index_path = articulation / "catalogs/artvip_index.json"
    measured_path.write_text(
        json.dumps(
            {
                "joint_hypotheses": [
                    {
                        "joint_type": "prismatic",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    prompts_path.write_text(
        json.dumps({"objects": [{"semantic_label": "cabinet"}]}),
        encoding="utf-8",
    )
    index_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "asset_id": "asset_001",
                        "category": "cabinet",
                        "link_count": 2,
                        "joint_types": ["prismatic"],
                        "license_record": license_record,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    request = {
        "source_family": "artvip",
        "articulated_object_id": "cabinet_0001",
        "measured_motion_path": measured_path.relative_to(root).as_posix(),
        "measured_motion_sha256": sha256_file(measured_path),
        "part_prompt_manifest_path": prompts_path.relative_to(root).as_posix(),
        "asset_index_path": index_path.relative_to(root).as_posix(),
        "asset_index_sha256": sha256_file(index_path),
        "selected_assets": [
            {
                "asset_id": "asset_001",
                "source_candidate_path": source_bundle.relative_to(root).as_posix(),
                "visual_path_mapping": {
                    "assets/base.ply": source_base.relative_to(root).as_posix(),
                    "assets/drawer.ply": source_drawer.relative_to(root).as_posix(),
                },
            }
        ],
        "maximum_candidates": 1,
        "output_path": "reconstruction/articulation/artvip_retrieval.json",
    }
    worker.retrieve(request, root)
    result = ArticulatedRetrievalResult.model_validate_json(
        (articulation / "artvip_retrieval.json").read_text(encoding="utf-8")
    )
    assert len(result.candidates) == 1
    retrieved = result.candidates[0]
    assert retrieved.candidate_bundle_path is not None
    candidate = ArticulatedCandidate.model_validate_json(
        (root / retrieved.candidate_bundle_path).read_text(encoding="utf-8")
    )
    assert candidate.candidate_id == "cabinet_0001__artvip__asset_001"
    assert {path for link in candidate.links for path in link.visual_asset_paths} == set(
        retrieved.visual_asset_paths
    )


def test_fake_phase5c_pipeline_resume_and_cli(tmp_path: Path) -> None:
    run_dir = tmp_path / "phase5c"
    runner = PipelineRunner(load_config(CONFIG), INPUT, run_dir)
    first = runner.run()
    assert all(stage["status"] == "succeeded" for stage in first["stages"].values())
    capture = ArticulationCaptureManifest.model_validate_json(
        (run_dir / "reconstruction/articulation/capture_manifest.json").read_text()
    )
    assert capture.evidence_level is ArticulationEvidenceLevel.MULTI_STATE_HELDOUT_VALIDATED
    candidates = ArticulatedCandidateManifest.model_validate_json(
        (run_dir / "reconstruction/articulation/candidate_manifest.json").read_text()
    )
    assert candidates.candidates
    assert {item.source_family for item in candidates.candidates} == {
        ArticulatedSourceFamily.MEASURED_MOTION,
        ArticulatedSourceFamily.ARTVIP,
        ArticulatedSourceFamily.PARTNET_MOBILITY,
        ArticulatedSourceFamily.PARTICULATE,
    }
    for candidate in candidates.candidates:
        for link in candidate.links:
            for path in link.visual_asset_paths:
                assert (run_dir / path).is_file()
    evaluation = ArticulatedEvaluationManifest.model_validate_json(
        (run_dir / "reconstruction/articulation/evaluation_manifest.json").read_text()
    )
    assert evaluation.candidate_structures_frozen_before_heldout
    selection = ArticulatedCandidateSelection.model_validate_json(
        (run_dir / "reconstruction/articulation/selection.json").read_text()
    )
    assert selection.objects[0].best_research_articulated_candidate
    report = Phase5CConsistencyReport.model_validate_json(
        (run_dir / "validation/phase5c_articulated_reconstruction.json").read_text()
    )
    assert report.passed
    assert len(report.checks) >= 28
    scene = SceneIR.model_validate_json((run_dir / "scene_ir/phase5c_scene.json").read_text())
    assert scene.collision_assets == []
    assert all(not item.sim_ready for item in scene.objects)
    assert (
        run_dir / "reconstruction/articulation/selected/cabinet_0001/preview_only.urdf"
    ).is_file()
    resumed = runner.run(resume=True)
    assert all(stage["last_execution"] == "cache_hit" for stage in resumed["stages"].values())
    cli = CliRunner()
    assert cli.invoke(app, ["articulation", "inspect", str(run_dir)]).exit_code == 0
    assert cli.invoke(app, ["validation", "verify-phase5c", str(run_dir)]).exit_code == 0


def test_rejected_state_alignment_cannot_enter_motion_or_heldout(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG)
    alignment_stage = config.stages["articulation_state_alignment"]
    alignment_adapter = alignment_stage.adapter.model_copy(
        update={
            "config": {
                **alignment_stage.adapter.config,
                "fake_mode": "state_alignment_failure",
            }
        }
    )
    config = config.model_copy(
        update={
            "stages": {
                **config.stages,
                "articulation_state_alignment": alignment_stage.model_copy(
                    update={"adapter": alignment_adapter}
                ),
            }
        }
    )
    run_dir = tmp_path / "failed-alignment"
    result = PipelineRunner(config, INPUT, run_dir).run()
    assert all(stage["status"] == "succeeded" for stage in result["stages"].values())
    motion = MeasuredPartMotionArtifact.model_validate_json(
        (run_dir / "reconstruction/articulation/measured_motion.json").read_text()
    )
    assert motion.evidence_level is ArticulationEvidenceLevel.SINGLE_STATE_PRIOR_ONLY
    assert all(
        {state.state_id for state in joint.states} == {"state_000"}
        for joint in motion.joint_hypotheses
    )
    evaluation_request = json.loads(
        (
            run_dir / "work/articulation_evaluation/attempt_1/reconstruction/articulation/raw/"
            "articulation_evaluation_request.json"
        ).read_text()
    )
    assert evaluation_request["heldout_state_ids"] == []
    selection = ArticulatedCandidateSelection.model_validate_json(
        (run_dir / "reconstruction/articulation/selection.json").read_text()
    )
    assert selection.objects[0].selected_candidate_id is None


def test_heldout_state_payload_does_not_enter_fitting_request(
    phase5c_run: Path,
) -> None:
    request = json.loads(
        (
            phase5c_run / "work/articulation_fitting/attempt_1/reconstruction/articulation/raw/"
            "articulation_fitting_request.json"
        ).read_text()
    )
    heldout = set(request["heldout_state_ids"])
    assert not heldout & set(request["fitting_state_ids"])
    assert all(
        state["state_id"] not in heldout
        for joint in request["joint_hypotheses"]
        for state in joint["states"]
    )
    attempt = phase5c_run / "work/articulation_fitting/attempt_1/reconstruction/articulation"
    assert not (attempt / "measured_states/heldout_manifest.json").exists()
    assert not (attempt / "measured_states/state_002").exists()
    evaluation_attempt = (
        phase5c_run / "work/articulation_evaluation/attempt_1/reconstruction/articulation"
    )
    assert (evaluation_attempt / "measured_states/state_002/evidence").is_dir()
    assert not (evaluation_attempt / "measured_states/state_000/evidence").exists()
    assert not (evaluation_attempt / "measured_states/state_001/evidence").exists()


@pytest.fixture
def phase5c_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("phase5c")
    PipelineRunner(load_config(CONFIG), INPUT, run_dir).run()
    return run_dir
