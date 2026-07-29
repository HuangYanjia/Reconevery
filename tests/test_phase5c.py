from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from recon2sim.adapters.articulation_selection import (
    ArticulationSelectionAdapter,
    articulated_candidate_geometry_source,
)
from recon2sim.adapters.particulate import ParticulateAdapterConfig
from recon2sim.articulation import (
    capture_evidence_tier,
    effective_evidence_level,
    estimate_analytic_joint,
    evidence_level,
    invert_sim3,
    ordered_motion_state_ids,
    proper_positive_sim3,
    select_articulated_candidate,
    sha256_file,
    split_articulation_evidence,
)
from recon2sim.artifacts import (
    ArticulatedAssetSpace,
    ArticulatedCandidate,
    ArticulatedCandidateEvaluation,
    ArticulatedCandidateManifest,
    ArticulatedCandidateSelection,
    ArticulatedCandidateStatus,
    ArticulatedEvaluationManifest,
    ArticulatedJoint,
    ArticulatedJointType,
    ArticulatedKinematicBundle,
    ArticulatedLicenseMode,
    ArticulatedLink,
    ArticulatedRetrievalResult,
    ArticulatedSelectedIdentityManifest,
    ArticulatedSourceFamily,
    ArticulationCaptureManifest,
    ArticulationEvidenceLevel,
    ArticulationFittingManifest,
    ArticulationHeldoutViewEvaluation,
    ArticulationPartPromptManifest,
    ArticulationStateEvaluation,
    ArticulationStateRecord,
    ArticulationStateTransform,
    FittedArticulatedKinematicModel,
    MeasuredPartMotionArtifact,
    Phase5CConsistencyReport,
)
from recon2sim.cli import app
from recon2sim.config import load_config
from recon2sim.ir import GeometrySourceType, SceneIR
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
    view_evaluations = [
        {
            "frame_id": f"frame_{index:06d}",
            "camera_reconstruction_sha256": "0" * 64,
            "depth_path": f"dense/depth_{index:06d}.bin",
            "depth_sha256": "1" * 64,
            "valid_depth": True,
            "target_mask_paths": {"part": f"masks/part_{index:06d}.png"},
            "target_mask_hashes": {"part": "2" * 64},
            "target_masks_complete": True,
            "required_link_ids": ["base", "part"],
            "rendered_link_ids": ["base", "part"],
            "missing_link_ids": [],
            "usable": True,
            "failure_reasons": [],
            "render_path": f"renders/frame_{index:06d}.png",
            "render_sha256": "3" * 64,
            "raw_candidate_pixel_count": 100,
            "visible_candidate_pixel_count": 80,
            "target_mask_pixel_count": 90,
        }
        for index in range(2)
    ]
    state = ArticulationStateEvaluation(
        state_id="state_heldout",
        heldout=True,
        requested_heldout_view_count=2,
        usable_heldout_view_count=2,
        rendered_heldout_view_count=2,
        views_with_target_masks=2,
        views_with_valid_depth=2,
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
        render_paths={item["frame_id"]: item["render_path"] for item in view_evaluations},
        view_evaluations=view_evaluations,
    )
    return ArticulatedCandidateEvaluation(
        candidate_id=candidate_id,
        status=ArticulatedCandidateStatus.MULTI_STATE,
        fitting_sha256="0" * 64,
        candidate_sha256="1" * 64,
        fitted_model_sha256="2" * 64,
        link_assignment_sha256="3" * 64,
        heldout_evidence_sha256="4" * 64,
        state_evaluations=[state],
        passed_hard_gates=True,
        failed_gates=[],
        heldout_state_validation_used=True,
        capture_state_count=3,
        accepted_alignment_state_ids=["closed", "half", "open"],
        selected_candidate_validation_level="multi_state_heldout_validated",
        link_assignment_confidence=0.9,
        heldout_q_objective_path=f"raw/open_q_objective_{candidate_id}.json",
        heldout_q_objective_sha256="5" * 64,
        heldout_q_objective_preview_path=f"previews/open_q_objective_{candidate_id}.png",
        heldout_q_objective_preview_sha256="6" * 64,
        runtime_seconds=0,
    )


IDENTITY = [
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


def test_articulation_evidence_levels_are_not_overstated() -> None:
    assert evidence_level(1) is ArticulationEvidenceLevel.SINGLE_STATE_PRIOR_ONLY
    assert evidence_level(2) is ArticulationEvidenceLevel.TWO_STATE_MOTION_SUPPORTED
    assert evidence_level(3) is ArticulationEvidenceLevel.MULTI_STATE_HELDOUT_AVAILABLE
    assert capture_evidence_tier(3) is ArticulationEvidenceLevel.MULTI_STATE_HELDOUT_AVAILABLE
    assert (
        effective_evidence_level(
            3,
            valid_measured_motion=True,
            heldout_evaluation_ran=True,
            heldout_candidate_passed=True,
        )
        is ArticulationEvidenceLevel.MULTI_STATE_HELDOUT_VALIDATED
    )
    with pytest.raises(ValueError, match="at least one"):
        evidence_level(0)


def test_stable_part_ids_are_separate_from_prompts_and_unique() -> None:
    manifest = ArticulationPartPromptManifest.model_validate(
        {
            "schema_version": "0.2.0",
            "objects": [
                {
                    "articulated_object_id": "cabinet_0001",
                    "semantic_label": "cabinet",
                    "base": {
                        "part_id": "cabinet_body",
                        "prompt_id": "cabinet body",
                        "label": "cabinet body",
                    },
                    "movable_parts": [
                        {
                            "part_id": "drawer",
                            "prompt_id": "drawer prompt",
                            "label": "drawer",
                            "expected_joint_hint": "prismatic",
                        }
                    ],
                }
            ],
        }
    )
    assert manifest.objects[0].base.part_id == "cabinet_body"
    assert manifest.objects[0].base.prompt_id == "cabinet body"
    with pytest.raises(ValueError):
        ArticulationPartPromptManifest.model_validate(
            {
                "schema_version": "0.2.0",
                "objects": [
                    {
                        "articulated_object_id": "cabinet_0001",
                        "semantic_label": "cabinet",
                        "base": {
                            "part_id": "drawer",
                            "prompt_id": "cabinet",
                            "label": "cabinet",
                        },
                        "movable_parts": [
                            {
                                "part_id": "drawer",
                                "prompt_id": "drawer",
                                "label": "drawer",
                            }
                        ],
                    }
                ],
            }
        )


def test_state_track_mapping_rejects_duplicate_track_identity() -> None:
    with pytest.raises(ValueError, match="multiple stable parts"):
        ArticulationStateRecord.model_validate(
            {
                "state_id": "closed",
                "run_dir": "/tmp/closed",
                "semantic_state_label": "closed",
                "part_track_ids": {
                    "cabinet_body": "same_track",
                    "drawer": "same_track",
                },
                "ingest_manifest_sha256": "0" * 64,
                "frame_sequence_digest": "0" * 64,
                "camera_reconstruction_sha256": "0" * 64,
                "segmentation_tracking_sha256": "0" * 64,
                "dense_depth_manifest_sha256": "0" * 64,
                "measured_geometry_sha256": "0" * 64,
                "phase5a_consistency_passed": True,
                "part_mask_hashes": {},
                "measured_part_cloud_hashes": {},
                "registered_frame_ids": [],
                "camera_evidence_path": "camera/reconstruction.json",
                "segmentation_evidence_path": "observations/object_tracks.json",
                "undistortion_evidence_path": ("reconstruction/dense/undistortion_manifest.json"),
                "depth_evidence_path": "reconstruction/dense/depth_manifest.json",
                "dense_map_hashes": {},
            }
        )


def test_declared_reference_state_is_first_even_when_manifest_order_differs() -> None:
    assert ordered_motion_state_ids(
        "closed",
        ["half", "closed", "open"],
        {"closed", "half", "open"},
    ) == ["closed", "half", "open"]
    with pytest.raises(ValueError, match="reference state was not accepted"):
        ordered_motion_state_ids("closed", ["half", "closed"], {"half"})


def test_part_diagonal_normalization_and_prismatic_q_scaling() -> None:
    kinematics = _evaluation_worker_module("articulation_evaluation_worker.kinematics")
    assert kinematics.normalized_residual(0.2, 4.0) == pytest.approx(0.05)
    assert kinematics.prismatic_candidate_q_scale(2.0) == pytest.approx(0.5)
    assert kinematics.revolute_candidate_q_scale() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="positive"):
        kinematics.prismatic_candidate_q_scale(0.0)


def test_axis_sign_is_provenance_only_in_fitted_q_scale() -> None:
    payload = {
        "candidate_id": "candidate",
        "matrix_reference_world_from_candidate_base": [
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "scale": 2.0,
        "link_assignments": [],
        "fitted_joints": [
            {
                "candidate_joint_id": "drawer_joint",
                "measured_joint_id": "drawer_joint",
                "parent_observed_part_id": "base",
                "child_observed_part_id": "drawer",
                "joint_type": "prismatic",
                "fitted_axis": [1.0, 0.0, 0.0],
                "axis_sign": -1,
                "q_scale": 0.5,
                "q_offset": 0.0,
                "fitting_state_q": {"closed": 0.0, "open": 0.5},
                "axis_refinement_degrees": 0.0,
                "fitting_residual_arbitrary_units": 0.01,
                "fitting_residual_part_diagonals": 0.01,
            }
        ],
        "generation_state_ids": ["closed"],
        "fitting_state_ids": ["open"],
        "heldout_state_ids": ["heldout"],
        "fit_residual_arbitrary_units": 0.01,
        "fit_residual_scene_diagonals": 0.01,
    }
    model = FittedArticulatedKinematicModel.model_validate(payload)
    assert model.fitted_joints[0].axis_sign == -1
    assert model.fitted_joints[0].q_scale == 0.5
    payload["fitted_joints"][0]["q_scale"] = -0.5
    with pytest.raises(ValueError, match="canonical axis convention"):
        FittedArticulatedKinematicModel.model_validate(payload)


def test_articulated_visual_asset_space_is_explicit() -> None:
    path = "candidates/link.glb"
    link = ArticulatedLink(
        link_id="link",
        name="link",
        visual_asset_paths=[path],
        visual_asset_hashes={path: "0" * 64},
        visual_asset_spaces={path: "candidate_base"},
        visual_asset_transforms_candidate_base={path: tuple(IDENTITY)},
        native_bounds_min=(0.0, 0.0, 0.0),
        native_bounds_max=(1.0, 1.0, 1.0),
    )
    assert link.visual_asset_spaces[path].value == "candidate_base"
    translated = list(IDENTITY)
    translated[3] = 1.0
    local = ArticulatedLink(
        link_id="local",
        name="local",
        visual_asset_paths=[path],
        visual_asset_hashes={path: "0" * 64},
        visual_asset_spaces={path: "link_local"},
        visual_asset_transforms_candidate_base={path: translated},
        native_bounds_min=(0.0, 0.0, 0.0),
        native_bounds_max=(1.0, 1.0, 1.0),
    )
    assert local.visual_asset_transforms_candidate_base[path][3] == 1.0
    measured = ArticulatedLink(
        link_id="measured",
        name="measured",
        visual_asset_paths=[path],
        visual_asset_hashes={path: "0" * 64},
        visual_asset_spaces={path: "reference_world"},
        visual_asset_transforms_candidate_base={path: tuple(IDENTITY)},
        native_bounds_min=(0.0, 0.0, 0.0),
        native_bounds_max=(1.0, 1.0, 1.0),
    )
    assert measured.visual_asset_spaces[path] is ArticulatedAssetSpace.REFERENCE_WORLD
    with pytest.raises(ValueError, match="identity transform"):
        ArticulatedLink(
            link_id="link",
            name="link",
            visual_asset_paths=[path],
            visual_asset_hashes={path: "0" * 64},
            visual_asset_spaces={path: "candidate_base"},
            visual_asset_transforms_candidate_base={path: translated},
            native_bounds_min=(0.0, 0.0, 0.0),
            native_bounds_max=(1.0, 1.0, 1.0),
        )


def test_reference_world_worker_asset_requires_identity_transform(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    evaluator = _evaluation_worker_module("articulation_evaluation_worker.cli")
    path = tmp_path / "measured.ply"
    path.write_bytes(b"measured reference-world fixture")
    link = {
        "visual_asset_spaces": {"measured.ply": "reference_world"},
        "visual_asset_hashes": {"measured.ply": sha256_file(path)},
        "visual_asset_transforms_candidate_base": {"measured.ply": IDENTITY},
    }
    matrix = evaluator._visual_asset_to_candidate_matrix(tmp_path, link, "measured.ply")
    assert np.allclose(matrix, np.eye(4))
    translated = list(IDENTITY)
    translated[3] = 1.0
    link["visual_asset_transforms_candidate_base"] = {"measured.ply": translated}
    with pytest.raises(ValueError, match="identity transform"):
        evaluator._visual_asset_to_candidate_matrix(tmp_path, link, "measured.ply")


def test_heldout_view_requires_complete_link_coverage() -> None:
    with pytest.raises(ValueError, match="usability"):
        ArticulationHeldoutViewEvaluation(
            frame_id="frame",
            camera_reconstruction_sha256="0" * 64,
            depth_path="dense/depth.bin",
            depth_sha256="1" * 64,
            valid_depth=True,
            target_mask_paths={"drawer": "masks/drawer.png"},
            target_mask_hashes={"drawer": "2" * 64},
            target_masks_complete=True,
            required_link_ids=["base", "drawer"],
            rendered_link_ids=["base"],
            missing_link_ids=["drawer"],
            usable=True,
            failure_reasons=[],
            render_path="renders/frame.png",
            render_sha256="3" * 64,
            raw_candidate_pixel_count=10,
            visible_candidate_pixel_count=10,
            target_mask_pixel_count=10,
        )


def test_heldout_revolute_axis_and_signed_angle_are_measured() -> None:
    np = pytest.importorskip("numpy")
    evaluator = _evaluation_worker_module("articulation_evaluation_worker.cli")
    angle = math.radians(-35.0)
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    axis, measured_angle = evaluator._rotation_axis_and_signed_angle(
        rotation,
        np.asarray([0.0, 0.0, 1.0]),
    )
    assert axis is not None
    assert abs(float(axis[2])) == pytest.approx(1.0)
    assert measured_angle == pytest.approx(angle)


def test_capture_template_and_real_preflight_use_state_local_tracks(
    tmp_path: Path,
) -> None:
    cli = CliRunner()
    template = cli.invoke(
        app,
        [
            "articulation",
            "capture-template",
            "--object-id",
            "cabinet_0001",
            "--states",
            "closed,half_open,open",
        ],
    )
    assert template.exit_code == 0
    template_payload = yaml.safe_load(template.stdout)
    assert template_payload["schema_version"] == "0.2.0"
    assert len(template_payload["states"]) == 3

    states = []
    for index, label in enumerate(("closed", "half_open", "open")):
        run_dir = tmp_path / label
        (run_dir / "validation").mkdir(parents=True)
        (run_dir / "camera").mkdir()
        (run_dir / "observations/masks").mkdir(parents=True)
        (run_dir / "reconstruction/dense/maps").mkdir(parents=True)
        (run_dir / "reconstruction/measured_objects/objects").mkdir(parents=True)
        (run_dir / "validation/phase5a_measured_geometry.json").write_text(
            json.dumps({"passed": True})
        )
        (run_dir / "camera/reconstruction.json").write_text(
            json.dumps({"registered_frame_ids": ["frame_000", "frame_001"]})
        )
        track_ids = {
            "cabinet_body": f"cabinet_body_{index:04d}",
            "drawer": f"drawer_{index:04d}",
        }
        tracks = []
        hypotheses = []
        for track_id in track_ids.values():
            observations = []
            for frame_id in ("frame_000", "frame_001"):
                mask = run_dir / f"observations/masks/{track_id}_{frame_id}.png"
                mask.write_bytes(b"mask")
                observations.append(
                    {
                        "frame_id": frame_id,
                        "mask_path": mask.relative_to(run_dir).as_posix(),
                    }
                )
            tracks.append({"object_id": track_id, "observations": observations})
            point_path = run_dir / f"reconstruction/measured_objects/objects/{track_id}.ply"
            point_path.write_text("ply\n", encoding="ascii")
            hypotheses.append(
                {
                    "object_id": track_id,
                    "point_cloud": {"relative_path": point_path.relative_to(run_dir).as_posix()},
                    "supporting_view_count": 2,
                    "observations": [
                        {
                            "frame_id": "frame_000",
                            "registered": True,
                            "validated_sample_count": 12,
                        },
                        {
                            "frame_id": "frame_001",
                            "registered": True,
                            "validated_sample_count": 9,
                        },
                    ],
                }
            )
        (run_dir / "observations/object_tracks.json").write_text(json.dumps({"tracks": tracks}))
        depth_records = []
        for frame_id in ("frame_000", "frame_001"):
            depth = run_dir / f"reconstruction/dense/maps/{frame_id}.bin"
            depth.write_bytes(b"depth")
            depth_records.append(
                {
                    "frame_id": frame_id,
                    "depth_path": depth.relative_to(run_dir).as_posix(),
                }
            )
        (run_dir / "reconstruction/dense/depth_manifest.json").write_text(
            json.dumps({"records": depth_records})
        )
        (run_dir / "reconstruction/measured_objects/geometry_manifest.json").write_text(
            json.dumps({"hypotheses": hypotheses})
        )
        states.append(
            {
                "state_id": f"state_{index:03d}_{label}",
                "run_dir": str(run_dir),
                "semantic_state_label": label,
                "part_track_ids": track_ids,
            }
        )
    capture_path = tmp_path / "capture.yaml"
    capture_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "0.2.0",
                "articulated_object_id": "cabinet_0001",
                "reference_state_id": "state_000_closed",
                "states": states,
            }
        )
    )
    result = cli.invoke(
        app,
        [
            "articulation",
            "preflight-capture",
            "--capture-manifest",
            str(capture_path),
            "--part-manifest",
            str(ROOT / "configs/articulation_parts/cabinet_drawer.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "capture preflight passed: 3 states" in result.output


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


def test_prismatic_prior_uses_translation_only_for_partial_geometry() -> None:
    np = pytest.importorskip("numpy")
    sim3 = _worker_module("articulation_alignment_worker.sim3")
    generator = np.random.default_rng(17)
    shared = generator.normal(size=(1000, 3))
    translation = np.array([0.35, -0.2, 0.55])
    partial_view_density = shared[:500] - translation
    source = np.concatenate([shared - translation, partial_view_density], axis=0)

    fit = sim3.robust_translation_registration(source, shared)

    assert fit.matrix[:3, :3] == pytest.approx(np.eye(3))
    assert fit.matrix[:3, 3] == pytest.approx(translation, abs=1e-6)


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
    position, residual = evaluator._heldout_joint_position(
        input_root=tmp_path,
        link={
            "visual_asset_paths": ["candidate.ply"],
            "visual_asset_hashes": {"candidate.ply": sha256_file(candidate_path)},
            "visual_asset_spaces": {"candidate.ply": "candidate_base"},
            "visual_asset_transforms_candidate_base": {"candidate.ply": IDENTITY},
        },
        joint={
            "joint_id": "drawer_joint",
            "joint_type": "prismatic",
            "axis": [1.0, 0.0, 0.0],
            "pivot": None,
            "candidate_limit_lower": 0.0,
            "candidate_limit_upper": 1.0,
            "limit_source": "candidate_prior",
        },
        fitted_joint={
            "fitted_axis": [1.0, 0.0, 0.0],
            "fitted_pivot": None,
            "fitting_state_q": {"state_000": 0.0, "state_001": 0.5},
        },
        measured_joint=None,
        measured_part={"measured_point_cloud_path": "measured.ply"},
        base_matrix=np.eye(4),
        reference_from_state=np.eye(4),
    )
    assert position == pytest.approx(0.7, abs=2e-3)
    assert residual is not None
    assert residual < 2e-3


def test_heldout_q_search_finds_global_minimum_across_multiple_basins() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    evaluator = _evaluation_worker_module("articulation_evaluation_worker.cli")

    def objective(q_value: float) -> float:
        return min((q_value + 0.65) ** 2 + 0.02, (q_value - 0.55) ** 2)

    result = evaluator._deterministic_global_q_search(
        objective,
        -1.0,
        1.0,
        grid_sample_count=401,
    )

    assert result["grid_sample_count"] == 401
    assert len(result["samples"]) == 401
    assert result["refined_global_minimum_q"] == pytest.approx(0.55, abs=1e-5)
    assert result["refined_global_minimum_objective"] == pytest.approx(0.0, abs=1e-9)
    assert len(result["all_local_minima"]) >= 2
    assert np.isfinite(result["legacy_optimizer_q"])
    assert result["optimizer_global_minimum_verified"] is True


@pytest.mark.parametrize("expected_q", [-0.7, 0.7])
def test_heldout_q_search_supports_both_prismatic_signs(expected_q: float) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    evaluator = _evaluation_worker_module("articulation_evaluation_worker.cli")
    result = evaluator._deterministic_global_q_search(
        lambda q_value: (q_value - expected_q) ** 2,
        -1.0,
        1.0,
        grid_sample_count=401,
    )
    assert result["refined_global_minimum_q"] == pytest.approx(expected_q, abs=1e-5)


def test_heldout_q_search_verifies_boundary_global_minimum() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    evaluator = _evaluation_worker_module("articulation_evaluation_worker.cli")
    result = evaluator._deterministic_global_q_search(
        lambda q_value: (q_value + 1.0) ** 2,
        -1.0,
        1.0,
        grid_sample_count=401,
    )
    assert result["refined_global_minimum_q"] == -1.0
    assert result["optimizer_global_minimum_verified"] is True
    assert result["legacy_optimizer_matches_global_minimum"] is True


def test_semantic_q_ordering_reports_inconsistency_without_forcing_q() -> None:
    pytest.importorskip("numpy")
    evaluator = _evaluation_worker_module("articulation_evaluation_worker.cli")
    fitted_joint = {
        "fitting_state_q": {"closed": 0.0, "half": 3.0},
        "q_scale": 1.0,
        "q_offset": 0.0,
    }
    ordering = evaluator._semantic_q_ordering(
        fitted_joint=fitted_joint,
        heldout_state_id="open",
        heldout_q=-3.8,
        semantic_state_labels={
            "closed": "closed",
            "half": "half_open",
            "open": "open",
        },
        local_minima=[{"q": -3.8, "total_objective": 0.1}],
    )
    assert ordering["ordering_consistent"] is False
    assert ordering["direction"] == "inconsistent"
    assert ordering["measured_q_by_state"]["open"] == -3.8


def test_semantic_q_ordering_accepts_consistent_sign_reversal() -> None:
    pytest.importorskip("numpy")
    evaluator = _evaluation_worker_module("articulation_evaluation_worker.cli")
    fitted_joint = {
        "fitting_state_q": {"closed": 0.0, "half": -2.0},
        "q_scale": 1.0,
        "q_offset": 0.0,
    }
    ordering = evaluator._semantic_q_ordering(
        fitted_joint=fitted_joint,
        heldout_state_id="open",
        heldout_q=-4.0,
        semantic_state_labels={
            "closed": "closed",
            "half": "half_open",
            "open": "open",
        },
        local_minima=[{"q": -4.0, "total_objective": 0.1}],
    )
    assert ordering["ordering_consistent"] is True
    assert ordering["direction"] == "decreasing"


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
                "visual_asset_spaces": {},
                "visual_asset_transforms_candidate_base": {},
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


@pytest.mark.parametrize(
    ("source_family", "expected_source"),
    [
        (ArticulatedSourceFamily.ARTVIP, GeometrySourceType.RETRIEVED),
        (ArticulatedSourceFamily.PARTNET_MOBILITY, GeometrySourceType.RETRIEVED),
        (ArticulatedSourceFamily.PARTICULATE, GeometrySourceType.GENERATED),
        (ArticulatedSourceFamily.MEASURED_MOTION, GeometrySourceType.MEASURED),
    ],
)
def test_articulated_candidate_scene_source_mapping(
    source_family: ArticulatedSourceFamily,
    expected_source: GeometrySourceType,
) -> None:
    assert articulated_candidate_geometry_source(source_family) is expected_source


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
                        "visual_asset_spaces": {"assets/base.ply": "candidate_base"},
                        "visual_asset_transforms_candidate_base": {"assets/base.ply": IDENTITY},
                        "native_bounds_min": [0, 0, 0],
                        "native_bounds_max": [1, 1, 1],
                    },
                    {
                        "link_id": "drawer_0001",
                        "name": "drawer",
                        "visual_asset_paths": ["assets/drawer.ply"],
                        "visual_asset_hashes": {"assets/drawer.ply": "0" * 64},
                        "visual_asset_spaces": {"assets/drawer.ply": "candidate_base"},
                        "visual_asset_transforms_candidate_base": {"assets/drawer.ply": IDENTITY},
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


def test_particulate_accepts_one_explicit_local_retrieval_manifest() -> None:
    config = ParticulateAdapterConfig(
        execution_mode="fake_worker",
        worker_script="tests/fixtures/fake_articulation_worker.py",
        retrieval_manifests=["artvip_retrieval.json"],
    )
    assert config.retrieval_manifests == ["artvip_retrieval.json"]

    with pytest.raises(ValueError, match="must be unique"):
        ParticulateAdapterConfig(
            execution_mode="fake_worker",
            worker_script="tests/fixtures/fake_articulation_worker.py",
            retrieval_manifests=[
                "artvip_retrieval.json",
                "artvip_retrieval.json",
            ],
        )


def test_fake_phase5c_pipeline_resume_and_cli(tmp_path: Path) -> None:
    run_dir = tmp_path / "phase5c"
    runner = PipelineRunner(load_config(CONFIG), INPUT, run_dir)
    first = runner.run()
    assert all(stage["status"] == "succeeded" for stage in first["stages"].values())
    capture = ArticulationCaptureManifest.model_validate_json(
        (run_dir / "reconstruction/articulation/capture_manifest.json").read_text()
    )
    assert capture.capture_evidence_tier is ArticulationEvidenceLevel.MULTI_STATE_HELDOUT_AVAILABLE
    assert capture.capture_state_count == 3
    assert len({state.part_track_ids["cabinet_body"] for state in capture.states}) == 3
    assert all(set(state.part_track_ids) == {"cabinet_body", "drawer"} for state in capture.states)
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
        if candidate.source_family is ArticulatedSourceFamily.PARTICULATE:
            assert candidate.working_frame_hypothesis == "+Z"
            assert candidate.working_frame_hypotheses_evaluated == ["+Z"]
            assert candidate.working_frame_selection_evidence
    evaluation = ArticulatedEvaluationManifest.model_validate_json(
        (run_dir / "reconstruction/articulation/evaluation_manifest.json").read_text()
    )
    assert evaluation.candidate_structures_frozen_before_heldout
    fitting = ArticulationFittingManifest.model_validate_json(
        (run_dir / "reconstruction/articulation/fitting_manifest.json").read_text()
    )
    fitting_by_id = {item.candidate_id: item for item in fitting.fittings}
    assert any(
        item.passed_hard_gates
        and item.selected_candidate_validation_level
        is ArticulationEvidenceLevel.MULTI_STATE_HELDOUT_VALIDATED
        and {
            "state_000",
            "state_001",
            "state_002",
        }
        == {
            *fitting_by_id[item.candidate_id].fitted_model.generation_state_ids,
            *fitting_by_id[item.candidate_id].fitted_model.fitting_state_ids,
            *(state.state_id for state in item.state_evaluations),
        }
        for item in evaluation.evaluations
        if fitting_by_id[item.candidate_id].fitted_model is not None
    )
    selection = ArticulatedCandidateSelection.model_validate_json(
        (run_dir / "reconstruction/articulation/selection.json").read_text()
    )
    assert selection.objects[0].best_research_articulated_candidate
    report = Phase5CConsistencyReport.model_validate_json(
        (run_dir / "validation/phase5c_articulated_reconstruction.json").read_text()
    )
    assert report.passed
    assert len(report.checks) >= 40
    scene = SceneIR.model_validate_json((run_dir / "scene_ir/phase5c_scene.json").read_text())
    assert scene.collision_assets == []
    assert all(not item.sim_ready for item in scene.objects)
    selected_candidate = next(
        item
        for item in candidates.candidates
        if item.candidate_id == selection.objects[0].selected_candidate_id
    )
    expected_visual_source = articulated_candidate_geometry_source(selected_candidate.source_family)
    articulated_visuals = [
        item for item in scene.geometry_assets if item.asset_role == "articulated_visual_link"
    ]
    assert articulated_visuals
    assert all(item.source is expected_visual_source for item in articulated_visuals)
    assert all(item.provenance.source is expected_visual_source for item in articulated_visuals)
    assert (
        run_dir / "reconstruction/articulation/selected/cabinet_0001/preview_only.urdf"
    ).is_file()
    bundle = json.loads(
        (
            run_dir / "reconstruction/articulation/selected/cabinet_0001/kinematic_bundle.json"
        ).read_text()
    )
    typed_bundle = ArticulatedKinematicBundle.model_validate(bundle)
    selected = selection.objects[0]
    assert typed_bundle.fitted_kinematic_model.path == selected.fitted_model_path
    assert typed_bundle.fitted_kinematic_model.sha256 == selected.fitted_model_sha256
    selected_root = run_dir / "reconstruction/articulation/selected/cabinet_0001"
    identity = ArticulatedSelectedIdentityManifest.model_validate_json(
        (selected_root / "selected_identity_manifest.json").read_text()
    )
    assert identity.selected_candidate.sha256 == sha256_file(
        selected_root / "selected_candidate.json"
    )
    assert identity.fitted_kinematic_model.sha256 == sha256_file(
        selected_root / "fitted_kinematic_model.json"
    )
    assert identity.selected_link_assignment.sha256 == sha256_file(
        selected_root / "selected_link_assignment.json"
    )
    assert identity.selected_evaluation.sha256 == sha256_file(
        selected_root / "selected_evaluation.json"
    )
    scene_object = next(item for item in scene.objects if item.object_id == "cabinet_0001")
    assert scene_object.articulation is not None
    assert scene_object.articulation.fitting_artifact_sha256
    scene_assets = {item.asset_id: item for item in scene.geometry_assets}
    measured_asset_ids = {
        item.asset_id for item in scene.geometry_assets if item.asset_role == "measured_anchor"
    }
    assert measured_asset_ids <= set(scene_object.geometry_asset_ids)
    assert all(
        scene_assets[asset_id].articulated_asset_space == "reference_world"
        for asset_id in measured_asset_ids
    )
    assert not any(
        asset_id in measured_asset_ids
        for link in scene_object.articulation.links
        for asset_id in link.geometry_asset_ids
    )
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
    assert (
        motion.effective_motion_evidence_level is ArticulationEvidenceLevel.SINGLE_STATE_PRIOR_ONLY
    )
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
    assert evaluation_request["semantic_state_labels"] == {
        "state_000": "closed",
        "state_001": "half_open",
        "state_002": "open",
    }
    selection = ArticulatedCandidateSelection.model_validate_json(
        (run_dir / "reconstruction/articulation/selection.json").read_text()
    )
    assert selection.objects[0].selected_candidate_id is None


@pytest.mark.parametrize(
    ("fake_mode", "expected_gate"),
    [
        ("missing_heldout_views", "minimum_usable_heldout_views"),
        ("no_target_mask", "target_part_mask_unavailable"),
    ],
)
def test_missing_heldout_render_evidence_fails_closed(
    tmp_path: Path,
    fake_mode: str,
    expected_gate: str,
) -> None:
    config = load_config(CONFIG)
    stage = config.stages["articulation_evaluation"]
    adapter = stage.adapter.model_copy(
        update={"config": {**stage.adapter.config, "fake_mode": fake_mode}}
    )
    config = config.model_copy(
        update={
            "stages": {
                **config.stages,
                "articulation_evaluation": stage.model_copy(update={"adapter": adapter}),
            }
        }
    )
    run_dir = tmp_path / fake_mode
    result = PipelineRunner(config, INPUT, run_dir).run()
    assert all(item["status"] == "succeeded" for item in result["stages"].values())
    evaluation = ArticulatedEvaluationManifest.model_validate_json(
        (run_dir / "reconstruction/articulation/evaluation_manifest.json").read_text()
    )
    assert evaluation.evaluations
    assert all(not item.passed_hard_gates for item in evaluation.evaluations)
    assert all(expected_gate in item.failed_gates for item in evaluation.evaluations)
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


def test_link_local_preview_urdf_preserves_visual_transform(phase5c_run: Path) -> None:
    selection = ArticulatedCandidateSelection.model_validate_json(
        (phase5c_run / "reconstruction/articulation/selection.json").read_text()
    )
    selected_id = selection.objects[0].selected_candidate_id
    assert selected_id is not None
    candidates = ArticulatedCandidateManifest.model_validate_json(
        (phase5c_run / "reconstruction/articulation/candidate_manifest.json").read_text()
    )
    candidate = next(item for item in candidates.candidates if item.candidate_id == selected_id)
    fitting = ArticulationFittingManifest.model_validate_json(
        (phase5c_run / "reconstruction/articulation/fitting_manifest.json").read_text()
    )
    fitted = next(item for item in fitting.fittings if item.candidate_id == selected_id)
    link = candidate.links[0]
    visual_path = link.visual_asset_paths[0]
    transform = (
        0.0,
        -2.0,
        0.0,
        1.0,
        2.0,
        0.0,
        0.0,
        2.0,
        0.0,
        0.0,
        2.0,
        3.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    updated_link = link.model_copy(
        update={
            "visual_asset_spaces": {
                **link.visual_asset_spaces,
                visual_path: ArticulatedAssetSpace.LINK_LOCAL,
            },
            "visual_asset_transforms_candidate_base": {
                **link.visual_asset_transforms_candidate_base,
                visual_path: transform,
            },
        }
    )
    updated_candidate = candidate.model_copy(
        update={
            "links": [
                updated_link if item.link_id == updated_link.link_id else item
                for item in candidate.links
            ]
        }
    )
    output = phase5c_run / "link_local_preview.urdf"
    ArticulationSelectionAdapter._write_preview_urdf(
        output,
        "cabinet_0001",
        updated_candidate,
        fitted,
    )
    text = output.read_text(encoding="utf-8")
    assert 'reconevery_asset_space="link_local"' in text
    assert 'origin xyz="1 2 3" rpy="0 -0 1.57079632679"' in text
    assert f'filename="{visual_path}" scale="2 2 2"' in text
    assert "matrix_reference_world_from_candidate_base=" in text


@pytest.mark.parametrize(
    "selection_path_field",
    [
        "selected_candidate_path",
        "fitted_model_path",
        "link_assignment_path",
        "evaluation_path",
        "selected_identity_manifest_path",
        "kinematic_bundle_path",
    ],
)
def test_selected_file_tamper_is_detected(
    phase5c_run: Path,
    selection_path_field: str,
) -> None:
    selection = ArticulatedCandidateSelection.model_validate_json(
        (phase5c_run / "reconstruction/articulation/selection.json").read_text()
    )
    selected = selection.objects[0]
    path_value = getattr(selected, selection_path_field)
    assert isinstance(path_value, str)
    path = phase5c_run / path_value
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    runner = PipelineRunner(load_config(CONFIG), INPUT, phase5c_run)
    try:
        runner.run(from_stage="phase5c_consistency_validation")
    except (RuntimeError, ValueError):
        return
    report = Phase5CConsistencyReport.model_validate_json(
        (phase5c_run / "validation/phase5c_articulated_reconstruction.json").read_text()
    )
    assert not report.passed
    assert not next(
        item for item in report.checks if item.check_id == "selected_file_hashes_exact"
    ).passed


@pytest.fixture
def phase5c_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("phase5c")
    PipelineRunner(load_config(CONFIG), INPUT, run_dir).run()
    return run_dir
