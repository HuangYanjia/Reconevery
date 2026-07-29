from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from recon2sim.adapters.canonical_scene import (
    _asset_transform_policy,
    _canonical_scene,
    _joint_motion_matrix,
    _prismatic_unit_mappings,
    _requires_direct_world_wrapper,
    _transform_matrix,
)
from recon2sim.adapters.world_calibration import _dataset_split
from recon2sim.artifacts import (
    CanonicalSceneWrapper,
    Phase6AConsistencyReport,
    WorldCalibrationArtifact,
    WorldCalibrationManifest,
    WorldCalibrationTransform,
)
from recon2sim.calibration import (
    build_sim3,
    canonical_rotation,
    invert_sim3,
    maximum_roundtrip_error,
    multiply_matrix4,
    rotation_determinant,
    sha256_file,
    transform_point,
)
from recon2sim.cli import app
from recon2sim.config import load_config
from recon2sim.ir import SceneIR
from recon2sim.pipeline import PipelineRunner

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "examples/tabletop"
CONFIG = ROOT / "configs/phase6a_e2e_fake.yaml"


def _transform_record() -> dict[str, object]:
    return {
        "scale_m_per_colmap": 2.0,
        "rotation_canonical_from_colmap": [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "translation_canonical_m": [1.0, -2.0, 0.5],
        "matrix_canonical_from_colmap": [
            2.0,
            0.0,
            0.0,
            1.0,
            0.0,
            2.0,
            0.0,
            -2.0,
            0.0,
            0.0,
            2.0,
            0.5,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "matrix_colmap_from_canonical": [
            0.5,
            0.0,
            0.0,
            -0.5,
            0.0,
            0.5,
            0.0,
            1.0,
            0.0,
            0.0,
            0.5,
            -0.25,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "rotation_determinant": 1.0,
        "orthonormal_error": 0.0,
        "inverse_roundtrip_error": 0.0,
        "covariance_diagonal": None,
    }


def _artifact() -> WorldCalibrationArtifact:
    transform = _transform_record()
    return WorldCalibrationArtifact.model_validate(
        {
            "status": "accepted_full_canonical",
            "evidence_tier": "full_canonical",
            "manifest_path": "calibration/evidence_manifest.json",
            "manifest_sha256": "0" * 64,
            "dataset_split": {
                "fitting_evidence_ids": ["fit"],
                "heldout_evidence_ids": ["hold"],
                "fitting_frame_ids": ["frame_0"],
                "heldout_frame_ids": ["frame_1"],
                "split_policy": "synthetic",
            },
            "candidates": [
                {
                    "candidate_id": "candidate",
                    "evidence_tier": "full_canonical",
                    "selected_by_fitting_only": True,
                    "transform": transform,
                    "fitting_objective": 0.0,
                    "evidence_ids": ["fit"],
                }
            ],
            "selected_candidate_id": "candidate",
            "accepted_transform": transform,
            "metrics": {"sim3_roundtrip_error": 0.0},
            "metric_scale_known": True,
            "gravity_alignment_known": True,
            "canonical_forward_known": True,
            "canonical_origin_known": True,
            "full_canonical_world_available": True,
            "source_cameras_unchanged": True,
            "source_geometry_unchanged": True,
        }
    )


def _articulated_scene() -> SceneIR:
    confidence = {"score": 1.0, "method": "synthetic"}
    provenance = {
        "adapter_name": "synthetic",
        "adapter_version": "0.1.0",
        "configuration": {},
        "input_artifact_paths": [],
        "output_artifact_paths": [],
        "confidence": confidence,
        "source": "measured",
    }
    return SceneIR.model_validate(
        {
            "schema_version": "0.1.6",
            "metadata": {
                "scene_id": "synthetic",
                "name": "synthetic articulation",
                "coordinate_convention": {
                    "world_frame": "colmap_arbitrary",
                    "alignment_status": "unoriented",
                    "camera_axes": "x_right_y_down_z_forward",
                    "linear_units": "arbitrary_units",
                    "scale_status": "scale_ambiguous",
                    "transform_direction": "world_from_camera",
                },
                "source": "measured",
                "provenance": [provenance],
            },
            "objects": [
                {
                    "object_id": "cabinet",
                    "name": "cabinet",
                    "asset_type": "articulated",
                    "transform": {
                        "translation": [1.0, 0.0, 0.0],
                        "rotation_xyzw": [
                            0.0,
                            0.0,
                            0.7071067811865475,
                            0.7071067811865476,
                        ],
                        "scale": [1.7, 1.7, 1.7],
                    },
                    "physics": {"is_static": True},
                    "articulation": {
                        "articulation_id": "cabinet_articulation",
                        "links": [
                            {"link_id": "base", "name": "base"},
                            {"link_id": "drawer", "name": "drawer"},
                            {"link_id": "door", "name": "door"},
                        ],
                        "joints": [
                            {
                                "joint_id": "drawer_joint",
                                "parent_link_id": "base",
                                "child_link_id": "drawer",
                                "joint_type": "prismatic",
                                "axis_xyz": [1.0, 0.0, 0.0],
                                "limits": [0.0, 0.5],
                                "observed_position_range": [0.0, 0.4],
                                "observed_state_positions": {
                                    "closed": 0.0,
                                    "open": 0.4 / 1.7,
                                    "negative": -0.2 / 1.7,
                                },
                            },
                            {
                                "joint_id": "door_joint",
                                "parent_link_id": "base",
                                "child_link_id": "door",
                                "joint_type": "revolute",
                                "axis_xyz": [0.0, 0.0, 1.0],
                                "origin_xyz": [1.0, 2.0, 3.0],
                                "limits": [0.0, 1.57],
                                "observed_state_positions": {"open": 1.0},
                            },
                        ],
                    },
                    "provenance": [provenance],
                    "confidence": confidence,
                }
            ],
        }
    )


def test_canonical_axes_are_right_handed_and_reject_parallel_forward() -> None:
    rotation = canonical_rotation((0.0, 1.0, 0.0), (1.0, 1.0, 0.0))
    assert rotation_determinant(rotation) == pytest.approx(1.0)
    assert rotation[6:9] == pytest.approx((0.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="zero"):
        canonical_rotation((0.0, 0.0, 1.0), (0.0, 0.0, 2.0))


def test_sim3_roundtrip_and_known_point() -> None:
    rotation = canonical_rotation((0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    matrix = build_sim3(2.0, rotation, (1.0, -2.0, 0.5))
    inverse = invert_sim3(matrix)
    assert maximum_roundtrip_error(matrix, inverse) < 1e-12
    assert transform_point(matrix, (1.0, 2.0, 3.0)) == pytest.approx((3.0, 2.0, 6.5))


def test_world_transform_rejects_negative_scale_and_improper_rotation() -> None:
    payload = _transform_record()
    payload["scale_m_per_colmap"] = -1.0
    with pytest.raises(ValidationError, match="greater than 0"):
        WorldCalibrationTransform.model_validate(payload)
    payload = _transform_record()
    payload["rotation_determinant"] = -1.0
    with pytest.raises(ValidationError, match="proper"):
        WorldCalibrationTransform.model_validate(payload)


def test_articulated_metric_propagation_preserves_local_joint_quantities() -> None:
    source = _articulated_scene()
    calibrated = _canonical_scene(source, _artifact())
    instance = calibrated.objects[0]
    assert instance.transform.translation == pytest.approx((3.0, -2.0, 0.5))
    assert instance.articulation is not None
    drawer, door = instance.articulation.joints
    assert drawer.axis_xyz == pytest.approx((1.0, 0.0, 0.0))
    assert drawer.limits == pytest.approx((0.0, 0.5))
    assert drawer.observed_state_positions["open"] == pytest.approx(0.4 / 1.7)
    assert door.limits == pytest.approx((0.0, 1.57))
    assert door.observed_state_positions["open"] == pytest.approx(1.0)
    assert door.origin_xyz == pytest.approx((1.0, 2.0, 3.0))
    assert source.objects[0].transform.translation == (1.0, 0.0, 0.0)


def test_articulated_world_space_pose_parity_with_non_unit_root_scale() -> None:
    source = _articulated_scene()
    rotation = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    matrix = build_sim3(2.0, rotation, (1.0, -2.0, 0.5))
    inverse = invert_sim3(matrix)
    transform = {
        "scale_m_per_colmap": 2.0,
        "rotation_canonical_from_colmap": rotation,
        "translation_canonical_m": [1.0, -2.0, 0.5],
        "matrix_canonical_from_colmap": matrix,
        "matrix_colmap_from_canonical": inverse,
        "rotation_determinant": 1.0,
        "orthonormal_error": 0.0,
        "inverse_roundtrip_error": maximum_roundtrip_error(matrix, inverse),
        "covariance_diagonal": None,
    }
    payload = _artifact().model_dump(mode="json")
    payload["accepted_transform"] = transform
    payload["candidates"][0]["transform"] = transform
    calibration = WorldCalibrationArtifact.model_validate(payload)
    calibrated = _canonical_scene(source, calibration)
    source_object = source.objects[0]
    canonical_object = calibrated.objects[0]
    assert source_object.articulation is not None
    transform = calibration.accepted_transform
    assert transform is not None
    source_root = _transform_matrix(source_object.transform)
    canonical_root = _transform_matrix(canonical_object.transform)
    for joint in source_object.articulation.joints:
        for q in (-0.4, 0.0, 0.3, 1.1):
            local_motion = _joint_motion_matrix(
                joint.joint_type,
                joint.axis_xyz,
                joint.origin_xyz,
                q,
            )
            expected = multiply_matrix4(
                transform.matrix_canonical_from_colmap,
                multiply_matrix4(source_root, local_motion),
            )
            actual = multiply_matrix4(canonical_root, local_motion)
            assert actual == pytest.approx(expected, abs=1e-9)
    mappings = _prismatic_unit_mappings(source, calibration)
    assert len(mappings) == 1
    assert mappings[0].prismatic_position_space == "object_local"
    assert mappings[0].source_object_scale_colmap_per_local_unit == pytest.approx(1.7)
    assert mappings[0].prismatic_position_scale_to_m == pytest.approx(3.4)
    assert mappings[0].raw_joint_values_unchanged


def test_asset_space_chooses_direct_wrapper_or_hierarchy_but_never_both() -> None:
    assert _asset_transform_policy("reference_world") == "wrapper_sim3"
    assert _requires_direct_world_wrapper("reference_world", full=True)
    for asset_space in ("candidate_base", "link_local"):
        assert _asset_transform_policy(asset_space) == "hierarchy_root_composition"
        assert not _requires_direct_world_wrapper(asset_space, full=True)
    assert not _requires_direct_world_wrapper("reference_world", full=False)


def test_calibration_split_rejects_heldout_leakage() -> None:
    payload = _artifact().model_dump(mode="json")
    payload["dataset_split"]["heldout_frame_ids"] = ["frame_0"]
    with pytest.raises(ValidationError, match="disjoint"):
        WorldCalibrationArtifact.model_validate(payload)


def test_fake_phase6a_pipeline_resume_cli_and_source_immutability(tmp_path: Path) -> None:
    run_dir = tmp_path / "phase6a"
    runner = PipelineRunner(load_config(CONFIG), INPUT, run_dir)
    result = runner.run()
    assert all(item["last_execution"] == "executed" for item in result["stages"].values())
    manifest = WorldCalibrationManifest.model_validate_json(
        (run_dir / "calibration/evidence_manifest.json").read_text(encoding="utf-8")
    )
    camera_before = (run_dir / manifest.camera_reconstruction_path).read_bytes()
    report = Phase6AConsistencyReport.model_validate_json(
        (run_dir / "validation/phase6a_world_calibration.json").read_text(encoding="utf-8")
    )
    wrapper = CanonicalSceneWrapper.model_validate_json(
        (run_dir / "calibration/canonical_scene_wrapper.json").read_text(encoding="utf-8")
    )
    assert report.passed
    assert len(report.checks) == 33
    assert report.full_canonical_world_available
    assert not report.camera_poses_rewritten
    assert not report.source_geometry_rewritten
    assert not report.collision_generation_implemented
    assert not report.physics_identification_implemented
    assert wrapper.source_camera_reconstruction_path == manifest.camera_reconstruction_path
    assert wrapper.source_camera_reconstruction_sha256 == manifest.camera_reconstruction_sha256
    assert wrapper.fiducial_world_derivation_path == ("calibration/apriltag_world_derivation.json")
    assert wrapper.fiducial_world_derivation_sha256 == sha256_file(
        run_dir / "calibration/apriltag_world_derivation.json"
    )
    canonical_scene = SceneIR.model_validate_json(
        (run_dir / "scene_ir/phase6a_canonical_scene.json").read_text(encoding="utf-8")
    )
    scene_reference = canonical_scene.metadata.world_calibration
    assert scene_reference is not None
    assert scene_reference.source_scene_ir_sha256 == sha256_file(
        run_dir / manifest.source_scene_ir_path
    )
    assert scene_reference.world_calibration_artifact_sha256 == sha256_file(
        run_dir / "calibration/world_calibration.json"
    )
    assert scene_reference.canonical_scene_wrapper_sha256 == sha256_file(
        run_dir / "calibration/canonical_scene_wrapper.json"
    )
    resumed = runner.run(resume=True)
    assert all(item["last_execution"] == "cache_hit" for item in resumed["stages"].values())
    assert (run_dir / manifest.camera_reconstruction_path).read_bytes() == camera_before

    cli = CliRunner()
    verify = cli.invoke(app, ["validation", "verify-phase6a", str(run_dir)])
    assert verify.exit_code == 0, verify.output
    inspect = cli.invoke(app, ["calibration", "inspect", str(run_dir)])
    assert inspect.exit_code == 0, inspect.output
    assert json.loads(inspect.output)["status"] == "accepted_full_canonical"
    preview = cli.invoke(app, ["calibration", "render-previews", str(run_dir)])
    assert preview.exit_code == 0, preview.output


def test_landmark_manifest_requires_two_frames_per_point() -> None:
    with pytest.raises(ValidationError, match="at least two"):
        WorldCalibrationManifest.model_validate(
            {
                "run_id": "invalid",
                "frame_sequence_digest": "0" * 64,
                "camera_reconstruction_path": "camera.json",
                "camera_reconstruction_sha256": "0" * 64,
                "source_scene_ir_path": "scene.json",
                "source_scene_ir_sha256": "0" * 64,
                "evidence": [],
                "known_distance": {
                    "landmarks": [
                        {
                            "landmark_id": "width",
                            "point_a_id": "a",
                            "point_b_id": "b",
                            "known_distance_m": 1.0,
                            "role": "fitting",
                        }
                    ],
                    "observations": [
                        {
                            "frame_id": "f0",
                            "point_id": "a",
                            "pixel_xy": [0.0, 0.0],
                            "role": "fitting",
                        },
                        {
                            "frame_id": "f0",
                            "point_id": "b",
                            "pixel_xy": [1.0, 0.0],
                            "role": "fitting",
                        },
                        {
                            "frame_id": "f1",
                            "point_id": "a",
                            "pixel_xy": [0.0, 0.0],
                            "role": "heldout",
                        },
                        {
                            "frame_id": "f1",
                            "point_id": "b",
                            "pixel_xy": [1.0, 0.0],
                            "role": "heldout",
                        },
                    ],
                },
                "evidence_tier": "scale_only",
            }
        )


def test_apriltag_image_source_requires_exact_evidence_reference() -> None:
    payload = {
        "run_id": "tag",
        "frame_sequence_digest": "0" * 64,
        "camera_reconstruction_path": "camera.json",
        "camera_reconstruction_sha256": "1" * 64,
        "source_scene_ir_path": "scene.json",
        "source_scene_ir_sha256": "2" * 64,
        "evidence": [
            {
                "evidence_id": "tag_fit",
                "evidence_type": "apriltag",
                "trust": "metric_fiducial",
                "role": "fitting",
                "source_files": [
                    {
                        "relative_path": "images/tag.png",
                        "sha256": "3" * 64,
                        "media_type": "image/png",
                    }
                ],
                "supports_metric_scale": True,
            }
        ],
        "apriltag": {
            "official_commit": "0e16a12dd380fd607e4afd54712ee9b1ffb9ec8f",
            "tag_family": "tagStandard41h12",
            "tag_id": 0,
            "detection_edge_size_m": 0.1,
            "detector_source_path": "apriltag_pose.h::estimate_tag_pose",
            "image_sources": [
                {
                    "frame_id": "frame_0",
                    "image_path": "images/tag.png",
                    "image_sha256": "3" * 64,
                    "width": 640,
                    "height": 480,
                    "intrinsics_fx_fy_cx_cy": [500.0, 500.0, 320.0, 240.0],
                    "split": "fitting",
                }
            ],
        },
        "evidence_tier": "scale_only",
    }
    manifest = WorldCalibrationManifest.model_validate(payload)
    assert manifest.apriltag is not None
    assert manifest.apriltag.image_sources[0].image_coordinate_space == ("registered_undistorted")
    payload["apriltag"]["image_sources"][0]["image_sha256"] = "4" * 64
    with pytest.raises(ValidationError, match="exact matching"):
        WorldCalibrationManifest.model_validate(payload)


def test_dataset_split_uses_typed_roles_not_misleading_evidence_ids() -> None:
    manifest = WorldCalibrationManifest.model_validate(
        {
            "run_id": "roles",
            "frame_sequence_digest": "0" * 64,
            "camera_reconstruction_path": "camera.json",
            "camera_reconstruction_sha256": "1" * 64,
            "source_scene_ir_path": "scene.json",
            "source_scene_ir_sha256": "2" * 64,
            "evidence": [
                {
                    "evidence_id": "heldout_in_name_but_fit",
                    "evidence_type": "external_metric",
                    "trust": "surveyed",
                    "role": "fitting",
                    "supports_metric_scale": True,
                },
                {
                    "evidence_id": "fitting_in_name_but_hold",
                    "evidence_type": "external_metric",
                    "trust": "surveyed",
                    "role": "heldout",
                    "supports_metric_scale": True,
                },
            ],
            "evidence_tier": "scale_only",
        }
    )
    split = _dataset_split(manifest)
    assert split.fitting_evidence_ids == ["heldout_in_name_but_fit"]
    assert split.heldout_evidence_ids == ["fitting_in_name_but_hold"]


def test_world_calibration_status_flags_are_mutually_consistent() -> None:
    payload = _artifact().model_dump(mode="json")
    payload["status"] = "accepted_metric_only"
    payload["evidence_tier"] = "scale_only"
    payload["candidates"][0]["evidence_tier"] = "scale_only"
    payload["gravity_alignment_known"] = False
    payload["canonical_forward_known"] = False
    payload["canonical_origin_known"] = False
    payload["full_canonical_world_available"] = False
    artifact = WorldCalibrationArtifact.model_validate(payload)
    assert artifact.metric_scale_known
    payload["gravity_alignment_known"] = True
    with pytest.raises(ValidationError, match="metric-only"):
        WorldCalibrationArtifact.model_validate(payload)


def test_fiducial_orientation_cannot_bypass_pose_bound_world_contract() -> None:
    with pytest.raises(ValidationError, match="world_contract"):
        WorldCalibrationManifest.model_validate(
            {
                "run_id": "unbound_fiducial",
                "frame_sequence_digest": "0" * 64,
                "camera_reconstruction_path": "camera.json",
                "camera_reconstruction_sha256": "1" * 64,
                "source_scene_ir_path": "scene.json",
                "source_scene_ir_sha256": "2" * 64,
                "evidence": [],
                "gravity": [
                    {
                        "evidence_id": "unbound",
                        "source": "fiducial_orientation",
                        "trust": "surveyed",
                        "up_vector_colmap": [0.0, 0.0, 1.0],
                        "sign_evidence": "manually pasted vector",
                        "fitting_residual_degrees": 0.0,
                        "heldout_residual_degrees": 0.0,
                        "angular_uncertainty_degrees": 0.0,
                        "supporting_ids": ["not_pose_bound"],
                    }
                ],
                "evidence_tier": "gravity_only",
            }
        )
