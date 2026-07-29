from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from world_calibration_worker.apriltag_detection import detect_official
from world_calibration_worker.canonical_axes import canonical_rotation
from world_calibration_worker.floor_plane import robust_plane
from world_calibration_worker.gravity import combine_up_vectors
from world_calibration_worker.landmark_triangulation import (
    reprojection_errors,
    triangulate,
    undistort_observations,
)
from world_calibration_worker.sim3 import matrix, transform_points, umeyama
from world_calibration_worker.solver import (
    _known_distance_solution,
    _landmark_world_derivation,
    _tag_world_derivation,
)


def test_known_sim3_and_apriltag_camera_trajectory() -> None:
    source = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    angle = np.deg2rad(30.0)
    expected_rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    expected = matrix(0.25, expected_rotation, np.asarray([1.0, -2.0, 0.5]))
    target = transform_points(source, expected)
    scale, rotation, translation = umeyama(source, target)
    recovered = matrix(scale, rotation, translation)
    assert recovered == pytest.approx(expected, abs=1e-10)


def test_known_distance_landmark_triangulation_and_heldout_reprojection() -> None:
    intrinsic = np.asarray([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    projections = {}
    observations = []
    point = np.asarray([0.2, -0.1, 3.0])
    for index, center_x in enumerate((-0.5, 0.0, 0.5)):
        extrinsic = np.column_stack((np.eye(3), np.asarray([-center_x, 0.0, 0.0])))
        projection = intrinsic @ extrinsic
        frame = f"frame_{index}"
        projections[frame] = projection
        pixel_h = projection @ np.append(point, 1.0)
        pixel = pixel_h[:2] / pixel_h[2]
        observations.append({"frame_id": frame, "pixel_xy": pixel.tolist()})
    recovered = triangulate(observations[:2], projections)
    assert recovered == pytest.approx(point, abs=1e-10)
    assert reprojection_errors(recovered, observations[2:], projections)[0] < 1e-9


def test_opencv_distorted_landmarks_are_undistorted_before_triangulation() -> None:
    intrinsic = np.asarray([[1184.0, 0.0, 360.0], [0.0, 1193.0, 640.0], [0.0, 0.0, 1.0]])
    distortion = np.asarray([0.0105, 0.0258, -0.0039, 0.0030])
    centers = (-0.8, 0.0, 0.8)
    points = {
        "a": np.asarray([-0.25, -0.45, 3.5]),
        "b": np.asarray([-0.25, 0.45, 3.5]),
    }
    poses = []
    observations = []
    for index, center_x in enumerate(centers):
        frame_id = f"frame_{index}"
        poses.append(
            {
                "frame_id": frame_id,
                "transform_world_from_camera": {
                    "translation": [center_x, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "scale": [1.0, 1.0, 1.0],
                },
            }
        )
        for point_id, point in points.items():
            camera_point = point - np.asarray([center_x, 0.0, 0.0])
            projected, _ = cv2.projectPoints(
                camera_point.reshape(1, 1, 3),
                np.zeros(3),
                np.zeros(3),
                intrinsic,
                distortion,
            )
            observations.append(
                {
                    "frame_id": frame_id,
                    "point_id": point_id,
                    "pixel_xy": projected.reshape(2).tolist(),
                    "role": "fitting" if index < 2 else "heldout",
                }
            )
    camera = {
        "model": "OPENCV",
        "intrinsics": {
            "fx": 1184.0,
            "fy": 1193.0,
            "cx": 360.0,
            "cy": 640.0,
            "distortion": distortion.tolist(),
        },
        "poses": poses,
    }
    manifest = {
        "known_distance": {
            "landmarks": [
                {
                    "landmark_id": "height",
                    "point_a_id": "a",
                    "point_b_id": "b",
                    "known_distance_m": 0.9,
                    "role": "fitting",
                }
            ],
            "observations": observations,
        }
    }
    result = _known_distance_solution(
        manifest,
        camera,
        {"frame_0", "frame_1"},
        {"frame_2"},
    )
    assert result["fitting_scales"] == pytest.approx([1.0], abs=1e-9)
    assert result["heldout_reprojection_error_px"] == pytest.approx(0.0, abs=1e-8)
    fitting = [item for item in observations if item["role"] == "fitting"]
    assert undistort_observations(fitting, camera) != fitting


def test_gravity_plane_with_noise_and_outliers() -> None:
    generator = np.random.default_rng(7)
    xy = generator.uniform(-2.0, 2.0, size=(500, 2))
    z = generator.normal(0.0, 0.002, size=(500, 1))
    points = np.column_stack((xy, z))
    points = np.vstack((points, generator.uniform(-2.0, 2.0, size=(30, 3))))
    normal, _, residuals = robust_plane(points)
    if normal[2] < 0:
        normal *= -1
    assert normal == pytest.approx((0.0, 0.0, 1.0), abs=0.01)
    assert float(np.median(residuals)) < 0.01


def test_right_handed_axis_construction_and_gravity_conflict() -> None:
    rotation = canonical_rotation(
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([1.0, 0.0, 0.0]),
    )
    assert float(np.linalg.det(rotation)) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="incompatible"):
        combine_up_vectors(
            [
                {"up_vector_colmap": [0.0, 0.0, 1.0]},
                {"up_vector_colmap": [0.0, 0.0, -1.0]},
            ]
        )


def test_heldout_changes_do_not_change_fitted_sim3() -> None:
    fitting_source = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    fitting_target = 2.0 * fitting_source + np.asarray([1.0, 2.0, 3.0])
    first = umeyama(fitting_source, fitting_target)
    heldout_a = np.asarray([[0.0, 0.0, 1.0]])
    heldout_b = np.asarray([[100.0, -50.0, 20.0]])
    assert not np.array_equal(heldout_a, heldout_b)
    second = umeyama(fitting_source, fitting_target)
    assert first[0] == second[0]
    assert first[1] == pytest.approx(second[1])
    assert first[2] == pytest.approx(second[2])


def test_official_pose_is_inverted_to_tag_from_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDetector:
        def detect(self, image: np.ndarray) -> list[dict[str, object]]:
            assert image.shape == (8, 8)
            return [
                {
                    "id": 0,
                    "lb-rb-rt-lt": [[1.0, 6.0], [6.0, 6.0], [6.0, 1.0], [1.0, 1.0]],
                    "margin": 80.0,
                    "hamming": 0,
                }
            ]

        def estimate_tag_pose(
            self,
            detection: dict[str, object],
            tag_size_m: float,
            fx: float,
            fy: float,
            cx: float,
            cy: float,
        ) -> dict[str, object]:
            assert tag_size_m == 0.1
            assert (fx, fy, cx, cy) == (100.0, 100.0, 4.0, 4.0)
            return {
                "R": np.eye(3),
                "t": np.asarray([1.0, 2.0, 3.0]),
                "error": 0.01,
            }

    monkeypatch.setitem(
        sys.modules,
        "apriltag",
        SimpleNamespace(apriltag=lambda family: FakeDetector()),
    )
    records = detect_official(
        np.zeros((8, 8), dtype=np.uint8),
        family="tagStandard41h12",
        tag_id=0,
        tag_size_m=0.1,
        camera_params=(100.0, 100.0, 4.0, 4.0),
    )
    assert records[0]["camera_center_tag_m"] == pytest.approx((-1.0, -2.0, -3.0))
    assert records[0]["rotation_tag_from_camera"] == pytest.approx(np.eye(3).reshape(-1))


def _known_distance_fixture(*, independent_holdout: bool) -> tuple[dict, dict]:
    intrinsic = np.asarray([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    centers = (-0.5, 0.0, 0.5)
    points = {
        "a": np.asarray([0.0, 0.0, 3.0]),
        "b": np.asarray([2.0, 0.0, 3.0]),
        "c": np.asarray([0.0, 0.0, 4.0]),
        "d": np.asarray([0.0, 1.0, 4.0]),
    }
    poses = []
    observations = []
    for index, center_x in enumerate(centers):
        frame_id = f"frame_{index}"
        poses.append(
            {
                "frame_id": frame_id,
                "transform_world_from_camera": {
                    "translation": [center_x, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "scale": [1.0, 1.0, 1.0],
                },
            }
        )
        extrinsic = np.column_stack((np.eye(3), np.asarray([-center_x, 0.0, 0.0])))
        projection = intrinsic @ extrinsic
        for point_id, point in points.items():
            projected = projection @ np.append(point, 1.0)
            observations.append(
                {
                    "frame_id": frame_id,
                    "point_id": point_id,
                    "pixel_xy": (projected[:2] / projected[2]).tolist(),
                    "role": "fitting" if index < 2 else "heldout",
                }
            )
    landmarks = [
        {
            "landmark_id": "fit_width",
            "point_a_id": "a",
            "point_b_id": "b",
            "known_distance_m": 1.0,
            "role": "fitting",
        }
    ]
    if independent_holdout:
        landmarks.append(
            {
                "landmark_id": "heldout_height",
                "point_a_id": "c",
                "point_b_id": "d",
                "known_distance_m": 0.5,
                "role": "heldout",
            }
        )
    manifest = {"known_distance": {"landmarks": landmarks, "observations": observations}}
    camera = {
        "intrinsics": {
            "fx": 500.0,
            "fy": 500.0,
            "cx": 320.0,
            "cy": 240.0,
        },
        "poses": poses,
    }
    return manifest, camera


def test_single_known_distance_uses_image_holdout_without_fake_length_holdout() -> None:
    manifest, camera = _known_distance_fixture(independent_holdout=False)
    result = _known_distance_solution(
        manifest,
        camera,
        {"frame_0", "frame_1"},
        {"frame_2"},
    )
    assert result["fitting_scales"] == pytest.approx([0.5])
    assert result["fitting_error"] == pytest.approx(0.0, abs=1e-10)
    assert result["heldout_error"] is None
    assert not result["independent_length_holdout"]
    assert result["heldout_reprojection_error_px"] == pytest.approx(0.0, abs=1e-8)


def test_dual_known_distance_has_independent_frozen_scale_holdout() -> None:
    manifest, camera = _known_distance_fixture(independent_holdout=True)
    result = _known_distance_solution(
        manifest,
        camera,
        {"frame_0", "frame_1"},
        {"frame_2"},
    )
    assert result["fitting_scales"] == pytest.approx([0.5])
    assert result["heldout_error"] == pytest.approx(0.0, abs=1e-10)
    assert result["independent_length_holdout"]
    assert result["heldout_residuals"] == pytest.approx({"heldout_height": 0.0}, abs=1e-10)


def test_scale_uncertainty_includes_jackknife_and_physical_measurement() -> None:
    manifest, camera = _known_distance_fixture(independent_holdout=True)
    manifest["known_distance"]["landmarks"][0]["measurement_uncertainty_m"] = 0.01
    result = _known_distance_solution(
        manifest,
        camera,
        {"frame_0", "frame_1"},
        {"frame_2"},
    )
    assert result["scale_annotation_jackknife_p90"] == 0.0
    assert result["scale_measurement_uncertainty"] == pytest.approx(0.005)
    assert result["scale_uncertainty"] == pytest.approx(0.005)
    assert result["scale_relative_uncertainty"] == pytest.approx(0.01)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_landmark_world_derivation_is_bound_to_current_solve_and_sources(
    tmp_path: Path,
) -> None:
    dependencies = {}
    for name in ("landmarks.yaml", "triangulated.json", "measured_motion.json"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        dependencies[name] = _sha256(path)
    derivation = {
        "camera_reconstruction_sha256": "1" * 64,
        "landmark_manifest_sha256": dependencies["landmarks.yaml"],
        "triangulated_landmarks_sha256": dependencies["triangulated.json"],
        "measured_motion_sha256": dependencies["measured_motion.json"],
        "source_scene_ir_sha256": "2" * 64,
        "point_coordinates_colmap": {
            "O": [0.0, 0.0, 0.0],
            "U": [0.0, 0.0, 2.0],
            "R": [1.0, 0.0, 0.0],
        },
        "up_vector_colmap": [0.0, 0.0, 1.0],
        "forward_vector_colmap": [0.0, 1.0, 0.0],
        "origin_colmap": [0.0, 0.0, 0.0],
        "scale_m_per_colmap": 0.1,
        "scale_annotation_jackknife_p90_m_per_colmap": 0.001,
        "scale_measurement_uncertainty_m_per_colmap": 0.002,
        "scale_uncertainty_m_per_colmap": 0.003,
        "scale_relative_uncertainty": 0.03,
    }
    derivation_path = tmp_path / "derivation.json"
    derivation_path.write_text(json.dumps(derivation), encoding="utf-8")
    manifest = {
        "camera_reconstruction_sha256": "1" * 64,
        "source_scene_ir_sha256": "2" * 64,
        "landmark_world_derivation_path": "derivation.json",
        "landmark_world_derivation_sha256": _sha256(derivation_path),
        "evidence": [
            {
                "source_files": [
                    {"relative_path": name, "sha256": digest}
                    for name, digest in dependencies.items()
                ]
            }
        ],
        "gravity": [
            {
                "source": "user_up_landmarks",
                "up_vector_colmap": [0.0, 0.0, 1.0],
            }
        ],
        "forward": {"forward_vector_colmap": [0.0, 1.0, 0.0]},
        "origin": {"origin_colmap": [0.0, 0.0, 0.0]},
    }
    known_distance = {
        "landmarks": [
            {"point_id": point_id, "point_colmap": coordinates}
            for point_id, coordinates in derivation["point_coordinates_colmap"].items()
        ],
        "robust_scale": 0.1,
        "scale_annotation_jackknife_p90": 0.001,
        "scale_measurement_uncertainty": 0.002,
        "scale_uncertainty": 0.003,
        "scale_relative_uncertainty": 0.03,
    }
    assert _landmark_world_derivation(manifest, tmp_path, known_distance) == derivation

    tampered_coordinates = json.loads(json.dumps(known_distance))
    tampered_coordinates["landmarks"][0]["point_colmap"][0] = 0.1
    with pytest.raises(ValueError, match="current solve"):
        _landmark_world_derivation(manifest, tmp_path, tampered_coordinates)

    tampered_manifest = json.loads(json.dumps(manifest))
    tampered_manifest["evidence"][0]["source_files"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="exact evidence sources"):
        _landmark_world_derivation(tampered_manifest, tmp_path, known_distance)


def test_apriltag_world_contract_derives_axes_and_origin_from_fitted_pose() -> None:
    manifest = {
        "apriltag": {
            "official_commit": "0e16a12dd380fd607e4afd54712ee9b1ffb9ec8f",
            "tag_family": "tagStandard41h12",
            "tag_id": 0,
            "world_contract": {
                "tag_origin_policy": "tag_center",
                "canonical_up_from_tag_axis": "+Z_tag",
                "canonical_forward_from_tag_axis": "-Y_tag",
                "mounting_description": "surveyed vertical board",
                "mounting_uncertainty_degrees": 0.5,
                "origin_uncertainty_m": 0.002,
            },
        }
    }
    detections = [
        {"frame_id": "fit", "split": "fitting"},
        {"frame_id": "hold", "split": "heldout"},
    ]
    derivation = _tag_world_derivation(
        manifest,
        detections,
        2.0,
        np.eye(3),
        np.asarray([1.0, -2.0, 0.5]),
        {"translation_error": 0.004, "rotation_error": 0.6},
    )
    assert derivation is not None
    assert derivation["derived_up_vector_colmap"] == pytest.approx((0.0, 0.0, 1.0))
    assert derivation["derived_forward_vector_colmap"] == pytest.approx((0.0, -1.0, 0.0))
    assert derivation["derived_origin_colmap"] == pytest.approx((-0.5, 1.0, -0.25))
