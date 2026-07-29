from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from world_calibration_worker.apriltag_detection import detect_official
from world_calibration_worker.canonical_axes import canonical_rotation
from world_calibration_worker.floor_plane import robust_plane
from world_calibration_worker.gravity import combine_up_vectors
from world_calibration_worker.landmark_triangulation import (
    reprojection_errors,
    triangulate,
)
from world_calibration_worker.sim3 import matrix, transform_points, umeyama


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
