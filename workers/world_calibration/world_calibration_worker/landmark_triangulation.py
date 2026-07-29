from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def quaternion_rotation(value: list[float]) -> np.ndarray:
    x, y, z, w = value
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def projection_by_frame(camera: dict[str, Any]) -> dict[str, np.ndarray]:
    intrinsics = camera["intrinsics"]
    intrinsic = np.asarray(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    result: dict[str, np.ndarray] = {}
    for pose in camera["poses"]:
        transform = pose["transform_world_from_camera"]
        rotation_world_camera = quaternion_rotation(transform["rotation_xyzw"])
        translation_world_camera = np.asarray(transform["translation"], dtype=np.float64)
        rotation_camera_world = rotation_world_camera.T
        translation_camera_world = -rotation_camera_world @ translation_world_camera
        extrinsic = np.column_stack((rotation_camera_world, translation_camera_world))
        result[str(pose["frame_id"])] = intrinsic @ extrinsic
    return result


def undistort_observations(
    observations: list[dict[str, Any]],
    camera: dict[str, Any],
) -> list[dict[str, Any]]:
    intrinsics = camera["intrinsics"]
    distortion = [float(value) for value in intrinsics.get("distortion", [])]
    if not distortion:
        return [dict(item) for item in observations]
    model = str(camera["model"])
    if model != "OPENCV" or len(distortion) != 4:
        raise ValueError(
            "known-distance triangulation supports distortion only for the "
            "four-parameter COLMAP OPENCV model"
        )
    intrinsic = np.asarray(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pixels = np.asarray(
        [item["pixel_xy"] for item in observations],
        dtype=np.float64,
    ).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(
        pixels,
        intrinsic,
        np.asarray(distortion, dtype=np.float64),
        P=intrinsic,
    ).reshape(-1, 2)
    result = []
    for item, pixel in zip(observations, undistorted, strict=True):
        normalized = dict(item)
        normalized["pixel_xy"] = [float(pixel[0]), float(pixel[1])]
        result.append(normalized)
    return result


def camera_centers(camera: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        str(pose["frame_id"]): np.asarray(
            pose["transform_world_from_camera"]["translation"], dtype=np.float64
        )
        for pose in camera["poses"]
    }


def triangulate(
    observations: list[dict[str, Any]], projections: dict[str, np.ndarray]
) -> np.ndarray:
    rows = []
    for observation in observations:
        projection = projections[str(observation["frame_id"])]
        x, y = observation["pixel_xy"]
        rows.extend((x * projection[2] - projection[0], y * projection[2] - projection[1]))
    _, _, right_t = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    homogeneous = right_t[-1]
    if abs(float(homogeneous[3])) <= 1e-12:
        raise ValueError("landmark triangulation is at infinity")
    point = homogeneous[:3] / homogeneous[3]
    if not np.all(np.isfinite(point)):
        raise ValueError("landmark triangulation is non-finite")
    return point


def reprojection_errors(
    point: np.ndarray,
    observations: list[dict[str, Any]],
    projections: dict[str, np.ndarray],
) -> list[float]:
    homogeneous = np.append(point, 1.0)
    result = []
    for observation in observations:
        projected = projections[str(observation["frame_id"])] @ homogeneous
        if projected[2] <= 0:
            result.append(float("inf"))
            continue
        pixel = projected[:2] / projected[2]
        result.append(
            float(np.linalg.norm(pixel - np.asarray(observation["pixel_xy"], dtype=np.float64)))
        )
    return result


__all__ = [
    "camera_centers",
    "projection_by_frame",
    "quaternion_rotation",
    "reprojection_errors",
    "triangulate",
    "undistort_observations",
]
