from __future__ import annotations

import math


def quaternion_xyzw_to_rotation(
    quaternion: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    x, y, z, w = quaternion
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        raise ValueError("camera quaternion must be non-zero")
    x, y, z, w = (component / norm for component in quaternion)
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _matvec(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        sum(matrix[0][column] * vector[column] for column in range(3)),
        sum(matrix[1][column] * vector[column] for column in range(3)),
        sum(matrix[2][column] * vector[column] for column in range(3)),
    )


def _transpose(
    matrix: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def backproject_pixel_world(
    *,
    pixel_xy: tuple[float, float],
    depth: float,
    intrinsics: tuple[float, float, float, float],
    translation_world_from_camera: tuple[float, float, float],
    rotation_world_from_camera_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    if not math.isfinite(depth) or depth <= 0:
        raise ValueError("measured camera depth must be finite and positive")
    u, v = pixel_xy
    fx, fy, cx, cy = intrinsics
    if fx <= 0 or fy <= 0:
        raise ValueError("camera focal lengths must be positive")
    camera_point = ((u - cx) * depth / fx, (v - cy) * depth / fy, depth)
    rotation = quaternion_xyzw_to_rotation(rotation_world_from_camera_xyzw)
    rotated = _matvec(rotation, camera_point)
    return (
        rotated[0] + translation_world_from_camera[0],
        rotated[1] + translation_world_from_camera[1],
        rotated[2] + translation_world_from_camera[2],
    )


def project_world_pixel(
    *,
    point_world: tuple[float, float, float],
    intrinsics: tuple[float, float, float, float],
    translation_world_from_camera: tuple[float, float, float],
    rotation_world_from_camera_xyzw: tuple[float, float, float, float],
) -> tuple[tuple[float, float], float]:
    rotation = quaternion_xyzw_to_rotation(rotation_world_from_camera_xyzw)
    relative = (
        point_world[0] - translation_world_from_camera[0],
        point_world[1] - translation_world_from_camera[1],
        point_world[2] - translation_world_from_camera[2],
    )
    x, y, depth = _matvec(_transpose(rotation), relative)
    if not math.isfinite(depth) or depth <= 0:
        raise ValueError("world point lies behind the camera")
    fx, fy, cx, cy = intrinsics
    return ((fx * x / depth + cx, fy * y / depth + cy), depth)


def relative_depth_agrees(
    predicted_depth: float, measured_depth: float, maximum_relative_residual: float
) -> bool:
    if (
        not math.isfinite(predicted_depth)
        or not math.isfinite(measured_depth)
        or predicted_depth <= 0
        or measured_depth <= 0
        or maximum_relative_residual <= 0
    ):
        return False
    return (
        abs(predicted_depth - measured_depth) / max(abs(predicted_depth), 1e-12)
        <= maximum_relative_residual
    )


def observed_triangle_is_local(
    depths: tuple[float, float, float], maximum_relative_discontinuity: float
) -> bool:
    if any(not math.isfinite(value) or value <= 0 for value in depths):
        return False
    minimum = min(depths)
    maximum = max(depths)
    return (maximum - minimum) / minimum <= maximum_relative_discontinuity
