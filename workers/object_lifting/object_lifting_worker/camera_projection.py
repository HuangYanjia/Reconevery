from __future__ import annotations

import math
from collections.abc import Sequence


def normalize_quaternion_xyzw(
    value: Sequence[float],
) -> tuple[float, float, float, float]:
    if len(value) != 4:
        raise ValueError("quaternion must contain four xyzw components")
    norm = math.sqrt(sum(component * component for component in value))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("quaternion must be finite and nonzero")
    return tuple(component / norm for component in value)  # type: ignore[return-value]


def quaternion_xyzw_to_matrix(
    value: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    x, y, z, w = normalize_quaternion_xyzw(value)
    return (
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
        ),
        (
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
        ),
        (
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
    )


def transpose3(
    matrix: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(matrix[column][row] for column in range(3)) for row in range(3))


def matvec3(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(
        sum(matrix[row][column] * vector[column] for column in range(3)) for row in range(3)
    )  # type: ignore[return-value]


def camera_from_world(
    translation_world_from_camera: Sequence[float],
    rotation_xyzw_world_from_camera: Sequence[float],
) -> tuple[tuple[tuple[float, float, float], ...], tuple[float, float, float]]:
    """Invert Reconevery's rigid transform_world_from_camera."""
    if len(translation_world_from_camera) != 3:
        raise ValueError("camera translation must contain three components")
    rotation_world_from_camera = quaternion_xyzw_to_matrix(rotation_xyzw_world_from_camera)
    rotation_camera_from_world = transpose3(rotation_world_from_camera)
    inverse_translation = matvec3(
        rotation_camera_from_world,
        tuple(-value for value in translation_world_from_camera),
    )
    return rotation_camera_from_world, inverse_translation


def transform_world_point_to_camera(
    point_world: Sequence[float],
    translation_world_from_camera: Sequence[float],
    rotation_xyzw_world_from_camera: Sequence[float],
) -> tuple[float, float, float]:
    rotation, translation = camera_from_world(
        translation_world_from_camera,
        rotation_xyzw_world_from_camera,
    )
    rotated = matvec3(rotation, point_world)
    return tuple(rotated[index] + translation[index] for index in range(3))  # type: ignore[return-value]


def project_pinhole(
    point_camera: Sequence[float],
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float] | None:
    x, y, z = point_camera
    if z <= 0:
        return None
    return fx * x / z + cx, fy * y / z + cy


def pixel_to_ndc(
    u: float,
    v: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    if width <= 0 or height <= 0:
        raise ValueError("raster dimensions must be positive")
    return (
        2.0 * (u + 0.5) / width - 1.0,
        1.0 - 2.0 * (v + 0.5) / height,
    )


def ndc_to_pixel(
    x_ndc: float,
    y_ndc: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    return (
        ((x_ndc + 1.0) * width / 2.0) - 0.5,
        ((1.0 - y_ndc) * height / 2.0) - 0.5,
    )
