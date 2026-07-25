from __future__ import annotations

import math
from typing import cast

from recon2sim.ir import Transform

Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def normalize_quaternion_xyzw(
    value: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(component * component for component in value))
    if norm <= 1e-15:
        raise ValueError("quaternion norm is zero")
    normalized = tuple(component / norm for component in value)
    if normalized[3] < 0:
        normalized = tuple(-component for component in normalized)
    return cast(tuple[float, float, float, float], normalized)


def qvec_to_rotation_matrix(qvec_wxyz: tuple[float, float, float, float]) -> Matrix3:
    qw, qx, qy, qz = qvec_wxyz
    qx, qy, qz, qw = normalize_quaternion_xyzw((qx, qy, qz, qw))
    return (
        (
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw),
        ),
        (
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx * qx + qz * qz),
            2 * (qy * qz - qx * qw),
        ),
        (
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx * qx + qy * qy),
        ),
    )


def quaternion_xyzw_to_rotation_matrix(
    quaternion: tuple[float, float, float, float],
) -> Matrix3:
    qx, qy, qz, qw = normalize_quaternion_xyzw(quaternion)
    return qvec_to_rotation_matrix((qw, qx, qy, qz))


def rotation_matrix_to_quaternion_xyzw(matrix: Matrix3) -> tuple[float, float, float, float]:
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        qw = 0.25 * scale
        qx = (m21 - m12) / scale
        qy = (m02 - m20) / scale
        qz = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / scale
        qx = 0.25 * scale
        qy = (m01 + m10) / scale
        qz = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / scale
        qx = (m01 + m10) / scale
        qy = 0.25 * scale
        qz = (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / scale
        qx = (m02 + m20) / scale
        qy = (m12 + m21) / scale
        qz = 0.25 * scale
    return normalize_quaternion_xyzw((qx, qy, qz, qw))


def _transpose(matrix: Matrix3) -> Matrix3:
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def _multiply_vector(
    matrix: Matrix3,
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        sum(matrix[0][column] * vector[column] for column in range(3)),
        sum(matrix[1][column] * vector[column] for column in range(3)),
        sum(matrix[2][column] * vector[column] for column in range(3)),
    )


def colmap_pose_to_world_from_camera(
    qvec_wxyz: tuple[float, float, float, float],
    tvec: tuple[float, float, float],
) -> Transform:
    rotation_world_to_camera = qvec_to_rotation_matrix(qvec_wxyz)
    rotation_camera_to_world = _transpose(rotation_world_to_camera)
    rotated_translation = _multiply_vector(rotation_camera_to_world, tvec)
    camera_center = (
        -rotated_translation[0],
        -rotated_translation[1],
        -rotated_translation[2],
    )
    return Transform(
        translation_m=camera_center,
        rotation_xyzw=rotation_matrix_to_quaternion_xyzw(rotation_camera_to_world),
    )


__all__ = [
    "Matrix3",
    "colmap_pose_to_world_from_camera",
    "normalize_quaternion_xyzw",
    "quaternion_xyzw_to_rotation_matrix",
    "qvec_to_rotation_matrix",
    "rotation_matrix_to_quaternion_xyzw",
]
