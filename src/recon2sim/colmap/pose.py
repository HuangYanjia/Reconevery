from __future__ import annotations

import math

from recon2sim.ir import Transform

Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def normalize_quaternion_wxyz(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("COLMAP quaternion must be finite and non-zero")
    normalized = tuple(value / norm for value in quaternion)
    return (normalized[0], normalized[1], normalized[2], normalized[3])


def quaternion_wxyz_to_matrix(
    quaternion: tuple[float, float, float, float],
) -> Matrix3:
    w, x, y, z = normalize_quaternion_wxyz(quaternion)
    return (
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
        ),
        (
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
        ),
        (
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
    )


def transpose(matrix: Matrix3) -> Matrix3:
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def matrix_vector_product(
    matrix: Matrix3, vector: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        sum(matrix[0][index] * vector[index] for index in range(3)),
        sum(matrix[1][index] * vector[index] for index in range(3)),
        sum(matrix[2][index] * vector[index] for index in range(3)),
    )


def matrix_to_quaternion_xyzw(matrix: Matrix3) -> tuple[float, float, float, float]:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        w = 0.25 * scale
        x = (matrix[2][1] - matrix[1][2]) / scale
        y = (matrix[0][2] - matrix[2][0]) / scale
        z = (matrix[1][0] - matrix[0][1]) / scale
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2
        w = (matrix[2][1] - matrix[1][2]) / scale
        x = 0.25 * scale
        y = (matrix[0][1] + matrix[1][0]) / scale
        z = (matrix[0][2] + matrix[2][0]) / scale
    elif matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2
        w = (matrix[0][2] - matrix[2][0]) / scale
        x = (matrix[0][1] + matrix[1][0]) / scale
        y = 0.25 * scale
        z = (matrix[1][2] + matrix[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2
        w = (matrix[1][0] - matrix[0][1]) / scale
        x = (matrix[0][2] + matrix[2][0]) / scale
        y = (matrix[1][2] + matrix[2][1]) / scale
        z = 0.25 * scale
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("rotation matrix produced an invalid quaternion")
    result = (x / norm, y / norm, z / norm, w / norm)
    if result[3] < 0:
        return (-result[0], -result[1], -result[2], -result[3])
    return result


def colmap_world_to_camera_to_world_from_camera(
    qvec_wxyz: tuple[float, float, float, float],
    tvec: tuple[float, float, float],
) -> Transform:
    """Invert COLMAP's world-to-camera rigid transform.

    COLMAP's reconstruction scale and world orientation remain unchanged here.
    Callers must label monocular scale as ambiguous and the world frame as
    COLMAP-unaligned until an explicit alignment step is performed.
    """
    if not all(math.isfinite(value) for value in tvec):
        raise ValueError("COLMAP translation must contain finite values")
    rotation_world_to_camera = quaternion_wxyz_to_matrix(qvec_wxyz)
    rotation_world_from_camera = transpose(rotation_world_to_camera)
    rotated_translation = matrix_vector_product(rotation_world_from_camera, tvec)
    translation_world_from_camera = (
        -rotated_translation[0],
        -rotated_translation[1],
        -rotated_translation[2],
    )
    return Transform(
        translation_m=translation_world_from_camera,
        rotation_xyzw=matrix_to_quaternion_xyzw(rotation_world_from_camera),
    )
