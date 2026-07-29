from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

Matrix4 = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]
Matrix3 = tuple[float, float, float, float, float, float, float, float, float]
Vector3 = tuple[float, float, float]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def vector_norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(component * component for component in value))


def normalize_vector(value: Sequence[float]) -> Vector3:
    if len(value) != 3:
        raise ValueError("3D vectors require exactly three values")
    norm = vector_norm(value)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return (value[0] / norm, value[1] / norm, value[2] / norm)


def cross(left: Sequence[float], right: Sequence[float]) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def canonical_rotation(up_colmap: Sequence[float], forward_colmap: Sequence[float]) -> Matrix3:
    up = normalize_vector(up_colmap)
    projected = tuple(
        forward_colmap[index] - dot(forward_colmap, up) * up[index] for index in range(3)
    )
    forward = normalize_vector(projected)
    left = normalize_vector(cross(up, forward))
    forward = normalize_vector(cross(left, up))
    # Rows map COLMAP vectors into canonical coordinates.
    return (
        forward[0],
        forward[1],
        forward[2],
        left[0],
        left[1],
        left[2],
        up[0],
        up[1],
        up[2],
    )


def rotation_determinant(rotation: Sequence[float]) -> float:
    if len(rotation) != 9:
        raise ValueError("rotation matrices require nine values")
    return (
        rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7])
        - rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6])
        + rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6])
    )


def orthonormal_error(rotation: Sequence[float]) -> float:
    rows = [rotation[0:3], rotation[3:6], rotation[6:9]]
    errors = [
        abs(dot(rows[index], rows[column]) - (1.0 if index == column else 0.0))
        for index in range(3)
        for column in range(3)
    ]
    return max(errors)


def build_sim3(scale: float, rotation: Sequence[float], translation: Sequence[float]) -> Matrix4:
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("Sim(3) scale must be finite and positive")
    if len(rotation) != 9 or len(translation) != 3:
        raise ValueError("invalid Sim(3) dimensions")
    if abs(rotation_determinant(rotation) - 1.0) > 1e-6:
        raise ValueError("Sim(3) rotation must be proper")
    if orthonormal_error(rotation) > 1e-6:
        raise ValueError("Sim(3) rotation must be orthonormal")
    return (
        scale * rotation[0],
        scale * rotation[1],
        scale * rotation[2],
        translation[0],
        scale * rotation[3],
        scale * rotation[4],
        scale * rotation[5],
        translation[1],
        scale * rotation[6],
        scale * rotation[7],
        scale * rotation[8],
        translation[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )


def invert_sim3(matrix: Sequence[float]) -> Matrix4:
    if len(matrix) != 16:
        raise ValueError("Sim(3) matrices require sixteen values")
    scale = vector_norm((matrix[0], matrix[4], matrix[8]))
    if scale <= 0:
        raise ValueError("Sim(3) scale must be positive")
    rotation = (
        matrix[0] / scale,
        matrix[1] / scale,
        matrix[2] / scale,
        matrix[4] / scale,
        matrix[5] / scale,
        matrix[6] / scale,
        matrix[8] / scale,
        matrix[9] / scale,
        matrix[10] / scale,
    )
    inverse_rotation = (
        rotation[0],
        rotation[3],
        rotation[6],
        rotation[1],
        rotation[4],
        rotation[7],
        rotation[2],
        rotation[5],
        rotation[8],
    )
    inverse_scale = 1.0 / scale
    translation = (matrix[3], matrix[7], matrix[11])
    inverse_translation = tuple(
        -inverse_scale
        * sum(inverse_rotation[row * 3 + column] * translation[column] for column in range(3))
        for row in range(3)
    )
    return build_sim3(inverse_scale, inverse_rotation, inverse_translation)


def multiply_matrix4(left: Sequence[float], right: Sequence[float]) -> Matrix4:
    if len(left) != 16 or len(right) != 16:
        raise ValueError("4x4 matrix multiplication requires sixteen values per matrix")
    return tuple(
        sum(left[row * 4 + inner] * right[inner * 4 + column] for inner in range(4))
        for row in range(4)
        for column in range(4)
    )  # type: ignore[return-value]


def transform_point(matrix: Sequence[float], point: Sequence[float]) -> Vector3:
    if len(matrix) != 16 or len(point) != 3:
        raise ValueError("point transformation requires a 4x4 matrix and 3D point")
    return (
        sum(matrix[column] * point[column] for column in range(3)) + matrix[3],
        sum(matrix[4 + column] * point[column] for column in range(3)) + matrix[7],
        sum(matrix[8 + column] * point[column] for column in range(3)) + matrix[11],
    )


def rotate_vector(rotation: Sequence[float], vector: Sequence[float]) -> Vector3:
    if len(rotation) != 9 or len(vector) != 3:
        raise ValueError("vector rotation requires a 3x3 matrix and 3D vector")
    return normalize_vector(
        tuple(
            sum(rotation[row * 3 + column] * vector[column] for column in range(3))
            for row in range(3)
        )
    )


def maximum_roundtrip_error(matrix: Sequence[float], inverse: Sequence[float]) -> float:
    product = multiply_matrix4(matrix, inverse)
    identity = (
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
    )
    return max(abs(actual - expected) for actual, expected in zip(product, identity, strict=True))


__all__ = [
    "Matrix3",
    "Matrix4",
    "Vector3",
    "build_sim3",
    "canonical_rotation",
    "cross",
    "dot",
    "invert_sim3",
    "maximum_roundtrip_error",
    "multiply_matrix4",
    "normalize_vector",
    "orthonormal_error",
    "rotate_vector",
    "rotation_determinant",
    "sha256_file",
    "stable_digest",
    "transform_point",
    "vector_norm",
]
