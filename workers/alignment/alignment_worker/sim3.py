from __future__ import annotations

import math
from typing import Any


def apply_transform(points: Any, matrix: Any) -> Any:
    import numpy as np

    values = np.asarray(points, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    return values @ transform[:3, :3].T + transform[:3, 3]


def umeyama_similarity(source: Any, target: Any, weights: Any | None = None) -> Any:
    import numpy as np

    source_values = np.asarray(source, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if len(source_values) < 3 or source_values.shape != target_values.shape:
        raise ValueError("Sim(3) estimation requires at least three paired 3D points")
    if weights is None:
        normalized_weights = np.full(len(source_values), 1.0 / len(source_values))
    else:
        raw_weights = np.asarray(weights, dtype=np.float64)
        normalized_weights = raw_weights / max(raw_weights.sum(), 1e-12)
    source_center = np.sum(source_values * normalized_weights[:, None], axis=0)
    target_center = np.sum(target_values * normalized_weights[:, None], axis=0)
    source_centered = source_values - source_center
    target_centered = target_values - target_center
    covariance = (target_centered * normalized_weights[:, None]).T @ source_centered
    left, singular, right_transpose = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(left @ right_transpose) < 0:
        sign[-1, -1] = -1
    rotation = left @ sign @ right_transpose
    variance = float(np.sum(normalized_weights * np.sum(source_centered**2, axis=1)))
    scale = float(np.trace(np.diag(singular) @ sign) / max(variance, 1e-15))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("estimated Sim(3) scale is not positive and finite")
    translation = target_center - scale * (rotation @ source_center)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return matrix


def decompose_similarity(matrix: Any, scene_diagonal: float) -> dict[str, object]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    value = np.asarray(matrix, dtype=np.float64)
    determinant = float(np.linalg.det(value[:3, :3]))
    scale = float(abs(determinant) ** (1.0 / 3.0))
    rotation = value[:3, :3] / max(scale, 1e-15)
    rotation_object = Rotation.from_matrix(rotation)
    axis_angle = rotation_object.as_rotvec()
    degrees = float(np.linalg.norm(axis_angle) * 180.0 / math.pi)
    inverse = np.linalg.inv(value)
    roundtrip = float(np.linalg.norm(value @ inverse - np.eye(4), ord="fro"))
    translation = value[:3, 3]
    return {
        "matrix_original_mesh_to_aligned_colmap": value.tolist(),
        "inverse_matrix": inverse.tolist(),
        "scale": scale,
        "rotation_matrix": rotation.tolist(),
        "rotation_axis_angle": axis_angle.tolist(),
        "rotation_degrees": degrees,
        "translation": translation.tolist(),
        "translation_scene_diagonal_ratio": float(
            np.linalg.norm(translation) / max(scene_diagonal, 1e-12)
        ),
        "determinant": determinant,
        "roundtrip_error": roundtrip,
    }


def initialization_matrices(mesh_points: Any, sparse_points: Any) -> list[dict[str, object]]:
    import numpy as np

    mesh = np.asarray(mesh_points, dtype=np.float64)
    sparse = np.asarray(sparse_points, dtype=np.float64)
    identity = np.eye(4, dtype=np.float64)
    mesh_center = np.median(mesh, axis=0)
    sparse_center = np.median(sparse, axis=0)
    mesh_extent = np.percentile(mesh, 95, axis=0) - np.percentile(mesh, 5, axis=0)
    sparse_extent = np.percentile(sparse, 95, axis=0) - np.percentile(sparse, 5, axis=0)
    valid = mesh_extent > 1e-12
    scale = float(np.median(sparse_extent[valid] / mesh_extent[valid]))
    scale = max(scale, 1e-8)
    centroid = identity.copy()
    centroid[:3, 3] = sparse_center - mesh_center
    extent = identity.copy()
    extent[:3, :3] *= scale
    extent[:3, 3] = sparse_center - scale * mesh_center
    source_covariance = np.cov((mesh - mesh_center).T)
    target_covariance = np.cov((sparse - sparse_center).T)
    _, source_basis = np.linalg.eigh(source_covariance)
    _, target_basis = np.linalg.eigh(target_covariance)
    rotation = target_basis @ source_basis.T
    if np.linalg.det(rotation) < 0:
        target_basis[:, 0] *= -1
        rotation = target_basis @ source_basis.T
    pca = identity.copy()
    pca[:3, :3] = scale * rotation
    pca[:3, 3] = sparse_center - scale * (rotation @ mesh_center)
    return [
        {"id": "identity", "strategy": "identity", "matrix": identity},
        {"id": "centroid", "strategy": "centroid_alignment", "matrix": centroid},
        {"id": "extent", "strategy": "robust_extent_scale", "matrix": extent},
        {"id": "pca_0", "strategy": "pca_axis_hypothesis", "matrix": pca},
    ]
