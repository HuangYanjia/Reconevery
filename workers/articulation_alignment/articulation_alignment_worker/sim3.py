from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Sim3Fit:
    matrix: np.ndarray
    inverse: np.ndarray
    scale: float
    residuals: np.ndarray
    correspondence_count: int


def apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def umeyama(source: np.ndarray, target: np.ndarray, *, with_scale: bool) -> np.ndarray:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Sim(3) correspondences must be matching Nx3 arrays")
    if source.shape[0] < 3:
        raise ValueError("Sim(3) requires at least three correspondences")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    covariance = centered_target.T @ centered_source / source.shape[0]
    left, singular, right = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(left @ right) < 0:
        sign[-1] = -1
    rotation = left @ np.diag(sign) @ right
    scale = 1.0
    if with_scale:
        variance = float(np.mean(np.sum(centered_source**2, axis=1)))
        if variance <= np.finfo(np.float64).eps:
            raise ValueError("Sim(3) source variance collapsed")
        scale = float(np.sum(singular * sign) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Sim(3) produced a non-positive scale")
    translation = target_mean - scale * (rotation @ source_mean)
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return matrix


def robust_icp_sim3(
    source: np.ndarray,
    target: np.ndarray,
    *,
    with_scale: bool = True,
    iterations: int = 20,
    trim_fraction: float = 0.8,
) -> Sim3Fit:
    if min(len(source), len(target)) < 3:
        raise ValueError("alignment requires at least three points per state")
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("alignment points must be finite")
    matrix = np.eye(4)
    source_extent = np.percentile(source, 90, axis=0) - np.percentile(source, 10, axis=0)
    target_extent = np.percentile(target, 90, axis=0) - np.percentile(target, 10, axis=0)
    valid = source_extent > np.finfo(np.float64).eps
    initial_scale = (
        float(np.median(target_extent[valid] / source_extent[valid]))
        if with_scale and np.any(valid)
        else 1.0
    )
    matrix[:3, :3] *= initial_scale
    matrix[:3, 3] = np.median(target, axis=0) - initial_scale * np.median(source, axis=0)
    tree = cKDTree(target)
    keep_count = max(3, int(min(len(source), len(target)) * trim_fraction))
    for _ in range(iterations):
        transformed = apply_transform(source, matrix)
        distances, indices = tree.query(transformed, k=1)
        keep = np.argsort(distances, kind="stable")[:keep_count]
        delta = umeyama(
            transformed[keep],
            target[indices[keep]],
            with_scale=with_scale,
        )
        updated = delta @ matrix
        if np.max(np.abs(updated - matrix)) < 1e-10:
            matrix = updated
            break
        matrix = updated
    transformed = apply_transform(source, matrix)
    distances, _ = tree.query(transformed, k=1)
    linear = matrix[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    rotation = linear / scale
    if np.linalg.det(rotation) < 0.999:
        raise ValueError("alignment produced an improper rotation")
    return Sim3Fit(
        matrix=matrix,
        inverse=np.linalg.inv(matrix),
        scale=scale,
        residuals=np.asarray(distances, dtype=np.float64),
        correspondence_count=keep_count,
    )


def rotation_axis_angle(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    rotation = matrix[:3, :3]
    scale = float(np.cbrt(np.linalg.det(rotation)))
    rotation = rotation / scale
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0]), 0.0
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    axis /= np.linalg.norm(axis)
    return axis, angle
