from __future__ import annotations

import numpy as np


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Sim(3) correspondences must be matching Nx3 arrays")
    if len(source) < 3:
        raise ValueError("Sim(3) requires at least three correspondences")
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    covariance = centered_target.T @ centered_source / len(source)
    left, singular_values, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_t) < 0:
        correction[2, 2] = -1
    rotation = left @ correction @ right_t
    variance = float(np.mean(np.sum(centered_source * centered_source, axis=1)))
    if variance <= np.finfo(np.float64).eps:
        raise ValueError("degenerate Sim(3) source correspondences")
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Sim(3) produced non-positive scale")
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def matrix(scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = scale * rotation
    value[:3, 3] = translation
    return value


def transform_points(points: np.ndarray, value: np.ndarray) -> np.ndarray:
    return points @ value[:3, :3].T + value[:3, 3]


__all__ = ["matrix", "transform_points", "umeyama"]
