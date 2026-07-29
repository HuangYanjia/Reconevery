from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def load_points(path: str) -> np.ndarray:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    points = np.asarray(loaded.vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"invalid point geometry: {path}")
    return points


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _umeyama(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    left, singular, right = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(left @ right) < 0:
        sign[-1] = -1
    rotation = left @ np.diag(sign) @ right
    variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    scale = float(np.sum(singular * sign) / max(variance, 1e-12))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("candidate registration produced invalid scale")
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = target_mean - scale * rotation @ source_mean
    return matrix


def register_sim3(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if min(len(source), len(target)) < 3:
        raise ValueError("candidate registration requires at least three points")
    stride_source = max(1, len(source) // 30_000)
    stride_target = max(1, len(target) // 30_000)
    source = source[::stride_source]
    target = target[::stride_target]
    matrix = np.eye(4)
    source_extent = np.ptp(source, axis=0)
    target_extent = np.ptp(target, axis=0)
    valid = source_extent > 1e-12
    scale = float(np.median(target_extent[valid] / source_extent[valid]))
    matrix[:3, :3] *= scale
    matrix[:3, 3] = np.median(target, axis=0) - scale * np.median(source, axis=0)
    tree = cKDTree(target)
    for _ in range(20):
        current = transform(source, matrix)
        distances, indices = tree.query(current, k=1)
        keep = np.argsort(distances, kind="stable")[: max(3, int(0.8 * len(source)))]
        delta = _umeyama(current[keep], target[indices[keep]])
        updated = delta @ matrix
        if np.max(np.abs(updated - matrix)) < 1e-10:
            matrix = updated
            break
        matrix = updated
    residuals, _ = tree.query(transform(source, matrix), k=1)
    return matrix, np.asarray(residuals)


def joint_transform(
    joint_type: str,
    axis: np.ndarray,
    pivot: np.ndarray | None,
    position: float,
) -> np.ndarray:
    result = np.eye(4)
    axis = axis / np.linalg.norm(axis)
    if joint_type == "prismatic":
        result[:3, 3] = axis * position
    elif joint_type in {"revolute", "continuous_candidate"}:
        if pivot is None:
            raise ValueError("revolute joint requires pivot")
        cross = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        rotation = np.eye(3) + np.sin(position) * cross + (1.0 - np.cos(position)) * (cross @ cross)
        result[:3, :3] = rotation
        result[:3, 3] = pivot - rotation @ pivot
    return result
