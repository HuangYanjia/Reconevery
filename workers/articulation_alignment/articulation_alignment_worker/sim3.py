from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

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


def _right_handed_axis_rotations() -> tuple[np.ndarray, ...]:
    rotations: list[np.ndarray] = []
    for permutation in permutations(range(3)):
        basis = np.eye(3)[:, permutation]
        for signs in product((-1.0, 1.0), repeat=3):
            rotation = basis @ np.diag(signs)
            if np.linalg.det(rotation) > 0.0:
                rotations.append(rotation)
    rotations.sort(key=lambda value: tuple(float(item) for item in value.reshape(-1)))
    return tuple(rotations)


def _initial_matrix(
    source: np.ndarray,
    target: np.ndarray,
    rotation: np.ndarray,
    scale: float,
) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = np.median(target, axis=0) - scale * (rotation @ np.median(source, axis=0))
    return matrix


def _refine_icp(
    source: np.ndarray,
    target: np.ndarray,
    tree: cKDTree,
    initial: np.ndarray,
    *,
    with_scale: bool,
    iterations: int,
    keep_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = initial.copy()
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
    return matrix, np.asarray(distances, dtype=np.float64)


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
    source_extent = np.percentile(source, 90, axis=0) - np.percentile(source, 10, axis=0)
    target_extent = np.percentile(target, 90, axis=0) - np.percentile(target, 10, axis=0)
    valid = source_extent > np.finfo(np.float64).eps
    initial_scale = (
        float(np.median(target_extent[valid] / source_extent[valid]))
        if with_scale and np.any(valid)
        else 1.0
    )
    tree = cKDTree(target)
    keep_count = max(3, int(min(len(source), len(target)) * trim_fraction))
    scored_initializations: list[tuple[float, np.ndarray]] = []
    for rotation in _right_handed_axis_rotations():
        initial = _initial_matrix(source, target, rotation, initial_scale)
        distances, _ = tree.query(apply_transform(source, initial), k=1)
        score = float(np.median(np.partition(distances, keep_count - 1)[:keep_count]))
        scored_initializations.append((score, initial))
    scored_initializations.sort(
        key=lambda item: (
            item[0],
            tuple(float(value) for value in item[1][:3, :3].reshape(-1)),
        )
    )

    refined: list[tuple[float, np.ndarray, np.ndarray]] = []
    for _, initial in scored_initializations[:4]:
        candidate_matrix, candidate_distances = _refine_icp(
            source,
            target,
            tree,
            initial,
            with_scale=with_scale,
            iterations=iterations,
            keep_count=keep_count,
        )
        robust_score = float(
            np.median(np.partition(candidate_distances, keep_count - 1)[:keep_count])
        )
        refined.append((robust_score, candidate_matrix, candidate_distances))
    refined.sort(
        key=lambda item: (
            item[0],
            tuple(float(value) for value in item[1][:3, :3].reshape(-1)),
        )
    )
    _, matrix, distances = refined[0]
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


def robust_translation_registration(
    source: np.ndarray,
    target: np.ndarray,
    *,
    iterations: int = 30,
    trim_fraction: float = 0.5,
) -> Sim3Fit:
    if min(len(source), len(target)) < 3:
        raise ValueError("translation registration requires at least three points per state")
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("translation registration points must be finite")
    tree = cKDTree(target)
    keep_count = max(3, int(min(len(source), len(target)) * trim_fraction))
    initializations = (
        np.zeros(3, dtype=np.float64),
        target.mean(axis=0) - source.mean(axis=0),
        np.median(target, axis=0) - np.median(source, axis=0),
    )
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for initial in initializations:
        translation = initial.copy()
        for _ in range(iterations):
            transformed = source + translation
            distances, indices = tree.query(transformed, k=1)
            keep = np.argsort(distances, kind="stable")[:keep_count]
            delta = np.median(
                target[indices[keep]] - transformed[keep],
                axis=0,
            )
            translation += delta
            if np.linalg.norm(delta) < 1e-10:
                break
        distances, _ = tree.query(source + translation, k=1)
        score = float(np.median(np.partition(distances, keep_count - 1)[:keep_count]))
        candidates.append((score, translation, np.asarray(distances, dtype=np.float64)))
    candidates.sort(key=lambda item: (item[0], tuple(float(value) for value in item[1])))
    _, translation, distances = candidates[0]
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    return Sim3Fit(
        matrix=matrix,
        inverse=np.linalg.inv(matrix),
        scale=1.0,
        residuals=distances,
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
