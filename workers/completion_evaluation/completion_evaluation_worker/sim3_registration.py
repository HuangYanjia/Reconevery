from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Sim3Result:
    matrix: np.ndarray
    inverse: np.ndarray
    scale: float
    median_residual: float
    p90_residual: float
    initialization: str
    symmetry_ambiguous: bool


def unsigned_normal_agreement(
    candidate: np.ndarray,
    candidate_normals: np.ndarray,
    measured: np.ndarray,
    measured_normals: np.ndarray,
    matrix_world_from_candidate: np.ndarray,
    *,
    trimmed_fraction: float = 0.8,
) -> float:
    linear = matrix_world_from_candidate[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    rotation = linear / scale
    transformed = candidate @ linear.T + matrix_world_from_candidate[:3, 3]
    distances, indices = cKDTree(transformed).query(measured, workers=1)
    keep_count = max(3, int(len(measured) * trimmed_fraction))
    keep = np.argsort(distances, kind="stable")[:keep_count]
    transformed_normals = candidate_normals @ rotation.T
    dot = np.sum(transformed_normals[indices[keep]] * measured_normals[keep], axis=1)
    return float(np.mean(np.abs(np.clip(dot, -1, 1))))


def measured_surface_residuals(
    candidate: np.ndarray,
    measured: np.ndarray,
    matrix_world_from_candidate: np.ndarray,
) -> tuple[float, float]:
    transformed = (
        candidate @ matrix_world_from_candidate[:3, :3].T + matrix_world_from_candidate[:3, 3]
    )
    distances, _ = cKDTree(transformed).query(measured, workers=1)
    return float(np.median(distances)), float(np.percentile(distances, 90))


def _umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = target_zero.T @ source_zero / len(source)
    u, singular, vh = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vh) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vh
    variance = np.square(source_zero).sum() / len(source)
    scale = float((singular * sign).sum() / max(variance, 1e-12))
    translation = target_center - scale * rotation @ source_center
    return scale, rotation, translation


def _basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    covariance = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    values = eigenvalues[order]
    basis = eigenvectors[:, order]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return values, basis


def _right_handed_axis_hypotheses() -> list[np.ndarray]:
    hypotheses = []
    for permutation in itertools.permutations(range(3)):
        permutation_matrix = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = permutation_matrix @ np.diag(signs)
            if np.linalg.det(matrix) > 0:
                hypotheses.append(matrix)
    return hypotheses


def _initial_transform(
    candidate: np.ndarray,
    measured: np.ndarray,
    rotation: np.ndarray,
) -> tuple[float, np.ndarray]:
    candidate_extent = np.percentile(candidate, 95, axis=0) - np.percentile(candidate, 5, axis=0)
    measured_extent = np.percentile(measured, 95, axis=0) - np.percentile(measured, 5, axis=0)
    scale = float(
        np.median(measured_extent[measured_extent > 1e-12])
        / max(np.median(candidate_extent[candidate_extent > 1e-12]), 1e-12)
    )
    translation = measured.mean(axis=0) - scale * rotation @ candidate.mean(axis=0)
    return scale, translation


def _residuals(
    candidate: np.ndarray,
    measured: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    transformed = scale * (candidate @ rotation.T) + translation
    distances, _ = cKDTree(transformed).query(measured, workers=1)
    return np.asarray(distances, dtype=np.float64)


def _icp(
    candidate: np.ndarray,
    measured: np.ndarray,
    *,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    iterations: int,
    trimmed_fraction: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    for _ in range(iterations):
        transformed = scale * (candidate @ rotation.T) + translation
        distances, indices = cKDTree(transformed).query(measured, workers=1)
        keep_count = max(3, int(len(measured) * trimmed_fraction))
        keep = np.argsort(distances, kind="stable")[:keep_count]
        next_scale, next_rotation, next_translation = _umeyama(
            candidate[indices[keep]], measured[keep]
        )
        delta = (
            abs(next_scale - scale)
            + float(np.linalg.norm(next_rotation - rotation))
            + float(np.linalg.norm(next_translation - translation))
        )
        scale, rotation, translation = next_scale, next_rotation, next_translation
        if delta < 1e-9:
            break
    return scale, rotation, translation


def _bounded_subset(points: np.ndarray, maximum: int = 30_000) -> np.ndarray:
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[indices]


def register_asymmetric_sim3(
    candidate: np.ndarray,
    measured: np.ndarray,
    *,
    iterations: int = 30,
    trimmed_fraction: float = 0.8,
) -> Sim3Result:
    if candidate.shape[1:] != (3,) or measured.shape[1:] != (3,):
        raise ValueError("candidate and measured point sets must be Nx3")
    if len(candidate) < 3 or len(measured) < 3:
        raise ValueError("candidate registration requires at least three points per set")
    candidate_values, candidate_basis = _basis(candidate)
    _, measured_basis = _basis(measured)
    initializations: list[tuple[str, np.ndarray]] = [("identity", np.eye(3))]
    initializations.extend(
        (
            f"pca_{index:02d}",
            measured_basis @ hypothesis @ candidate_basis.T,
        )
        for index, hypothesis in enumerate(_right_handed_axis_hypotheses())
    )
    candidate_subset = _bounded_subset(candidate)
    measured_subset = _bounded_subset(measured)
    scored = []
    for name, rotation in initializations:
        scale, translation = _initial_transform(candidate_subset, measured_subset, rotation)
        distances = _residuals(
            candidate_subset,
            measured_subset,
            scale,
            rotation,
            translation,
        )
        keep_count = max(3, int(len(distances) * trimmed_fraction))
        score = float(np.mean(np.partition(distances, keep_count - 1)[:keep_count]))
        scored.append((score, name, scale, rotation, translation))
    finalists = sorted(scored, key=lambda item: (item[0], item[1]))[:4]
    refined = []
    warmup_iterations = min(iterations, 12)
    for _, name, scale, rotation, translation in finalists:
        scale, rotation, translation = _icp(
            candidate_subset,
            measured_subset,
            scale=scale,
            rotation=rotation,
            translation=translation,
            iterations=warmup_iterations,
            trimmed_fraction=trimmed_fraction,
        )
        distances = _residuals(
            candidate_subset,
            measured_subset,
            scale,
            rotation,
            translation,
        )
        refined.append((float(np.median(distances)), name, scale, rotation, translation))
    _, initialization, scale, rotation, translation = min(
        refined, key=lambda item: (item[0], item[1])
    )
    scale, rotation, translation = _icp(
        candidate,
        measured,
        scale=scale,
        rotation=rotation,
        translation=translation,
        iterations=max(1, iterations - warmup_iterations),
        trimmed_fraction=trimmed_fraction,
    )
    distances = _residuals(candidate, measured, scale, rotation, translation)
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    normalized_values = candidate_values / max(float(candidate_values.max()), 1e-12)
    symmetry_ambiguous = bool(
        abs(normalized_values[0] - normalized_values[1]) < 0.05
        or abs(normalized_values[1] - normalized_values[2]) < 0.05
    )
    return Sim3Result(
        matrix=matrix,
        inverse=np.linalg.inv(matrix),
        scale=scale,
        median_residual=float(np.median(distances)),
        p90_residual=float(np.percentile(distances, 90)),
        initialization=initialization,
        symmetry_ambiguous=symmetry_ambiguous,
    )
