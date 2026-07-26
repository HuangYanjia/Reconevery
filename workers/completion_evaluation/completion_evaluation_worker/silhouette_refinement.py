from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class FittingFrame:
    camera_from_world: np.ndarray
    intrinsics: tuple[float, float, float, float]
    mask: np.ndarray
    scene_depth: np.ndarray


@dataclass(frozen=True)
class FittingRefinement:
    matrix: np.ndarray
    objective_before: float
    objective_after: float
    refined: bool
    evaluations: int


def _matrix_from_parameters(parameters: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    scale = math.exp(float(parameters[0]))
    rotation = Rotation.from_rotvec(parameters[1:4]).as_matrix()
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = parameters[4:7]
    return matrix


def _parameters_from_matrix(matrix: np.ndarray) -> np.ndarray:
    linear = matrix[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("fitting refinement requires a proper positive-scale Sim(3)")
    rotation = linear / scale
    return np.concatenate(
        [
            np.asarray([math.log(scale)]),
            Rotation.from_matrix(rotation).as_rotvec(),
            matrix[:3, 3],
        ]
    )


def _bounded_points(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    return points[np.linspace(0, len(points) - 1, maximum, dtype=np.int64)]


def _fitting_objective(
    parameters: np.ndarray,
    points: np.ndarray,
    frames: list[FittingFrame],
    initial_parameters: np.ndarray,
) -> float:
    matrix = _matrix_from_parameters(parameters)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    world = (matrix @ homogeneous.T).T
    losses = []
    for frame in frames:
        camera = (frame.camera_from_world @ world.T).T
        z = camera[:, 2]
        valid = np.isfinite(camera).all(axis=1) & (z > 1e-8)
        fx, fy, cx, cy = frame.intrinsics
        columns = np.full(len(points), -1, dtype=np.int64)
        rows = np.full(len(points), -1, dtype=np.int64)
        columns[valid] = np.rint(fx * camera[valid, 0] / z[valid] + cx).astype(np.int64)
        rows[valid] = np.rint(fy * camera[valid, 1] / z[valid] + cy).astype(np.int64)
        height, width = frame.mask.shape
        valid &= (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
        if np.count_nonzero(valid) < 16:
            losses.append(2.0)
            continue
        sampled_mask = frame.mask[rows[valid], columns[valid]]
        sampled_depth = frame.scene_depth[rows[valid], columns[valid]]
        projected_depth = z[valid]
        depth_valid = np.isfinite(sampled_depth) & (sampled_depth > 0)
        inside = sampled_mask
        supported = inside & depth_valid
        if np.any(supported):
            relative_depth = np.abs(
                projected_depth[supported] - sampled_depth[supported]
            ) / np.maximum(np.abs(sampled_depth[supported]), 1e-8)
            depth_loss = float(np.median(np.minimum(relative_depth, 1.0)))
        else:
            depth_loss = 1.0
        in_mask_fraction = float(np.mean(inside))
        front_violation = ~inside & depth_valid & (projected_depth < sampled_depth * (1.0 - 0.03))
        front_fraction = float(np.mean(front_violation))
        losses.append(depth_loss + 0.4 * (1.0 - in_mask_fraction) + 0.6 * front_fraction)
    regularization = (
        0.01 * float(np.square(parameters[0] - initial_parameters[0]))
        + 0.001 * float(np.square(parameters[1:4] - initial_parameters[1:4]).sum())
        + 0.001 * float(np.square(parameters[4:7] - initial_parameters[4:7]).sum())
    )
    return float(np.mean(losses) + regularization) if losses else 2.0 + regularization


def refine_sim3_on_fitting_views(
    matrix: np.ndarray,
    candidate_points: np.ndarray,
    measured_points: np.ndarray,
    frames: list[FittingFrame],
    *,
    maximum_iterations: int,
    maximum_points: int,
    maximum_rotation_degrees: float,
    maximum_scale_ratio: float,
    translation_extent_ratio: float,
    minimum_scale: float,
    maximum_scale: float,
) -> FittingRefinement:
    initial = _parameters_from_matrix(matrix)
    points = _bounded_points(candidate_points, maximum_points)
    before = _fitting_objective(initial, points, frames, initial)
    if maximum_iterations == 0 or not frames:
        return FittingRefinement(matrix, before, before, False, 1)
    rotation_delta = math.radians(maximum_rotation_degrees)
    measured_extent = np.percentile(measured_points, 95, axis=0) - np.percentile(
        measured_points, 5, axis=0
    )
    translation_delta = max(float(np.linalg.norm(measured_extent)), 1e-6) * (
        translation_extent_ratio
    )
    scale_lower = max(
        math.log(minimum_scale),
        initial[0] - math.log(maximum_scale_ratio),
    )
    scale_upper = min(
        math.log(maximum_scale),
        initial[0] + math.log(maximum_scale_ratio),
    )
    bounds = [
        (scale_lower, scale_upper),
        *[
            (initial[index] - rotation_delta, initial[index] + rotation_delta)
            for index in range(1, 4)
        ],
        *[
            (initial[index] - translation_delta, initial[index] + translation_delta)
            for index in range(4, 7)
        ],
    ]
    result = minimize(
        _fitting_objective,
        initial,
        args=(points, frames, initial),
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": maximum_iterations,
            "xtol": 1e-5,
            "ftol": 1e-5,
        },
    )
    after = float(result.fun)
    if not np.isfinite(after) or after >= before - 1e-8:
        return FittingRefinement(matrix, before, before, False, int(result.nfev) + 1)
    return FittingRefinement(
        _matrix_from_parameters(np.asarray(result.x)),
        before,
        after,
        True,
        int(result.nfev) + 1,
    )
