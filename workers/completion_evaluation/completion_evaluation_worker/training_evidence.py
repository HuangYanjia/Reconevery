from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class TrainingView:
    frame_id: str
    depth: np.ndarray
    normal: np.ndarray
    consistency: np.ndarray
    mask: np.ndarray
    core_mask: np.ndarray
    intrinsics: np.ndarray
    world_from_camera: np.ndarray
    frame_score: float


@dataclass
class BackprojectionResult:
    points: np.ndarray
    normals: np.ndarray
    raw_sample_count: int
    boundary_rejected_count: int
    invalid_geometry_rejected_count: int
    sam_score_rejected_count: int
    consistency_rejected_count: int
    depth_discontinuity_rejected_count: int


def build_training_view(
    *,
    frame_id: str,
    depth: np.ndarray,
    normal: np.ndarray,
    consistency: np.ndarray,
    mask: np.ndarray,
    intrinsics: list[float],
    world_from_camera: np.ndarray,
    frame_score: float,
    backprojection_configuration: dict[str, Any],
) -> TrainingView:
    if depth.ndim != 2 or normal.shape != (*depth.shape, 3):
        raise ValueError("dense depth and normal dimensions are inconsistent")
    if consistency.shape != depth.shape or mask.shape != depth.shape:
        raise ValueError("training consistency or mask dimensions differ from depth")
    source_mask = mask.astype(bool)
    core_mask = source_mask.copy()
    if bool(backprojection_configuration["exclude_mask_boundary"]):
        core_mask = (
            cv2.erode(
                source_mask.astype(np.uint8) * 255,
                np.ones((3, 3), dtype=np.uint8),
                iterations=1,
            )
            > 0
        )
    return TrainingView(
        frame_id=frame_id,
        depth=depth,
        normal=normal,
        consistency=consistency,
        mask=source_mask,
        core_mask=core_mask,
        intrinsics=np.asarray(intrinsics, dtype=np.float64),
        world_from_camera=world_from_camera,
        frame_score=frame_score,
    )


def _depth_discontinuity(depth: np.ndarray, threshold: float) -> np.ndarray:
    discontinuity = np.zeros(depth.shape, dtype=bool)
    valid_depth = np.isfinite(depth) & (depth > 0)
    for row_shift, column_shift in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor = np.roll(depth, (row_shift, column_shift), axis=(0, 1))
        pair = valid_depth & np.isfinite(neighbor) & (neighbor > 0)
        relative = np.zeros(depth.shape, dtype=np.float32)
        relative[pair] = np.abs(neighbor[pair] - depth[pair]) / np.maximum(
            np.abs(depth[pair]),
            1e-8,
        )
        discontinuity |= pair & (relative > threshold)
    return discontinuity


def backproject_training_view(
    view: TrainingView,
    configuration: dict[str, Any],
) -> BackprojectionResult:
    raw = view.mask
    core = view.core_mask
    geometry = (
        np.isfinite(view.depth)
        & (view.depth > 0)
        & np.isfinite(view.normal).all(axis=2)
        & (np.linalg.norm(view.normal, axis=2) > 1e-8)
    )
    sam_score_valid = view.frame_score >= float(configuration["minimum_sam_frame_score"])
    consistent = view.consistency >= int(configuration["minimum_consistent_source_views"])
    discontinuity = _depth_discontinuity(
        view.depth,
        float(configuration["maximum_relative_depth_discontinuity"]),
    )
    valid = core & geometry
    valid &= sam_score_valid
    valid &= consistent
    valid &= ~discontinuity
    stride = int(configuration["pixel_stride"])
    if stride > 1:
        sampled = np.zeros(valid.shape, dtype=bool)
        sampled[::stride, ::stride] = valid[::stride, ::stride]
        valid = sampled

    rows, columns = np.nonzero(valid)
    if rows.size:
        depth = view.depth[rows, columns].astype(np.float64)
        fx, fy, cx, cy = view.intrinsics
        camera_points = np.column_stack(
            (
                (columns - cx) * depth / fx,
                (rows - cy) * depth / fy,
                depth,
                np.ones_like(depth),
            )
        )
        points = (view.world_from_camera @ camera_points.T).T[:, :3]
        normals = (view.world_from_camera[:3, :3] @ view.normal[rows, columns].T).T
        lengths = np.linalg.norm(normals, axis=1)
        finite = (
            np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1) & (lengths > 1e-8)
        )
        points = points[finite]
        normals = normals[finite]
        normals /= np.linalg.norm(normals, axis=1)[:, None]
    else:
        points = np.empty((0, 3), dtype=np.float64)
        normals = np.empty((0, 3), dtype=np.float64)

    geometry_core = core & geometry
    return BackprojectionResult(
        points=points,
        normals=normals,
        raw_sample_count=int(np.count_nonzero(raw)),
        boundary_rejected_count=int(np.count_nonzero(raw & ~core)),
        invalid_geometry_rejected_count=int(np.count_nonzero(core & ~geometry)),
        sam_score_rejected_count=(
            int(np.count_nonzero(geometry_core)) if not sam_score_valid else 0
        ),
        consistency_rejected_count=int(
            np.count_nonzero(geometry_core & sam_score_valid & ~consistent)
        ),
        depth_discontinuity_rejected_count=int(
            np.count_nonzero(geometry_core & sam_score_valid & consistent & discontinuity)
        ),
    )


def _project(view: TrainingView, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    camera = (np.linalg.inv(view.world_from_camera) @ homogeneous.T).T
    depth = camera[:, 2]
    fx, fy, cx, cy = view.intrinsics
    pixels = np.column_stack(
        (
            fx * camera[:, 0] / np.maximum(depth, 1e-12) + cx,
            fy * camera[:, 1] / np.maximum(depth, 1e-12) + cy,
        )
    )
    inside = (
        (depth > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 0] < view.depth.shape[1])
        & (pixels[:, 1] < view.depth.shape[0])
    )
    return pixels, depth, inside


def validate_training_points(
    points: np.ndarray,
    source_frame_id: str,
    views: list[TrainingView],
    configuration: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    supporting = np.ones(len(points), dtype=np.int32)
    contradicting = np.zeros(len(points), dtype=np.int32)
    residual_sum = np.zeros(len(points), dtype=np.float64)
    residual_count = np.zeros(len(points), dtype=np.int32)
    maximum = float(configuration["maximum_relative_depth_residual"])
    for view in views:
        if view.frame_id == source_frame_id:
            continue
        pixels, predicted_depth, inside = _project(view, points)
        indices = np.nonzero(inside)[0]
        if not len(indices):
            continue
        columns = np.clip(
            np.rint(pixels[indices, 0]).astype(int),
            0,
            view.depth.shape[1] - 1,
        )
        rows = np.clip(
            np.rint(pixels[indices, 1]).astype(int),
            0,
            view.depth.shape[0] - 1,
        )
        measured = view.depth[rows, columns]
        finite = np.isfinite(measured) & (measured > 0)
        residual = np.full(indices.size, np.inf)
        residual[finite] = np.abs(measured[finite] - predicted_depth[indices][finite]) / np.maximum(
            np.abs(predicted_depth[indices][finite]),
            1e-8,
        )
        mask_support = view.core_mask[rows, columns]
        support = finite & mask_support & (residual <= maximum)
        contradiction = finite & ((~mask_support) | (residual > maximum))
        supporting[indices[support]] += 1
        contradicting[indices[contradiction]] += 1
        good = finite & np.isfinite(residual)
        residual_sum[indices[good]] += residual[good]
        residual_count[indices[good]] += 1
    keep = (supporting >= int(configuration["minimum_supporting_views"])) & (
        contradicting <= int(configuration["maximum_contradicting_views"])
    )
    residual = residual_sum / np.maximum(residual_count, 1)
    return keep, supporting, residual
