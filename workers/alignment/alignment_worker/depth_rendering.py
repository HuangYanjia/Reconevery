from __future__ import annotations

import statistics
from typing import Any

from object_lifting_worker.rasterization import NvdiffrastRasterizer

from alignment_worker.sim3 import apply_transform


def _percentile(values: list[float], percentile: float) -> float | None:
    import numpy as np

    return float(np.percentile(values, percentile)) if values else None


def _metric_payload(
    residuals: list[float],
    log_residuals: list[float],
    coverages: list[float],
    frame_medians: list[float],
    point_metrics: dict[str, float | None],
    *,
    bad_frame_threshold: float,
) -> dict[str, object]:
    return {
        "observation_count": len(residuals),
        "sparse_depth_residual_median": (statistics.median(residuals) if residuals else None),
        "sparse_depth_residual_p75": _percentile(residuals, 75),
        "sparse_depth_residual_p90": _percentile(residuals, 90),
        "sparse_depth_residual_p95": _percentile(residuals, 95),
        "log_depth_residual_median": (statistics.median(log_residuals) if log_residuals else None),
        "inlier_fractions": {
            "0.05": (
                sum(value <= 0.05 for value in residuals) / len(residuals) if residuals else 0.0
            ),
            "0.10": (
                sum(value <= 0.10 for value in residuals) / len(residuals) if residuals else 0.0
            ),
            "0.20": (
                sum(value <= 0.20 for value in residuals) / len(residuals) if residuals else 0.0
            ),
        },
        "mesh_pixel_coverage": statistics.mean(coverages) if coverages else 0.0,
        "point_to_surface_median_scene_diagonal": point_metrics["median"],
        "point_to_surface_p90_scene_diagonal": point_metrics["p90"],
        "point_to_plane_median_scene_diagonal": None,
        "bad_frame_fraction": (
            sum(value > bad_frame_threshold for value in frame_medians) / len(frame_medians)
            if frame_medians
            else 1.0
        ),
    }


def render_alignment_metrics(
    *,
    vertices: Any,
    faces: Any,
    camera: dict[str, Any],
    observations: list[dict[str, object]],
    frame_ids: list[str],
    undistortion_records: dict[str, dict[str, object]],
    matrix: Any,
    face_chunk_size: int,
    point_metrics: dict[str, float | None],
    bad_frame_threshold: float,
    split: str,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[tuple[float, float]],
    list[dict[str, object]],
]:
    import math

    import numpy as np
    import torch

    transformed_vertices = apply_transform(vertices, matrix).astype(np.float32)
    rasterizer = NvdiffrastRasterizer(
        transformed_vertices,
        faces,
        face_chunk_size=face_chunk_size,
    )
    pose_by_frame = {pose["frame_id"]: pose for pose in camera["poses"]}
    observations_by_frame: dict[str, list[dict[str, object]]] = {}
    for observation in observations:
        if observation["frame_id"] in frame_ids:
            observations_by_frame.setdefault(str(observation["frame_id"]), []).append(observation)
    all_residuals: list[float] = []
    all_log_residuals: list[float] = []
    coverages: list[float] = []
    frame_medians: list[float] = []
    frame_metrics: list[dict[str, object]] = []
    pairs: list[tuple[float, float]] = []
    comparable_records: list[dict[str, object]] = []
    for frame_id in frame_ids:
        if frame_id not in pose_by_frame:
            continue
        intrinsics = undistortion_records[frame_id]["undistorted_intrinsics"]
        raster = rasterizer.rasterize(pose_by_frame[frame_id], intrinsics)
        coverage = float(np.asarray(raster.valid).mean())
        coverages.append(coverage)
        residuals: list[float] = []
        log_residuals: list[float] = []
        for observation in observations_by_frame.get(frame_id, []):
            x, y = observation["undistorted_pixel"]
            column = int(round(float(x)))
            row = int(round(float(y)))
            if (
                row < 0
                or column < 0
                or row >= raster.valid.shape[0]
                or column >= raster.valid.shape[1]
                or not bool(raster.valid[row, column])
            ):
                continue
            rendered_depth = float(raster.depth[row, column])
            sparse_depth = float(observation["camera_depth"])
            if not (
                math.isfinite(rendered_depth)
                and rendered_depth > 0
                and math.isfinite(sparse_depth)
                and sparse_depth > 0
            ):
                continue
            residual = abs(rendered_depth - sparse_depth) / max(abs(sparse_depth), 1e-12)
            log_residual = abs(math.log(rendered_depth) - math.log(sparse_depth))
            residuals.append(residual)
            log_residuals.append(log_residual)
            all_residuals.append(residual)
            all_log_residuals.append(log_residual)
            pairs.append((sparse_depth, rendered_depth))
            comparable_records.append(
                {
                    "frame_id": frame_id,
                    "point3d_id": observation["point3d_id"],
                    "point_world": observation["point_world"],
                    "residual": residual,
                }
            )
        median = statistics.median(residuals) if residuals else None
        if median is not None:
            frame_medians.append(median)
        frame_metrics.append(
            {
                "frame_id": frame_id,
                "camera_id": camera["camera_id"],
                "valid_sparse_observations": len(residuals),
                "mesh_pixel_coverage": coverage,
                "median_residual": median,
                "p90_residual": _percentile(residuals, 90),
                "inlier_fraction": (
                    sum(value <= 0.10 for value in residuals) / len(residuals)
                    if residuals
                    else None
                ),
                "visible_mesh_face_count": int(np.unique(raster.face_ids[raster.valid]).size),
                "outlier": median is None or median > bad_frame_threshold,
                "outlier_reason": (
                    "no_comparable_sparse_depth"
                    if median is None
                    else "median_depth_residual_above_threshold"
                    if median > bad_frame_threshold
                    else None
                ),
                "split": split,
            }
        )
    metrics = _metric_payload(
        all_residuals,
        all_log_residuals,
        coverages,
        frame_medians,
        point_metrics,
        bad_frame_threshold=bad_frame_threshold,
    )
    del rasterizer
    torch.cuda.empty_cache()
    return metrics, frame_metrics, pairs, comparable_records


def merge_camera_metrics(
    baseline: list[dict[str, object]],
    aligned: list[dict[str, object]],
) -> list[dict[str, object]]:
    aligned_by_frame = {str(item["frame_id"]): item for item in aligned}
    output = []
    for item in baseline:
        candidate = aligned_by_frame[str(item["frame_id"])]
        output.append(
            {
                "frame_id": item["frame_id"],
                "camera_id": item["camera_id"],
                "valid_sparse_observations": candidate["valid_sparse_observations"],
                "mesh_pixel_coverage": candidate["mesh_pixel_coverage"],
                "baseline_median_residual": item["median_residual"],
                "aligned_median_residual": candidate["median_residual"],
                "baseline_p90_residual": item["p90_residual"],
                "aligned_p90_residual": candidate["p90_residual"],
                "baseline_inlier_fraction": item["inlier_fraction"],
                "aligned_inlier_fraction": candidate["inlier_fraction"],
                "visible_mesh_face_count": candidate["visible_mesh_face_count"],
                "outlier": candidate["outlier"],
                "outlier_reason": candidate["outlier_reason"],
                "split": candidate["split"],
            }
        )
    return output
