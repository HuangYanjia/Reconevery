from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _canvas(title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (960, 600), (246, 247, 249))
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), title, fill=(20, 25, 32))
    if subtitle:
        draw.text((24, 42), subtitle, fill=(70, 78, 88))
    return image, draw


def _bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    baseline: list[float],
    aligned: list[float],
) -> None:
    image, draw = _canvas(title, "blue=baseline, green=best global Sim(3)")
    maximum = max([*baseline, *aligned, 1e-9])
    width = max(8, 840 // max(len(labels), 1))
    for index, label in enumerate(labels):
        x = 60 + index * width
        base_height = int(430 * baseline[index] / maximum)
        aligned_height = int(430 * aligned[index] / maximum)
        draw.rectangle((x, 520 - base_height, x + width // 3, 520), fill=(45, 105, 190))
        draw.rectangle(
            (x + width // 3 + 2, 520 - aligned_height, x + 2 * width // 3, 520),
            fill=(35, 150, 95),
        )
        draw.text((x, 530), label[-5:], fill=(40, 45, 55))
    image.save(path, format="PNG")


def _scatter(
    path: Path,
    title: str,
    baseline_pairs: list[tuple[float, float]],
    aligned_pairs: list[tuple[float, float]],
) -> None:
    image, draw = _canvas(title, "x=COLMAP sparse depth; y=rendered mesh depth")
    values = [value for pair in [*baseline_pairs, *aligned_pairs] for value in pair]
    maximum = max(values, default=1.0)
    draw.line((80, 530, 880, 80), fill=(90, 90, 90), width=2)
    for pairs, color in ((baseline_pairs, (35, 90, 190)), (aligned_pairs, (25, 155, 85))):
        for sparse_depth, mesh_depth in pairs[:5000]:
            x = 80 + int(800 * sparse_depth / maximum)
            y = 530 - int(450 * mesh_depth / maximum)
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
    image.save(path, format="PNG")


def _summary(path: Path, title: str, lines: list[str]) -> None:
    image, draw = _canvas(title)
    y = 85
    for line in lines:
        draw.text((55, y), line, fill=(35, 42, 50))
        y += 34
    image.save(path, format="PNG")


def _project_points(
    points: np.ndarray,
    *,
    axes: tuple[int, int],
    low: np.ndarray,
    extent: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    left, top, width, height = bounds
    normalized = (points[:, axes] - low) / extent
    x = left + np.clip(normalized[:, 0], 0.0, 1.0) * width
    y = top + (1.0 - np.clip(normalized[:, 1], 0.0, 1.0)) * height
    return x.astype(np.int32), y.astype(np.int32)


def _point_cloud_overlay(
    path: Path,
    title: str,
    mesh_points: np.ndarray,
    sparse_points: np.ndarray,
    subtitle: str,
) -> None:
    image, draw = _canvas(title, subtitle)
    mesh_stride = max(1, len(mesh_points) // 20_000)
    sparse_stride = max(1, len(sparse_points) // 10_000)
    mesh = np.asarray(mesh_points[::mesh_stride], dtype=np.float64)
    sparse = np.asarray(sparse_points[::sparse_stride], dtype=np.float64)
    combined = np.concatenate((mesh, sparse), axis=0)
    views = (("XY", 0, 1), ("XZ", 0, 2))
    for view_index, (label, axis_x, axis_y) in enumerate(views):
        left = 45 + view_index * 460
        top = 80
        width = 410
        height = 465
        low = np.percentile(combined[:, (axis_x, axis_y)], 1.0, axis=0)
        high = np.percentile(combined[:, (axis_x, axis_y)], 99.0, axis=0)
        extent = np.maximum(high - low, 1e-12)
        draw.rectangle((left, top, left + width, top + height), outline=(190, 196, 205))
        draw.text((left + 8, top + 8), label, fill=(45, 50, 58))
        projection = {
            "axes": (axis_x, axis_y),
            "low": low,
            "extent": extent,
            "bounds": (left, top, width, height),
        }
        mesh_x, mesh_y = _project_points(mesh, **projection)
        for x, y in zip(mesh_x.tolist(), mesh_y.tolist(), strict=True):
            draw.point((x, y), fill=(170, 176, 184))
        sparse_x, sparse_y = _project_points(sparse, **projection)
        for x, y in zip(sparse_x.tolist(), sparse_y.tolist(), strict=True):
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(32, 96, 190))
    image.save(path, format="PNG")


def write_previews(
    *,
    output_dir: Path,
    audit: dict[str, Any],
    baseline_metrics: dict[str, Any],
    aligned_metrics: dict[str, Any],
    camera_metrics: list[dict[str, Any]],
    chunk_metrics: list[dict[str, Any]],
    baseline_pairs: list[tuple[float, float]],
    aligned_pairs: list[tuple[float, float]],
    status: str,
    transform: dict[str, Any],
    mesh_samples: np.ndarray,
    sparse_points: np.ndarray,
    aligned_mesh_samples: np.ndarray,
) -> dict[str, str]:
    previews = output_dir / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    _summary(
        previews / "transform_chain_comparison.png",
        "Phase 3 transform-chain audit",
        [
            f"status: {audit['status']}",
            f"COLMAP/working roundtrip: {audit['colmap_working_roundtrip_error']:.3e}",
            f"camera roundtrip: {audit['camera_basis_roundtrip_error']:.3e}",
            f"mesh roundtrip: {audit['sampled_mesh_roundtrip_error']:.3e}",
            f"pre/post equivalent: {audit['pre_post_render_equivalent']}",
        ],
    )
    _summary(
        previews / "baseline_depth_residual.png",
        "Identity baseline depth residual",
        [
            f"observations: {baseline_metrics['observation_count']}",
            f"median: {baseline_metrics['sparse_depth_residual_median']}",
            f"p90: {baseline_metrics['sparse_depth_residual_p90']}",
            f"inlier@0.10: {baseline_metrics['inlier_fractions']['0.10']:.4f}",
            f"mesh coverage: {baseline_metrics['mesh_pixel_coverage']:.4f}",
        ],
    )
    _summary(
        previews / "aligned_depth_residual.png",
        "Best global Sim(3) held-out depth residual",
        [
            f"status: {status}",
            f"observations: {aligned_metrics['observation_count']}",
            f"median: {aligned_metrics['sparse_depth_residual_median']}",
            f"p90: {aligned_metrics['sparse_depth_residual_p90']}",
            f"inlier@0.10: {aligned_metrics['inlier_fractions']['0.10']:.4f}",
            f"mesh coverage: {aligned_metrics['mesh_pixel_coverage']:.4f}",
        ],
    )
    _scatter(
        previews / "baseline_vs_aligned_scatter.png",
        "Baseline vs aligned sparse/rendered depth",
        baseline_pairs,
        aligned_pairs,
    )
    labels = [str(item["frame_id"]) for item in camera_metrics]
    baseline = [float(item["baseline_median_residual"] or 0.0) for item in camera_metrics]
    aligned = [float(item["aligned_median_residual"] or 0.0) for item in camera_metrics]
    _bar_chart(
        previews / "per_camera_residuals.png",
        "Per-camera held-out residuals",
        labels,
        baseline,
        aligned,
    )
    _bar_chart(
        previews / "per_chunk_residuals.png",
        "Per-chunk residual structure",
        [str(item["chunk_id"]) for item in chunk_metrics],
        [float(item["baseline_median_residual"] or 0.0) for item in chunk_metrics],
        [float(item["aligned_median_residual"] or 0.0) for item in chunk_metrics],
    )
    _point_cloud_overlay(
        previews / "sparse_points_and_mesh_before.png",
        "Sparse points and mesh before alignment",
        mesh_samples,
        sparse_points,
        "gray=global mesh samples, blue=COLMAP sparse points; arbitrary coordinates",
    )
    _point_cloud_overlay(
        previews / "sparse_points_and_mesh_after.png",
        "Sparse points and mesh after candidate alignment",
        aligned_mesh_samples,
        sparse_points,
        (
            f"candidate s={transform['scale']:.4f}, "
            f"R={transform['rotation_degrees']:.2f} deg; not necessarily accepted"
        ),
    )
    _summary(
        previews / "heldout_validation_summary.png",
        "Held-out validation decision",
        [
            f"status: {status}",
            f"baseline median: {baseline_metrics['sparse_depth_residual_median']}",
            f"aligned median: {aligned_metrics['sparse_depth_residual_median']}",
            f"baseline p90: {baseline_metrics['sparse_depth_residual_p90']}",
            f"aligned p90: {aligned_metrics['sparse_depth_residual_p90']}",
            "Cameras and original mesh bytes remain unchanged.",
        ],
    )
    return {
        "transform_chain_comparison_path": (
            "reconstruction/alignment/previews/transform_chain_comparison.png"
        ),
        "baseline_depth_residual_path": (
            "reconstruction/alignment/previews/baseline_depth_residual.png"
        ),
        "aligned_depth_residual_path": (
            "reconstruction/alignment/previews/aligned_depth_residual.png"
        ),
        "baseline_vs_aligned_scatter_path": (
            "reconstruction/alignment/previews/baseline_vs_aligned_scatter.png"
        ),
        "per_camera_residuals_path": ("reconstruction/alignment/previews/per_camera_residuals.png"),
        "per_chunk_residuals_path": ("reconstruction/alignment/previews/per_chunk_residuals.png"),
        "sparse_points_and_mesh_before_path": (
            "reconstruction/alignment/previews/sparse_points_and_mesh_before.png"
        ),
        "sparse_points_and_mesh_after_path": (
            "reconstruction/alignment/previews/sparse_points_and_mesh_after.png"
        ),
        "heldout_validation_summary_path": (
            "reconstruction/alignment/previews/heldout_validation_summary.png"
        ),
    }
