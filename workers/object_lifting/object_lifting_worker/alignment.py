from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from object_lifting_worker.camera_projection import transform_world_point_to_camera
from object_lifting_worker.distortion import undistort_points
from object_lifting_worker.previews import add_title, contact_sheet


def _data_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def read_sparse_observations(
    *,
    images_path: Path,
    points3d_path: Path,
    registered_frames_path: Path,
) -> tuple[dict[str, list[tuple[float, float, tuple[float, float, float]]]], int]:
    import json

    points = {}
    for line in _data_lines(points3d_path):
        fields = line.split()
        points[int(fields[0])] = (float(fields[1]), float(fields[2]), float(fields[3]))
    registered = json.loads(registered_frames_path.read_text(encoding="utf-8"))
    frame_by_name = {item["package_image_name"]: item["frame_id"] for item in registered["frames"]}
    lines = _data_lines(images_path)
    observations: dict[str, list[tuple[float, float, tuple[float, float, float]]]] = {}
    for index in range(0, len(lines), 2):
        image_fields = lines[index].split()
        frame_id = frame_by_name.get(image_fields[9])
        if frame_id is None:
            continue
        values = lines[index + 1].split()
        frame_observations = []
        for offset in range(0, len(values), 3):
            point_id = int(values[offset + 2])
            if point_id >= 0 and point_id in points:
                frame_observations.append(
                    (float(values[offset]), float(values[offset + 1]), points[point_id])
                )
        observations[frame_id] = frame_observations
    return observations, len(points)


def _percentile(values: Any, percentile: float) -> float:
    import numpy as np

    return float(np.percentile(values, percentile)) if len(values) else 0.0


def compute_alignment(
    *,
    frames: list[Any],
    camera: dict[str, Any],
    images_path: Path,
    points3d_path: Path,
    registered_frames_path: Path,
    raster_scale: float,
    scene_diagonal: float,
    inlier_threshold: float,
    minimum_inlier_fraction: float,
    frame_sequence_digest: str,
    camera_reconstruction_sha256: str,
    global_mesh_sha256: str,
) -> tuple[dict[str, Any], list[tuple[float, float]], list[Image.Image], list[Image.Image]]:
    import numpy as np

    sparse_by_frame, _point_count = read_sparse_observations(
        images_path=images_path,
        points3d_path=points3d_path,
        registered_frames_path=registered_frames_path,
    )
    camera_model = str(camera["model"])
    source_intrinsics = camera["intrinsics"]
    frame_records = []
    all_residuals: list[float] = []
    depth_pairs: list[tuple[float, float]] = []
    depth_tiles: list[Image.Image] = []
    edge_tiles: list[Image.Image] = []
    for frame in frames:
        valid = frame.raster.valid
        finite_depth = np.isfinite(frame.raster.depth) & valid
        depths = frame.raster.depth[finite_depth]
        visible_faces = int(np.unique(frame.raster.face_ids[valid]).size)
        residuals: list[float] = []
        sparse = sparse_by_frame.get(frame.frame_id, [])
        if sparse:
            mapped = undistort_points(
                [(item[0], item[1]) for item in sparse],
                camera_model=camera_model,
                intrinsics=source_intrinsics,
                raster_scale=raster_scale,
            )
            transform = frame.pose["transform_world_from_camera"]
            for pixel, (_x, _y, point_world) in zip(mapped, sparse, strict=True):
                column = int(round(float(pixel[0])))
                row = int(round(float(pixel[1])))
                if (
                    row < 0
                    or column < 0
                    or row >= valid.shape[0]
                    or column >= valid.shape[1]
                    or not finite_depth[row, column]
                ):
                    continue
                sparse_depth = transform_world_point_to_camera(
                    point_world,
                    transform["translation"],
                    transform["rotation_xyzw"],
                )[2]
                if sparse_depth <= 0:
                    continue
                mesh_depth = float(frame.raster.depth[row, column])
                residual = abs(mesh_depth - sparse_depth) / max(
                    abs(sparse_depth),
                    scene_diagonal * 1e-6,
                )
                residuals.append(residual)
                all_residuals.append(residual)
                depth_pairs.append((sparse_depth, mesh_depth))
        coverage = float(valid.mean())
        frame_records.append(
            {
                "frame_id": frame.frame_id,
                "mesh_pixel_coverage": coverage,
                "depth_finite_ratio": float(finite_depth.mean()),
                "visible_global_face_count": visible_faces,
                "depth_percentiles": {
                    "p05": _percentile(depths, 5),
                    "p50": _percentile(depths, 50),
                    "p95": _percentile(depths, 95),
                },
                "sparse_observation_count": len(residuals),
                "normalized_depth_residual_median": (
                    statistics.median(residuals) if residuals else None
                ),
                "normalized_depth_residual_p90": (
                    _percentile(residuals, 90) if residuals else None
                ),
                "depth_inlier_fraction": (
                    sum(value <= inlier_threshold for value in residuals) / len(residuals)
                    if residuals
                    else None
                ),
            }
        )
        depth_image = _depth_image(frame.raster.depth, finite_depth)
        depth_tiles.append(add_title(depth_image, f"{frame.frame_id} mesh depth"))
        edge_tiles.append(add_title(_edge_image(valid), f"{frame.frame_id} mesh edges"))
    inlier_fraction = (
        sum(value <= inlier_threshold for value in all_residuals) / len(all_residuals)
        if all_residuals
        else None
    )
    mean_coverage = statistics.mean(item["mesh_pixel_coverage"] for item in frame_records)
    sufficient = (
        inlier_fraction is not None
        and inlier_fraction >= minimum_inlier_fraction
        and mean_coverage >= 0.05
    )
    if not all_residuals:
        diagnosis = "No sparse observations coincided with rendered mesh depth."
    elif sufficient:
        diagnosis = (
            "Camera/global-mesh depth alignment is sufficient for lifting; remaining "
            "quality limits are likely mesh granularity or missing geometry."
        )
    else:
        diagnosis = (
            "Camera/global-mesh depth alignment is weak; association quality is limited "
            "before exact-face or surface-sample granularity."
        )
    artifact = {
        "schema_version": "0.1.0",
        "frame_sequence_digest": frame_sequence_digest,
        "camera_reconstruction_sha256": camera_reconstruction_sha256,
        "global_mesh_sha256": global_mesh_sha256,
        "frames": frame_records,
        "mesh_pixel_coverage_mean": mean_coverage,
        "sparse_depth_residual_median": (
            statistics.median(all_residuals) if all_residuals else None
        ),
        "sparse_depth_residual_p90": (_percentile(all_residuals, 90) if all_residuals else None),
        "sparse_depth_inlier_fraction": inlier_fraction,
        "alignment_sufficient_for_lifting": sufficient,
        "diagnosis": diagnosis,
        "warnings": ([] if all_residuals else ["No comparable sparse/mesh depth samples"]),
    }
    return artifact, depth_pairs, depth_tiles, edge_tiles


def _depth_image(depth: Any, valid: Any) -> Image.Image:
    import numpy as np

    output = np.zeros((*depth.shape, 3), dtype=np.uint8)
    values = depth[valid]
    if values.size:
        low, high = np.percentile(values, [5, 95])
        normalized = np.clip((depth - low) / max(high - low, 1e-12), 0, 1)
        output[:, :, 0] = np.where(valid, 255 * normalized, 24).astype(np.uint8)
        output[:, :, 1] = np.where(valid, 255 * (1 - normalized), 24).astype(np.uint8)
        output[:, :, 2] = np.where(valid, 180, 24).astype(np.uint8)
    return Image.fromarray(output, mode="RGB")


def _edge_image(valid: Any) -> Image.Image:
    import numpy as np

    edges = np.zeros_like(valid, dtype=bool)
    edges[1:, :] |= valid[1:, :] != valid[:-1, :]
    edges[:, 1:] |= valid[:, 1:] != valid[:, :-1]
    output = np.full((*valid.shape, 3), 28, dtype=np.uint8)
    output[valid] = (70, 70, 70)
    output[edges] = (0, 220, 255)
    return Image.fromarray(output, mode="RGB")


def write_alignment_previews(
    *,
    output_root: Path,
    depth_tiles: list[Image.Image],
    edge_tiles: list[Image.Image],
    depth_pairs: list[tuple[float, float]],
) -> None:
    contact_sheet(
        depth_tiles[:12],
        output_root / "global_mesh_depth_contact_sheet.png",
        columns=3,
    )
    contact_sheet(
        edge_tiles[:12],
        output_root / "global_mesh_edge_overlay.png",
        columns=3,
    )
    canvas = Image.new("RGB", (800, 500), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 15), "Sparse-point depth vs rendered global-mesh depth", fill=(20, 20, 20))
    if depth_pairs:
        maximum = max(max(left, right) for left, right in depth_pairs)
        maximum = max(maximum, 1e-12)
        draw.line((60, 440, 760, 60), fill=(120, 120, 120), width=2)
        for sparse_depth, mesh_depth in depth_pairs[:5000]:
            x = 60 + int(700 * sparse_depth / maximum)
            y = 440 - int(380 * mesh_depth / maximum)
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(0, 90, 180))
    else:
        draw.text((250, 240), "No comparable depth samples", fill=(100, 100, 100))
    canvas.save(
        output_root / "sparse_point_vs_mesh_depth.png",
        format="PNG",
        compress_level=6,
        optimize=False,
    )
