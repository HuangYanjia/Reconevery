from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from measured_geometry_worker.spacing import estimate_spacing
from measured_geometry_worker.version import __version__


@dataclass
class View:
    frame_id: str
    depth: np.ndarray
    normal: np.ndarray
    consistency: np.ndarray
    mask: np.ndarray
    intrinsics: np.ndarray
    rotation_world_from_camera: np.ndarray
    translation_world_from_camera: np.ndarray
    frame_score: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_array(path: Path, channels: int) -> np.ndarray:
    with path.open("rb") as file:
        width, height, actual = np.genfromtxt(
            file, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        file.seek(0)
        delimiters = 0
        while delimiters < 3:
            byte = file.read(1)
            if not byte:
                raise ValueError(f"truncated dense map {path}")
            delimiters += byte == b"&"
        payload = np.fromfile(file, dtype="<f4")
    if actual != channels or payload.size != width * height * actual:
        raise ValueError(f"invalid dense map dimensions or channels: {path}")
    result = payload.reshape((width, height, actual), order="F").transpose(1, 0, 2)
    return result.squeeze() if channels == 1 else result


def read_consistency(path: Path, shape: tuple[int, int], image_count: int) -> np.ndarray:
    with path.open("rb") as file:
        width, height, _ = np.genfromtxt(
            file, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        file.seek(0)
        delimiters = 0
        while delimiters < 3:
            byte = file.read(1)
            if not byte:
                raise ValueError(f"truncated consistency graph {path}")
            delimiters += byte == b"&"
        values = np.fromfile(file, dtype="<i4")
    if (height, width) != shape:
        raise ValueError(f"consistency graph dimensions differ from depth map: {path}")
    counts = np.zeros(shape, dtype=np.int32)
    cursor = 0
    while cursor < values.size:
        if cursor + 3 > values.size:
            raise ValueError(f"truncated consistency graph entry: {path}")
        column, row, count = (int(value) for value in values[cursor : cursor + 3])
        cursor += 3
        if (
            row < 0
            or row >= shape[0]
            or column < 0
            or column >= shape[1]
            or count < 0
            or cursor + count > values.size
        ):
            raise ValueError(f"invalid consistency graph entry: {path}")
        sources = values[cursor : cursor + count]
        cursor += count
        if np.any(sources < 0) or np.any(sources >= image_count):
            raise ValueError(f"invalid consistency graph source index: {path}")
        counts[row, column] = count
    return counts


def quaternion_matrix(values: list[float]) -> np.ndarray:
    x, y, z, w = values
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        raise ValueError("zero camera quaternion")
    x, y, z, w = (value / norm for value in (x, y, z, w))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def distortion_coefficients(model: str, values: list[float]) -> np.ndarray:
    if model in {"PINHOLE", "SIMPLE_PINHOLE"}:
        return np.zeros(5, dtype=np.float64)
    padded = [*values, 0, 0, 0, 0]
    if model == "SIMPLE_RADIAL":
        return np.array([padded[0], 0, 0, 0, 0], dtype=np.float64)
    if model == "RADIAL":
        return np.array([padded[0], padded[1], 0, 0, 0], dtype=np.float64)
    if model == "OPENCV":
        return np.array([padded[0], padded[1], padded[2], padded[3], 0], dtype=np.float64)
    raise ValueError(f"unsupported camera model {model}")


def undistort_mask(mask_path: Path, record: dict[str, Any]) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"could not read canonical mask {mask_path}")
    source_width, source_height = record["source_dimensions"]
    if mask.shape != (source_height, source_width):
        raise ValueError(f"mask dimensions differ from normalized RGB: {mask_path}")
    fx, fy, cx, cy = record["source_intrinsics"]
    dfx, dfy, dcx, dcy = record["dense_intrinsics"]
    dense_width, dense_height = record["dense_dimensions"]
    source_k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dense_k = np.array([[dfx, 0, dcx], [0, dfy, dcy], [0, 0, 1]], dtype=np.float64)
    map_x, map_y = cv2.initUndistortRectifyMap(
        source_k,
        distortion_coefficients(record["source_camera_model"], record["source_distortion"]),
        None,
        dense_k,
        (dense_width, dense_height),
        cv2.CV_32FC1,
    )
    remapped = cv2.remap(
        mask,
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    remapped = np.where(remapped >= 128, 255, 0).astype(np.uint8)
    if not set(np.unique(remapped)).issubset({0, 255}):
        raise RuntimeError("undistorted canonical mask is not binary")
    return remapped


def write_points(path: Path, points: np.ndarray, normals: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as file:
        file.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float nx\nproperty float ny\nproperty float nz\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "element face 0\nproperty list uchar int vertex_indices\nend_header\n"
        )
        for point, normal in zip(points, normals, strict=True):
            file.write(
                f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g} 170 190 210\n"
            )


def _labeled_tile(image: np.ndarray, label: str) -> Image.Image:
    tile = Image.fromarray(image.astype(np.uint8), mode="RGB")
    tile.thumbnail((360, 240), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (360, 272), "white")
    canvas.paste(tile, ((360 - tile.width) // 2, 28))
    ImageDraw.Draw(canvas).text((8, 7), label, fill=(20, 25, 30))
    return canvas


def contact_sheet(path: Path, title: str, tiles: list[tuple[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = 2
    rows = max(1, math.ceil(len(tiles) / columns))
    image = Image.new("RGB", (columns * 360, 42 + rows * 272), (238, 240, 242))
    draw = ImageDraw.Draw(image)
    draw.text((14, 14), title, fill=(20, 25, 30))
    for index, (label, tile) in enumerate(tiles):
        image.paste(
            _labeled_tile(tile, label), ((index % columns) * 360, 42 + index // columns * 272)
        )
    image.save(path)


def depth_preview(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth[valid], [2, 98])
        if high > low:
            normalized[valid] = np.clip(255 * (depth[valid] - low) / (high - low), 0, 255).astype(
                np.uint8
            )
    colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def point_cloud_preview(points: np.ndarray) -> np.ndarray:
    sample = points[:: max(1, len(points) // 150_000)]
    centered = sample - np.median(sample, axis=0)
    _, eigenvectors = np.linalg.eigh(np.cov(centered.T))
    projected = centered @ eigenvectors[:, -2:]
    depth = centered @ eigenvectors[:, -3]
    low = np.percentile(projected, 1, axis=0)
    high = np.percentile(projected, 99, axis=0)
    extent = np.maximum(high - low, 1e-8)
    pixels = np.clip((projected - low) / extent * np.array([699, 419]), 0, [699, 419]).astype(int)
    colors = cv2.applyColorMap(
        np.clip(255 * (depth - depth.min()) / max(float(np.ptp(depth)), 1e-8), 0, 255).astype(
            np.uint8
        ),
        cv2.COLORMAP_TURBO,
    ).reshape(-1, 3)
    image = np.full((420, 700, 3), 245, dtype=np.uint8)
    rows = 419 - pixels[:, 1]
    columns = pixels[:, 0]
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            target_rows = np.clip(rows + row_offset, 0, 419)
            target_columns = np.clip(columns + column_offset, 0, 699)
            image[target_rows, target_columns] = colors[:, ::-1]
    return image


def project(view: View, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_points = (
        view.rotation_world_from_camera.T @ (points_world - view.translation_world_from_camera).T
    ).T
    depth = camera_points[:, 2]
    pixels = np.empty((len(points_world), 2), dtype=np.float64)
    pixels[:, 0] = view.intrinsics[0] * camera_points[:, 0] / depth + view.intrinsics[2]
    pixels[:, 1] = view.intrinsics[1] * camera_points[:, 1] / depth + view.intrinsics[3]
    inside = (
        (depth > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 0] < view.depth.shape[1])
        & (pixels[:, 1] < view.depth.shape[0])
    )
    return pixels, depth, inside


def build_view(
    *,
    frame_id: str,
    mask_path: Path,
    frame_score: float,
    depth_record: dict[str, Any],
    undistortion_record: dict[str, Any],
    pose: dict[str, Any],
    input_root: Path,
    image_count: int,
    backprojection_config: dict[str, Any],
) -> View:
    depth = read_array(input_root / depth_record["depth_path"], 1)
    normal = read_array(input_root / depth_record["normal_path"], 3)
    consistency = read_consistency(
        input_root / depth_record["consistency_graph_path"], depth.shape, image_count
    )
    mask = undistort_mask(mask_path, undistortion_record)
    if mask.shape != depth.shape:
        mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask >= 128, 255, 0).astype(np.uint8)
    if backprojection_config["exclude_mask_boundary"]:
        mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    transform = pose["transform_world_from_camera"]
    return View(
        frame_id=frame_id,
        depth=depth,
        normal=normal,
        consistency=consistency,
        mask=mask,
        intrinsics=np.asarray(undistortion_record["dense_intrinsics"], dtype=np.float64),
        rotation_world_from_camera=quaternion_matrix(transform["rotation_xyzw"]),
        translation_world_from_camera=np.asarray(transform["translation"], dtype=np.float64),
        frame_score=frame_score,
    )


def backproject(view: View, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    stride = int(config["pixel_stride"])
    valid = (
        np.isfinite(view.depth)
        & (view.depth > 0)
        & np.isfinite(view.normal).all(axis=2)
        & (view.mask > 0)
        & (view.consistency >= int(config["minimum_consistent_source_views"]))
        & (view.frame_score >= float(config["minimum_sam_frame_score"]))
    )
    threshold = float(config["maximum_relative_depth_discontinuity"])
    discontinuity = np.zeros_like(valid)
    for row_shift, column_shift in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor = np.roll(view.depth, (row_shift, column_shift), axis=(0, 1))
        pair = np.isfinite(neighbor) & (neighbor > 0) & np.isfinite(view.depth) & (view.depth > 0)
        relative = np.zeros_like(view.depth, dtype=np.float32)
        relative[pair] = np.abs(neighbor[pair] - view.depth[pair]) / np.maximum(
            np.abs(view.depth[pair]), 1e-8
        )
        discontinuity |= pair & (relative > threshold)
    valid &= ~discontinuity
    valid[::stride, ::stride] &= True
    if stride > 1:
        sampled = np.zeros_like(valid)
        sampled[::stride, ::stride] = valid[::stride, ::stride]
        valid = sampled
    rows, columns = np.nonzero(valid)
    if rows.size == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    depth = view.depth[rows, columns].astype(np.float64)
    fx, fy, cx, cy = view.intrinsics
    camera_points = np.column_stack(((columns - cx) * depth / fx, (rows - cy) * depth / fy, depth))
    points = (
        view.rotation_world_from_camera @ camera_points.T
    ).T + view.translation_world_from_camera
    normals = (view.rotation_world_from_camera @ view.normal[rows, columns].T).T
    norms = np.linalg.norm(normals, axis=1)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1) & (norms > 0)
    normals[finite] /= norms[finite, None]
    return points[finite], normals[finite]


def validate_points(
    points: np.ndarray,
    source_frame: str,
    views: list[View],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    supporting = np.ones(len(points), dtype=np.int32)
    contradicting = np.zeros(len(points), dtype=np.int32)
    residual_sum = np.zeros(len(points), dtype=np.float64)
    residual_count = np.zeros(len(points), dtype=np.int32)
    maximum = float(config["maximum_relative_depth_residual"])
    for view in views:
        if view.frame_id == source_frame:
            continue
        pixels, predicted_depth, inside = project(view, points)
        indices = np.nonzero(inside)[0]
        if indices.size == 0:
            continue
        columns = np.rint(pixels[indices, 0]).astype(int)
        rows = np.rint(pixels[indices, 1]).astype(int)
        columns = np.clip(columns, 0, view.depth.shape[1] - 1)
        rows = np.clip(rows, 0, view.depth.shape[0] - 1)
        measured = view.depth[rows, columns]
        finite = np.isfinite(measured) & (measured > 0)
        residual = np.full(indices.size, np.inf)
        residual[finite] = np.abs(measured[finite] - predicted_depth[indices][finite]) / np.maximum(
            np.abs(predicted_depth[indices][finite]), 1e-8
        )
        mask_support = view.mask[rows, columns] > 0
        support = finite & mask_support & (residual <= maximum)
        contradiction = finite & ((~mask_support) | (residual > maximum))
        supporting[indices[support]] += 1
        contradicting[indices[contradiction]] += 1
        good = finite & np.isfinite(residual)
        residual_sum[indices[good]] += residual[good]
        residual_count[indices[good]] += 1
    keep = (supporting >= int(config["minimum_supporting_views"])) & (
        contradicting <= int(config["maximum_contradicting_views"])
    )
    residual = residual_sum / np.maximum(residual_count, 1)
    return keep, supporting, residual


def fuse(
    points: np.ndarray, normals: np.ndarray, multiplier: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | str]]:
    if len(points) == 0:
        raise ValueError("surfel fusion requires at least one measured point")
    spacing = estimate_spacing(
        (tuple(float(value) for value in point) for point in points),
        multiplier=multiplier,
    )
    voxel = float(spacing["voxel_size"])
    keys = np.floor(points / voxel).astype(np.int64)
    order = np.lexsort(
        (
            normals[:, 2],
            normals[:, 1],
            normals[:, 0],
            points[:, 2],
            points[:, 1],
            points[:, 0],
            keys[:, 2],
            keys[:, 1],
            keys[:, 0],
        )
    )
    keys = keys[order]
    points = points[order]
    normals = normals[order]
    unique, starts, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    fused_points = np.add.reduceat(points, starts, axis=0) / counts[:, None]
    fused_normals = np.add.reduceat(normals, starts, axis=0)
    lengths = np.linalg.norm(fused_normals, axis=1)
    fused_normals /= np.maximum(lengths[:, None], 1e-12)
    return fused_points, fused_normals, counts, unique, spacing


def voxel_component_count(keys: np.ndarray) -> int:
    remaining = {tuple(int(value) for value in key) for key in keys}
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            x, y, z = stack.pop()
            for neighbor in (
                (x - 1, y, z),
                (x + 1, y, z),
                (x, y - 1, z),
                (x, y + 1, z),
                (x, y, z - 1),
                (x, y, z + 1),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return count


def reprojection_metrics(
    points: np.ndarray,
    views: list[View],
    *,
    maximum_relative_depth_residual: float,
    splat_radius_pixels: int,
) -> tuple[float, float, float, float]:
    true_positive = predicted_positive = mask_positive = 0
    per_view_ious: list[float] = []
    kernel_size = 2 * splat_radius_pixels + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    for view in views:
        pixels, predicted_depth, inside = project(view, points)
        indices = np.nonzero(inside)[0]
        rendered = np.zeros(view.depth.shape, dtype=np.uint8)
        if indices.size:
            columns = np.clip(np.rint(pixels[indices, 0]).astype(int), 0, view.depth.shape[1] - 1)
            rows = np.clip(np.rint(pixels[indices, 1]).astype(int), 0, view.depth.shape[0] - 1)
            measured = view.depth[rows, columns]
            finite = np.isfinite(measured) & (measured > 0)
            residual = np.full(indices.size, np.inf)
            residual[finite] = np.abs(
                measured[finite] - predicted_depth[indices][finite]
            ) / np.maximum(np.abs(predicted_depth[indices][finite]), 1e-8)
            visible = finite & (residual <= maximum_relative_depth_residual)
            rendered[rows[visible], columns[visible]] = 1
            if splat_radius_pixels:
                rendered = cv2.dilate(rendered, kernel, iterations=1)
        mask = view.mask > 0
        rendered_bool = rendered > 0
        intersection = int(np.count_nonzero(rendered_bool & mask))
        predicted = int(np.count_nonzero(rendered_bool))
        target = int(np.count_nonzero(mask))
        union = predicted + target - intersection
        true_positive += intersection
        predicted_positive += predicted
        mask_positive += target
        per_view_ious.append(intersection / union if union else 0.0)
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / mask_positive if mask_positive else 0.0
    union = predicted_positive + mask_positive - true_positive
    iou = true_positive / union if union else 0.0
    return precision, recall, iou, float(np.median(per_view_ious)) if per_view_ious else 0.0


def rendered_point_mask(
    points: np.ndarray,
    view: View,
    *,
    maximum_relative_depth_residual: float,
    splat_radius_pixels: int,
) -> np.ndarray:
    pixels, predicted_depth, inside = project(view, points)
    indices = np.nonzero(inside)[0]
    rendered = np.zeros(view.depth.shape, dtype=np.uint8)
    if indices.size:
        columns = np.clip(np.rint(pixels[indices, 0]).astype(int), 0, view.depth.shape[1] - 1)
        rows = np.clip(np.rint(pixels[indices, 1]).astype(int), 0, view.depth.shape[0] - 1)
        measured = view.depth[rows, columns]
        finite = np.isfinite(measured) & (measured > 0)
        residual = np.full(indices.size, np.inf)
        residual[finite] = np.abs(measured[finite] - predicted_depth[indices][finite]) / np.maximum(
            np.abs(predicted_depth[indices][finite]), 1e-8
        )
        visible = finite & (residual <= maximum_relative_depth_residual)
        rendered[rows[visible], columns[visible]] = 1
        if splat_radius_pixels:
            size = 2 * splat_radius_pixels + 1
            rendered = cv2.dilate(rendered, np.ones((size, size), dtype=np.uint8), iterations=1)
    return rendered > 0


def reprojection_overlay(points: np.ndarray, view: View, config: dict[str, Any]) -> np.ndarray:
    rendered = rendered_point_mask(
        points,
        view,
        maximum_relative_depth_residual=float(config["maximum_relative_depth_residual"]),
        splat_radius_pixels=int(config["splat_radius_pixels"]),
    )
    mask = view.mask > 0
    image = depth_preview(view.depth).astype(np.float32)
    colors = np.zeros_like(image)
    colors[rendered & mask] = (45, 190, 95)
    colors[rendered & ~mask] = (230, 65, 55)
    colors[~rendered & mask] = (50, 115, 230)
    highlighted = rendered | mask
    image[highlighted] = 0.35 * image[highlighted] + 0.65 * colors[highlighted]
    return image.astype(np.uint8)


def infer(request_path: Path, input_root: Path, output_dir: Path) -> dict[str, object]:
    started = time.monotonic()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    manifest = json.loads((input_root / request["manifest_path"]).read_text(encoding="utf-8"))
    camera = json.loads(
        (input_root / request["camera_reconstruction_path"]).read_text(encoding="utf-8")
    )
    undistortion = json.loads(
        (input_root / request["undistortion_manifest_path"]).read_text(encoding="utf-8")
    )
    depth_manifest = json.loads(
        (input_root / request["depth_manifest_path"]).read_text(encoding="utf-8")
    )
    pose_by_frame = {pose["frame_id"]: pose for pose in camera["poses"]}
    undistortion_by_frame = {record["frame_id"]: record for record in undistortion["records"]}
    depth_by_frame = {record["frame_id"]: record for record in depth_manifest["records"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    hypotheses = []
    total_raw = total_validated = total_surfels = 0
    object_tiles: list[tuple[str, np.ndarray]] = []
    depth_mask_tiles: list[tuple[str, np.ndarray]] = []
    reprojection_tiles: list[tuple[str, np.ndarray]] = []
    mask_mapping_seconds = 0.0
    backprojection_seconds = 0.0
    multiview_validation_seconds = 0.0
    surfel_fusion_seconds = 0.0
    preview_seconds = 0.0
    for track in request["object_tracks"]:
        available_frames = [
            frame_id
            for frame_id in track["mask_paths_by_frame"]
            if frame_id in depth_by_frame
            and frame_id in undistortion_by_frame
            and frame_id in pose_by_frame
        ]
        operation_started = time.monotonic()
        views = [
            build_view(
                frame_id=frame_id,
                mask_path=input_root / track["mask_paths_by_frame"][frame_id],
                frame_score=float(track["frame_scores"][frame_id]),
                depth_record=depth_by_frame[frame_id],
                undistortion_record=undistortion_by_frame[frame_id],
                pose=pose_by_frame[frame_id],
                input_root=input_root,
                image_count=len(depth_by_frame),
                backprojection_config=request["backprojection_configuration"],
            )
            for frame_id in available_frames
        ]
        mask_mapping_seconds += time.monotonic() - operation_started
        all_points: list[np.ndarray] = []
        all_normals: list[np.ndarray] = []
        observations = []
        object_raw = object_valid = 0
        for view in views:
            operation_started = time.monotonic()
            points, normals = backproject(view, request["backprojection_configuration"])
            backprojection_seconds += time.monotonic() - operation_started
            object_raw += len(points)
            operation_started = time.monotonic()
            keep, supporting, residual = validate_points(
                points,
                view.frame_id,
                views,
                request["consistency_configuration"],
            )
            multiview_validation_seconds += time.monotonic() - operation_started
            kept_points = points[keep]
            kept_normals = normals[keep]
            object_valid += len(kept_points)
            all_points.append(kept_points)
            all_normals.append(kept_normals)
            observations.append(
                {
                    "frame_id": view.frame_id,
                    "registered": True,
                    "raw_sample_count": len(points),
                    "validated_sample_count": len(kept_points),
                    "supporting_view_count": (int(np.max(supporting[keep])) if np.any(keep) else 0),
                    "contradicting_view_count": 0,
                    "depth_residual_median": (
                        float(np.median(residual[keep])) if np.any(keep) else None
                    ),
                    "mask_support_fraction": (float(np.mean(keep)) if len(keep) else 0.0),
                }
            )
        total_raw += object_raw
        total_validated += object_valid
        provenance = {
            "adapter_name": "measured_object_geometry",
            "adapter_version": "0.1.0",
            "configuration": request["backprojection_configuration"],
            "input_artifact_paths": [
                request["camera_reconstruction_path"],
                request["segmentation_tracking_path"],
                request["depth_manifest_path"],
            ],
            "output_artifact_paths": [],
            "timestamp": manifest["provenance"]["timestamp"],
            "confidence": {
                "score": 0.0,
                "method": "measured_multiview_depth_support",
                "notes": "visible measured surface only",
            },
            "source": "measured",
        }
        common = {
            "object_id": track["object_id"],
            "semantic_label": track["semantic_label"],
            "prompt_id": track["prompt_id"],
            "asset_type_hint": track["asset_type_hint"],
            "registered_mask_observations": len(available_frames),
            "observations_with_valid_dense_depth": len(views),
            "raw_measured_sample_count": object_raw,
            "validated_sample_count": object_valid,
            "supporting_view_count": len(views),
            "observations": observations,
            "completeness_confidence": 0.0,
            "geometry_source": "measured",
            "geometry_status": "partial_measured",
            "hidden_surface_completion": "not_implemented",
            "watertight": False,
            "sim_ready": False,
            "metric_scale_known": False,
            "canonical_gravity_alignment_known": False,
            "coordinate_convention": request["coordinate_convention"],
            "scale_status": "scale_ambiguous",
            "provenance": provenance,
        }
        if object_valid == 0:
            hypotheses.append(
                {
                    **common,
                    "status": "unresolved",
                    "reason": "insufficient_multiview_dense_support",
                    "fused_surfel_count": 0,
                    "point_cloud": None,
                    "surfel_cloud": None,
                    "observed_surface": None,
                    "depth_consistency": 0.0,
                    "normal_consistency": 0.0,
                    "reprojection_precision": 0.0,
                    "reprojection_recall": 0.0,
                    "reprojection_iou": 0.0,
                    "visible_mask_coverage": 0.0,
                    "connected_component_count": 0,
                    "measurement_confidence": 0.0,
                    "warnings": ["no valid multi-view geometric samples"],
                }
            )
            object_tiles.append(
                (
                    f"{track['object_id']} unresolved",
                    np.full((300, 420, 3), 235, dtype=np.uint8),
                )
            )
            continue
        points = np.concatenate(all_points)
        normals = np.concatenate(all_normals)
        maximum = int(request["backprojection_configuration"]["maximum_samples_per_object"])
        if len(points) > maximum:
            indices = np.linspace(0, len(points) - 1, maximum, dtype=int)
            points, normals = points[indices], normals[indices]
        operation_started = time.monotonic()
        fused_points, fused_normals, counts, voxel_keys, spacing = fuse(
            points,
            normals,
            float(request["surfel_fusion_configuration"]["voxel_size_multiplier"]),
        )
        surfel_fusion_seconds += time.monotonic() - operation_started
        operation_started = time.monotonic()
        precision, recall, reprojection_iou, median_view_iou = reprojection_metrics(
            fused_points,
            views,
            maximum_relative_depth_residual=float(
                request["consistency_configuration"]["maximum_relative_depth_residual"]
            ),
            splat_radius_pixels=int(request["reprojection_configuration"]["splat_radius_pixels"]),
        )
        multiview_validation_seconds += time.monotonic() - operation_started
        total_surfels += len(fused_points)
        root = output_dir / "objects" / track["object_id"]
        points_path = root / "measured_points.ply"
        surfels_path = root / "surfels.ply"
        write_points(points_path, points, normals)
        write_points(surfels_path, fused_points, fused_normals)
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            root / "surfels.npz",
            positions=fused_points,
            normals=fused_normals,
            sample_counts=counts,
        )
        write_json(
            root / "view_support.json",
            {"object_id": track["object_id"], "observations": observations},
        )
        relative_points = points_path.relative_to(input_root).as_posix()
        relative_surfels = surfels_path.relative_to(input_root).as_posix()
        provenance["output_artifact_paths"] = [relative_points, relative_surfels]
        coverage = min(1.0, object_valid / max(object_raw, 1))
        multiview_support = min(
            1.0,
            len(views) / max(1, request["consistency_configuration"]["minimum_supporting_views"]),
        )
        confidence = float(
            np.cbrt(max(coverage, 1e-8) * max(precision, 1e-8) * max(multiview_support, 1e-8))
        )
        provenance["confidence"]["score"] = confidence
        hypotheses.append(
            {
                **common,
                "status": (
                    "accepted"
                    if len(views)
                    >= int(request["consistency_configuration"]["minimum_supporting_views"])
                    and reprojection_iou
                    >= float(request["reprojection_configuration"]["minimum_accepted_iou"])
                    else "partial"
                ),
                "reason": None,
                "fused_surfel_count": len(fused_points),
                "point_cloud": {
                    "relative_path": relative_points,
                    "sha256": sha256(points_path),
                    "point_count": len(points),
                    "has_normals": True,
                    "has_colors": True,
                },
                "surfel_cloud": {
                    "relative_path": relative_surfels,
                    "sha256": sha256(surfels_path),
                    "point_count": len(fused_points),
                    "has_normals": True,
                    "has_colors": True,
                },
                "observed_surface": None,
                "depth_consistency": coverage,
                "normal_consistency": float(np.mean(np.isfinite(normals).all(axis=1))),
                "reprojection_precision": precision,
                "reprojection_recall": recall,
                "reprojection_iou": reprojection_iou,
                "visible_mask_coverage": recall,
                "connected_component_count": voxel_component_count(voxel_keys),
                "surfel_spacing": spacing,
                "measurement_confidence": confidence,
                "warnings": [
                    "visible surfels only; hidden surfaces and watertight completion are absent",
                    f"median per-view point-splat reprojection IoU: {median_view_iou:.6f}",
                ],
            }
        )
        operation_started = time.monotonic()
        point_image = point_cloud_preview(fused_points)
        object_label = (
            f"{track['object_id']} surfels={len(fused_points)} IoU={reprojection_iou:.3f}"
        )
        object_tiles.append((object_label, point_image))
        object_preview = Image.new("RGB", (700, 460), (245, 245, 245))
        object_preview.paste(Image.fromarray(point_image, mode="RGB"), (0, 40))
        ImageDraw.Draw(object_preview).text((12, 14), object_label, fill=(20, 25, 30))
        object_preview.save(output_dir / "previews" / "objects" / f"{track['object_id']}.png")
        representative = views[len(views) // 2]
        depth_mask = depth_preview(representative.depth)
        mask_outline = cv2.morphologyEx(
            (representative.mask > 0).astype(np.uint8),
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), dtype=np.uint8),
        )
        depth_mask[mask_outline > 0] = (60, 235, 90)
        depth_mask_tiles.append((f"{track['object_id']} / {representative.frame_id}", depth_mask))
        reprojection_tiles.append(
            (
                f"{track['object_id']} / {representative.frame_id}",
                reprojection_overlay(
                    fused_points,
                    representative,
                    {
                        "maximum_relative_depth_residual": request["consistency_configuration"][
                            "maximum_relative_depth_residual"
                        ],
                        "splat_radius_pixels": request["reprojection_configuration"][
                            "splat_radius_pixels"
                        ],
                    },
                ),
            )
        )
        preview_seconds += time.monotonic() - operation_started
    operation_started = time.monotonic()
    contact_sheet(
        output_dir / "previews" / "measured_object_contact_sheet.png",
        "Measured partial objects: PCA diagnostic views",
        object_tiles,
    )
    contact_sheet(
        output_dir / "previews" / "object_point_clouds.png",
        "Measured object surfel clouds (arbitrary axes and scale)",
        object_tiles,
    )
    contact_sheet(
        output_dir / "previews" / "depth_mask_contact_sheet.png",
        "Dense depth with undistorted canonical mask outline",
        depth_mask_tiles,
    )
    contact_sheet(
        output_dir / "previews" / "reprojection_contact_sheet.png",
        "Measured surfel reprojection: green intersection, red false positive, blue missed mask",
        reprojection_tiles,
    )
    preview_seconds += time.monotonic() - operation_started
    statuses = [item["status"] for item in hypotheses]
    write_json(
        output_dir / "geometry_manifest.json",
        {
            "schema_version": "0.1.0",
            "manifest_sha256": request["manifest_sha256"],
            "frame_sequence_digest": request["frame_sequence_digest"],
            "camera_reconstruction_sha256": request["camera_reconstruction_sha256"],
            "segmentation_tracking_sha256": request["segmentation_tracking_sha256"],
            "dense_workspace_manifest_sha256": request["dense_workspace_manifest_sha256"],
            "undistortion_manifest_sha256": request["undistortion_manifest_sha256"],
            "depth_manifest_sha256": request["depth_manifest_sha256"],
            "hypotheses": hypotheses,
            "coordinate_convention": request["coordinate_convention"],
            "scale_status": "scale_ambiguous",
            "generated_geometry_used_as_source": False,
        },
    )
    total = time.monotonic() - started
    peak_host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    write_json(
        output_dir / "diagnostics.json",
        {
            "schema_version": "0.1.0",
            "track_count": len(hypotheses),
            "accepted_object_count": statuses.count("accepted"),
            "partial_object_count": statuses.count("partial"),
            "unresolved_object_count": statuses.count("unresolved"),
            "raw_sample_count": total_raw,
            "validated_sample_count": total_validated,
            "fused_surfel_count": total_surfels,
            "mask_mapping_seconds": mask_mapping_seconds,
            "backprojection_seconds": backprojection_seconds,
            "multiview_validation_seconds": multiview_validation_seconds,
            "surfel_fusion_seconds": surfel_fusion_seconds,
            "observed_mesh_seconds": 0.0,
            "preview_seconds": preview_seconds,
            "total_runtime_seconds": total,
            "peak_gpu_memory_bytes": 0,
            "peak_host_memory_bytes": peak_host,
            "warnings": [],
        },
    )
    write_json(
        output_dir / "worker_manifest.json",
        {
            "schema_version": "0.1.0",
            "worker_version": __version__,
            "backend": "numpy_opencv",
            "request_sha256": sha256(request_path),
            "manifest_sha256": request["manifest_sha256"],
            "frame_sequence_digest": request["frame_sequence_digest"],
            "camera_reconstruction_sha256": request["camera_reconstruction_sha256"],
            "segmentation_tracking_sha256": request["segmentation_tracking_sha256"],
            "depth_manifest_sha256": request["depth_manifest_sha256"],
            "runtime_seconds": total,
            "peak_gpu_memory_bytes": 0,
            "peak_host_memory_bytes": peak_host,
            "raw_output_paths": [],
            "warnings": [],
        },
    )
    return {
        "objects": len(hypotheses),
        "measured_objects": sum(status != "unresolved" for status in statuses),
    }
