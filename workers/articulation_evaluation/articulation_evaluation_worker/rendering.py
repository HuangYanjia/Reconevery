from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def rotation_xyzw(values: list[float]) -> np.ndarray:
    x, y, z, w = np.asarray(values, dtype=np.float64)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0:
        raise ValueError("camera quaternion is zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def camera_world_matrices(camera: dict[str, Any]) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    for pose in camera["poses"]:
        transform = pose["transform_world_from_camera"]
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation_xyzw(transform["rotation_xyzw"])
        matrix[:3, 3] = transform["translation"]
        matrices[str(pose["frame_id"])] = matrix
    return matrices


def undistort_mask(path: Path, record: dict[str, Any]) -> np.ndarray:
    source = np.asarray(Image.open(path).convert("L"))
    source_width, source_height = record["source_dimensions"]
    if source.shape != (source_height, source_width):
        raise ValueError(f"mask dimensions do not match source camera: {path}")
    fx, fy, cx, cy = record["source_intrinsics"]
    dense_fx, dense_fy, dense_cx, dense_cy = record["dense_intrinsics"]
    dense_width, dense_height = record["dense_dimensions"]
    source_k = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    dense_k = np.asarray(
        [[dense_fx, 0, dense_cx], [0, dense_fy, dense_cy], [0, 0, 1]],
        np.float64,
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        source_k,
        np.asarray(record["source_distortion"], dtype=np.float64),
        None,
        dense_k,
        (dense_width, dense_height),
        cv2.CV_32FC1,
    )
    remapped = cv2.remap(
        source,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return remapped >= 128


def render_mesh_depth(
    path: Path,
    matrix_reference_from_link: np.ndarray,
    camera_from_reference: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    dimensions: tuple[int, int],
) -> np.ndarray:
    import torch
    import trimesh

    try:
        import nvdiffrast.torch as dr
    except ImportError as exc:
        raise RuntimeError("nvdiffrast is required for articulated rendering") from exc
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.to_geometry()
    vertices = np.ascontiguousarray(loaded.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(
        getattr(loaded, "faces", np.empty((0, 3), dtype=np.int32)),
        dtype=np.int32,
    )
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError(f"invalid articulated link vertices: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        return render_point_depth(
            vertices,
            matrix_reference_from_link,
            camera_from_reference,
            intrinsics,
            dimensions,
        )
    homogeneous = np.column_stack((vertices, np.ones(len(vertices), dtype=np.float32)))
    camera_vertices = (camera_from_reference @ matrix_reference_from_link @ homogeneous.T).T[:, :3]
    width, height = dimensions
    fx, fy, cx, cy = intrinsics
    positive = camera_vertices[:, 2][camera_vertices[:, 2] > 1e-8]
    if not len(positive):
        return np.full((height, width), np.nan, dtype=np.float32)
    near = max(float(np.percentile(positive, 1)) * 0.25, 1e-6)
    far = max(float(np.percentile(positive, 99)) * 2.0, near * 100.0)
    x, y, z = camera_vertices.T
    clip = np.ascontiguousarray(
        np.stack(
            (
                (2 * fx / width) * x + (2 * (cx + 0.5) / width - 1) * z,
                (-2 * fy / height) * y + (1 - 2 * (cy + 0.5) / height) * z,
                ((far + near) / (far - near)) * z - (2 * far * near / (far - near)),
                z,
            ),
            axis=1,
        ),
        dtype=np.float32,
    )
    device = torch.device("cuda")
    positions = torch.from_numpy(clip).to(device).unsqueeze(0).contiguous()
    triangles = torch.from_numpy(faces).to(device).contiguous()
    raster, _ = dr.rasterize(
        dr.RasterizeCudaContext(device=device),
        positions,
        triangles,
        (height, width),
    )
    depths, _ = dr.interpolate(
        torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32))
        .to(device)
        .reshape(1, len(z), 1),
        raster,
        triangles,
    )
    valid = torch.flip(raster[0, ..., 3] > 0, dims=(0,))
    depth = torch.flip(depths[0, ..., 0], dims=(0,)).detach().cpu().numpy()
    depth[~valid.detach().cpu().numpy()] = np.nan
    return depth


def render_point_depth(
    points: np.ndarray,
    matrix_reference_from_link: np.ndarray,
    camera_from_reference: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    dimensions: tuple[int, int],
    radius: int = 1,
) -> np.ndarray:
    width, height = dimensions
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    camera = (camera_from_reference @ matrix_reference_from_link @ homogeneous.T).T[:, :3]
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-8)
    fx, fy, cx, cy = intrinsics
    columns = np.rint(fx * camera[valid, 0] / camera[valid, 2] + cx).astype(int)
    rows = np.rint(fy * camera[valid, 1] / camera[valid, 2] + cy).astype(int)
    depths = camera[valid, 2]
    result = np.full((height, width), np.nan, dtype=np.float32)
    order = np.argsort(depths, kind="stable")[::-1]
    for index in order:
        column, row, depth = columns[index], rows[index], depths[index]
        if column < 0 or column >= width or row < 0 or row >= height:
            continue
        left, right = max(0, column - radius), min(width, column + radius + 1)
        top, bottom = max(0, row - radius), min(height, row + radius + 1)
        patch = result[top:bottom, left:right]
        replace = ~np.isfinite(patch) | (depth < patch)
        patch[replace] = depth
    return result


def classify_depth(
    candidate_depth: np.ndarray,
    scene_depth: np.ndarray,
    target_mask: np.ndarray,
    relative_tolerance: float = 0.03,
) -> dict[str, np.ndarray]:
    candidate = np.isfinite(candidate_depth) & (candidate_depth > 0)
    scene = np.isfinite(scene_depth) & (scene_depth > 0)
    tolerance = relative_tolerance * np.maximum(scene_depth, 1e-8)
    occluded = candidate & scene & (candidate_depth > scene_depth + tolerance)
    visible = candidate & ~occluded
    return {
        "visible": visible,
        "occluded": occluded,
        "negative": candidate & ~target_mask & ~occluded,
        "front": candidate & ~target_mask & scene & (candidate_depth < scene_depth - tolerance),
    }


def mask_metrics(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    intersection = int(np.count_nonzero(predicted & target))
    predicted_count = int(np.count_nonzero(predicted))
    target_count = int(np.count_nonzero(target))
    union = predicted_count + target_count - intersection
    return (
        intersection / max(predicted_count, 1),
        intersection / max(target_count, 1),
        intersection / max(union, 1),
    )


def depth_metrics(
    candidate: np.ndarray,
    scene: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float]:
    selected = valid & np.isfinite(candidate) & np.isfinite(scene)
    if not selected.any():
        return 1_000_000.0, 0.0
    residual = np.abs(candidate[selected] - scene[selected]) / np.maximum(
        np.abs(scene[selected]), 1e-8
    )
    return float(np.median(residual)), float(np.mean(residual <= 0.08))
