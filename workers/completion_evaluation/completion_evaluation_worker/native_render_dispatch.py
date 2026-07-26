from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NativeRender:
    rgba: np.ndarray
    depth: np.ndarray
    valid: np.ndarray
    renderer: str


def render_mesh_candidate(
    asset_path: Path,
    matrix_world_from_candidate: np.ndarray,
    camera: dict[str, object],
) -> NativeRender:
    import torch
    import trimesh

    try:
        import nvdiffrast.torch as dr
    except ImportError as exc:
        raise RuntimeError("nvdiffrast is required for held-out mesh rendering") from exc
    mesh = trimesh.load(asset_path, force="mesh", process=False)
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"candidate mesh has invalid vertices: {asset_path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"candidate mesh has invalid triangle faces: {asset_path}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"candidate mesh contains non-finite vertices: {asset_path}")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError(f"candidate mesh contains out-of-range face indices: {asset_path}")
    world = (
        matrix_world_from_candidate[:3, :3] @ vertices.T + matrix_world_from_candidate[:3, 3:4]
    ).T
    camera_from_world = np.asarray(camera["camera_from_world"], dtype=np.float32)
    homogeneous = np.concatenate([world, np.ones((len(world), 1), np.float32)], axis=1)
    camera_vertices = (camera_from_world @ homogeneous.T).T[:, :3]
    width, height = int(camera["width"]), int(camera["height"])
    fx, fy, cx, cy = (
        float(camera["fx"]),
        float(camera["fy"]),
        float(camera["cx"]),
        float(camera["cy"]),
    )
    near, far = float(camera["near"]), float(camera["far"])
    if width <= 0 or height <= 0:
        raise ValueError("target camera dimensions must be positive")
    if not (0 < near < far):
        raise ValueError("target camera clipping planes must satisfy 0 < near < far")
    x, y, z = camera_vertices.T
    clip = np.ascontiguousarray(
        np.stack(
            [
                (2 * fx / width) * x + (2 * (cx + 0.5) / width - 1) * z,
                (-2 * fy / height) * y + (1 - 2 * (cy + 0.5) / height) * z,
                ((far + near) / (far - near)) * z - 2 * far * near / (far - near),
                z,
            ],
            axis=1,
        ),
        dtype=np.float32,
    )
    if clip.shape != (len(vertices), 4) or not np.isfinite(clip).all():
        raise ValueError(f"candidate mesh produced invalid clip coordinates: {asset_path}")
    device = torch.device("cuda")
    clip_tensor = torch.from_numpy(clip).to(device=device).unsqueeze(0).contiguous()
    face_tensor = torch.from_numpy(faces).to(device=device).contiguous()
    context = dr.RasterizeCudaContext(device=device)
    try:
        raster, _ = dr.rasterize(context, clip_tensor, face_tensor, (height, width))
    except RuntimeError as exc:
        raise RuntimeError(
            "candidate rasterization failed for "
            f"{asset_path} with pos={tuple(clip_tensor.shape)}, "
            f"tri={tuple(face_tensor.shape)}, resolution={(height, width)}"
        ) from exc
    # nvdiffrast framebuffers are bottom-origin; canonical RGB/masks are top-origin.
    triangle = torch.flip(raster[0, ..., 3] > 0, dims=(0,))
    z_tensor = (
        torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32))
        .to(device=device)
        .reshape(1, len(vertices), 1)
        .contiguous()
    )
    depth_values, _ = dr.interpolate(
        z_tensor,
        raster,
        face_tensor,
    )
    depth = torch.flip(depth_values[0, ..., 0], dims=(0,)).detach().cpu().numpy()
    valid = triangle.detach().cpu().numpy()
    depth[~valid] = np.nan
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[valid] = (180, 190, 210, 255)
    return NativeRender(rgba=rgba, depth=depth, valid=valid, renderer="nvdiffrast")
