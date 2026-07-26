from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _inside(root: Path, relative_path: str) -> Path:
    if Path(relative_path).is_absolute():
        raise ValueError("render asset paths must be relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative_path).resolve(strict=True)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"render asset escapes the input root: {relative_path}")
    return resolved


def _matrix(value: object, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    return matrix


def _render_mesh(
    asset_path: Path,
    matrix_world_from_candidate: np.ndarray,
    camera: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    import trimesh

    try:
        import nvdiffrast.torch as dr
    except ImportError as exc:
        raise RuntimeError("nvdiffrast is required for target-camera rendering") from exc

    mesh = trimesh.load(asset_path, force="mesh", process=False)
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("render asset has no valid vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("render asset has no triangle faces")
    if not np.isfinite(vertices).all():
        raise ValueError("render asset contains non-finite vertices")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError("render asset has out-of-range face indices")

    world = (
        matrix_world_from_candidate[:3, :3] @ vertices.T + matrix_world_from_candidate[:3, 3:4]
    ).T
    camera_from_world = _matrix(camera["camera_from_world"], "camera_from_world")
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
    if width <= 0 or height <= 0 or not (0 < near < far):
        raise ValueError("target camera dimensions or clipping planes are invalid")

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
    if not np.isfinite(clip).all():
        raise ValueError("render asset produced non-finite clip coordinates")

    device = torch.device("cuda")
    clip_tensor = torch.from_numpy(clip).to(device=device).unsqueeze(0).contiguous()
    face_tensor = torch.from_numpy(faces).to(device=device).contiguous()
    context = dr.RasterizeCudaContext(device=device)
    raster, _ = dr.rasterize(context, clip_tensor, face_tensor, (height, width))
    valid_tensor = raster[0, ..., 3] > 0
    depth_values, _ = dr.interpolate(
        torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32))
        .to(device=device)
        .reshape(1, len(vertices), 1)
        .contiguous(),
        raster,
        face_tensor,
    )
    depth = depth_values[0, ..., 0].detach().cpu().numpy()
    valid = valid_tensor.detach().cpu().numpy()
    depth[~valid] = np.nan
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[valid] = (180, 190, 210, 255)
    return rgba, depth, valid


def render_candidate(request_path: Path, input_root: Path, output_dir: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, dict) or request.get("schema_version") != "0.1.0":
        raise ValueError("unsupported target-camera render request")
    camera = request.get("camera")
    if not isinstance(camera, dict):
        raise ValueError("render request camera must be an object")
    asset_path = _inside(input_root, str(request["asset_path"]))
    matrix = _matrix(
        request["matrix_world_from_candidate"],
        "matrix_world_from_candidate",
    )
    rgba, depth, valid = _render_mesh(asset_path, matrix, camera)

    output_dir.mkdir(parents=True, exist_ok=True)
    rgba_path = output_dir / "rgba.png"
    depth_path = output_dir / "depth.npy"
    valid_path = output_dir / "valid.png"
    Image.fromarray(rgba, mode="RGBA").save(rgba_path, format="PNG", optimize=False)
    np.save(depth_path, depth, allow_pickle=False)
    Image.fromarray(valid.astype(np.uint8) * 255, mode="L").save(
        valid_path,
        format="PNG",
        optimize=False,
    )
    manifest = {
        "schema_version": "0.1.0",
        "renderer": "nvdiffrast",
        "asset_path": str(request["asset_path"]),
        "rgba_path": rgba_path.name,
        "depth_path": depth_path.name,
        "valid_path": valid_path.name,
        "width": int(camera["width"]),
        "height": int(camera["height"]),
        "valid_pixel_count": int(valid.sum()),
        "sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (rgba_path, depth_path, valid_path)
        },
    }
    (output_dir / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
