from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sam3d_objects_worker.commit_verification import verify_checkout


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
    # nvdiffrast framebuffers are bottom-origin; target cameras use top-origin pixels.
    valid_tensor = torch.flip(raster[0, ..., 3] > 0, dims=(0,))
    depth_values, _ = dr.interpolate(
        torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32))
        .to(device=device)
        .reshape(1, len(vertices), 1)
        .contiguous(),
        raster,
        face_tensor,
    )
    depth = torch.flip(depth_values[0, ..., 0], dims=(0,)).detach().cpu().numpy()
    valid = valid_tensor.detach().cpu().numpy()
    depth[~valid] = np.nan
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[valid] = (180, 190, 210, 255)
    return rgba, depth, valid


def _render_gaussian(
    asset_path: Path,
    matrix_world_from_candidate: np.ndarray,
    camera: dict[str, Any],
    *,
    checkout: Path,
    expected_commit: str,
    alpha_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    verify_checkout(checkout, expected_commit)
    os.environ["LIDRA_SKIP_INIT"] = "true"
    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))
    import torch
    from easydict import EasyDict
    from plyfile import PlyData
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply
    from sam3d_objects.model.backbone.tdfy_dit.renderers.gaussian_render import (
        render as official_render,
    )
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian import (
        Gaussian,
    )

    ply = PlyData.read(asset_path)
    vertex = ply["vertex"]
    xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float32)
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("native Gaussian PLY has no finite Gaussian centers")
    minimum = xyz.min(axis=0)
    extent = np.maximum(xyz.max(axis=0) - minimum, 1e-6)
    rest_count = sum(prop.name.startswith("f_rest_") for prop in vertex.properties)
    sh_degree = int(round(np.sqrt(rest_count / 3 + 1) - 1)) if rest_count else 0
    gaussian = Gaussian(
        aabb=[*minimum.tolist(), *extent.tolist()],
        sh_degree=sh_degree,
        device="cuda",
    )
    gaussian.load_ply(str(asset_path))

    matrix = torch.as_tensor(matrix_world_from_candidate, dtype=torch.float32, device="cuda")
    linear = matrix[:3, :3]
    scale = torch.linalg.det(linear).clamp_min(1e-12).pow(1 / 3)
    rotation = linear / scale
    if not torch.allclose(
        rotation.T @ rotation,
        torch.eye(3, device="cuda"),
        rtol=1e-4,
        atol=1e-4,
    ):
        raise ValueError("Gaussian target transform is not a uniform positive-scale Sim(3)")
    transformed_xyz = scale * (gaussian.get_xyz @ rotation.T) + matrix[:3, 3]
    gaussian.from_xyz(transformed_xyz)
    gaussian.from_scaling(gaussian.get_scaling * scale)
    rotation_quaternion = matrix_to_quaternion(rotation.unsqueeze(0))
    gaussian.from_rotation(
        quaternion_multiply(
            rotation_quaternion.expand_as(gaussian.get_rotation),
            gaussian.get_rotation,
        )
    )

    width, height = int(camera["width"]), int(camera["height"])
    fx, fy, cx, cy = (float(camera[name]) for name in ("fx", "fy", "cx", "cy"))
    intrinsics = torch.tensor(
        [
            [fx / width, 0.0, cx / width],
            [0.0, fy / height, cy / height],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    camera_from_world = torch.as_tensor(
        _matrix(camera["camera_from_world"], "camera_from_world"),
        dtype=torch.float32,
        device="cuda",
    )
    camera_dict = EasyDict(
        image_height=height,
        image_width=width,
        world_view_transform=camera_from_world.T.contiguous(),
        camera_center=torch.linalg.inv(camera_from_world)[:3, 3],
        intrinsics=intrinsics,
    )
    pipe = EasyDict(
        kernel_size=0.1,
        convert_SHs_python=False,
        compute_cov3D_python=False,
        scale_modifier=1.0,
        debug=False,
    )
    result = official_render(
        camera_dict,
        gaussian,
        pipe,
        torch.zeros(3, dtype=torch.float32, device="cuda"),
        backend="gsplat",
    )
    if "alpha" not in result:
        raise RuntimeError("pinned official Gaussian renderer did not return alpha")
    alpha = result["alpha"].squeeze().detach().clamp(0, 1).cpu().numpy()
    color = result["render"].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    valid = alpha >= alpha_threshold
    rgba = np.concatenate((color, alpha[..., None]), axis=2)
    return (
        np.rint(rgba * 255).astype(np.uint8),
        valid,
        {
            "official_renderer_source_path": (
                "sam3d_objects/model/backbone/tdfy_dit/renderers/gaussian_render.py"
            ),
            "official_renderer_backend": "gsplat",
            "official_code_commit": expected_commit,
            "dependency_license_identity": "SAM License plus pinned gsplat dependencies",
            "camera_conversion": "opencv_pixel_intrinsics_to_normalized_gsplat",
            "valid_gaussian_count": int(len(xyz)),
            "rendered_alpha_count": int(np.count_nonzero(valid)),
            "depth_supported": False,
        },
    )


def render_candidate(request_path: Path, input_root: Path, output_dir: Path) -> None:
    started = time.monotonic()
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
    asset_format = str(request.get("asset_format", "mesh_glb"))
    gaussian_metadata: dict[str, Any] | None = None
    if asset_format == "gaussian_splat_ply":
        checkout = Path(str(request["official_checkout_path"]))
        rgba, valid, gaussian_metadata = _render_gaussian(
            asset_path,
            matrix,
            camera,
            checkout=checkout,
            expected_commit=str(request["official_code_commit"]),
            alpha_threshold=float(request.get("alpha_threshold", 1e-3)),
        )
        depth = None
    else:
        rgba, depth, valid = _render_mesh(asset_path, matrix, camera)

    output_dir.mkdir(parents=True, exist_ok=True)
    rgba_path = output_dir / "rgba.png"
    valid_path = output_dir / "valid.png"
    Image.fromarray(rgba, mode="RGBA").save(rgba_path, format="PNG", optimize=False)
    depth_path = output_dir / "depth.npy"
    if depth is not None:
        np.save(depth_path, depth, allow_pickle=False)
    Image.fromarray(valid.astype(np.uint8) * 255, mode="L").save(
        valid_path,
        format="PNG",
        optimize=False,
    )
    manifest = {
        "schema_version": "0.1.0",
        "renderer": (
            "official_sam3d_gaussian_gsplat" if gaussian_metadata is not None else "nvdiffrast"
        ),
        "asset_path": str(request["asset_path"]),
        "asset_format": asset_format,
        "rgba_path": rgba_path.name,
        "depth_path": depth_path.name if depth is not None else None,
        "valid_path": valid_path.name,
        "width": int(camera["width"]),
        "height": int(camera["height"]),
        "valid_pixel_count": int(valid.sum()),
        "runtime_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": _peak_gpu_memory(),
        "sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (rgba_path, valid_path, *([depth_path] if depth is not None else []))
        },
    }
    if gaussian_metadata is not None:
        manifest.update(gaussian_metadata)
    (output_dir / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _peak_gpu_memory() -> int | None:
    try:
        import torch

        return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    except ImportError:
        return None
