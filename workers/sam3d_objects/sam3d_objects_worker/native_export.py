from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def official_anchor_camera(
    intrinsics_tensor: Any,
    *,
    width: int,
    height: int,
) -> dict[str, object]:
    import numpy as np

    if not hasattr(intrinsics_tensor, "detach"):
        raise RuntimeError("official SAM 3D point-map path omitted its intrinsics tensor")
    intrinsics = intrinsics_tensor.detach().cpu().numpy()
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise RuntimeError("official SAM 3D point-map intrinsics are not a finite 3x3 matrix")
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if fx <= 0 or fy <= 0:
        raise RuntimeError("official SAM 3D point-map focal lengths must be positive")
    return {
        "width": width,
        "height": height,
        "normalized_intrinsics": tuple(float(value) for value in intrinsics.reshape(-1)),
        "pixel_intrinsics": (fx * width, fy * height, cx * width, cy * height),
        "camera_axes": "x_right_y_down_z_forward",
        "source": "official_pointmap_intrinsics",
    }


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as file:
        header = file.read(64 * 1024)
    end = header.find(b"end_header")
    match = re.search(rb"(?:^|\n)element vertex ([0-9]+)(?:\r?\n)", header[:end])
    if end < 0 or match is None or int(match.group(1)) <= 0:
        raise RuntimeError("official SAM 3D Objects Gaussian PLY has no vertices")
    return int(match.group(1))


def serializable_layout(output: dict[str, Any]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in ("scale", "rotation", "translation", "layout"):
        value = output.get(key)
        if value is None:
            continue
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        elif hasattr(value, "tolist"):
            value = value.tolist()
        try:
            json.dumps(value)
        except TypeError:
            value = str(value)
        result[key] = value
    return result


def export_native(output: dict[str, Any], native_root: Path) -> list[tuple[Path, str, str]]:
    native_root.mkdir(parents=True, exist_ok=True)
    assets: list[tuple[Path, str, str]] = []
    gaussian = output.get("gs")
    if gaussian is None:
        raw_gaussian = output.get("gaussian")
        if isinstance(raw_gaussian, (list, tuple)) and raw_gaussian:
            gaussian = raw_gaussian[0]
    if gaussian is None or not hasattr(gaussian, "save_ply"):
        raise RuntimeError("official SAM 3D Objects output omitted its Gaussian representation")
    gaussian_path = native_root / "gaussians.ply"
    gaussian.save_ply(str(gaussian_path))
    assets.append((gaussian_path, "gaussian_splat_ply", "official_gaussian_splat"))
    glb = output.get("glb")
    if glb is not None and hasattr(glb, "export"):
        mesh_path = native_root / "visual_asset.glb"
        glb.export(str(mesh_path))
        assets.append((mesh_path, "pbr_glb", "official_optional_visual_glb"))
    return assets
