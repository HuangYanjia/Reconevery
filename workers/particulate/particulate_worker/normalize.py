from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def _rotation(up: str) -> np.ndarray:
    rotations = {
        "X": [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        "-X": [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        "Y": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        "-Y": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
        "Z": np.eye(3).tolist(),
        "-Z": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
    }
    if up not in rotations:
        raise ValueError(f"unsupported Particulate up axis: {up}")
    return np.asarray(rotations[up], dtype=np.float64)


def source_to_working(mesh: trimesh.Trimesh, up: str) -> tuple[np.ndarray, np.ndarray]:
    rotation = _rotation(up)
    rotated = np.asarray(mesh.vertices, dtype=np.float64) @ rotation.T
    bounds_min = rotated.min(axis=0)
    bounds_max = rotated.max(axis=0)
    center = (bounds_min + bounds_max) / 2.0
    extent = float(np.max(bounds_max - bounds_min))
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Particulate source mesh has collapsed bounds")
    matrix = np.eye(4)
    matrix[:3, :3] = rotation / extent
    matrix[:3, 3] = -center / extent
    return matrix, np.linalg.inv(matrix)


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError("Particulate source must be a non-empty triangle mesh")
    return loaded


def axis_point_from_prediction(
    axis_working: np.ndarray,
    point_working: np.ndarray | None,
    inverse: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    axis_source = inverse[:3, :3] @ axis_working
    axis_source /= np.linalg.norm(axis_source)
    point_source = None
    if point_working is not None:
        point_source = inverse[:3, :3] @ point_working + inverse[:3, 3]
    return axis_source, point_source
