from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def read_ascii_ply_points_and_normals(
    path: Path,
) -> tuple[np.ndarray, np.ndarray | None]:
    lines = path.read_text(encoding="ascii").splitlines()
    end = lines.index("end_header")
    vertex_line = next(line for line in lines[:end] if line.startswith("element vertex "))
    count = int(vertex_line.rsplit(" ", 1)[1])
    properties = [
        line.rsplit(" ", 1)[1]
        for line in lines[:end]
        if line.startswith("property ") and " list " not in line
    ]
    rows = [[float(value) for value in line.split()] for line in lines[end + 1 : end + 1 + count]]
    points = [[row[properties.index(axis)] for axis in ("x", "y", "z")] for row in rows]
    result = np.asarray(points, dtype=np.float64)
    if result.shape != (count, 3) or not np.isfinite(result).all():
        raise ValueError(f"candidate point data is invalid: {path}")
    if not {"nx", "ny", "nz"}.issubset(properties):
        return result, None
    normals = np.asarray(
        [[row[properties.index(axis)] for axis in ("nx", "ny", "nz")] for row in rows],
        dtype=np.float64,
    )
    lengths = np.linalg.norm(normals, axis=1)
    if normals.shape != (count, 3) or not np.isfinite(normals).all() or np.any(lengths <= 1e-12):
        raise ValueError(f"candidate normal data is invalid: {path}")
    return result, normals / lengths[:, None]


def read_ascii_ply_points(path: Path) -> np.ndarray:
    return read_ascii_ply_points_and_normals(path)[0]


def load_candidate_surface_with_normals(
    path: Path,
    *,
    maximum_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if path.suffix.lower() == ".ply":
        return read_ascii_ply_points_and_normals(path)
    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"candidate is not a non-empty triangle mesh: {path}")
    rng = np.random.default_rng(seed)
    count = min(maximum_samples, max(10_000, len(loaded.faces)))
    samples, face_ids = trimesh.sample.sample_surface(loaded, count, seed=rng)
    normals = np.asarray(loaded.face_normals[face_ids], dtype=np.float64)
    return np.asarray(samples, dtype=np.float64), normals


def load_candidate_surface(path: Path, *, maximum_samples: int, seed: int) -> np.ndarray:
    return load_candidate_surface_with_normals(
        path,
        maximum_samples=maximum_samples,
        seed=seed,
    )[0]
