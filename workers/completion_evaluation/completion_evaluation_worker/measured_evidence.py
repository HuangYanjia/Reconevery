from pathlib import Path

import numpy as np

from completion_evaluation_worker.candidate_io import read_ascii_ply_points_and_normals


def load_measured_points(path: Path) -> np.ndarray:
    points, _ = read_ascii_ply_points_and_normals(path)
    if len(points) < 3:
        raise ValueError("candidate registration requires at least three measured points")
    return points


def load_measured_evidence(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    points, normals = read_ascii_ply_points_and_normals(path)
    if len(points) < 3:
        raise ValueError("candidate registration requires at least three measured points")
    return points, normals
