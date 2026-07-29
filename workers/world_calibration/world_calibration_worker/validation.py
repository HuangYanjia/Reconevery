from __future__ import annotations

import numpy as np


def sim3_diagnostics(matrix: np.ndarray) -> tuple[float, float, float]:
    scale = float(np.linalg.norm(matrix[:3, 0]))
    rotation = matrix[:3, :3] / scale
    determinant = float(np.linalg.det(rotation))
    orthonormal_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    roundtrip = float(np.max(np.abs(matrix @ np.linalg.inv(matrix) - np.eye(4))))
    return determinant, orthonormal_error, roundtrip


__all__ = ["sim3_diagnostics"]
