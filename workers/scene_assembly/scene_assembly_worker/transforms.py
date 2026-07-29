from __future__ import annotations

import numpy as np


def matrix4(values: list[float]) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise ValueError("asset transform contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("asset transform is not affine")
    return matrix


__all__ = ["matrix4"]
