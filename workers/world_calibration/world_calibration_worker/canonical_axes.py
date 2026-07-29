from __future__ import annotations

import numpy as np


def canonical_rotation(up: np.ndarray, forward: np.ndarray) -> np.ndarray:
    up = up / np.linalg.norm(up)
    horizontal = forward - float(np.dot(forward, up)) * up
    norm = float(np.linalg.norm(horizontal))
    if norm <= 1e-8:
        raise ValueError("up and forward evidence are nearly parallel")
    forward = horizontal / norm
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    forward = np.cross(left, up)
    rotation = np.stack((forward, left, up), axis=0)
    if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-8:
        raise ValueError("canonical axes are not right handed")
    return rotation


__all__ = ["canonical_rotation"]
