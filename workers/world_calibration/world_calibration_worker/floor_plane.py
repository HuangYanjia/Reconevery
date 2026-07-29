from __future__ import annotations

import numpy as np


def robust_plane(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    if len(points) < 3:
        raise ValueError("floor-plane fitting requires at least three points")
    center = np.median(points, axis=0)
    _, _, right_t = np.linalg.svd(points - center)
    normal = right_t[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(np.dot(normal, center))
    residuals = np.abs(points @ normal + offset)
    cutoff = float(np.percentile(residuals, 80))
    inliers = points[residuals <= cutoff]
    center = np.mean(inliers, axis=0)
    _, _, right_t = np.linalg.svd(inliers - center)
    normal = right_t[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(np.dot(normal, center))
    return normal, offset, np.abs(points @ normal + offset)


__all__ = ["robust_plane"]
