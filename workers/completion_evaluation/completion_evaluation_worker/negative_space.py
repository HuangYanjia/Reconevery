from __future__ import annotations

import numpy as np


def classify_candidate_pixels(
    candidate_depth: np.ndarray,
    scene_depth: np.ndarray,
    object_mask: np.ndarray,
    *,
    relative_tolerance: float,
) -> dict[str, np.ndarray]:
    candidate = np.isfinite(candidate_depth) & (candidate_depth > 0)
    scene = np.isfinite(scene_depth) & (scene_depth > 0)
    tolerance = relative_tolerance * np.maximum(scene_depth, 1e-8)
    occluded = candidate & scene & (candidate_depth > scene_depth + tolerance)
    front = candidate & ~object_mask & scene & (candidate_depth < scene_depth - tolerance)
    negative = candidate & ~object_mask & ~occluded
    visible = candidate & ~occluded
    return {
        "visible": visible,
        "occluded": occluded,
        "negative": negative,
        "front": front,
    }
