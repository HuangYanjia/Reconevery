from __future__ import annotations

from typing import Any


def cauchy_loss(residuals: Any, scale: float) -> float:
    import numpy as np

    values = np.asarray(residuals, dtype=np.float64) / max(scale, 1e-12)
    return float(np.mean(np.log1p(values * values))) if len(values) else float("inf")


def robust_inlier_mask(distances: Any, multiplier: float) -> Any:
    import numpy as np

    values = np.asarray(distances, dtype=np.float64)
    if not len(values):
        return np.zeros(0, dtype=bool)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = median + multiplier * max(1.4826 * mad, median * 0.05, 1e-12)
    return values <= threshold
