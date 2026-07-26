from __future__ import annotations

import numpy as np


def relative_depth_metrics(
    candidate_depth: np.ndarray,
    measured_depth: np.ndarray,
    valid: np.ndarray,
    *,
    inlier_threshold: float,
) -> tuple[float, float]:
    selected = valid & np.isfinite(candidate_depth) & np.isfinite(measured_depth)
    if not selected.any():
        # A finite sentinel keeps the canonical JSON standards-compliant while
        # making the preconfigured residual gate fail unambiguously.
        return 1_000_000.0, 0.0
    residual = np.abs(candidate_depth[selected] - measured_depth[selected]) / np.maximum(
        np.abs(measured_depth[selected]), 1e-8
    )
    return float(np.median(residual)), float(np.mean(residual <= inlier_threshold))
