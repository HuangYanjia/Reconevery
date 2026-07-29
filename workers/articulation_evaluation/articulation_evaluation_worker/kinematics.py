from __future__ import annotations

import math


def prismatic_candidate_q_scale(global_sim3_scale: float) -> float:
    if not math.isfinite(global_sim3_scale) or global_sim3_scale <= 0:
        raise ValueError("global Sim(3) scale must be finite and positive")
    return 1.0 / global_sim3_scale


def revolute_candidate_q_scale() -> float:
    return 1.0


def normalized_residual(raw_residual: float, normalization_diagonal: float) -> float:
    if not math.isfinite(raw_residual) or raw_residual < 0:
        raise ValueError("residual must be finite and non-negative")
    if not math.isfinite(normalization_diagonal) or normalization_diagonal <= 0:
        raise ValueError("normalization diagonal must be finite and positive")
    return raw_residual / normalization_diagonal


__all__ = [
    "normalized_residual",
    "prismatic_candidate_q_scale",
    "revolute_candidate_q_scale",
]
