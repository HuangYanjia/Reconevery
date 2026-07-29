from __future__ import annotations

import numpy as np


def combine_up_vectors(records: list[dict[str, object]]) -> tuple[np.ndarray, float]:
    if not records:
        raise ValueError("no gravity evidence")
    vectors = np.asarray([record["up_vector_colmap"] for record in records], dtype=np.float64)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    reference = vectors[0]
    angles = np.degrees(np.arccos(np.clip(vectors @ reference, -1.0, 1.0)))
    if float(np.max(angles)) > 10.0:
        raise ValueError("incompatible high-trust gravity evidence")
    combined = np.median(vectors, axis=0)
    combined /= np.linalg.norm(combined)
    residual = float(np.max(np.degrees(np.arccos(np.clip(vectors @ combined, -1.0, 1.0)))))
    return combined, residual


__all__ = ["combine_up_vectors"]
