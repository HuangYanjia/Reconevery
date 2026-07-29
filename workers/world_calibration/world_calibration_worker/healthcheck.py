from __future__ import annotations

import platform

import cv2
import numpy
import scipy

from world_calibration_worker import (
    OFFICIAL_APRILTAG_COMMIT,
    OFFICIAL_APRILTAG_LICENSE,
    OFFICIAL_APRILTAG_REPOSITORY,
    OFFICIAL_APRILTAG_VERSION,
)


def health() -> dict[str, object]:
    official_binding_available = False
    official_binding_error = None
    try:
        from apriltag import apriltag as official_apriltag  # type: ignore[import-not-found]

        official_binding_available = callable(official_apriltag)
    except (ImportError, OSError) as exc:
        official_binding_error = str(exc)
    return {
        "ok": official_binding_available,
        "official_apriltag_repository": OFFICIAL_APRILTAG_REPOSITORY,
        "official_apriltag_commit": OFFICIAL_APRILTAG_COMMIT,
        "official_apriltag_version": OFFICIAL_APRILTAG_VERSION,
        "official_apriltag_license": OFFICIAL_APRILTAG_LICENSE,
        "runtime": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "opencv": cv2.__version__,
            "cuda": "not_used",
        },
        "official_binding_available": official_binding_available,
        "official_binding_error": official_binding_error,
        "capabilities": [
            "official_apriltag",
            "known_distance_triangulation",
            "gravity",
            "canonical_sim3",
        ],
    }


__all__ = ["health"]
