from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from measured_geometry_worker.version import __version__


def healthcheck(config_path: Path) -> dict[str, object]:
    json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "worker_version": __version__,
        "backend": "numpy_opencv",
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
    }
