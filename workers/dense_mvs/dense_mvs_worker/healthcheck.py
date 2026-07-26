from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from dense_mvs_worker.colmap_version import inspect_colmap
from dense_mvs_worker.version import __version__


def healthcheck(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    version, banner, commit_prefix = inspect_colmap(config["executable"])
    if version != config["official_version"]:
        raise RuntimeError(
            f"configured COLMAP is {version}; exact official "
            f"{config['official_version']} is required"
        )
    if not config["official_commit"].startswith(commit_prefix):
        raise RuntimeError(
            f"configured COLMAP binary commit {commit_prefix} does not match "
            f"official pin {config['official_commit']}"
        )
    return {
        "available": True,
        "worker_version": __version__,
        "colmap_version": version,
        "colmap_commit": config["official_commit"],
        "verified_binary_commit_prefix": commit_prefix,
        "banner": banner,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
    }
