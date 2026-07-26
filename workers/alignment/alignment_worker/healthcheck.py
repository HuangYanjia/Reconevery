from __future__ import annotations

import json
import platform
from pathlib import Path

from alignment_worker.schema import WorkerConfig
from alignment_worker.version import __version__


def run_healthcheck(config_path: Path) -> dict[str, object]:
    config = WorkerConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    if config.worker_version != __version__:
        raise RuntimeError(
            "alignment worker version mismatch: "
            f"expected {config.worker_version}, got {__version__}"
        )
    import cv2
    import numpy
    import nvdiffrast.torch  # noqa: F401
    import scipy
    import torch
    import trimesh

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to the alignment worker")
    payload = {
        "available": True,
        "worker_version": __version__,
        "python_version": platform.python_version(),
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "opencv_version": cv2.__version__,
        "trimesh_version": trimesh.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "backend": config.backend,
    }
    return payload


def print_healthcheck(config_path: Path) -> None:
    print(json.dumps(run_healthcheck(config_path), sort_keys=True))
