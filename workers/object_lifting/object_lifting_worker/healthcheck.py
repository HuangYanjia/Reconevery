from __future__ import annotations

import json
import platform
from pathlib import Path

from object_lifting_worker.schema import WorkerConfig
from object_lifting_worker.version import __version__


def run_healthcheck(config_path: Path) -> dict[str, object]:
    config = WorkerConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    if config.worker_version != __version__:
        raise RuntimeError(
            f"worker version mismatch: requested {config.worker_version}, installed {__version__}"
        )
    import cv2
    import numpy
    import nvdiffrast
    import nvdiffrast.torch as dr
    import torch
    import trimesh

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to the object-lifting worker")
    device_name = torch.cuda.get_device_name(0)
    context = dr.RasterizeCudaContext(device=torch.device("cuda"))
    del context
    return {
        "available": True,
        "backend": config.backend,
        "worker_version": __version__,
        "python_version": platform.python_version(),
        "numpy_version": numpy.__version__,
        "opencv_version": cv2.__version__,
        "trimesh_version": trimesh.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "nvdiffrast_version": getattr(nvdiffrast, "__version__", "unknown"),
        "device_name": device_name,
    }


def healthcheck_json(config_path: Path) -> str:
    return json.dumps(run_healthcheck(config_path), sort_keys=True)
