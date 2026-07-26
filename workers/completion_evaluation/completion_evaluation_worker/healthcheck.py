from __future__ import annotations

import platform

from completion_evaluation_worker.version import WORKER_VERSION


def run_healthcheck() -> dict[str, object]:
    import cv2
    import numpy
    import scipy
    import torch
    import trimesh

    if not torch.cuda.is_available():
        raise RuntimeError("completion evaluation requires CUDA")
    import nvdiffrast.torch  # noqa: F401

    return {
        "available": True,
        "worker_version": WORKER_VERSION,
        "python_version": platform.python_version(),
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "opencv_version": cv2.__version__,
        "trimesh_version": trimesh.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
    }
