from __future__ import annotations

import json
import platform
from importlib import metadata
from pathlib import Path
from typing import Any

from sam3_worker.model_loader import (
    check_checkpoint_access,
    validate_official_code_commit,
)
from sam3_worker.schema import WorkerConfiguration
from sam3_worker.version import (
    PYTORCH_VERSION,
    TORCHVISION_VERSION,
    WORKER_VERSION,
)


def _version_tuple(value: str) -> tuple[int, ...]:
    components: list[int] = []
    for part in value.split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        components.append(int(digits))
    return tuple(components)


def run_healthcheck(config_path: Path) -> dict[str, Any]:
    config = WorkerConfiguration.model_validate_json(config_path.read_text(encoding="utf-8"))
    import sam3  # noqa: F401
    import torch
    import torchvision

    if torch.__version__.split("+", 1)[0] != PYTORCH_VERSION:
        raise RuntimeError(f"PyTorch {PYTORCH_VERSION} is required, found {torch.__version__}")
    if torchvision.__version__.split("+", 1)[0] != TORCHVISION_VERSION:
        raise RuntimeError(
            f"torchvision {TORCHVISION_VERSION} is required, found {torchvision.__version__}"
        )
    if config.device != "cuda":
        raise RuntimeError("the pinned official SAM backend requires device=cuda")
    if config.precision != "bfloat16":
        raise RuntimeError(
            "the pinned official video predictor currently supports precision=bfloat16"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; a compatible NVIDIA GPU is required")
    cuda_version = torch.version.cuda
    if cuda_version is None or _version_tuple(cuda_version) < (12, 6):
        raise RuntimeError(f"CUDA 12.6 or newer is required, found {cuda_version}")
    commit = validate_official_code_commit(config)
    checkpoint = check_checkpoint_access(config)
    return {
        "available": True,
        "worker_version": WORKER_VERSION,
        "official_code_commit": commit,
        "checkpoint_repository": config.checkpoint_repository,
        "checkpoint_revision": config.checkpoint_revision,
        "checkpoint_access_mode": checkpoint["access_mode"],
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": cuda_version,
        "device_name": torch.cuda.get_device_name(0),
        "sam_package_version": metadata.version("sam3"),
    }


def format_healthcheck(config_path: Path) -> str:
    return json.dumps(run_healthcheck(config_path), sort_keys=True)
