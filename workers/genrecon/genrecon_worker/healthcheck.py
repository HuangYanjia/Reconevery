from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

from genrecon_worker.checkpoint_loader import verify_checkpoints
from genrecon_worker.commit_verification import verify_checkout
from genrecon_worker.runtime_assets import (
    DINOV3_REPOSITORY,
    resolve_runtime_repository_revisions,
)
from genrecon_worker.schema import WorkerConfiguration
from genrecon_worker.version import (
    PYTHON_VERSION,
    PYTORCH_VERSION,
    TORCHVISION_VERSION,
)


def run_healthcheck(config: WorkerConfiguration) -> dict[str, object]:
    checkout = Path(config.official_checkout_path).resolve()
    submodules = verify_checkout(
        checkout,
        config.official_code_commit,
        config.submodule_commits,
    )
    checkpoints = verify_checkpoints(config.checkpoint_paths, config.checkpoint_hashes)
    python_version = platform.python_version()
    if ".".join(python_version.split(".")[:2]) != PYTHON_VERSION:
        raise RuntimeError(f"GenRecon requires Python {PYTHON_VERSION}, found {python_version}")

    import torch
    import torchvision

    if torch.__version__.split("+", 1)[0] != PYTORCH_VERSION:
        raise RuntimeError(f"GenRecon requires torch {PYTORCH_VERSION}, found {torch.__version__}")
    if torchvision.__version__.split("+", 1)[0] != TORCHVISION_VERSION:
        raise RuntimeError(
            f"GenRecon requires torchvision {TORCHVISION_VERSION}, found {torchvision.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("GenRecon requires a CUDA GPU")
    extension_modules = (
        "flash_attn",
        "nvdiffrast.torch",
        "nvdiffrec_render",
        "cumesh",
        "o_voxel",
        "flex_gemm",
    )
    checkout_path = str(checkout)
    sys.path.insert(0, checkout_path)
    try:
        imported = importlib.import_module("genrecon")
    finally:
        sys.path.remove(checkout_path)
    imported_path = Path(imported.__file__ or "").resolve()
    if checkout not in imported_path.parents:
        raise RuntimeError(
            f"official GenRecon import resolved outside the verified checkout: {imported_path}"
        )
    for module in extension_modules:
        importlib.import_module(module)
    runtime_revisions = resolve_runtime_repository_revisions()
    return {
        "available": True,
        "official_code_commit": config.official_code_commit,
        "submodule_commits": submodules,
        "checkpoint_hashes": {record["checkpoint_id"]: record["sha256"] for record in checkpoints},
        "python_version": python_version,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "extensions": list(extension_modules),
        "runtime_model_repository": DINOV3_REPOSITORY,
        "runtime_model_revision": runtime_revisions[DINOV3_REPOSITORY],
        "runtime_repository_revisions": runtime_revisions,
    }
