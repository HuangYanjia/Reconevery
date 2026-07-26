from __future__ import annotations

import json
import platform
from pathlib import Path

from sam3d_objects_worker.checkpoint_loader import verify_checkpoint_files
from sam3d_objects_worker.commit_verification import verify_checkout
from sam3d_objects_worker.official_api import load_inference_class
from sam3d_objects_worker.version import WORKER_VERSION


def run_healthcheck(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    generation = config.get("generation_configuration", {})
    checkout_value = generation.get("official_checkout_path")
    checkpoint_value = generation.get("checkpoint_root")
    if checkout_value is None or checkpoint_value is None:
        return {
            "available": False,
            "worker_version": WORKER_VERSION,
            "reason": "official_checkout_path and checkpoint_root are required",
        }
    commit = verify_checkout(Path(checkout_value), config["official_code_commit"])
    checkpoint_root = Path(checkpoint_value)
    hashes = verify_checkpoint_files(
        checkpoint_root,
        config.get("checkpoint_hashes", {}),
    )
    pipeline_value = generation.get("pipeline_config")
    if pipeline_value is None:
        raise RuntimeError("pipeline_config is required")
    pipeline_config = Path(pipeline_value)
    if not pipeline_config.is_file() or not pipeline_config.resolve().is_relative_to(
        checkpoint_root.resolve()
    ):
        raise RuntimeError("pipeline_config must be a file inside the verified checkpoint root")
    load_inference_class(Path(checkout_value))
    import nvdiffrast.torch as dr
    import torch
    import trimesh

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return {
        "available": True,
        "worker_version": WORKER_VERSION,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "nvdiffrast_import": dr.__name__,
        "trimesh_version": trimesh.__version__,
        "official_code_commit": commit,
        "checkpoint_revision": config["checkpoint_revision"],
        "checkpoint_hashes": hashes,
    }
