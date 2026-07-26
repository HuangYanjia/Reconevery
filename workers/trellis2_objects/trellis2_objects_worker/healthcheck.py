from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

from trellis2_objects_worker.commit_verification import EXPECTED_SUBMODULES, verify_checkout
from trellis2_objects_worker.runtime_assets import verify_assets
from trellis2_objects_worker.version import WORKER_VERSION


def run_healthcheck(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    generation = config.get("generation_configuration", {})
    checkout = generation.get("official_checkout_path")
    cache = generation.get("checkpoint_snapshot_path")
    if checkout is None or cache is None:
        return {
            "available": False,
            "worker_version": WORKER_VERSION,
            "reason": "official_checkout_path and checkpoint_snapshot_path are required",
        }
    commit = verify_checkout(Path(checkout), config["official_code_commit"])
    hashes = verify_assets(Path(cache), config.get("checkpoint_hashes", {}))
    runtime_paths = {
        name: Path(str(path)) for name, path in generation.get("runtime_model_paths", {}).items()
    }
    runtime_hashes = config.get("runtime_model_hashes", {})
    runtime_revisions = config.get("runtime_model_revisions", {})
    if set(runtime_paths) != set(runtime_hashes) or set(runtime_hashes) != set(runtime_revisions):
        raise RuntimeError("TRELLIS.2 runtime model paths, revisions, and hashes must match")
    verified_runtime = {
        name: verify_assets(runtime_paths[name], expected)
        for name, expected in runtime_hashes.items()
    }
    sys.path.insert(0, str(Path(checkout)))
    import nvdiffrast.torch as dr
    import torch
    import trimesh
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

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
        "official_submodule_commits": EXPECTED_SUBMODULES,
        "checkpoint_revision": config["checkpoint_revision"],
        "checkpoint_hashes": hashes,
        "runtime_model_revisions": runtime_revisions,
        "runtime_model_hashes": verified_runtime,
        "pipeline_import": Trellis2ImageTo3DPipeline.__name__,
    }
