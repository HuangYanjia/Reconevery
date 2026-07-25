from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from sam3_worker.commit_verification import require_official_commit
from sam3_worker.official_compat import apply_sam31_start_session_compatibility
from sam3_worker.schema import WorkerConfiguration
from sam3_worker.version import OFFICIAL_CODE_COMMIT


def checkpoint_filename(model_mode: str) -> str:
    return "sam3.1_multiplex.pt" if model_mode == "sam3.1" else "sam3.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_official_code_commit(config: WorkerConfiguration) -> str:
    if config.official_code_commit != OFFICIAL_CODE_COMMIT:
        raise RuntimeError(
            f"worker supports official SAM commit {OFFICIAL_CODE_COMMIT}, "
            f"not {config.official_code_commit}"
        )
    return require_official_commit(OFFICIAL_CODE_COMMIT)


def check_checkpoint_access(config: WorkerConfiguration) -> dict[str, Any]:
    if config.local_checkpoint_path:
        path = Path(config.local_checkpoint_path)
        if not path.is_file() or not os.access(path, os.R_OK):
            raise RuntimeError(f"local official checkpoint is unreadable: {path}")
        return {"access_mode": "local_path", "path": str(path)}
    filename = checkpoint_filename(config.model_mode)
    if config.offline:
        path = Path(
            hf_hub_download(
                repo_id=config.checkpoint_repository,
                filename=filename,
                revision=config.checkpoint_revision,
                local_files_only=True,
                cache_dir=config.model_cache_path,
            )
        )
        return {"access_mode": "offline_cache", "path": str(path)}
    info = HfApi().model_info(
        config.checkpoint_repository,
        revision=config.checkpoint_revision,
        token=os.environ.get("HF_TOKEN"),
        files_metadata=False,
    )
    sibling_names = {item.rfilename for item in info.siblings or []}
    if filename not in sibling_names:
        raise RuntimeError(
            f"official checkpoint file {filename!r} was not found at revision "
            f"{config.checkpoint_revision}"
        )
    return {"access_mode": "authenticated_remote", "path": None}


def resolve_checkpoint(config: WorkerConfiguration) -> Path:
    if config.local_checkpoint_path:
        path = Path(config.local_checkpoint_path)
        if not path.is_file():
            raise RuntimeError(f"local official checkpoint is missing: {path}")
        return path
    return Path(
        hf_hub_download(
            repo_id=config.checkpoint_repository,
            filename=checkpoint_filename(config.model_mode),
            revision=config.checkpoint_revision,
            local_files_only=config.offline,
            cache_dir=config.model_cache_path,
        )
    )


def load_predictor(config: WorkerConfiguration) -> tuple[Any, Path]:
    validate_official_code_commit(config)
    checkpoint = resolve_checkpoint(config)
    from sam3.model_builder import build_sam3_predictor

    predictor = build_sam3_predictor(
        checkpoint_path=str(checkpoint),
        version=config.model_mode,
        compile=False,
        warm_up=False,
        use_fa3=False,
        async_loading_frames=False,
    )
    if config.model_mode == "sam3.1":
        apply_sam31_start_session_compatibility(predictor)
    return predictor, checkpoint
