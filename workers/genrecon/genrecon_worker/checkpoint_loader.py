from __future__ import annotations

import hashlib
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

CHECKPOINT_URLS = {
    "sparse_structure": "https://kaldir.vc.cit.tum.de/genrecon/sparse_structure.pt",
    "shape_slat": "https://kaldir.vc.cit.tum.de/genrecon/shape_slat.pt",
    "texture_slat": "https://kaldir.vc.cit.tum.de/genrecon/texture_slat.pt",
}

CHECKPOINT_TRAIN_CONFIGS = {
    "sparse_structure": "configs/gen/ss_flow_img/genrecon.json",
    "shape_slat": "configs/gen/slat_flow_img2shape/genrecon_512.json",
    "texture_slat": "configs/gen/slat_flow_imgshape2tex/genrecon_512.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoints(
    checkpoint_paths: dict[str, str],
    expected_hashes: dict[str, str],
) -> list[dict[str, object]]:
    expected_names = set(CHECKPOINT_URLS)
    if set(checkpoint_paths) != expected_names or set(expected_hashes) != expected_names:
        raise RuntimeError("all three official GenRecon checkpoints are required")
    records: list[dict[str, object]] = []
    for checkpoint_id in ("sparse_structure", "shape_slat", "texture_slat"):
        path = Path(checkpoint_paths[checkpoint_id]).expanduser()
        if not path.is_file():
            raise RuntimeError(f"official GenRecon checkpoint is missing: {path}")
        digest = sha256_file(path)
        if digest != expected_hashes[checkpoint_id]:
            raise RuntimeError(
                f"{checkpoint_id} checkpoint SHA-256 is {digest}, "
                f"expected {expected_hashes[checkpoint_id]}"
            )
        records.append(
            {
                "checkpoint_id": checkpoint_id,
                "source_url": CHECKPOINT_URLS[checkpoint_id],
                "local_filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": digest,
                "resolved_at": datetime.fromtimestamp(
                    path.stat().st_mtime,
                    timezone.utc,  # noqa: UP017 - worker supports Python 3.10
                ).isoformat(),
                "access_mode": "local_cache",
            }
        )
    return records


def stage_checkpoints(
    checkout: Path,
    temporary: Path,
    checkpoint_paths: dict[str, str],
) -> dict[str, str]:
    staged: dict[str, str] = {}
    for checkpoint_id, config_relative_path in CHECKPOINT_TRAIN_CONFIGS.items():
        source = Path(checkpoint_paths[checkpoint_id]).resolve()
        stage_root = temporary / "model_inputs" / checkpoint_id
        checkpoint_root = stage_root / "checkpoints"
        checkpoint_root.mkdir(parents=True)
        config_source = checkout / config_relative_path
        if not config_source.is_file():
            raise RuntimeError(
                f"pinned official GenRecon training config is missing: {config_relative_path}"
            )
        shutil.copy2(config_source, stage_root / "config.json")
        destination = checkpoint_root / source.name
        try:
            os.link(source, destination)
        except OSError:
            destination.symlink_to(source)
        staged[checkpoint_id] = str(destination)
    return staged
