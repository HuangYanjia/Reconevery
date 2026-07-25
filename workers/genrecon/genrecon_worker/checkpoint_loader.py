from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

CHECKPOINT_URLS = {
    "sparse_structure": "https://kaldir.vc.cit.tum.de/genrecon/sparse_structure.pt",
    "shape_slat": "https://kaldir.vc.cit.tum.de/genrecon/shape_slat.pt",
    "texture_slat": "https://kaldir.vc.cit.tum.de/genrecon/texture_slat.pt",
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
