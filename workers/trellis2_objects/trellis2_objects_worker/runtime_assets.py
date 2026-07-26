from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_assets(root: Path, expected: dict[str, str]) -> dict[str, str]:
    if not expected:
        raise RuntimeError("exact TRELLIS.2 runtime model hashes are required")
    actual = {}
    for relative, expected_hash in expected.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"TRELLIS.2 runtime asset is missing: {relative}")
        actual[relative] = sha256(path)
        if actual[relative] != expected_hash:
            raise RuntimeError(f"TRELLIS.2 runtime asset hash mismatch: {relative}")
    return actual
