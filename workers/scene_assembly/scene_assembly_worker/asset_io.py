from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def safe_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or relative in {"", "."}:
        raise ValueError(f"unsafe assembly worker path: {relative!r}")
    resolved = root.joinpath(*path.parts).resolve()
    if root.resolve() not in resolved.parents:
        raise ValueError(f"assembly worker path escapes input root: {relative!r}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["read_json", "safe_path", "sha256_file", "write_json"]
