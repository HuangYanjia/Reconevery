from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(
        path, json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n"
    )


def atomic_write_yaml(path: Path, data: Any) -> None:
    atomic_write_text(path, yaml.safe_dump(data, sort_keys=False))
