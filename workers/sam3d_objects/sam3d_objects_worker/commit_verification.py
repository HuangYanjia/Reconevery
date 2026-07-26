from __future__ import annotations

import subprocess
from pathlib import Path


def verify_checkout(path: Path, expected_commit: str) -> str:
    if not (path / ".git").exists():
        raise RuntimeError("official SAM 3D Objects checkout is not a verifiable Git checkout")
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    actual = result.stdout.strip()
    if result.returncode != 0 or actual != expected_commit:
        raise RuntimeError(
            f"official SAM 3D Objects commit mismatch: expected {expected_commit}, got "
            f"{actual or 'unavailable'}"
        )
    return actual
