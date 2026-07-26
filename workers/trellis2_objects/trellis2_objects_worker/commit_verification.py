from __future__ import annotations

import subprocess
from pathlib import Path

EXPECTED_SUBMODULES = {
    "o-voxel/third_party/eigen": "21e4582d1739107337a03460c81412981130373e",
}


def verify_checkout(path: Path, expected_commit: str) -> str:
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
            f"official TRELLIS.2 commit mismatch: expected {expected_commit}, got "
            f"{actual or 'unavailable'}"
        )
    status = subprocess.run(
        ["git", "-C", str(path), "submodule", "status", "--recursive"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    submodules = {}
    for line in status.stdout.splitlines():
        fields = line.strip().split()
        if len(fields) >= 2:
            submodules[fields[1]] = fields[0].lstrip("-+U")
    if status.returncode != 0 or submodules != EXPECTED_SUBMODULES:
        raise RuntimeError(
            "official TRELLIS.2 submodule mismatch: "
            f"expected {EXPECTED_SUBMODULES}, got {submodules}"
        )
    return actual
