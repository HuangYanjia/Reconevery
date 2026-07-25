from __future__ import annotations

import subprocess
from pathlib import Path


def git_commit(checkout: Path) -> str:
    if not (checkout / ".git").exists():
        raise RuntimeError(f"official GenRecon checkout is not a Git checkout: {checkout}")
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"could not inspect official GenRecon commit: {result.stderr.strip()}")
    return result.stdout.strip()


def submodule_commits(checkout: Path) -> dict[str, str]:
    result = subprocess.run(
        ["git", "-C", str(checkout), "submodule", "status", "--recursive"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"could not inspect GenRecon submodules: {result.stderr.strip()}")
    commits: dict[str, str] = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split()
        if len(fields) < 2:
            continue
        commits[fields[1]] = fields[0].lstrip("-+U")
    return commits


def verify_checkout(
    checkout: Path,
    expected_commit: str,
    expected_submodules: dict[str, str],
) -> dict[str, str]:
    actual_commit = git_commit(checkout)
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"official GenRecon checkout is at {actual_commit}, expected {expected_commit}"
        )
    actual_submodules = submodule_commits(checkout)
    if actual_submodules != expected_submodules:
        raise RuntimeError(
            f"official GenRecon submodules are {actual_submodules}, expected {expected_submodules}"
        )
    return actual_submodules
