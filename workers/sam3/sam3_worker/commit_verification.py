from __future__ import annotations

import json
import subprocess
from importlib import import_module, metadata
from pathlib import Path
from urllib.parse import unquote, urlparse


def _git_commit(checkout: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and len(commit) == 40 else None


def _editable_checkout(payload: dict[str, object]) -> Path | None:
    directory = payload.get("dir_info")
    if not isinstance(directory, dict) or directory.get("editable") is not True:
        return None
    url = payload.get("url")
    if not isinstance(url, str):
        return None
    parsed = urlparse(url)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        return None
    return Path(unquote(parsed.path))


def commit_from_direct_url(direct_url: str) -> str | None:
    try:
        payload = json.loads(direct_url)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    vcs = payload.get("vcs_info")
    if isinstance(vcs, dict) and vcs.get("vcs") == "git":
        commit = vcs.get("commit_id")
        if isinstance(commit, str) and len(commit) == 40:
            return commit
    checkout = _editable_checkout(payload)
    return _git_commit(checkout) if checkout is not None else None


def installed_sam_commit() -> str | None:
    try:
        distribution = metadata.distribution("sam3")
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            commit = commit_from_direct_url(direct_url)
            if commit is not None:
                return commit
    except metadata.PackageNotFoundError:
        pass

    try:
        module = import_module("sam3")
        package_file = getattr(module, "__file__", None)
        if package_file is None:
            return None
        return _git_commit(Path(package_file).resolve().parents[1])
    except (ImportError, OSError):
        return None


def require_official_commit(expected: str) -> str:
    installed = installed_sam_commit()
    if installed is None:
        raise RuntimeError(
            "official SAM installation commit could not be verified; install the exact "
            "checkout with `pip install -e /path/to/sam3` or use an exact git+https VCS URL"
        )
    if installed != expected:
        raise RuntimeError(f"official SAM installation is at {installed}, expected {expected}")
    return installed
