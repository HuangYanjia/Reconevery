from __future__ import annotations

import re
import subprocess


def inspect_colmap(executable: str) -> tuple[str, str, str]:
    result = subprocess.run(
        [executable, "-h"], capture_output=True, text=True, check=False, timeout=30
    )
    output = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        raise RuntimeError(output.strip() or "COLMAP healthcheck failed")
    match = re.search(r"COLMAP\s+([0-9]+\.[0-9]+\.[0-9]+)", output)
    if match is None:
        raise RuntimeError("could not determine COLMAP version from `colmap -h`")
    commit = re.search(r"\(Commit\s+([0-9a-f]+)\b", output)
    if commit is None:
        raise RuntimeError("could not determine COLMAP commit prefix from `colmap -h`")
    return (
        match.group(1),
        output.splitlines()[0] if output.splitlines() else output.strip(),
        commit.group(1),
    )
