from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def conversion_command(checkout: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(checkout / "chunked_to_glb.py"),
        "--inputs",
        str(output_dir / "to_glb_inputs.pt"),
        "--chunk_inputs",
        str(output_dir / "chunk_inputs.pt"),
        "--output_dir",
        str(output_dir),
    ]


def run_glb_conversion(
    checkout: Path,
    output_dir: Path,
    environment: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        result = subprocess.run(
            conversion_command(checkout, output_dir),
            cwd=checkout,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return result.returncode
