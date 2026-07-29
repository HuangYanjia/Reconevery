from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

PARTICULATE_COMMIT = "dee37a75c449f324d9989993461ee09eaccc1686"
PARTICULATE_MODEL_SHA256 = "ad6f14067dadf85335119199b94e8249401376d5700c9b627c3608594ea99b5c"
PARTFIELD_MODEL_SHA256 = "463efc8a3afd3913142aa025e0125c00f16ef452b8de6a132ebe32bbe7877ee4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_official_runtime(
    repository: Path,
    checkpoint: Path,
    partfield_checkpoint: Path,
) -> None:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or result.stdout.strip() != PARTICULATE_COMMIT:
        raise RuntimeError("official Particulate commit verification failed")
    if sha256(checkpoint) != PARTICULATE_MODEL_SHA256:
        raise RuntimeError("official Particulate checkpoint hash mismatch")
    if sha256(partfield_checkpoint) != PARTFIELD_MODEL_SHA256:
        raise RuntimeError("official PartField checkpoint hash mismatch")
    expected = repository / "PartField/model/model_objaverse.ckpt"
    expected.parent.mkdir(parents=True, exist_ok=True)
    if expected.exists() or expected.is_symlink():
        if sha256(expected.resolve()) != PARTFIELD_MODEL_SHA256:
            raise RuntimeError("Particulate repository contains a wrong PartField checkpoint")
    else:
        expected.symlink_to(partfield_checkpoint.resolve())
