from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np


def _read_header(stream: BinaryIO, path: Path) -> tuple[int, int, int]:
    fields: list[int] = []
    current = bytearray()
    while len(fields) < 3:
        byte = stream.read(1)
        if not byte:
            raise ValueError(f"truncated COLMAP dense header: {path}")
        if byte == b"&":
            try:
                fields.append(int(current.decode("ascii")))
            except (UnicodeDecodeError, ValueError) as exc:
                raise ValueError(f"invalid COLMAP dense header: {path}") from exc
            current.clear()
        else:
            current.extend(byte)
    width, height, channels = fields
    if width <= 0 or height <= 0 or channels <= 0:
        raise ValueError(f"invalid COLMAP dense dimensions: {path}")
    return width, height, channels


def read_dense_array(path: Path, channels: int) -> np.ndarray:
    """Decode the official COLMAP Fortran-ordered dense-array format."""
    with path.open("rb") as stream:
        width, height, actual_channels = _read_header(stream, path)
        payload = np.fromfile(stream, dtype="<f4")
    if actual_channels != channels:
        raise ValueError(f"{path.name} has {actual_channels} channels, expected {channels}")
    expected = width * height * actual_channels
    if payload.size != expected:
        raise ValueError(f"{path.name} has {payload.size} floats, expected {expected}")
    result = payload.reshape((width, height, actual_channels), order="F").transpose(1, 0, 2)
    return result[..., 0] if channels == 1 else result
