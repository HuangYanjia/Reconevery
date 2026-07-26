from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np


def _read_header(file: BinaryIO, path: Path) -> tuple[int, int, int]:
    values: list[int] = []
    field = bytearray()
    while len(values) < 3:
        byte = file.read(1)
        if not byte:
            raise ValueError(f"truncated COLMAP header: {path}")
        if byte == b"&":
            try:
                values.append(int(field.decode("ascii")))
            except (UnicodeDecodeError, ValueError) as exc:
                raise ValueError(f"invalid COLMAP header: {path}") from exc
            field.clear()
        else:
            field.extend(byte)
    width, height, channels = values
    if width <= 0 or height <= 0 or channels <= 0:
        raise ValueError(f"invalid COLMAP dimensions: {path}")
    return width, height, channels


def read_array(path: Path, channels: int, *, require_finite: bool = False) -> np.ndarray:
    """Read an official COLMAP dense array using its Fortran-ordered payload contract."""
    with path.open("rb") as file:
        width, height, actual = _read_header(file, path)
        payload = np.fromfile(file, dtype="<f4")
    if actual != channels:
        raise ValueError(f"{path.name} has {actual} channels, expected {channels}")
    expected = int(width * height * actual)
    if payload.size != expected:
        raise ValueError(f"{path.name} has {payload.size} floats, expected {expected}")
    result = payload.reshape((width, height, actual), order="F").transpose(1, 0, 2)
    if require_finite and not np.isfinite(result).all():
        raise ValueError(f"{path.name} contains non-finite values")
    return result.squeeze(axis=2) if channels == 1 else result


def read_consistency_graph(
    path: Path,
    shape: tuple[int, int],
    image_count: int,
) -> np.ndarray:
    with path.open("rb") as file:
        width, height, actual = _read_header(file, path)
        values = np.fromfile(file, dtype="<i4")
    if actual != 1:
        raise ValueError(f"{path.name} has {actual} channels, expected 1")
    if (int(height), int(width)) != shape:
        raise ValueError(f"{path.name} dimensions do not match its depth map")
    counts = np.zeros(shape, dtype=np.int32)
    cursor = 0
    while cursor < values.size:
        if cursor + 3 > values.size:
            raise ValueError(f"truncated consistency entry: {path}")
        column, row, count = (int(value) for value in values[cursor : cursor + 3])
        cursor += 3
        if (
            row < 0
            or row >= height
            or column < 0
            or column >= width
            or count < 0
            or cursor + count > values.size
        ):
            raise ValueError(f"invalid consistency entry: {path}")
        sources = values[cursor : cursor + count]
        cursor += count
        if np.any(sources < 0) or np.any(sources >= image_count):
            raise ValueError(f"invalid consistency image index: {path}")
        counts[row, column] = count
    return counts
