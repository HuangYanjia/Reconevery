from __future__ import annotations

from pathlib import Path

import numpy as np


def read_array(path: Path, channels: int) -> np.ndarray:
    with path.open("rb") as file:
        width, height, actual = np.genfromtxt(
            file, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        file.seek(0)
        delimiters = 0
        while delimiters < 3:
            byte = file.read(1)
            if not byte:
                raise ValueError(f"truncated COLMAP header: {path}")
            delimiters += byte == b"&"
        payload = np.fromfile(file, dtype="<f4")
    if actual != channels:
        raise ValueError(f"{path.name} has {actual} channels, expected {channels}")
    expected = int(width * height * actual)
    if payload.size != expected:
        raise ValueError(f"{path.name} has {payload.size} floats, expected {expected}")
    result = payload.reshape((width, height, actual), order="F").transpose(1, 0, 2)
    return result.squeeze() if channels == 1 else result


def read_consistency_graph(path: Path, image_count: int) -> tuple[np.ndarray, list[list[int]]]:
    with path.open("rb") as file:
        width, height, _ = np.genfromtxt(
            file, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        file.seek(0)
        delimiters = 0
        while delimiters < 3:
            byte = file.read(1)
            if not byte:
                raise ValueError(f"truncated consistency header: {path}")
            delimiters += byte == b"&"
        values = np.fromfile(file, dtype="<i4")
    counts = np.zeros((height, width), dtype=np.int32)
    sources: list[list[int]] = []
    cursor = 0
    while cursor < values.size:
        if cursor + 3 > values.size:
            raise ValueError(f"truncated consistency entry: {path}")
        column, row, count = (int(value) for value in values[cursor : cursor + 3])
        cursor += 3
        if row < 0 or row >= height or column < 0 or column >= width or count < 0:
            raise ValueError(f"invalid consistency entry: {path}")
        if cursor + count > values.size:
            raise ValueError(f"truncated consistency source list: {path}")
        entry = [int(value) for value in values[cursor : cursor + count]]
        cursor += count
        if any(value < 0 or value >= image_count for value in entry):
            raise ValueError(f"invalid consistency image index: {path}")
        counts[row, column] = count
        sources.append(entry)
    return counts, sources
