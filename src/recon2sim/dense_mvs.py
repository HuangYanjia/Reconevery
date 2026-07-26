from __future__ import annotations

import hashlib
import math
import struct
from array import array
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

OFFICIAL_COLMAP_REPOSITORY = "https://github.com/colmap/colmap"
OFFICIAL_COLMAP_VERSION = "4.0.4"
# Release tag 4.0.4 is resolved and verified by the worker healthcheck.  Keeping
# the commit configurable permits source builds to record the exact official SHA.
OFFICIAL_COLMAP_COMMIT = "9c23f6942fe69962e06030905e77067c8673382f"


class DenseMapFormatError(ValueError):
    pass


@dataclass(frozen=True)
class DenseArray:
    width: int
    height: int
    channels: int
    values: array[float]

    def value(self, x: int, y: int, channel: int = 0) -> float:
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError("dense-map pixel is outside the image")
        if not 0 <= channel < self.channels:
            raise IndexError("dense-map channel is outside the array")
        # COLMAP stores matrices in column-major order.
        return self.values[x + self.width * y + self.width * self.height * channel]


@dataclass(frozen=True)
class ConsistencyGraphEntry:
    row: int
    column: int
    source_image_indices: tuple[int, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_ascii_field(file: BinaryIO) -> int:
    raw = bytearray()
    while True:
        byte = file.read(1)
        if not byte:
            raise DenseMapFormatError("truncated COLMAP dense-map header")
        if byte == b"&":
            break
        if len(raw) >= 20 or not byte.isdigit():
            raise DenseMapFormatError("invalid COLMAP dense-map header")
        raw.extend(byte)
    if not raw:
        raise DenseMapFormatError("empty COLMAP dense-map header field")
    value = int(raw)
    if value <= 0:
        raise DenseMapFormatError("COLMAP dense-map dimensions must be positive")
    return value


def read_dense_array(
    path: Path,
    *,
    expected_channels: int | None = None,
    reject_non_finite: bool = False,
) -> DenseArray:
    with path.open("rb") as file:
        width = _read_ascii_field(file)
        height = _read_ascii_field(file)
        channels = _read_ascii_field(file)
        if expected_channels is not None and channels != expected_channels:
            raise DenseMapFormatError(
                f"{path.name} has {channels} channels; expected {expected_channels}"
            )
        count = width * height * channels
        payload = file.read()
    expected_bytes = count * 4
    if len(payload) != expected_bytes:
        raise DenseMapFormatError(
            f"{path.name} contains {len(payload)} data bytes; expected {expected_bytes}"
        )
    values = array("f")
    values.frombytes(payload)
    if struct.pack("=I", 1) != struct.pack("<I", 1):
        values.byteswap()
    if reject_non_finite and any(not math.isfinite(value) for value in values):
        raise DenseMapFormatError(f"{path.name} contains non-finite values")
    return DenseArray(width, height, channels, values)


def iter_consistency_graph(path: Path, *, image_count: int) -> Iterator[ConsistencyGraphEntry]:
    payload = path.read_bytes()
    width = height = None
    if b"&" in payload[:64]:
        cursor = 0
        fields: list[int] = []
        for _ in range(3):
            separator = payload.find(b"&", cursor)
            if separator < 0:
                raise DenseMapFormatError("truncated COLMAP consistency graph header")
            field = payload[cursor:separator]
            if not field.isdigit() or int(field) <= 0:
                raise DenseMapFormatError("invalid COLMAP consistency graph header")
            fields.append(int(field))
            cursor = separator + 1
        width, height, _ = fields
        payload = payload[cursor:]
    if len(payload) % 4:
        raise DenseMapFormatError("COLMAP consistency graph has truncated int32 data")
    values = struct.unpack(f"<{len(payload) // 4}i", payload)
    cursor = 0
    while cursor < len(values):
        if len(values) - cursor < 3:
            raise DenseMapFormatError("COLMAP consistency graph has a truncated entry")
        column, row, count = values[cursor : cursor + 3]
        cursor += 3
        if row < 0 or column < 0 or count < 0:
            raise DenseMapFormatError("COLMAP consistency graph contains negative metadata")
        if width is not None and height is not None and (column >= width or row >= height):
            raise DenseMapFormatError("COLMAP consistency graph contains an invalid pixel")
        if cursor + count > len(values):
            raise DenseMapFormatError("COLMAP consistency graph source list is truncated")
        sources = tuple(values[cursor : cursor + count])
        cursor += count
        if any(source < 0 or source >= image_count for source in sources):
            raise DenseMapFormatError("COLMAP consistency graph contains invalid image indices")
        yield ConsistencyGraphEntry(row, column, sources)


def ply_counts(path: Path) -> tuple[int, int]:
    with path.open("rb") as file:
        if file.readline().strip() != b"ply":
            raise ValueError(f"{path.name} is not a PLY file")
        vertices = faces = None
        format_name = None
        reading_vertices = False
        vertex_properties: list[tuple[str, str]] = []
        for raw in file:
            line = raw.decode("ascii", errors="strict").strip()
            if line.startswith("format "):
                format_name = line.split()[1]
            elif line.startswith("element vertex "):
                vertices = int(line.split()[2])
                reading_vertices = True
            elif line.startswith("element face "):
                faces = int(line.split()[2])
                reading_vertices = False
            elif line.startswith("element "):
                reading_vertices = False
            elif reading_vertices and line.startswith("property "):
                fields = line.split()
                if fields[1] == "list":
                    raise ValueError(f"{path.name} has unsupported list-valued vertices")
                vertex_properties.append((fields[2], fields[1]))
            elif line == "end_header":
                break
        else:
            raise ValueError(f"{path.name} has no PLY end_header")
        if vertices is None or vertices <= 0:
            raise ValueError(f"{path.name} has no vertices")
        property_names = [name for name, _ in vertex_properties]
        if not {"x", "y", "z"}.issubset(property_names):
            raise ValueError(f"{path.name} has no xyz vertex coordinates")
        if format_name == "ascii":
            xyz_indices = tuple(property_names.index(axis) for axis in ("x", "y", "z"))
            for _ in range(vertices):
                vertex_fields = file.readline().split()
                if len(vertex_fields) < len(vertex_properties):
                    raise ValueError(f"{path.name} has truncated ASCII vertex data")
                point = (float(vertex_fields[index]) for index in xyz_indices)
                if any(not math.isfinite(value) for value in point):
                    raise ValueError(f"{path.name} contains non-finite vertex coordinates")
        elif format_name == "binary_little_endian":
            type_codes = {
                "char": "b",
                "uchar": "B",
                "short": "h",
                "ushort": "H",
                "int": "i",
                "uint": "I",
                "float": "f",
                "double": "d",
            }
            try:
                record = struct.Struct(
                    "<"
                    + "".join(type_codes[property_type] for _, property_type in vertex_properties)
                )
            except KeyError as exc:
                raise ValueError(
                    f"{path.name} has unsupported PLY property type {exc.args[0]}"
                ) from exc
            xyz_indices = tuple(property_names.index(axis) for axis in ("x", "y", "z"))
            for _ in range(vertices):
                payload = file.read(record.size)
                if len(payload) != record.size:
                    raise ValueError(f"{path.name} has truncated binary vertex data")
                values = record.unpack(payload)
                if any(not math.isfinite(float(values[index])) for index in xyz_indices):
                    raise ValueError(f"{path.name} contains non-finite vertex coordinates")
        else:
            raise ValueError(f"{path.name} has unsupported PLY format {format_name!r}")
    return vertices, faces or 0
