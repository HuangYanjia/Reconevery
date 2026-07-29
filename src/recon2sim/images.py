from __future__ import annotations

import struct
import zlib
from pathlib import Path

from PIL import Image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def write_solid_png(
    path: Path, width: int, height: int, rgb: tuple[int, int, int] = (128, 128, 128)
) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("PNG dimensions must be positive")
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise ValueError("RGB channels must be in [0, 255]")
    path.parent.mkdir(parents=True, exist_ok=True)
    scanline = b"\x00" + bytes(rgb) * width
    raw = scanline * height
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        PNG_SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )


def png_dimensions(path: Path) -> tuple[int, int]:
    width, height, _ = _read_png(path)
    return width, height


def _read_png(path: Path) -> tuple[int, int, list[tuple[bytes, bytes]]]:
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"{path} is not a valid PNG file")
    chunks: list[tuple[bytes, bytes]] = []
    offset = len(PNG_SIGNATURE)
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError(f"{path} contains a truncated PNG chunk")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        end = offset + 12 + length
        if end > len(data):
            raise ValueError(f"{path} contains a truncated PNG payload")
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length : end])[0]
        if zlib.crc32(kind + payload) != expected_crc:
            raise ValueError(f"{path} contains a PNG chunk with an invalid checksum")
        chunks.append((kind, payload))
        offset = end
        if kind == b"IEND":
            break
    if offset != len(data):
        raise ValueError(f"{path} contains trailing data after IEND")
    if not chunks or chunks[0][0] != b"IHDR" or len(chunks[0][1]) != 13:
        raise ValueError(f"{path} has an invalid PNG header")
    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
        ">IIBBBBB", chunks[0][1]
    )
    if width <= 0 or height <= 0:
        raise ValueError(f"{path} has invalid PNG dimensions")
    if (bit_depth, color_type, compression, filter_method, interlace) != (8, 2, 0, 0, 0):
        raise ValueError(f"{path} must be an 8-bit, non-interlaced RGB PNG")
    if not any(kind == b"IDAT" for kind, _ in chunks):
        raise ValueError(f"{path} is missing required PNG chunks")
    if chunks[-1] != (b"IEND", b""):
        raise ValueError(f"{path} has an invalid IEND chunk")
    return width, height, chunks


def validate_png(path: Path) -> None:
    width, height, chunks = _read_png(path)
    compressed = b"".join(payload for kind, payload in chunks if kind == b"IDAT")
    try:
        pixels = zlib.decompress(compressed)
    except zlib.error as exc:
        raise ValueError(f"{path} contains invalid compressed PNG pixels") from exc
    row_bytes = 1 + width * 3
    if len(pixels) != row_bytes * height:
        raise ValueError(f"{path} contains an unexpected amount of PNG pixel data")
    if any(pixels[row * row_bytes] > 4 for row in range(height)):
        raise ValueError(f"{path} contains an invalid PNG row filter")


def validate_binary_mask_png(path: Path) -> None:
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG":
            raise ValueError(f"{path} is not a PNG image")
        if image.mode != "L":
            raise ValueError(f"{path} must be an 8-bit grayscale PNG mask")
        if not set(image.tobytes()) <= {0, 255}:
            raise ValueError(f"{path} must contain only binary mask values 0 and 255")
