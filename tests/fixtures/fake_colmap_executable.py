#!/usr/bin/env python3
from __future__ import annotations

import struct
import sys
from pathlib import Path


def option(arguments: list[str], name: str) -> str:
    return arguments[arguments.index(name) + 1]


def write_model(output: Path, frames: list[Path]) -> None:
    root = output / "0"
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("cameras.bin").write_bytes(
        struct.pack("<QiiQQdddd", 1, 1, 1, 32, 24, 30.0, 31.0, 16.0, 12.0)
    )
    image_data = struct.pack("<Q", len(frames))
    for index, frame in enumerate(frames):
        image_data += struct.pack(
            "<idddddddi",
            index + 1,
            1.0,
            0.0,
            0.0,
            0.0,
            float(index) * 0.1,
            0.0,
            0.0,
            1,
        )
        image_data += frame.name.encode("utf-8") + b"\0" + struct.pack("<Q", 0)
    root.joinpath("images.bin").write_bytes(image_data)
    point_data = struct.pack("<Q", 64)
    for index in range(64):
        x = float(index % 4) * 0.2
        y = float((index // 4) % 4) * 0.2
        z = float(index // 16) * 0.2
        point_data += struct.pack(
            "<QdddBBBdQ",
            index + 1,
            x,
            y,
            z,
            80,
            120,
            160,
            0.25,
            0,
        )
    root.joinpath("points3D.bin").write_bytes(point_data)


def main() -> int:
    arguments = sys.argv[1:]
    if arguments == ["-h"]:
        print("COLMAP 3.11.1 deterministic fake")
        return 0
    command = arguments[0]
    if command == "feature_extractor":
        database = Path(option(arguments, "--database_path"))
        database.parent.mkdir(parents=True, exist_ok=True)
        database.write_bytes(b"deterministic fake sqlite")
    elif command == "mapper":
        frames = sorted(Path(option(arguments, "--image_path")).glob("*.png"))
        write_model(Path(option(arguments, "--output_path")), frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
