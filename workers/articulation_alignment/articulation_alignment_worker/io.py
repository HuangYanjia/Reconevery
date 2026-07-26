from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_points(path: Path) -> np.ndarray:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [
            np.asarray(geometry.vertices, dtype=np.float64)
            for geometry in loaded.geometry.values()
            if hasattr(geometry, "vertices")
        ]
        if not geometries:
            raise ValueError(f"geometry has no vertices: {path}")
        points = np.concatenate(geometries, axis=0)
    else:
        points = np.asarray(loaded.vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"geometry vertices are not finite Nx3 data: {path}")
    return points


def write_preview(path: Path, title: str, lines: list[str]) -> None:
    image = Image.new("RGB", (1280, 720), (246, 247, 249))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1280, 72), fill=(24, 33, 43))
    draw.text((28, 24), title, fill=(255, 255, 255))
    for index, line in enumerate(lines):
        draw.text((40, 110 + index * 40), line, fill=(28, 37, 46))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)
