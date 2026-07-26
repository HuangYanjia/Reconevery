from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def object_color(object_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(object_id.encode()).digest()
    return tuple(64 + component % 160 for component in digest[:3])  # type: ignore[return-value]


def assignment_image(
    face_ids: Any,
    accepted_by_object: dict[str, set[int]],
    ambiguous_faces: set[int],
    conflict_faces: set[int],
) -> Image.Image:
    import numpy as np

    height, width = face_ids.shape
    output = np.full((height, width, 3), 96, dtype=np.uint8)
    output[face_ids < 0] = (25, 27, 30)
    for object_id in sorted(accepted_by_object):
        selected = np.isin(face_ids, np.fromiter(accepted_by_object[object_id], np.int64))
        output[selected] = object_color(object_id)
    if ambiguous_faces:
        output[np.isin(face_ids, np.fromiter(ambiguous_faces, np.int64))] = (255, 210, 35)
    if conflict_faces:
        output[np.isin(face_ids, np.fromiter(conflict_faces, np.int64))] = (220, 45, 45)
    return Image.fromarray(output, mode="RGB")


def annotated_tile(
    mask: Any,
    rendered: Any,
    *,
    title: str,
    iou: float,
) -> Image.Image:
    import numpy as np

    mask = mask.astype(bool)
    rendered = rendered.astype(bool)
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    output[mask & rendered] = (70, 190, 95)
    output[mask & ~rendered] = (55, 135, 230)
    output[~mask & rendered] = (225, 55, 55)
    image = Image.fromarray(output, mode="RGB")
    canvas = Image.new("RGB", (image.width, image.height + 28), "white")
    canvas.paste(image, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), f"{title}  IoU={iou:.3f}", fill="#111111")
    return canvas


def contact_sheet(
    images: list[Image.Image],
    path: Path,
    *,
    columns: int = 3,
    empty_title: str = "No resolved surfaces",
) -> None:
    if not images:
        canvas = Image.new("RGB", (640, 240), "white")
        ImageDraw.Draw(canvas).text((24, 24), empty_title, fill="#111111")
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, format="PNG", compress_level=6, optimize=False)
        return
    tile_width = max(image.width for image in images)
    tile_height = max(image.height for image in images)
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "#202328")
    for index, image in enumerate(images):
        canvas.paste(image, ((index % columns) * tile_width, (index // columns) * tile_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", compress_level=6, optimize=False)


def add_title(image: Image.Image, title: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 32), "white")
    canvas.paste(image, (0, 32))
    ImageDraw.Draw(canvas).text((8, 9), title, fill="#111111")
    return canvas
