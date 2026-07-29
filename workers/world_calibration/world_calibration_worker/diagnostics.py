from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

PREVIEW_NAMES = (
    "metric_evidence",
    "tag_detections",
    "landmark_reprojection",
    "floor_plane",
    "gravity_evidence",
    "canonical_axes",
    "camera_trajectory_before_after",
    "scene_bounds_before_after",
    "heldout_validation",
)


def render_previews(root: Path, status: str, lines: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in PREVIEW_NAMES:
        image = Image.new("RGB", (960, 540), (246, 247, 249))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 960, 66), fill=(24, 34, 44))
        draw.text((24, 22), f"Phase 6A: {name.replace('_', ' ')}", fill="white")
        draw.text((28, 98), f"status: {status}", fill=(25, 35, 45))
        for index, line in enumerate(lines[:8]):
            draw.text((28, 134 + 34 * index), line, fill=(45, 55, 65))
        image.save(root / f"{name}.png", format="PNG", optimize=False, compress_level=9)


__all__ = ["PREVIEW_NAMES", "render_previews"]
