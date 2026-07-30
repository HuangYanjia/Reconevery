from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw


def write_preview(
    path: Path,
    *,
    title: str,
    width: int,
    height: int,
    lines: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (width, height), (245, 246, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 60, width - 24, height - 24), outline=(54, 69, 79), width=2)
    draw.text((24, 24), title, fill=(20, 30, 40))
    for index, line in enumerate(lines[:20]):
        draw.text((44, 82 + 24 * index), line, fill=(45, 55, 65))
    image.save(path, format="PNG", optimize=False, compress_level=9)


def render_scene_preview(
    path: Path,
    *,
    title: str,
    scene: trimesh.Scene,
    width: int,
    height: int,
    lines: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (width, height), (245, 246, 248))
    draw = ImageDraw.Draw(image)
    draw.text((24, 20), title, fill=(20, 30, 40))
    points_by_node: list[np.ndarray] = []
    for node_name in sorted(scene.graph.nodes_geometry):
        transform, geometry_name = scene.graph[node_name]
        vertices = np.asarray(scene.geometry[geometry_name].vertices, dtype=np.float64)
        if not len(vertices):
            continue
        stride = max(1, int(np.ceil(len(vertices) / 40000)))
        sampled = trimesh.transform_points(vertices[::stride], transform)
        points_by_node.append(sampled)
    if not points_by_node:
        write_preview(path, title=title, width=width, height=height, lines=lines)
        return
    all_points = np.concatenate(points_by_node, axis=0)
    view = np.column_stack(
        (
            0.866 * all_points[:, 0] - 0.5 * all_points[:, 1],
            0.28 * all_points[:, 0] + 0.48 * all_points[:, 1] - 0.83 * all_points[:, 2],
        )
    )
    lower = np.min(view, axis=0)
    upper = np.max(view, axis=0)
    extent = np.maximum(upper - lower, 1e-9)
    margin_x = 36
    margin_top = 60
    margin_bottom = 80
    scale = min(
        (width - 2 * margin_x) / extent[0],
        (height - margin_top - margin_bottom) / extent[1],
    )
    offset = np.asarray(
        (
            margin_x + ((width - 2 * margin_x) - extent[0] * scale) / 2,
            margin_top + ((height - margin_top - margin_bottom) - extent[1] * scale) / 2,
        )
    )
    colors = (
        (42, 111, 151),
        (204, 92, 72),
        (67, 143, 94),
        (143, 98, 173),
        (214, 155, 49),
    )
    cursor = 0
    for index, points in enumerate(points_by_node):
        count = len(points)
        projected = view[cursor : cursor + count]
        cursor += count
        pixels = (projected - lower) * scale + offset
        pixels[:, 1] = height - margin_bottom - (pixels[:, 1] - margin_top)
        draw.point(
            [(int(point[0]), int(point[1])) for point in pixels],
            fill=colors[index % len(colors)],
        )
    draw.rectangle(
        (margin_x, margin_top, width - margin_x, height - margin_bottom),
        outline=(60, 75, 90),
        width=1,
    )
    summary = " | ".join(lines[:3])
    draw.text((24, height - 48), summary, fill=(45, 55, 65))
    image.save(path, format="PNG", optimize=False, compress_level=9)


__all__ = ["render_scene_preview", "write_preview"]
