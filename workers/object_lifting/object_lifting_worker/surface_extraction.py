from __future__ import annotations

import hashlib
import struct
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def connected_face_components(
    global_faces: Any,
    face_ids: list[int],
) -> list[list[int]]:
    if not face_ids:
        return []
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_id in face_ids:
        face = global_faces[face_id]
        for left, right in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            edge_faces[(min(left, right), max(left, right))].append(face_id)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for touching in edge_faces.values():
        for face_id in touching:
            adjacency[face_id].update(other for other in touching if other != face_id)
    remaining = set(face_ids)
    components: list[list[int]] = []
    while remaining:
        seed = min(remaining)
        queue = deque([seed])
        remaining.remove(seed)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda item: (-len(item), item[0]))


def filter_components(
    global_faces: Any,
    face_ids: list[int],
    *,
    min_faces: int,
    min_relative_area: float,
) -> tuple[list[int], list[dict[str, object]]]:
    components = connected_face_components(global_faces, face_ids)
    total = max(len(face_ids), 1)
    retained: list[int] = []
    diagnostics = []
    for index, component in enumerate(components, 1):
        ratio = len(component) / total
        keep = len(component) >= min_faces and ratio >= min_relative_area
        if keep:
            retained.extend(component)
        diagnostics.append(
            {
                "component_id": f"component_{index:04d}",
                "face_count": len(component),
                "relative_face_ratio": ratio,
                "retained": keep,
                "removal_reason": (None if keep else "below_scale_independent_component_threshold"),
            }
        )
    return sorted(retained), diagnostics


def write_face_ids(
    path: Path,
    values: list[int],
    *,
    global_mesh_sha256: str,
    relative_path: str,
) -> dict[str, object]:
    maximum = max(values, default=0)
    dtype = "uint32" if maximum <= 2**32 - 1 else "uint64"
    format_code = "I" if dtype == "uint32" else "Q"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        for value in values:
            file.write(struct.pack("<" + format_code, value))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "relative_path": relative_path,
        "dtype": dtype,
        "byte_order": "little",
        "count": len(values),
        "global_mesh_sha256": global_mesh_sha256,
        "minimum_face_id": min(values) if values else None,
        "maximum_face_id": max(values) if values else None,
        "content_sha256": digest,
    }


def extract_surface_assets(
    vertices: Any,
    faces: Any,
    face_ids: list[int],
    *,
    mesh_path: Path,
    points_path: Path,
) -> dict[str, object]:
    import numpy as np
    import trimesh

    if not face_ids:
        return {
            "vertex_count": 0,
            "face_count": 0,
            "bbox_min": None,
            "bbox_max": None,
            "bbox_extent": None,
            "centroid": None,
        }
    selected_faces = np.asarray(faces[face_ids], dtype=np.int64)
    unique_vertices, inverse = np.unique(selected_faces.reshape(-1), return_inverse=True)
    local_vertices = np.asarray(vertices[unique_vertices], dtype=np.float32)
    local_faces = inverse.reshape(-1, 3)
    surface = trimesh.Trimesh(
        vertices=local_vertices,
        faces=local_faces,
        process=False,
        validate=False,
    )
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    surface.export(mesh_path, file_type="ply", encoding="binary_little_endian")
    centroids = local_vertices[local_faces].mean(axis=1)
    trimesh.PointCloud(centroids).export(
        points_path,
        file_type="ply",
        encoding="binary_little_endian",
    )
    minimum = local_vertices.min(axis=0)
    maximum = local_vertices.max(axis=0)
    return {
        "vertex_count": int(len(local_vertices)),
        "face_count": int(len(local_faces)),
        "bbox_min": minimum.tolist(),
        "bbox_max": maximum.tolist(),
        "bbox_extent": (maximum - minimum).tolist(),
        "centroid": local_vertices.mean(axis=0).tolist(),
    }
