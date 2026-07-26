from __future__ import annotations

import hashlib
import math
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
    vertices: Any,
    global_faces: Any,
    face_ids: list[int],
    *,
    min_faces: int,
    min_relative_area: float,
) -> tuple[list[int], list[dict[str, object]]]:
    components = connected_face_components(global_faces, face_ids)
    area_by_face: dict[int, float] = {}
    for face_id in face_ids:
        face = global_faces[face_id]
        first = vertices[int(face[0])]
        second = vertices[int(face[1])]
        third = vertices[int(face[2])]
        left = tuple(float(second[axis]) - float(first[axis]) for axis in range(3))
        right = tuple(float(third[axis]) - float(first[axis]) for axis in range(3))
        cross = (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )
        area_by_face[face_id] = 0.5 * math.sqrt(sum(value * value for value in cross))
    total_faces = max(len(face_ids), 1)
    total_area = max(sum(area_by_face.values()), 1e-30)
    retained: list[int] = []
    diagnostics = []
    for index, component in enumerate(components, 1):
        face_ratio = len(component) / total_faces
        surface_area = sum(area_by_face[face_id] for face_id in component)
        area_ratio = surface_area / total_area
        keep = len(component) >= min_faces and area_ratio >= min_relative_area
        if keep:
            retained.extend(component)
        diagnostics.append(
            {
                "component_id": f"component_{index:04d}",
                "face_count": len(component),
                "surface_area_arbitrary_units_squared": surface_area,
                "relative_face_ratio": face_ratio,
                "relative_surface_area": area_ratio,
                "retained": keep,
                "removal_reason": (None if keep else "below_scale_independent_component_threshold"),
            }
        )
    return sorted(retained), diagnostics


def median_edge_length(
    vertices: Any,
    faces: Any,
    *,
    maximum_sample_faces: int = 200_000,
) -> float:
    import numpy as np

    face_count = len(faces)
    if face_count == 0:
        raise ValueError("cannot estimate edge scale from an empty mesh")
    stride = max(1, face_count // maximum_sample_faces)
    sampled = np.asarray(faces[::stride], dtype=np.int64)[:maximum_sample_faces]
    triangles = np.asarray(vertices, dtype=np.float64)[sampled]
    lengths = np.concatenate(
        [
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ]
    )
    positive = lengths[np.isfinite(lengths) & (lengths > 0)]
    if not positive.size:
        raise ValueError("global mesh has no finite positive-length edges")
    return float(np.median(positive))


def seam_aware_component_diagnostics(
    vertices: Any,
    faces: Any,
    face_ids: list[int],
    *,
    median_edge: float,
    centroid_distance_multiplier: float,
    endpoint_distance_multiplier: float,
    normal_cosine: float,
) -> dict[str, int]:
    """Group existing exact components across likely duplicated-vertex seams."""
    import itertools

    import numpy as np

    components = connected_face_components(faces, face_ids)
    if len(components) < 2:
        return {
            "exact_component_count": len(components),
            "seam_aware_component_count": len(components),
            "potential_chunk_seam_merges": 0,
        }
    vertices_array = np.asarray(vertices, dtype=np.float64)
    faces_array = np.asarray(faces, dtype=np.int64)
    component_by_face = {
        face_id: component_index
        for component_index, component in enumerate(components)
        for face_id in component
    }
    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    for face_id in face_ids:
        face = faces_array[face_id]
        for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_counts[(min(int(left), int(right)), max(int(left), int(right)))] += 1
    centroid_limit = median_edge * centroid_distance_multiplier
    endpoint_limit = median_edge * endpoint_distance_multiplier
    cell = max(centroid_limit, endpoint_limit, 1e-12)
    buckets: dict[tuple[int, int, int], list[tuple[int, Any, Any, Any]]] = defaultdict(list)
    for face_id in face_ids:
        face = faces_array[face_id]
        boundary_edges = [
            (int(left), int(right))
            for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
            if edge_counts[(min(int(left), int(right)), max(int(left), int(right)))] == 1
        ]
        if not boundary_edges:
            continue
        triangle = vertices_array[face]
        centroid = triangle.mean(axis=0)
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        norm = np.linalg.norm(normal)
        if norm <= 0:
            continue
        normal /= norm
        for left, right in boundary_edges:
            endpoints = vertices_array[[left, right]]
            midpoint = endpoints.mean(axis=0)
            key = tuple(np.floor(midpoint / cell).astype(np.int64).tolist())
            buckets[key].append((component_by_face[face_id], endpoints, centroid, normal))
    parent = list(range(len(components)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    offsets = list(itertools.product((-1, 0, 1), repeat=3))
    for key in sorted(buckets):
        candidates = buckets[key]
        neighbors = [
            item
            for offset in offsets
            for item in buckets.get(
                tuple(key[axis] + offset[axis] for axis in range(3)),
                [],
            )
        ]
        for left in candidates:
            for right in neighbors:
                if left[0] >= right[0] or find(left[0]) == find(right[0]):
                    continue
                if np.linalg.norm(left[2] - right[2]) > centroid_limit:
                    continue
                endpoint_distance = min(
                    np.linalg.norm(left_point - right_point)
                    for left_point in left[1]
                    for right_point in right[1]
                )
                if endpoint_distance > endpoint_limit:
                    continue
                if abs(float(np.dot(left[3], right[3]))) < normal_cosine:
                    continue
                union(left[0], right[0])
    seam_count = len({find(index) for index in range(len(components))})
    return {
        "exact_component_count": len(components),
        "seam_aware_component_count": seam_count,
        "potential_chunk_seam_merges": len(components) - seam_count,
    }


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
