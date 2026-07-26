from __future__ import annotations

from typing import Any


def sample_mesh_surface(
    vertices: Any,
    faces: Any,
    *,
    maximum_vertices: int,
    maximum_face_centroids: int,
) -> tuple[Any, Any, float]:
    import numpy as np

    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    vertex_stride = max(1, len(vertex_array) // maximum_vertices)
    sampled_vertices = vertex_array[::vertex_stride][:maximum_vertices]
    face_stride = max(1, len(face_array) // maximum_face_centroids)
    sampled_faces = face_array[::face_stride][:maximum_face_centroids]
    triangles = vertex_array[sampled_faces]
    centroids = triangles.mean(axis=1)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norms = np.linalg.norm(cross, axis=1)
    normals = np.zeros_like(cross)
    valid = norms > 1e-15
    normals[valid] = cross[valid] / norms[valid, None]
    points = np.concatenate([sampled_vertices, centroids], axis=0)
    point_normals = np.concatenate([np.zeros_like(sampled_vertices), normals], axis=0)
    low = np.percentile(vertex_array, 0.5, axis=0)
    high = np.percentile(vertex_array, 99.5, axis=0)
    diagonal = float(np.linalg.norm(high - low))
    if not diagonal > 0:
        raise ValueError("global mesh robust bounds are degenerate")
    return points, point_normals, diagonal
