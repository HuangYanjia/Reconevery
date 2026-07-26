from __future__ import annotations

from typing import Any


def _bounds(values: Any) -> tuple[list[float] | None, list[float] | None]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return None, None
    return array.min(axis=0).tolist(), array.max(axis=0).tolist()


def _matrix_record(
    stage_id: str,
    source: str,
    matrix: Any,
    *,
    mesh: Any | None = None,
    cameras: Any | None = None,
    sparse: Any | None = None,
) -> dict[str, object]:
    import numpy as np

    value = np.asarray(matrix, dtype=np.float64)
    inverse = np.linalg.inv(value)
    linear = value[:3, :3]
    determinant = float(np.linalg.det(linear))
    scale = float(abs(determinant) ** (1.0 / 3.0))
    rotation = linear / max(scale, 1e-15)
    orthogonality = float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
    roundtrip = float(np.linalg.norm(value @ inverse - np.eye(4), ord="fro"))
    mesh_min, mesh_max = _bounds(mesh if mesh is not None else [])
    camera_min, camera_max = _bounds(cameras if cameras is not None else [])
    sparse_min, sparse_max = _bounds(sparse if sparse is not None else [])
    return {
        "stage_id": stage_id,
        "transform_source": source,
        "matrix_from_previous": value.tolist(),
        "matrix_to_previous": inverse.tolist(),
        "determinant": determinant,
        "rotation_orthogonality_error": orthogonality,
        "scale": scale,
        "translation": value[:3, 3].tolist(),
        "roundtrip_error": roundtrip,
        "mesh_bounds_min": mesh_min,
        "mesh_bounds_max": mesh_max,
        "camera_center_bounds_min": camera_min,
        "camera_center_bounds_max": camera_max,
        "sparse_point_bounds_min": sparse_min,
        "sparse_point_bounds_max": sparse_max,
    }


def _transform_points(points: Any, matrix: Any) -> Any:
    import numpy as np

    values = np.asarray(points, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    return values @ transform[:3, :3].T + transform[:3, 3]


def _working_camera_pose(
    camera_pose: dict[str, Any],
    colmap_to_working: Any,
) -> tuple[dict[str, object], float]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    matrix = np.asarray(colmap_to_working, dtype=np.float64)
    scale = float(abs(np.linalg.det(matrix[:3, :3])) ** (1.0 / 3.0))
    world_rotation = matrix[:3, :3] / max(scale, 1e-15)
    source = camera_pose["transform_world_from_camera"]
    camera_rotation = Rotation.from_quat(source["rotation_xyzw"]).as_matrix()
    working_rotation = world_rotation @ camera_rotation
    working_translation = _transform_points([source["translation"]], matrix)[0]
    return (
        {
            "transform_world_from_camera": {
                "translation": working_translation.tolist(),
                "rotation_xyzw": Rotation.from_matrix(working_rotation).as_quat().tolist(),
            }
        },
        scale,
    )


def _render_equivalence(
    *,
    final_vertices: Any,
    working_vertices: Any,
    faces: Any,
    camera_pose: dict[str, Any],
    intrinsics: dict[str, Any],
    colmap_to_working: Any,
    face_chunk_size: int,
) -> tuple[float | None, float]:
    import numpy as np
    import torch
    from object_lifting_worker.rasterization import NvdiffrastRasterizer

    final_rasterizer = NvdiffrastRasterizer(
        final_vertices,
        faces,
        face_chunk_size=face_chunk_size,
    )
    final = final_rasterizer.rasterize(camera_pose, intrinsics)
    del final_rasterizer
    torch.cuda.empty_cache()

    working_pose, scale = _working_camera_pose(camera_pose, colmap_to_working)
    working_rasterizer = NvdiffrastRasterizer(
        working_vertices,
        faces,
        face_chunk_size=face_chunk_size,
    )
    working = working_rasterizer.rasterize(working_pose, intrinsics)
    del working_rasterizer
    torch.cuda.empty_cache()

    intersection = np.logical_and(final.valid, working.valid)
    union = np.logical_or(final.valid, working.valid)
    silhouette_iou = float(intersection.sum() / max(int(union.sum()), 1))
    if not intersection.any():
        return None, silhouette_iou
    working_depth_colmap_units = working.depth[intersection] / max(scale, 1e-15)
    final_depth = final.depth[intersection]
    relative_depth_error = np.abs(working_depth_colmap_units - final_depth) / np.maximum(
        np.abs(final_depth),
        1e-12,
    )
    return float(np.median(relative_depth_error)), silhouette_iou


def audit_transform_chain(
    *,
    working_transform: dict[str, Any],
    chunk_transforms: dict[str, Any],
    final_mesh_vertices: Any,
    camera_centers: Any,
    sparse_points: Any,
    working_mesh_vertices: Any | None,
    final_mesh_faces: Any,
    camera_pose: dict[str, Any],
    undistorted_intrinsics: dict[str, Any],
    face_chunk_size: int,
    tolerance: float,
) -> dict[str, object]:
    import numpy as np

    colmap_to_working = np.asarray(
        working_transform["matrix_colmap_to_working"],
        dtype=np.float64,
    )
    working_to_colmap = np.asarray(
        working_transform["matrix_working_to_colmap"],
        dtype=np.float64,
    )
    identity = np.eye(4, dtype=np.float64)
    sparse = np.asarray(sparse_points, dtype=np.float64)
    cameras = np.asarray(camera_centers, dtype=np.float64)
    final_vertices = np.asarray(final_mesh_vertices, dtype=np.float64)
    stride = max(1, len(final_vertices) // 10000)
    sampled_final = final_vertices[::stride][:10000]
    sparse_roundtrip = _transform_points(
        _transform_points(sparse, colmap_to_working),
        working_to_colmap,
    )
    camera_roundtrip = _transform_points(
        _transform_points(cameras, colmap_to_working),
        working_to_colmap,
    )
    mesh_roundtrip = _transform_points(
        _transform_points(sampled_final, colmap_to_working),
        working_to_colmap,
    )
    sparse_error = float(
        np.max(np.linalg.norm(sparse_roundtrip - sparse, axis=1)) if len(sparse) else 0.0
    )
    camera_error = float(
        np.max(np.linalg.norm(camera_roundtrip - cameras, axis=1)) if len(cameras) else 0.0
    )
    mesh_error = float(
        np.max(np.linalg.norm(mesh_roundtrip - sampled_final, axis=1))
        if len(sampled_final)
        else 0.0
    )
    inverse_error = float(
        np.linalg.norm(working_to_colmap @ colmap_to_working - identity, ord="fro")
    )
    if working_mesh_vertices is not None:
        working_vertices = np.asarray(working_mesh_vertices, dtype=np.float64)
        working_stride = max(1, len(working_vertices) // max(len(sampled_final), 1))
        sampled_working = working_vertices[::working_stride][: len(sampled_final)]
        transformed = _transform_points(sampled_working, working_to_colmap)
        if len(transformed) == len(sampled_final):
            pre_post_error = float(np.median(np.linalg.norm(transformed - sampled_final, axis=1)))
        else:
            pre_post_error = None
    else:
        reconstructed_working = _transform_points(sampled_final, colmap_to_working)
        transformed = _transform_points(reconstructed_working, working_to_colmap)
        pre_post_error = float(
            np.max(np.linalg.norm(transformed - sampled_final, axis=1))
            if len(sampled_final)
            else 0.0
        )
        working_vertices = _transform_points(final_vertices, colmap_to_working)
    render_depth_error, render_silhouette_iou = _render_equivalence(
        final_vertices=final_vertices,
        working_vertices=working_vertices,
        faces=final_mesh_faces,
        camera_pose=camera_pose,
        intrinsics=undistorted_intrinsics,
        colmap_to_working=colmap_to_working,
        face_chunk_size=face_chunk_size,
    )
    stages = [
        _matrix_record(
            "A_colmap_arbitrary",
            "identity canonical COLMAP source frame",
            identity,
            mesh=sampled_final,
            cameras=cameras,
            sparse=sparse,
        ),
        _matrix_record(
            "B_genrecon_working",
            "reconstruction/global/raw/working_transform.json",
            colmap_to_working,
            mesh=_transform_points(sampled_final, colmap_to_working),
            cameras=_transform_points(cameras, colmap_to_working),
            sparse=_transform_points(sparse, colmap_to_working),
        ),
    ]
    chunks = chunk_transforms.get("chunks", [])
    for index, chunk in enumerate(chunks):
        matrix = chunk.get("M_original_to_chunk")
        if matrix is None:
            continue
        stages.append(
            _matrix_record(
                f"C_chunk_{chunk.get('index', index)}",
                "reconstruction/global/raw/chunk_transforms.json",
                matrix,
            )
        )
    stages.extend(
        [
            _matrix_record(
                "D_official_working_output",
                "official GenRecon reconstructed working output",
                identity,
            ),
            _matrix_record(
                "E_glb_conversion_output",
                "official chunked_to_glb output",
                identity,
            ),
            _matrix_record(
                "F_final_colmap_mesh",
                "matrix_working_to_colmap applied by Reconevery",
                working_to_colmap,
                mesh=sampled_final,
                cameras=cameras,
                sparse=sparse,
            ),
        ]
    )
    matrix_checks = [
        abs(stage["determinant"]) > 1e-12 and stage["roundtrip_error"] <= max(tolerance, 1e-8)
        for stage in stages
    ]
    checks = {
        "working_matrices_are_inverse": inverse_error <= tolerance,
        "sparse_roundtrip": sparse_error <= tolerance,
        "camera_center_roundtrip": camera_error <= tolerance,
        "sampled_mesh_roundtrip": mesh_error <= tolerance,
        "stage_matrices_invertible": all(matrix_checks),
        "pre_post_coordinate_equivalence": (
            pre_post_error is not None and pre_post_error <= max(tolerance, 1e-6)
        ),
        "pre_post_render_depth_equivalence": (
            render_depth_error is not None and render_depth_error <= max(tolerance, 1e-5)
        ),
        "pre_post_render_silhouette_equivalence": render_silhouette_iou >= 0.999,
    }
    consistent = all(checks.values())
    warnings = []
    if working_mesh_vertices is None:
        warnings.append(
            "No immutable pre-canonical working mesh was available; coordinate and render "
            "equivalence reconstructed working vertices with the recorded inverse transform."
        )
    return {
        "schema_version": "0.1.0",
        "status": "consistent" if consistent else "transform_chain_bug",
        "stages": stages,
        "colmap_working_roundtrip_error": max(inverse_error, sparse_error),
        "camera_basis_roundtrip_error": camera_error,
        "sampled_mesh_roundtrip_error": mesh_error,
        "pre_post_render_depth_error": render_depth_error,
        "pre_post_render_silhouette_iou": render_silhouette_iou,
        "pre_post_render_equivalent": (
            checks["pre_post_render_depth_equivalence"]
            and checks["pre_post_render_silhouette_equivalence"]
        ),
        "raw_working_mesh_available": working_mesh_vertices is not None,
        "raw_working_scene_available": False,
        "checks": checks,
        "warnings": warnings,
    }
