from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from completion_evaluation_worker.candidate_io import (
    load_candidate_surface,
    load_candidate_surface_with_normals,
)
from completion_evaluation_worker.dense_depth_scoring import relative_depth_metrics
from completion_evaluation_worker.dense_io import read_array, read_consistency_graph
from completion_evaluation_worker.measured_evidence import load_measured_evidence
from completion_evaluation_worker.native_render_dispatch import render_mesh_candidate
from completion_evaluation_worker.negative_space import classify_candidate_pixels
from completion_evaluation_worker.ranking import hard_gate_failures
from completion_evaluation_worker.silhouette_refinement import (
    FittingFrame,
    refine_sim3_on_fitting_views,
)
from completion_evaluation_worker.sim3_registration import (
    measured_surface_residuals,
    register_asymmetric_sim3,
    unsigned_normal_agreement,
)
from completion_evaluation_worker.training_evidence import (
    backproject_training_view,
    build_training_view,
    validate_training_points,
)
from completion_evaluation_worker.version import WORKER_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _worker_manifest(
    output_dir: Path,
    request_path: Path,
    *,
    action: str,
    runtime: float,
) -> None:
    _write_json(
        output_dir
        / ("worker_manifest.json" if action == "evidence" else f"{action}_worker_manifest.json"),
        {
            "worker_name": "completion_evaluation_worker",
            "worker_version": WORKER_VERSION,
            "action": action,
            "backend": "numpy_scipy_nvdiffrast",
            "request_sha256": _sha256(request_path),
            "official_repository": None,
            "official_code_commit": None,
            "checkpoint_repository": None,
            "checkpoint_revision": None,
            "checkpoint_hashes": {},
            "runtime_seconds": runtime,
            "peak_gpu_memory_bytes": _peak_gpu(),
            "peak_host_memory_bytes": None,
            "warnings": [],
        },
    )


def _peak_gpu() -> int | None:
    try:
        import torch

        return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    except ImportError:
        return None


def _rotation_xyzw(value: list[float]) -> np.ndarray:
    x, y, z, w = np.asarray(value, dtype=np.float64)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _camera_matrices(camera: dict[str, Any]) -> dict[str, np.ndarray]:
    result = {}
    for pose in camera["poses"]:
        transform = pose["transform_world_from_camera"]
        matrix = np.eye(4)
        matrix[:3, :3] = _rotation_xyzw(transform["rotation_xyzw"])
        matrix[:3, 3] = transform["translation"]
        result[pose["frame_id"]] = matrix
    return result


def _wxyz_rotation(values: list[float]) -> np.ndarray:
    w, x, y, z = np.asarray(values, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0:
        raise ValueError("backend layout quaternion is zero")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _backend_layout_world_initialization(
    candidate: dict[str, Any],
    world_from_camera: dict[str, np.ndarray],
) -> np.ndarray | None:
    layout = candidate.get("backend_predicted_layout")
    if candidate.get("backend") != "sam3d_objects" or not isinstance(layout, dict):
        return None
    try:
        scale_values = np.asarray(layout["scale"], dtype=np.float64).reshape(-1)
        rotation_values = np.asarray(layout["rotation"], dtype=np.float64).reshape(-1)
        translation = np.asarray(layout["translation"], dtype=np.float64).reshape(-1)
        if len(scale_values) != 3 or len(rotation_values) != 4 or len(translation) != 3:
            return None
        if not np.allclose(scale_values, scale_values[0], rtol=1e-6, atol=1e-9):
            return None
        camera_from_candidate = np.eye(4, dtype=np.float64)
        camera_from_candidate[:3, :3] = (
            float(scale_values[0]) * _wxyz_rotation(rotation_values.tolist()).T
        )
        camera_from_candidate[:3, 3] = translation
        # The official layout is local-to-anchor-camera in PyTorch3D axes.
        pytorch3d_to_opencv = np.diag([-1.0, -1.0, 1.0, 1.0])
        return (
            world_from_camera[candidate["anchor_frame_id"]]
            @ pytorch3d_to_opencv
            @ camera_from_candidate
        )
    except (KeyError, TypeError, ValueError):
        return None


def _remap_mask(
    mask_path: Path,
    undistortion: dict[str, Any],
) -> np.ndarray:
    source = np.asarray(Image.open(mask_path).convert("L"))
    source_width, source_height = undistortion["source_dimensions"]
    if source.shape != (source_height, source_width):
        raise ValueError("canonical mask dimensions do not match undistortion source")
    fx, fy, cx, cy = undistortion["source_intrinsics"]
    dense_fx, dense_fy, dense_cx, dense_cy = undistortion["dense_intrinsics"]
    dense_width, dense_height = undistortion["dense_dimensions"]
    source_k = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    dense_k = np.asarray(
        [[dense_fx, 0, dense_cx], [0, dense_fy, dense_cy], [0, 0, 1]],
        np.float64,
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        source_k,
        np.asarray(undistortion["source_distortion"], np.float64),
        None,
        dense_k,
        (dense_width, dense_height),
        cv2.CV_32FC1,
    )
    remapped = cv2.remap(
        source,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return remapped > 0


def _write_ply(
    path: Path,
    points: np.ndarray,
    normals: np.ndarray | None = None,
) -> None:
    if normals is not None and normals.shape != points.shape:
        raise ValueError("measured normal count must match measured point count")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as file:
        file.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            + (
                "property float nx\nproperty float ny\nproperty float nz\n"
                if normals is not None
                else ""
            )
            + "element face 0\nproperty list uchar int vertex_indices\nend_header\n"
        )
        for index, point in enumerate(points):
            values = [*point]
            if normals is not None:
                values.extend(normals[index])
            file.write(" ".join(f"{value:.12g}" for value in values) + "\n")


def _write_renderer_control_mesh(
    path: Path,
    points: np.ndarray,
    normals: np.ndarray,
) -> tuple[int, float | None]:
    """Write open tangent triangles around fitting-only measured samples."""
    if not len(points):
        return 0, None
    from scipy.spatial import cKDTree

    hashes = np.asarray(
        [
            int.from_bytes(hashlib.sha256(point.tobytes()).digest()[:8], "little")
            for point in points
        ],
        dtype=np.uint64,
    )
    spacing_indices = np.argsort(hashes, kind="stable")[: min(len(points), 20_000)]
    nearest = cKDTree(points).query(points[spacing_indices], k=2)[0][:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 1e-12)]
    if not len(nearest):
        return 0, None
    radius = float(np.median(nearest)) * 0.75
    selected = np.argsort(hashes, kind="stable")[: min(len(points), 50_000)]
    selected_points = points[selected]
    selected_normals = normals[selected]
    reference = np.tile(np.asarray([1.0, 0.0, 0.0]), (len(selected), 1))
    reference[np.abs(np.sum(reference * selected_normals, axis=1)) > 0.9] = (0.0, 1.0, 0.0)
    tangent = np.cross(selected_normals, reference)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-12)
    bitangent = np.cross(selected_normals, tangent)
    vertices = np.stack(
        (
            selected_points + radius * tangent,
            selected_points + radius * (-0.5 * tangent + math.sqrt(3.0) * 0.5 * bitangent),
            selected_points + radius * (-0.5 * tangent - math.sqrt(3.0) * 0.5 * bitangent),
        ),
        axis=1,
    ).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as file:
        file.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(vertices)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            f"element face {len(selected)}\n"
            "property list uchar int vertex_indices\nend_header\n"
        )
        for vertex in vertices:
            file.write(" ".join(f"{value:.12g}" for value in vertex) + "\n")
        for index in range(len(selected)):
            base = 3 * index
            file.write(f"3 {base} {base + 1} {base + 2}\n")
    return len(selected), radius


def _prepare_evidence(
    request_path: Path,
    input_root: Path,
    output_dir: Path,
) -> None:
    started = time.monotonic()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    camera = json.loads(
        (input_root / request["camera_reconstruction_path"]).read_text(encoding="utf-8")
    )
    undistortion_manifest = json.loads(
        (input_root / request["dense_undistortion_manifest_path"]).read_text(encoding="utf-8")
    )
    undistortion = {item["frame_id"]: item for item in undistortion_manifest["records"]}
    world_from_camera = _camera_matrices(camera)
    objects = []
    for object_id, values in sorted(request["object_inputs"].items()):
        views = []
        image_count = len(camera["registered_frame_ids"])
        for frame_id in values["training_frame_ids"]:
            mask = _remap_mask(
                input_root / values["training_masks"][frame_id],
                undistortion[frame_id],
            )
            depth = read_array(
                input_root / values["training_dense_maps"][frame_id]["depth_path"],
                1,
            )
            normal = read_array(
                input_root / values["training_dense_maps"][frame_id]["normal_path"],
                3,
            )
            consistency = read_consistency_graph(
                input_root / values["training_dense_maps"][frame_id]["consistency_graph_path"],
                depth.shape,
                image_count,
            )
            views.append(
                build_training_view(
                    frame_id=frame_id,
                    depth=depth,
                    normal=normal,
                    consistency=consistency,
                    mask=mask,
                    intrinsics=undistortion[frame_id]["dense_intrinsics"],
                    world_from_camera=world_from_camera[frame_id],
                    frame_score=float(values["frame_scores"][frame_id]),
                    backprojection_configuration=request["backprojection_configuration"],
                )
            )
        collected = []
        collected_normals = []
        raw_sample_count = 0
        boundary_rejected_count = 0
        invalid_geometry_rejected_count = 0
        sam_score_rejected_count = 0
        consistency_rejected_count = 0
        depth_discontinuity_rejected_count = 0
        multi_view_rejected_count = 0
        frame_records = []
        for view in views:
            result = backproject_training_view(
                view,
                request["backprojection_configuration"],
            )
            keep, supporting, residual = validate_training_points(
                result.points,
                view.frame_id,
                views,
                request["consistency_configuration"],
            )
            raw_sample_count += result.raw_sample_count
            boundary_rejected_count += result.boundary_rejected_count
            invalid_geometry_rejected_count += result.invalid_geometry_rejected_count
            sam_score_rejected_count += result.sam_score_rejected_count
            consistency_rejected_count += result.consistency_rejected_count
            depth_discontinuity_rejected_count += result.depth_discontinuity_rejected_count
            multi_view_rejected_count += int(np.count_nonzero(~keep))
            collected.append(result.points[keep])
            collected_normals.append(result.normals[keep])
            frame_records.append(
                {
                    "frame_id": view.frame_id,
                    "raw_sample_count": result.raw_sample_count,
                    "backprojected_point_count": len(result.points),
                    "validated_point_count": int(np.count_nonzero(keep)),
                    "maximum_supporting_views": (
                        int(np.max(supporting[keep])) if np.any(keep) else 0
                    ),
                    "median_relative_depth_residual": (
                        float(np.median(residual[keep])) if np.any(keep) else None
                    ),
                }
            )
        points = np.concatenate(collected, axis=0) if collected else np.empty((0, 3))
        normals = (
            np.concatenate(collected_normals, axis=0) if collected_normals else np.empty((0, 3))
        )
        maximum_points = int(request["backprojection_configuration"]["maximum_samples_per_object"])
        if len(points) > maximum_points:
            hashes = np.asarray(
                [
                    int.from_bytes(hashlib.sha256(point.tobytes()).digest()[:8], "little")
                    for point in points
                ],
                dtype=np.uint64,
            )
            selected = np.argsort(hashes, kind="stable")[:maximum_points]
            points = points[selected]
            normals = normals[selected]
        point_path = output_dir / object_id / "training_points.ply"
        _write_ply(point_path, points, normals if len(normals) else None)
        control_path = output_dir / object_id / "training_renderer_control_mesh.ply"
        control_face_count, control_radius = _write_renderer_control_mesh(
            control_path,
            points,
            normals,
        )
        geometry_path = output_dir / object_id / "training_measured_geometry.json"
        normal_hash = hashlib.sha256(
            np.ascontiguousarray(normals, dtype="<f8").tobytes()
        ).hexdigest()
        _write_json(
            geometry_path,
            {
                "schema_version": "0.1.0",
                "object_id": object_id,
                "training_frame_ids": values["training_frame_ids"],
                "heldout_frame_ids": values["heldout_frame_ids"],
                "raw_sample_count": raw_sample_count,
                "boundary_rejected_count": boundary_rejected_count,
                "invalid_geometry_rejected_count": invalid_geometry_rejected_count,
                "sam_score_rejected_count": sam_score_rejected_count,
                "consistency_rejected_count": consistency_rejected_count,
                "depth_discontinuity_rejected_count": (depth_discontinuity_rejected_count),
                "multi_view_rejected_count": multi_view_rejected_count,
                "validated_point_count": len(points),
                "supporting_fitting_views": [
                    record["frame_id"]
                    for record in frame_records
                    if record["validated_point_count"] > 0
                ],
                "point_cloud_path": point_path.relative_to(input_root).as_posix(),
                "point_cloud_sha256": _sha256(point_path),
                "normal_sha256": normal_hash,
                "renderer_control_mesh_path": (
                    control_path.relative_to(input_root).as_posix() if control_face_count else None
                ),
                "renderer_control_mesh_sha256": (
                    _sha256(control_path) if control_face_count else None
                ),
                "renderer_control_face_count": control_face_count,
                "renderer_control_triangle_radius": control_radius,
                "phase5a_all_view_validated_point_count": values[
                    "phase5a_all_view_validated_point_count"
                ],
                "phase5a_point_cloud_sha256": values["phase5a_point_cloud_sha256"],
                "frame_records": frame_records,
                "backprojection_configuration": request["backprojection_configuration"],
                "consistency_configuration": request["consistency_configuration"],
            },
        )
        heldout_path = output_dir / object_id / "heldout_measurements.json"
        _write_json(
            heldout_path,
            {
                "object_id": object_id,
                "heldout_frame_ids": values["heldout_frame_ids"],
                "pixel_evidence_materialized": False,
            },
        )
        objects.append(
            {
                "object_id": object_id,
                "training_frame_ids": values["training_frame_ids"],
                "heldout_frame_ids": values["heldout_frame_ids"],
                "training_points_path": point_path.relative_to(input_root).as_posix(),
                "training_points_sha256": _sha256(point_path),
                "training_point_count": len(points),
                "training_normals_available": bool(len(normals)),
                "training_geometry_manifest_path": (
                    geometry_path.relative_to(input_root).as_posix()
                ),
                "training_geometry_manifest_sha256": _sha256(geometry_path),
                "renderer_control_mesh_path": (
                    control_path.relative_to(input_root).as_posix() if control_face_count else None
                ),
                "renderer_control_mesh_sha256": (
                    _sha256(control_path) if control_face_count else None
                ),
                "heldout_measurement_manifest_path": (
                    heldout_path.relative_to(input_root).as_posix()
                ),
                "heldout_measurement_manifest_sha256": _sha256(heldout_path),
            }
        )
    _write_json(
        output_dir / "evidence_package.json",
        {
            "manifest_sha256": request["manifest_sha256"],
            "frame_sequence_digest": request["frame_sequence_digest"],
            "camera_reconstruction_sha256": request["camera_reconstruction_sha256"],
            "segmentation_tracking_sha256": request["segmentation_tracking_sha256"],
            "dense_depth_manifest_sha256": request["dense_depth_manifest_sha256"],
            "measured_geometry_sha256": request["measured_geometry_sha256"],
            "evidence_split_sha256": request["evidence_split_sha256"],
            "crop_manifest_sha256": request["crop_manifest_sha256"],
            "objects": objects,
            "coordinate_convention": request["coordinate_convention"],
            "scale_status": "scale_ambiguous",
        },
    )
    _worker_manifest(
        output_dir,
        request_path,
        action="evidence",
        runtime=time.monotonic() - started,
    )


def _generations(input_root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    candidates = []
    by_id = {}
    for name in (
        "sam3d_generation_manifest.json",
        "trellis2_generation_manifest.json",
        "measured_generation_manifest.json",
    ):
        manifest = json.loads(
            (input_root / "reconstruction" / "completion" / name).read_text(encoding="utf-8")
        )
        for candidate in manifest["candidates"]:
            candidates.append(candidate)
            by_id[candidate["candidate_id"]] = candidate
    return candidates, by_id


def _register(request_path: Path, input_root: Path, output_dir: Path) -> None:
    started = time.monotonic()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    package = json.loads(
        (input_root / "reconstruction/completion/evidence/evidence_package.json").read_text()
    )
    split = json.loads((input_root / "reconstruction/completion/evidence_split.json").read_text())
    camera = json.loads(
        (input_root / request["camera_reconstruction_path"]).read_text(encoding="utf-8")
    )
    undistortion_manifest = json.loads(
        (input_root / request["dense_undistortion_manifest_path"]).read_text(encoding="utf-8")
    )
    undistortion = {item["frame_id"]: item for item in undistortion_manifest["records"]}
    world_from_camera = _camera_matrices(camera)
    candidates, _ = _generations(input_root)
    evidence = {item["object_id"]: item for item in package["objects"]}
    splits = {item["object_id"]: item for item in split["objects"]}
    config = request["registration_configuration"]
    registrations = []
    for candidate in candidates:
        object_id = candidate["object_id"]
        point_record = evidence[object_id]
        measured, measured_normals = load_measured_evidence(
            input_root / point_record["training_points_path"]
        )
        if candidate["backend"] == "measured_partial_baseline":
            matrix = np.eye(4)
            median = p90 = 0.0
            scale = 1.0
            initialization = "measured_identity"
            symmetry_ambiguous = False
            normal_agreement = 1.0
            fitting_refined = False
            fitting_before = fitting_after = 0.0
        else:
            registration_asset = next(
                asset
                for asset in candidate["native_assets"]
                if asset["asset_id"] == candidate["registration_asset_id"]
            )
            surface, surface_normals = load_candidate_surface_with_normals(
                input_root / registration_asset["relative_path"],
                maximum_samples=int(config["maximum_surface_samples"]),
                seed=request["seed"],
            )
            layout_initialization = _backend_layout_world_initialization(
                candidate,
                world_from_camera,
            )
            result = register_asymmetric_sim3(
                surface,
                measured,
                iterations=int(config["maximum_iterations"]),
                trimmed_fraction=float(config["trimmed_fraction"]),
                extra_initializations=(
                    [("backend_predicted_layout", layout_initialization)]
                    if layout_initialization is not None
                    else None
                ),
            )
            matrix, scale = result.matrix, result.scale
            median, p90 = result.median_residual, result.p90_residual
            initialization = result.initialization
            symmetry_ambiguous = result.symmetry_ambiguous
            fitting = request["fitting_inputs"][object_id]
            fitting_frames = [
                FittingFrame(
                    camera_from_world=np.linalg.inv(world_from_camera[frame_id]),
                    intrinsics=tuple(undistortion[frame_id]["dense_intrinsics"]),
                    mask=_remap_mask(
                        input_root / fitting["mask_paths"][frame_id],
                        undistortion[frame_id],
                    ),
                    scene_depth=read_array(
                        input_root / fitting["depth_paths"][frame_id],
                        1,
                    ),
                )
                for frame_id in fitting["frame_ids"]
            ]
            refinement = refine_sim3_on_fitting_views(
                matrix,
                surface,
                measured,
                fitting_frames,
                maximum_iterations=int(config["fitting_refinement_iterations"]),
                maximum_points=int(config["fitting_refinement_maximum_points"]),
                maximum_rotation_degrees=float(
                    config["fitting_refinement_maximum_rotation_degrees"]
                ),
                maximum_scale_ratio=float(config["fitting_refinement_maximum_scale_ratio"]),
                translation_extent_ratio=float(
                    config["fitting_refinement_translation_extent_ratio"]
                ),
                minimum_scale=float(config["minimum_scale"]),
                maximum_scale=float(config["maximum_scale"]),
            )
            matrix = refinement.matrix
            scale = float(np.cbrt(np.linalg.det(matrix[:3, :3])))
            median, p90 = measured_surface_residuals(surface, measured, matrix)
            fitting_refined = refinement.refined
            fitting_before = refinement.objective_before
            fitting_after = refinement.objective_after
            normal_agreement = (
                unsigned_normal_agreement(
                    surface,
                    surface_normals,
                    measured,
                    measured_normals,
                    matrix,
                    trimmed_fraction=float(config["trimmed_fraction"]),
                )
                if surface_normals is not None and measured_normals is not None
                else 0.0
            )
        inverse = np.linalg.inv(matrix)
        rotation = matrix[:3, :3] / scale
        angle = math.degrees(math.acos(float(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))
        transform = {
            "matrix_world_from_candidate": matrix.reshape(-1).tolist(),
            "inverse_matrix": inverse.reshape(-1).tolist(),
            "scale": scale,
            "rotation_determinant": float(np.linalg.det(rotation)),
            "rotation_degrees": angle,
            "translation": matrix[:3, 3].tolist(),
            "measured_surface_median_residual": median,
            "measured_surface_p90_residual": p90,
            "normal_agreement": normal_agreement,
            "symmetry_ambiguous": symmetry_ambiguous,
            "fitting_refined": fitting_refined,
            "fitting_objective_before": fitting_before,
            "fitting_objective_after": fitting_after,
        }
        registrations.append(
            {
                "candidate_id": candidate["candidate_id"],
                "object_id": object_id,
                "registration_asset_id": candidate["registration_asset_id"],
                "registration_asset_path": candidate["registration_asset_path"],
                "status": "symmetry_ambiguous" if symmetry_ambiguous else "registered",
                "frozen_transform": transform,
                "fitting_frame_ids": splits[object_id]["registration_fitting_frames"],
                "heldout_frame_ids": splits[object_id]["heldout_validation_frames"],
                "training_points_sha256": point_record["training_points_sha256"],
                "fitting_objective": fitting_after,
                "failure_reason": None,
                "warnings": [
                    f"selected training-only initialization: {initialization}",
                    (
                        "fitting-view Sim(3) refinement accepted"
                        if fitting_refined
                        else "fitting-view Sim(3) refinement retained the measured-fit transform"
                    ),
                ],
            }
        )
    runtime = time.monotonic() - started
    _write_json(
        output_dir / "registration_manifest.json",
        {
            "request_sha256": _sha256(request_path),
            "registrations": registrations,
            "runtime_seconds": runtime,
            "peak_gpu_memory_bytes": _peak_gpu(),
            "peak_host_memory_bytes": None,
        },
    )
    _worker_manifest(output_dir, request_path, action="registration", runtime=runtime)


def _point_splat(
    points: np.ndarray,
    matrix_world_from_candidate: np.ndarray,
    camera_from_world: np.ndarray,
    intrinsics: list[float],
    dimensions: list[int],
) -> np.ndarray:
    width, height = dimensions
    fx, fy, cx, cy = intrinsics
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    world = (matrix_world_from_candidate @ homogeneous.T).T
    camera = (camera_from_world @ world.T).T
    valid = camera[:, 2] > 1e-8
    u = np.full(len(camera), -1, dtype=int)
    v = np.full(len(camera), -1, dtype=int)
    u[valid] = np.rint(fx * camera[valid, 0] / camera[valid, 2] + cx).astype(int)
    v[valid] = np.rint(fy * camera[valid, 1] / camera[valid, 2] + cy).astype(int)
    valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    depth = np.full((height, width), np.nan, dtype=np.float32)
    for x, y, z in sorted(
        zip(u[valid], v[valid], camera[valid, 2], strict=True),
        key=lambda item: float(item[2]),
        reverse=True,
    ):
        depth[y, x] = z
    return depth


def _mask_metrics(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    intersection = int(np.count_nonzero(predicted & target))
    predicted_area = int(np.count_nonzero(predicted))
    target_area = int(np.count_nonzero(target))
    union = predicted_area + target_area - intersection
    return (
        intersection / max(predicted_area, 1),
        intersection / max(target_area, 1),
        intersection / max(union, 1),
    )


def _write_heldout_overlay(
    path: Path,
    rgb_path: Path,
    target: np.ndarray,
    classification: dict[str, np.ndarray],
    *,
    candidate_id: str,
    frame_id: str,
) -> None:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    if rgb.shape[:2] != target.shape:
        raise ValueError(f"dense RGB dimensions do not match held-out mask: {rgb_path}")
    target_panel = rgb.copy()
    target_panel[target] = (0.35 * target_panel[target] + 0.65 * np.asarray([40, 190, 90])).astype(
        np.uint8
    )
    candidate_panel = rgb.copy()
    for region, color in (
        ("visible", np.asarray([40, 210, 230])),
        ("occluded", np.asarray([240, 190, 45])),
        ("negative", np.asarray([235, 65, 65])),
        ("front", np.asarray([210, 55, 210])),
    ):
        selected = classification[region]
        candidate_panel[selected] = (0.35 * candidate_panel[selected] + 0.65 * color).astype(
            np.uint8
        )
    predicted = classification["visible"]
    error_panel = rgb.copy()
    intersection = predicted & target
    false_positive = predicted & ~target
    false_negative = target & ~predicted
    for selected, color in (
        (intersection, np.asarray([45, 200, 90])),
        (false_positive, np.asarray([235, 65, 65])),
        (false_negative, np.asarray([65, 110, 235])),
    ):
        error_panel[selected] = (0.25 * error_panel[selected] + 0.75 * color).astype(np.uint8)
    height, width = target.shape
    header = 44
    sheet = Image.new("RGB", (width * 2, height * 2 + header), (245, 247, 249))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), f"{candidate_id} | {frame_id}", fill=(20, 30, 45))
    draw.text(
        (10, 25),
        "RGB | target mask | candidate visibility | intersection / FP / FN",
        fill=(60, 70, 85),
    )
    for offset, panel in (
        ((0, header), rgb),
        ((width, header), target_panel),
        ((0, height + header), candidate_panel),
        ((width, height + header), error_panel),
    ):
        sheet.paste(Image.fromarray(panel), offset)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG", compress_level=9)


def _declared_asset(
    candidate: dict[str, Any],
    *,
    asset_id_field: str,
    asset_path_field: str,
) -> dict[str, Any]:
    asset_id = candidate[asset_id_field]
    asset_path = candidate[asset_path_field]
    matches = [
        item
        for item in candidate["native_assets"]
        if item["asset_id"] == asset_id and item["relative_path"] == asset_path
    ]
    if len(matches) != 1:
        raise ValueError(
            f"candidate {candidate['candidate_id']} does not declare "
            f"{asset_id_field}={asset_id!r} at {asset_path!r}"
        )
    return matches[0]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return None
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def _bbox_intersection(
    first: tuple[int, int, int, int] | None,
    second: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if first is None or second is None:
        return None
    result = (
        max(first[0], second[0]),
        max(first[1], second[1]),
        min(first[2], second[2]),
        min(first[3], second[3]),
    )
    return result if result[0] < result[2] and result[1] < result[3] else None


def _finite_percentiles(values: np.ndarray) -> tuple[float | None, float | None, float | None]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None, None, None
    return (
        float(finite.min()),
        float(np.median(finite)),
        float(finite.max()),
    )


def _candidate_depth(
    *,
    input_root: Path,
    asset: dict[str, Any],
    points: np.ndarray,
    transform: np.ndarray,
    camera_from_world: np.ndarray,
    undistortion: dict[str, Any],
    measured_baseline: bool,
) -> np.ndarray:
    dimensions = undistortion["dense_dimensions"]
    if measured_baseline and asset.get("role") != "fitting_only_open_surface_renderer_control":
        return _point_splat(
            points,
            transform,
            camera_from_world,
            undistortion["dense_intrinsics"],
            dimensions,
        )
    if asset["format"] not in {"mesh_glb", "mesh_ply", "pbr_glb"}:
        raise RuntimeError(
            "evaluation representation is not a mesh; use the official backend renderer "
            "for native Gaussian evaluation"
        )
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    camera_points = (camera_from_world @ (transform @ homogeneous.T)).T
    positive_depth = camera_points[:, 2][camera_points[:, 2] > 1e-8]
    width, height = dimensions
    if not len(positive_depth):
        return np.full((height, width), np.nan, dtype=np.float32)
    fx, fy, cx, cy = undistortion["dense_intrinsics"]
    near = max(float(np.percentile(positive_depth, 1)) * 0.25, 1e-6)
    far = max(float(np.percentile(positive_depth, 99)) * 2.0, near * 100)
    return render_mesh_candidate(
        input_root / asset["relative_path"],
        transform,
        {
            "camera_from_world": camera_from_world,
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "near": near,
            "far": far,
        },
    ).depth


def _evaluate_frames(
    *,
    input_root: Path,
    output_dir: Path,
    candidate: dict[str, Any],
    asset: dict[str, Any],
    points: np.ndarray,
    transform: np.ndarray,
    transform_source: str,
    frame_inputs: dict[str, Any],
    frame_ids: list[str],
    group: str,
    world_from_camera: dict[str, np.ndarray],
    undistortion: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    per_frame: list[dict[str, Any]] = []
    render_paths: dict[str, str] = {}
    precision_values: list[float] = []
    recall_values: list[float] = []
    iou_values: list[float] = []
    depth_values: list[float] = []
    inlier_values: list[float] = []
    visible_area = occluded_area = negative_area = front_area = raw_area = 0
    for frame_id in frame_ids:
        target = _remap_mask(
            input_root / frame_inputs["mask_paths"][frame_id],
            undistortion[frame_id],
        )
        scene_depth = read_array(input_root / frame_inputs["depth_paths"][frame_id], 1)
        candidate_depth = _candidate_depth(
            input_root=input_root,
            asset=asset,
            points=points,
            transform=transform,
            camera_from_world=np.linalg.inv(world_from_camera[frame_id]),
            undistortion=undistortion[frame_id],
            measured_baseline=asset["asset_id"] == "measured_partial_point_cloud",
        )
        classification = classify_candidate_pixels(
            candidate_depth,
            scene_depth,
            target,
            relative_tolerance=0.03,
        )
        render_path = output_dir / "renders" / candidate["candidate_id"] / group / f"{frame_id}.png"
        _write_heldout_overlay(
            render_path,
            input_root / frame_inputs["dense_image_paths"][frame_id],
            target,
            classification,
            candidate_id=candidate["candidate_id"],
            frame_id=f"{group}:{frame_id}",
        )
        render_paths[frame_id] = render_path.relative_to(input_root).as_posix()
        raw = np.isfinite(candidate_depth)
        predicted = classification["visible"]
        precision, recall, iou = _mask_metrics(predicted, target)
        depth_residual, depth_inliers = relative_depth_metrics(
            candidate_depth,
            scene_depth,
            predicted & target,
            inlier_threshold=0.08,
        )
        precision_values.append(precision)
        recall_values.append(recall)
        iou_values.append(iou)
        depth_values.append(depth_residual)
        inlier_values.append(depth_inliers)
        visible = int(np.count_nonzero(classification["visible"]))
        occluded = int(np.count_nonzero(classification["occluded"]))
        negative = int(np.count_nonzero(classification["negative"]))
        front = int(np.count_nonzero(classification["front"]))
        raw_count = int(np.count_nonzero(raw))
        visible_area += visible
        occluded_area += occluded
        negative_area += negative
        front_area += front
        raw_area += raw_count
        candidate_bbox = _bbox(raw)
        target_bbox = _bbox(target)
        candidate_depth_stats = _finite_percentiles(candidate_depth)
        scene_depth_stats = _finite_percentiles(scene_depth)
        per_frame.append(
            {
                "frame_id": frame_id,
                "raw_candidate_pixel_count": raw_count,
                "visible_pixel_count": visible,
                "occluded_pixel_count": occluded,
                "negative_space_pixel_count": negative,
                "front_of_scene_pixel_count": front,
                "candidate_depth_min": candidate_depth_stats[0],
                "candidate_depth_median": candidate_depth_stats[1],
                "candidate_depth_max": candidate_depth_stats[2],
                "scene_depth_min": scene_depth_stats[0],
                "scene_depth_median": scene_depth_stats[1],
                "scene_depth_max": scene_depth_stats[2],
                "candidate_projected_bbox": candidate_bbox,
                "target_mask_bbox": target_bbox,
                "bbox_intersection": _bbox_intersection(candidate_bbox, target_bbox),
                "mask_area": int(np.count_nonzero(target)),
                "candidate_area": raw_count,
                "mask_precision": precision,
                "mask_recall": recall,
                "mask_iou": iou,
            }
        )
    denominator = max(raw_area, 1)
    return (
        {
            "frame_ids": frame_ids,
            "transform_source": transform_source,
            "mask_precision": float(np.mean(precision_values)) if precision_values else 0.0,
            "mask_recall": float(np.mean(recall_values)) if recall_values else 0.0,
            "mask_iou": float(np.mean(iou_values)) if iou_values else 0.0,
            "dense_depth_relative_residual": (
                float(np.median(depth_values)) if depth_values else None
            ),
            "depth_inlier_fraction": (float(np.mean(inlier_values)) if inlier_values else None),
            "negative_space_violation_ratio": negative_area / denominator,
            "front_of_scene_violation_ratio": front_area / denominator,
            "valid_candidate_pixel_count": raw_area,
            "per_frame": per_frame,
            "_areas": {
                "visible": visible_area,
                "occluded": occluded_area,
                "negative": negative_area,
                "front": front_area,
            },
            "_per_frame_iou": {
                frame_id: value for frame_id, value in zip(frame_ids, iou_values, strict=True)
            },
        },
        render_paths,
    )


def _heldout_metrics(
    sanity: dict[str, Any],
    registration_item: dict[str, Any],
) -> dict[str, Any]:
    areas = sanity["_areas"]
    result = {
        "mask_precision": sanity["mask_precision"],
        "mask_recall": sanity["mask_recall"],
        "mask_iou": sanity["mask_iou"],
        "per_frame_iou": sanity["_per_frame_iou"],
        "dense_depth_relative_residual": sanity["dense_depth_relative_residual"] or 1.0,
        "depth_inlier_fraction": sanity["depth_inlier_fraction"] or 0.0,
        "negative_space_violation_ratio": sanity["negative_space_violation_ratio"],
        "front_of_scene_violation_ratio": sanity["front_of_scene_violation_ratio"],
        "measured_point_to_candidate_median": registration_item["frozen_transform"][
            "measured_surface_median_residual"
        ],
        "measured_point_to_candidate_p90": registration_item["frozen_transform"][
            "measured_surface_p90_residual"
        ],
        "normal_agreement": registration_item["frozen_transform"]["normal_agreement"],
        "candidate_visible_coverage": sanity["mask_recall"],
        "validation_view_count": len(sanity["frame_ids"]),
        "visible_candidate_area": areas["visible"],
        "occluded_candidate_area": areas["occluded"],
        "negative_space_violation_area": areas["negative"],
        "front_of_scene_violation_area": areas["front"],
    }
    for internal in ("_areas", "_per_frame_iou"):
        sanity.pop(internal)
    return result


def _failure_classification(
    candidate: dict[str, Any],
    anchor: dict[str, Any],
    fitting: dict[str, Any],
    heldout: dict[str, Any],
    failed_gates: list[str],
) -> str:
    if candidate["backend"] == "measured_partial_baseline":
        return "passed"
    if anchor["valid_candidate_pixel_count"] == 0:
        return "empty_candidate_render"
    if anchor["mask_iou"] <= 1e-8:
        return "registration_failed"
    if fitting["mask_iou"] <= 1e-8:
        return "fitting_view_inconsistent"
    if heldout["mask_iou"] <= 1e-8:
        return "fitting_overfit_heldout_failure"
    if "maximum_negative_space_violation_ratio" in failed_gates:
        return "negative_space_violation"
    if any("depth" in gate for gate in failed_gates):
        return "depth_inconsistent"
    if failed_gates:
        return "heldout_shape_inconsistent"
    return "passed"


def _evaluate(request_path: Path, input_root: Path, output_dir: Path) -> None:
    started = time.monotonic()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    registration = json.loads((input_root / request["registration_manifest_path"]).read_text())
    camera = json.loads((input_root / request["camera_reconstruction_path"]).read_text())
    undistortion_manifest = json.loads(
        (input_root / request["dense_undistortion_manifest_path"]).read_text()
    )
    undistortion = {item["frame_id"]: item for item in undistortion_manifest["records"]}
    world_from_camera = _camera_matrices(camera)
    _, candidates = _generations(input_root)
    config = request["evaluation_configuration"]
    raw_metrics: dict[str, dict[str, Any]] = {}
    sanity_by_candidate: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    render_paths_by_candidate: dict[str, dict[str, str]] = {}
    anchor_paths_by_candidate: dict[str, dict[str, str]] = {}
    fitting_paths_by_candidate: dict[str, dict[str, str]] = {}
    for registration_item in registration["registrations"]:
        if registration_item["status"] == "registration_failed":
            continue
        candidate = candidates[registration_item["candidate_id"]]
        asset = _declared_asset(
            candidate,
            asset_id_field="evaluation_asset_id",
            asset_path_field="evaluation_asset_path",
        )
        if (
            registration_item["registration_asset_id"] != candidate["registration_asset_id"]
            or registration_item["registration_asset_path"] != candidate["registration_asset_path"]
        ):
            raise ValueError("registration manifest does not match candidate representation")
        if candidate["registration_asset_id"] != candidate["evaluation_asset_id"]:
            raise ValueError(
                "registration/evaluation representation transfer requires an accepted parity "
                "artifact before evaluation"
            )
        points = load_candidate_surface(
            input_root / asset["relative_path"],
            maximum_samples=200_000,
            seed=request["seed"],
        )
        transform = np.asarray(
            registration_item["frozen_transform"]["matrix_world_from_candidate"]
        ).reshape(4, 4)
        anchor_frame_ids = (
            [candidate["anchor_frame_id"]]
            if candidate["anchor_frame_id"]
            in request["anchor_inputs"][candidate["object_id"]]["frame_ids"]
            else request["anchor_inputs"][candidate["object_id"]]["frame_ids"][:1]
        )
        anchor, anchor_paths = _evaluate_frames(
            input_root=input_root,
            output_dir=output_dir,
            candidate=candidate,
            asset=asset,
            points=points,
            transform=transform,
            transform_source="frozen_registration_anchor",
            frame_inputs=request["anchor_inputs"][candidate["object_id"]],
            frame_ids=anchor_frame_ids,
            group="anchor",
            world_from_camera=world_from_camera,
            undistortion=undistortion,
        )
        fitting, fitting_paths = _evaluate_frames(
            input_root=input_root,
            output_dir=output_dir,
            candidate=candidate,
            asset=asset,
            points=points,
            transform=transform,
            transform_source="frozen_registration",
            frame_inputs=request["fitting_inputs"][candidate["object_id"]],
            frame_ids=registration_item["fitting_frame_ids"],
            group="fitting",
            world_from_camera=world_from_camera,
            undistortion=undistortion,
        )
        heldout, candidate_render_paths = _evaluate_frames(
            input_root=input_root,
            output_dir=output_dir,
            candidate=candidate,
            asset=asset,
            points=points,
            transform=transform,
            transform_source="frozen_registration",
            frame_inputs=request["heldout_inputs"][candidate["object_id"]],
            frame_ids=registration_item["heldout_frame_ids"],
            group="heldout",
            world_from_camera=world_from_camera,
            undistortion=undistortion,
        )
        raw_metrics[candidate["candidate_id"]] = _heldout_metrics(
            heldout,
            registration_item,
        )
        for sanity in (anchor, fitting):
            sanity.pop("_areas")
            sanity.pop("_per_frame_iou")
        sanity_by_candidate[candidate["candidate_id"]] = (anchor, fitting)
        render_paths_by_candidate[candidate["candidate_id"]] = candidate_render_paths
        anchor_paths_by_candidate[candidate["candidate_id"]] = anchor_paths
        fitting_paths_by_candidate[candidate["candidate_id"]] = fitting_paths
    baselines = {
        candidates[candidate_id]["object_id"]: metrics
        for candidate_id, metrics in raw_metrics.items()
        if candidates[candidate_id]["backend"] == "measured_partial_baseline"
    }
    evaluations = []
    for registration_item in registration["registrations"]:
        candidate_id = registration_item["candidate_id"]
        if candidate_id not in raw_metrics:
            continue
        candidate = candidates[candidate_id]
        metrics = raw_metrics[candidate_id]
        baseline = baselines[candidate["object_id"]]
        gains = {
            "recall_gain_vs_measured_baseline": metrics["mask_recall"] - baseline["mask_recall"],
            "iou_gain_vs_measured_baseline": metrics["mask_iou"] - baseline["mask_iou"],
            "precision_change_vs_measured_baseline": metrics["mask_precision"]
            - baseline["mask_precision"],
            "depth_residual_change": metrics["dense_depth_relative_residual"]
            - baseline["dense_depth_relative_residual"],
            "visible_coverage_gain": metrics["candidate_visible_coverage"]
            - baseline["candidate_visible_coverage"],
            "negative_space_change": metrics["negative_space_violation_ratio"]
            - baseline["negative_space_violation_ratio"],
        }
        failed = (
            []
            if candidate["backend"] == "measured_partial_baseline"
            else hard_gate_failures(metrics, config)
        )
        if len(registration_item["heldout_frame_ids"]) < config["minimum_validation_views"]:
            failed.append("minimum_validation_views")
        if (
            candidate["backend"] != "measured_partial_baseline"
            and gains["recall_gain_vs_measured_baseline"]
            < config["minimum_recall_gain_over_measured_baseline"]
        ):
            failed.append("minimum_recall_gain_over_measured_baseline")
        if (
            candidate["backend"] != "measured_partial_baseline"
            and gains["precision_change_vs_measured_baseline"]
            < -config["maximum_precision_drop_from_measured_baseline"]
        ):
            failed.append("maximum_precision_drop_from_measured_baseline")
        evaluations.append(
            {
                "candidate_id": candidate_id,
                "object_id": candidate["object_id"],
                "backend": candidate["backend"],
                "registration_asset_id": candidate["registration_asset_id"],
                "registration_asset_path": candidate["registration_asset_path"],
                "evaluation_asset_id": candidate["evaluation_asset_id"],
                "evaluation_asset_path": candidate["evaluation_asset_path"],
                "selection_asset_id": candidate["selection_asset_id"],
                "selection_asset_path": candidate["selection_asset_path"],
                "transform_sha256": _digest(registration_item["frozen_transform"]),
                "anchor_sanity": sanity_by_candidate[candidate_id][0],
                "fitting_metrics": sanity_by_candidate[candidate_id][1],
                "heldout_frame_ids": registration_item["heldout_frame_ids"],
                "metrics": metrics,
                "measured_baseline_metrics": baseline,
                "completion_gain": gains,
                "passed_hard_gates": not failed,
                "failed_gates": sorted(set(failed)),
                "evaluation_runtime_seconds": 0.0,
                "license_record": candidate["license_record"],
                "render_paths": render_paths_by_candidate[candidate_id],
                "anchor_render_paths": anchor_paths_by_candidate[candidate_id],
                "fitting_render_paths": fitting_paths_by_candidate[candidate_id],
                "failure_classification": _failure_classification(
                    candidate,
                    sanity_by_candidate[candidate_id][0],
                    sanity_by_candidate[candidate_id][1],
                    metrics,
                    failed,
                ),
                "representation_parity_path": None,
                "representation_parity_accepted": False,
            }
        )
    runtime = time.monotonic() - started
    _write_json(
        output_dir / "evaluation_manifest.json",
        {
            "registration_manifest_sha256": request["registration_manifest_sha256"],
            "evaluation_configuration": config,
            "evaluations": evaluations,
            "transforms_frozen_before_heldout_evaluation": True,
            "runtime_seconds": runtime,
            "peak_gpu_memory_bytes": _peak_gpu(),
            "peak_host_memory_bytes": None,
        },
    )
    _worker_manifest(output_dir, request_path, action="evaluation", runtime=runtime)


def run_action(
    action: str,
    request_path: Path,
    input_root: Path,
    output_dir: Path,
) -> None:
    if action == "prepare-evidence":
        _prepare_evidence(request_path, input_root, output_dir)
    elif action == "register":
        _register(request_path, input_root, output_dir)
    elif action == "evaluate":
        _evaluate(request_path, input_root, output_dir)
    else:
        raise ValueError(f"unsupported completion evaluation action: {action}")
