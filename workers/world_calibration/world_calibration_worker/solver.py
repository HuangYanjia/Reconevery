from __future__ import annotations

import hashlib
import platform
import resource
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy

from world_calibration_worker.apriltag_detection import detect_official
from world_calibration_worker.canonical_axes import canonical_rotation
from world_calibration_worker.diagnostics import render_previews
from world_calibration_worker.gravity import combine_up_vectors
from world_calibration_worker.landmark_triangulation import (
    camera_centers,
    projection_by_frame,
    reprojection_errors,
    triangulate,
)
from world_calibration_worker.schema import read_json, write_json
from world_calibration_worker.sim3 import matrix, transform_points, umeyama
from world_calibration_worker.validation import sim3_diagnostics


def _flatten(value: np.ndarray) -> list[float]:
    return [float(item) for item in value.reshape(-1)]


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tag_detections(
    manifest: dict[str, Any],
    input_root: Path,
) -> list[dict[str, Any]]:
    tag = manifest.get("apriltag")
    if not isinstance(tag, dict):
        return []
    detections = [dict(item) for item in tag.get("detections", [])]
    detected_frames = {str(item["frame_id"]) for item in detections}
    for source in tag.get("image_sources", []):
        frame_id = str(source["frame_id"])
        if frame_id in detected_frames:
            continue
        image_path = input_root / str(source["image_path"])
        if _sha256(image_path) != str(source["image_sha256"]):
            raise ValueError(f"AprilTag source image hash mismatch: {frame_id}")
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"unable to read AprilTag source image: {frame_id}")
        if image.shape != (int(source["height"]), int(source["width"])):
            raise ValueError(f"AprilTag source image dimensions changed: {frame_id}")
        results = detect_official(
            image,
            family=str(tag["tag_family"]),
            tag_id=int(tag["tag_id"]),
            tag_size_m=float(tag["detection_edge_size_m"]),
            camera_params=tuple(float(value) for value in source["intrinsics_fx_fy_cx_cy"]),
        )
        if len(results) > 1:
            raise ValueError(f"ambiguous duplicate configured AprilTag ID in frame: {frame_id}")
        if not results:
            continue
        result = results[0]
        detections.append(
            {
                "frame_id": frame_id,
                "image_path": source["image_path"],
                "image_sha256": source["image_sha256"],
                "tag_id": result["tag_id"],
                "corners_xy": result["corners_xy"],
                "decision_margin": result["decision_margin"],
                "hamming": result["hamming"],
                "camera_center_tag_m": result["camera_center_tag_m"],
                "rotation_tag_from_camera": result["rotation_tag_from_camera"],
                "pose_error": result["pose_error"],
                "split": source["split"],
            }
        )
        detected_frames.add(frame_id)
    return sorted(detections, key=lambda item: (str(item["frame_id"]), str(item["split"])))


def _rotation_from_xyzw(values: list[float]) -> np.ndarray:
    x, y, z, w = (float(value) for value in values)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_error_degrees(left: np.ndarray, right: np.ndarray) -> float:
    relative = left @ right.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _known_distance_solution(
    manifest: dict[str, Any],
    camera: dict[str, Any],
    fitting_frames: set[str],
    heldout_frames: set[str],
) -> tuple[list[float], list[dict[str, Any]], dict[str, float], float | None]:
    known = manifest.get("known_distance")
    if not isinstance(known, dict):
        return [], [], {}, None
    projections = projection_by_frame(camera)
    by_point: dict[str, list[dict[str, Any]]] = {}
    for observation in known["observations"]:
        by_point.setdefault(str(observation["point_id"]), []).append(observation)
    triangulated: dict[str, np.ndarray] = {}
    records = []
    for point_id, observations in sorted(by_point.items()):
        fitting = [item for item in observations if str(item["frame_id"]) in fitting_frames]
        heldout = [item for item in observations if str(item["frame_id"]) in heldout_frames]
        if len(fitting) < 2:
            raise ValueError(f"landmark {point_id!r} has fewer than two fitting observations")
        point = triangulate(fitting, projections)
        fitting_errors = reprojection_errors(point, fitting, projections)
        heldout_errors = reprojection_errors(point, heldout, projections)
        triangulated[point_id] = point
        records.append(
            {
                "point_id": point_id,
                "point_colmap": point.tolist(),
                "fitting_frame_ids": sorted(str(item["frame_id"]) for item in fitting),
                "heldout_frame_ids": sorted(str(item["frame_id"]) for item in heldout),
                "fitting_reprojection_error_px": _median(fitting_errors),
                "heldout_reprojection_error_px": (
                    _median(heldout_errors) if heldout_errors else None
                ),
                "covariance_diagonal": None,
            }
        )
    scales = []
    anchor_values: dict[str, tuple[float, float]] = {}
    for landmark in known["landmarks"]:
        point_a = triangulated[str(landmark["point_a_id"])]
        point_b = triangulated[str(landmark["point_b_id"])]
        distance = float(np.linalg.norm(point_a - point_b))
        if distance <= np.finfo(np.float64).eps:
            raise ValueError("known-distance landmark triangulated to zero length")
        scale = float(landmark["known_distance_m"]) / distance
        scales.append(scale)
        anchor_values[str(landmark["landmark_id"])] = (
            distance,
            float(landmark["known_distance_m"]),
        )
    if not scales:
        return [], records, {}, None
    robust_scale = _median(scales)
    residuals = {
        landmark_id: abs(distance * robust_scale - known_m) / known_m
        for landmark_id, (distance, known_m) in anchor_values.items()
    }
    return scales, records, residuals, max(residuals.values(), default=None)


def _tag_solution(
    manifest: dict[str, Any],
    camera: dict[str, Any],
    detections: list[dict[str, Any]],
) -> tuple[float | None, np.ndarray | None, np.ndarray | None, dict[str, float | int | None]]:
    tag = manifest.get("apriltag")
    if not isinstance(tag, dict):
        return (
            None,
            None,
            None,
            {
                "heldout_count": 0,
                "translation_error": None,
                "rotation_error": None,
            },
        )
    centers = camera_centers(camera)
    fitting = [
        item
        for item in detections
        if item["split"] == "fitting" and item.get("camera_center_tag_m") is not None
    ]
    heldout = [
        item
        for item in detections
        if item["split"] == "heldout" and item.get("camera_center_tag_m") is not None
    ]
    if len(fitting) < 3:
        return (
            None,
            None,
            None,
            {
                "heldout_count": len(heldout),
                "translation_error": None,
                "rotation_error": None,
            },
        )
    source = np.asarray([centers[str(item["frame_id"])] for item in fitting])
    target = np.asarray([item["camera_center_tag_m"] for item in fitting], dtype=np.float64)
    scale, rotation, translation = umeyama(source, target)
    fit_matrix = matrix(scale, rotation, translation)
    translation_error = None
    rotation_error = None
    if heldout:
        heldout_source = np.asarray([centers[str(item["frame_id"])] for item in heldout])
        heldout_target = np.asarray(
            [item["camera_center_tag_m"] for item in heldout], dtype=np.float64
        )
        residuals = np.linalg.norm(
            transform_points(heldout_source, fit_matrix) - heldout_target, axis=1
        )
        translation_error = float(np.median(residuals))
        pose_by_frame = {str(item["frame_id"]): item for item in camera["poses"]}
        orientation_errors = []
        for item in heldout:
            observed = item.get("rotation_tag_from_camera")
            pose = pose_by_frame.get(str(item["frame_id"]))
            if observed is None or pose is None:
                continue
            rotation_colmap_from_camera = _rotation_from_xyzw(
                pose["transform_world_from_camera"]["rotation_xyzw"]
            )
            predicted = rotation @ rotation_colmap_from_camera
            orientation_errors.append(
                _rotation_error_degrees(
                    predicted,
                    np.asarray(observed, dtype=np.float64).reshape(3, 3),
                )
            )
        rotation_error = max(orientation_errors, default=None)
    return (
        scale,
        rotation,
        translation,
        {
            "heldout_count": len(heldout),
            "translation_error": translation_error,
            "rotation_error": rotation_error,
        },
    )


def solve(request_path: Path, input_root: Path, output_dir: Path) -> None:
    started = time.monotonic()
    request = read_json(request_path)
    manifest = read_json(input_root / str(request["manifest_path"]))
    camera = read_json(input_root / str(request["camera_reconstruction_path"]))
    split = request["dataset_split"]
    fitting_frames = {str(value) for value in split["fitting_frame_ids"]}
    heldout_frames = {str(value) for value in split["heldout_frame_ids"]}
    gates = request["acceptance_gates"]
    tag_detections = _tag_detections(manifest, input_root)

    landmark_scales, landmarks, distance_residuals, distance_error = _known_distance_solution(
        manifest,
        camera,
        fitting_frames,
        heldout_frames,
    )
    tag_scale, tag_rotation, tag_translation, tag_metrics = _tag_solution(
        manifest,
        camera,
        tag_detections,
    )
    scale_values = [*landmark_scales, *([tag_scale] if tag_scale is not None else [])]
    metric_scale = _median(scale_values) if scale_values else None
    inconsistent_metric = bool(
        scale_values
        and max(scale_values) / min(scale_values) - 1.0
        > float(gates.get("maximum_known_distance_relative_error", 0.02))
    )

    gravity_records = [
        item for item in manifest.get("gravity", []) if item.get("source") != "manhattan_diagnostic"
    ]
    accepted_floor_planes = [
        item
        for item in manifest.get("floor_planes", [])
        if int(item["point_count"]) >= int(gates.get("minimum_floor_point_count", 1000))
        and float(item["spatial_extent_colmap"])
        >= float(gates.get("minimum_floor_spatial_extent_colmap", 0.25))
    ]
    gravity_records.extend(
        {
            "evidence_id": item["evidence_id"],
            "source": "floor_plane",
            "trust": "geometry_plane",
            "up_vector_colmap": item["plane_normal_colmap"],
            "sign_evidence": item["sign_policy"],
            "fitting_residual_degrees": 0.0,
            "heldout_residual_degrees": item["heldout_normal_error_degrees"],
            "angular_uncertainty_degrees": item["heldout_normal_error_degrees"],
            "supporting_ids": item["floor_mask_paths"],
        }
        for item in accepted_floor_planes
    )
    gravity_vector = None
    gravity_error = None
    inconsistent_gravity = False
    if gravity_records:
        try:
            gravity_vector, gravity_error = combine_up_vectors(gravity_records)
        except ValueError:
            inconsistent_gravity = True
    forward_record = manifest.get("forward")
    origin_record = manifest.get("origin")

    rotation = None
    if gravity_vector is not None and isinstance(forward_record, dict):
        rotation = canonical_rotation(
            gravity_vector,
            np.asarray(forward_record["forward_vector_colmap"], dtype=np.float64),
        )
    elif tag_rotation is not None and metric_scale is not None:
        rotation = tag_rotation

    transform = None
    if metric_scale is not None:
        selected_rotation = rotation if rotation is not None else np.eye(3)
        if isinstance(origin_record, dict):
            origin = np.asarray(origin_record["origin_colmap"], dtype=np.float64)
            translation = -metric_scale * selected_rotation @ origin
        elif tag_translation is not None and rotation is tag_rotation:
            translation = tag_translation
        else:
            translation = np.zeros(3)
        transform = matrix(metric_scale, selected_rotation, translation)
    determinant = orthonormal = roundtrip = 0.0
    transform_record = None
    if transform is not None:
        determinant, orthonormal, roundtrip = sim3_diagnostics(transform)
        inverse = np.linalg.inv(transform)
        transform_record = {
            "scale_m_per_colmap": metric_scale,
            "rotation_canonical_from_colmap": _flatten(transform[:3, :3] / float(metric_scale)),
            "translation_canonical_m": transform[:3, 3].tolist(),
            "matrix_canonical_from_colmap": _flatten(transform),
            "matrix_colmap_from_canonical": _flatten(inverse),
            "rotation_determinant": determinant,
            "orthonormal_error": orthonormal,
            "inverse_roundtrip_error": roundtrip,
            "covariance_diagonal": None,
        }

    metric_evidence_count = sum(
        bool(item.get("supports_metric_scale")) for item in manifest["evidence"]
    )
    metric_valid = (
        metric_scale is not None
        and not inconsistent_metric
        and metric_evidence_count >= int(gates.get("minimum_metric_evidence_records", 1))
    )
    if distance_error is not None:
        metric_valid = metric_valid and distance_error <= float(
            gates.get("maximum_known_distance_relative_error", 0.02)
        )
    if manifest.get("known_distance") is not None:
        metric_valid = metric_valid and len(landmark_scales) >= int(
            gates.get("minimum_known_distance_anchors", 1)
        )
    if manifest.get("apriltag") is not None:
        metric_valid = metric_valid and int(tag_metrics["heldout_count"] or 0) >= int(
            gates.get("minimum_heldout_tag_detections", 3)
        )
        metric_valid = metric_valid and tag_metrics["translation_error"] is not None
        if tag_metrics["translation_error"] is not None:
            metric_valid = metric_valid and float(tag_metrics["translation_error"]) <= float(
                gates.get("maximum_heldout_tag_translation_error_m", 0.02)
            )
        if tag_metrics["rotation_error"] is not None:
            metric_valid = metric_valid and float(tag_metrics["rotation_error"]) <= float(
                gates.get("maximum_heldout_tag_rotation_error_degrees", 3.0)
            )
    gravity_valid = (
        gravity_vector is not None
        and not inconsistent_gravity
        and len(gravity_records) >= int(gates.get("minimum_gravity_evidence_records", 1))
        and all(
            item.get("heldout_residual_degrees") is not None
            and float(item["heldout_residual_degrees"])
            <= float(gates.get("maximum_gravity_heldout_error_degrees", 3.0))
            for item in gravity_records
        )
    )
    forward_valid = (
        isinstance(forward_record, dict)
        and float(forward_record["uncertainty_degrees"])
        <= float(gates.get("maximum_forward_uncertainty_degrees", 5.0))
        and int(forward_record is not None) >= int(gates.get("minimum_forward_evidence_records", 1))
    )
    origin_valid = isinstance(origin_record, dict)
    roundtrip_valid = transform is not None and roundtrip <= float(
        gates.get("maximum_sim3_roundtrip_error", 1e-8)
    )

    full = metric_valid and gravity_valid and forward_valid and origin_valid and roundtrip_valid
    if inconsistent_metric:
        status = "rejected_inconsistent_metric_evidence"
    elif inconsistent_gravity:
        status = "rejected_inconsistent_gravity_evidence"
    elif full:
        status = "accepted_full_canonical"
    elif metric_valid and not gravity_valid:
        status = "accepted_metric_only"
    elif gravity_valid and not metric_valid:
        status = "accepted_gravity_only"
    elif metric_valid and gravity_valid and not forward_valid:
        status = "insufficient_forward_evidence"
    elif metric_valid and gravity_valid and forward_valid and not origin_valid:
        status = "insufficient_origin_evidence"
    elif metric_scale is not None or gravity_vector is not None:
        status = "rejected_heldout_validation"
    else:
        status = "insufficient_evidence"

    evidence_tier = str(manifest["evidence_tier"])
    candidate_id = "world_calibration__fitting_v1"
    candidate = {
        "candidate_id": candidate_id,
        "evidence_tier": evidence_tier,
        "selected_by_fitting_only": True,
        "transform": transform_record,
        "fitting_objective": float(distance_error or gravity_error or 0.0),
        "evidence_ids": list(split["fitting_evidence_ids"]),
        "warnings": [],
    }
    metrics = {
        "fitting_metric_relative_error": distance_error,
        "heldout_metric_relative_error": distance_error,
        "heldout_tag_detection_count": int(tag_metrics["heldout_count"] or 0),
        "heldout_tag_translation_error_m": tag_metrics["translation_error"],
        "heldout_tag_rotation_error_degrees": tag_metrics["rotation_error"],
        "gravity_fitting_error_degrees": gravity_error,
        "gravity_heldout_error_degrees": (
            max(
                (
                    float(item["heldout_residual_degrees"])
                    for item in gravity_records
                    if item.get("heldout_residual_degrees") is not None
                ),
                default=None,
            )
        ),
        "forward_uncertainty_degrees": (
            float(forward_record["uncertainty_degrees"])
            if isinstance(forward_record, dict)
            else None
        ),
        "sim3_roundtrip_error": roundtrip,
        "known_distance_residuals": distance_residuals,
    }
    accepted_transform = transform_record if status.startswith("accepted_") else None
    selected_candidate_id = candidate_id if accepted_transform is not None else None
    write_json(
        output_dir / "world_calibration.json",
        {
            "schema_version": "0.1.0",
            "status": status,
            "evidence_tier": evidence_tier,
            "manifest_path": request["manifest_path"],
            "manifest_sha256": request["manifest_sha256"],
            "dataset_split": split,
            "candidates": [candidate],
            "selected_candidate_id": selected_candidate_id,
            "accepted_transform": accepted_transform,
            "metrics": metrics,
            "metric_scale_known": metric_valid,
            "gravity_alignment_known": gravity_valid,
            "canonical_forward_known": forward_valid,
            "canonical_origin_known": origin_valid,
            "full_canonical_world_available": full,
            "source_cameras_unchanged": True,
            "source_geometry_unchanged": True,
            "warnings": [],
        },
    )
    write_json(
        output_dir / "apriltag_detections.json",
        {
            "schema_version": "0.1.0",
            "official_repository": (
                manifest["apriltag"]["official_repository"]
                if isinstance(manifest.get("apriltag"), dict)
                else None
            ),
            "official_commit": (
                manifest["apriltag"]["official_commit"]
                if isinstance(manifest.get("apriltag"), dict)
                else None
            ),
            "detections": tag_detections,
        },
    )
    write_json(
        output_dir / "triangulated_landmarks.json",
        {
            "schema_version": "0.1.0",
            "landmarks": landmarks,
        },
    )
    write_json(
        output_dir / "diagnostics.json",
        {
            "schema_version": "0.1.0",
            "status": status,
            "metric_evidence_count": metric_evidence_count,
            "gravity_evidence_count": len(gravity_records),
            "forward_evidence_count": int(forward_record is not None),
            "origin_evidence_count": int(origin_record is not None),
            "fitting_evidence_count": len(split["fitting_evidence_ids"]),
            "heldout_evidence_count": len(split["heldout_evidence_ids"]),
            "total_runtime_seconds": time.monotonic() - started,
            "peak_host_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "runtime_environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "opencv": cv2.__version__,
                "cuda": "not_used",
            },
            "warnings": [],
        },
    )
    render_previews(
        output_dir / "previews",
        status,
        [
            f"metric scale: {metric_scale}",
            f"gravity: {gravity_vector.tolist() if gravity_vector is not None else None}",
            f"forward: {forward_record is not None}",
            f"origin: {origin_record is not None}",
            f"held-out tag detections: {tag_metrics['heldout_count']}",
        ],
    )


__all__ = ["solve"]
