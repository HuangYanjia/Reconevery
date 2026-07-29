from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

IDENTITY_ROTATION = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
MATRIX = [
    2.0,
    0.0,
    0.0,
    1.0,
    0.0,
    2.0,
    0.0,
    -2.0,
    0.0,
    0.0,
    2.0,
    0.5,
    0.0,
    0.0,
    0.0,
    1.0,
]
INVERSE = [
    0.5,
    0.0,
    0.0,
    -0.5,
    0.0,
    0.5,
    0.0,
    1.0,
    0.0,
    0.0,
    0.5,
    -0.25,
    0.0,
    0.0,
    0.0,
    1.0,
]
PREVIEWS = (
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


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("request must be an object")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_preview(path: Path, title: str) -> None:
    image = Image.new("RGB", (640, 360), (245, 247, 249))
    draw = ImageDraw.Draw(image)
    draw.text((24, 24), title, fill=(20, 30, 40))
    image.save(path, format="PNG", optimize=False, compress_level=9)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solve(request_path: Path, input_root: Path, output_dir: Path) -> None:
    request = read_json(request_path)
    mode = str(request.get("fake_mode") or "perfect_full_canonical")
    if mode == "timeout":
        time.sleep(60)
    if mode == "interruption":
        raise KeyboardInterrupt
    if mode == "malformed_output":
        write_json(output_dir / "world_calibration.json", {"broken": True})
        return
    if mode == "path_escape":
        print("path escape rejected", file=sys.stderr)
        raise SystemExit(7)
    if mode == "worker_modifying_upstream":
        (input_root / str(request["manifest_path"])).write_text("modified\n", encoding="utf-8")
        raise SystemExit(8)

    full = mode == "perfect_full_canonical"
    metric = full or mode in {"metric_only", "wrong_tag_size"}
    gravity = full or mode in {"gravity_only", "flipped_gravity"}
    forward = full
    origin = full
    status = {
        "perfect_full_canonical": "accepted_full_canonical",
        "metric_only": "accepted_metric_only",
        "gravity_only": "accepted_gravity_only",
        "wrong_tag_size": "rejected_heldout_validation",
        "inconsistent_tag_detections": "rejected_heldout_validation",
        "inconsistent_landmark_distances": "rejected_inconsistent_metric_evidence",
        "flipped_gravity": "rejected_inconsistent_gravity_evidence",
        "parallel_up_forward": "insufficient_forward_evidence",
        "heldout_regression": "rejected_heldout_validation",
        "negative_scale": "rejected_heldout_validation",
        "improper_rotation": "rejected_heldout_validation",
        "singular_transform": "rejected_heldout_validation",
    }.get(mode, "insufficient_evidence")
    transform = {
        "scale_m_per_colmap": 2.0,
        "rotation_canonical_from_colmap": IDENTITY_ROTATION,
        "translation_canonical_m": [1.0, -2.0, 0.5],
        "matrix_canonical_from_colmap": MATRIX,
        "matrix_colmap_from_canonical": INVERSE,
        "rotation_determinant": 1.0,
        "orthonormal_error": 0.0,
        "inverse_roundtrip_error": 0.0,
        "covariance_diagonal": [1e-6] * 7,
    }
    if mode == "negative_scale":
        transform["scale_m_per_colmap"] = -2.0
    if mode == "improper_rotation":
        transform["rotation_determinant"] = -1.0
    if mode == "singular_transform":
        transform["matrix_canonical_from_colmap"] = [0.0] * 16
    candidate = {
        "candidate_id": "fake_full_canonical",
        "evidence_tier": (
            "full_canonical"
            if full
            else "scale_only"
            if metric
            else "gravity_only"
            if gravity
            else "none"
        ),
        "selected_by_fitting_only": True,
        "transform": transform if metric else None,
        "fitting_objective": 0.001,
        "evidence_ids": request["dataset_split"]["fitting_evidence_ids"],
        "warnings": [],
    }
    accepted = status.startswith("accepted_") and metric
    derivation = (
        {
            "schema_version": "0.1.0",
            "official_commit": "0e16a12dd380fd607e4afd54712ee9b1ffb9ec8f",
            "tag_family": "tagStandard41h12",
            "tag_id": 0,
            "fitting_detection_frame_ids": ["frame_000000", "frame_000002", "frame_000004"],
            "heldout_detection_frame_ids": ["frame_000001", "frame_000003", "frame_000005"],
            "tag_pose_sha256_by_frame": {
                f"frame_{index:06d}": f"{index + 1:064x}" for index in range(6)
            },
            "matrix_tag_from_colmap": MATRIX,
            "world_contract": {
                "tag_origin_policy": "tag_center",
                "canonical_up_from_tag_axis": "+Z_tag",
                "canonical_forward_from_tag_axis": "+X_tag",
                "mounting_description": "surveyed fixed tag board",
                "mounting_uncertainty_degrees": 0.1,
                "origin_uncertainty_m": 0.001,
            },
            "derived_up_vector_colmap": [0.0, 0.0, 1.0],
            "derived_forward_vector_colmap": [1.0, 0.0, 0.0],
            "derived_origin_colmap": [-0.5, 1.0, -0.25],
            "heldout_translation_residual_m": 0.004,
            "heldout_orientation_residual_degrees": 0.5,
            "angular_uncertainty_degrees": 0.1,
            "origin_uncertainty_m": 0.001,
        }
        if full
        else None
    )
    derivation_path = output_dir / "apriltag_world_derivation.json"
    write_json(
        derivation_path,
        derivation
        if derivation is not None
        else {
            "schema_version": "0.1.0",
            "available": False,
            "reason": "no explicit AprilTag world contract derivation",
        },
    )
    landmark_derivation_path = output_dir / "landmark_world_derivation.json"
    write_json(
        landmark_derivation_path,
        {
            "schema_version": "0.1.0",
            "available": False,
            "reason": "fake full-canonical fixture uses an AprilTag world contract",
        },
    )
    write_json(
        output_dir / "world_calibration.json",
        {
            "schema_version": "0.2.0",
            "status": status,
            "evidence_tier": candidate["evidence_tier"],
            "manifest_path": request["manifest_path"],
            "manifest_sha256": request["manifest_sha256"],
            "dataset_split": request["dataset_split"],
            "candidates": [candidate],
            "selected_candidate_id": candidate["candidate_id"] if accepted else None,
            "accepted_transform": transform if accepted else None,
            "fiducial_world_derivation": derivation,
            "landmark_world_derivation": None,
            "metrics": {
                "fitting_metric_relative_error": 0.001 if metric else None,
                "heldout_metric_relative_error": None,
                "fitting_landmark_reprojection_error_px": None,
                "heldout_landmark_reprojection_error_px": None,
                "independent_metric_length_holdout_available": False,
                "heldout_tag_detection_count": 3 if metric else 0,
                "heldout_tag_translation_error_m": 0.004 if metric else None,
                "heldout_tag_rotation_error_degrees": 0.5 if metric else None,
                "gravity_fitting_error_degrees": 0.2 if gravity else None,
                "gravity_heldout_error_degrees": 0.4 if gravity else None,
                "forward_uncertainty_degrees": 0.5 if forward else None,
                "sim3_roundtrip_error": 0.0,
                "fitting_known_distance_residuals": {},
                "heldout_known_distance_residuals": {},
            },
            "metric_scale_known": metric and status not in {"rejected_heldout_validation"},
            "gravity_alignment_known": gravity and not status.startswith("rejected_"),
            "canonical_forward_known": forward and not status.startswith("rejected_"),
            "canonical_origin_known": origin and not status.startswith("rejected_"),
            "full_canonical_world_available": full,
            "source_cameras_unchanged": True,
            "source_geometry_unchanged": True,
            "warnings": [],
        },
    )
    write_json(
        output_dir / "apriltag_detections.json",
        {
            "schema_version": "0.2.0",
            "official_repository": "https://github.com/AprilRobotics/apriltag",
            "official_commit": "0e16a12dd380fd607e4afd54712ee9b1ffb9ec8f",
            "detections": [],
        },
    )
    write_json(
        output_dir / "triangulated_landmarks.json",
        {"schema_version": "0.1.0", "landmarks": []},
    )
    write_json(
        output_dir / "diagnostics.json",
        {
            "schema_version": "0.2.0",
            "status": status,
            "metric_evidence_count": int(metric),
            "gravity_evidence_count": int(gravity),
            "forward_evidence_count": int(forward),
            "origin_evidence_count": int(origin),
            "fitting_evidence_count": len(request["dataset_split"]["fitting_evidence_ids"]),
            "heldout_evidence_count": len(request["dataset_split"]["heldout_evidence_ids"]),
            "total_runtime_seconds": 0.01,
            "peak_host_memory_bytes": 1024,
            "runtime_environment": {
                "python": "fake",
                "numpy": "not_loaded",
                "scipy": "not_loaded",
                "opencv": "not_loaded",
                "cuda": "not_used",
            },
            "fiducial_world_derivation_path": (
                "calibration/apriltag_world_derivation.json" if derivation is not None else None
            ),
            "fiducial_world_derivation_sha256": (
                sha256_file(derivation_path) if derivation is not None else None
            ),
            "landmark_world_derivation_path": None,
            "landmark_world_derivation_sha256": None,
            "warnings": [],
        },
    )
    preview_root = output_dir / "previews"
    preview_root.mkdir(parents=True, exist_ok=True)
    for name in PREVIEWS:
        write_preview(preview_root / f"{name}.png", f"Fake Phase 6A {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["healthcheck", "solve"])
    parser.add_argument("--request", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    if arguments.action == "healthcheck":
        print('{"ok": true, "worker": "fake_world_calibration"}')
        return 0
    if arguments.request is None or arguments.input_root is None or arguments.output_dir is None:
        parser.error("solve requires --request, --input-root, and --output-dir")
    solve(arguments.request, arguments.input_root, arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
