from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

import numpy as np

from articulation_alignment_worker import __version__
from articulation_alignment_worker.io import (
    load_points,
    read_json,
    write_json,
    write_preview,
)
from articulation_alignment_worker.motion import JointEstimate, estimate_joint
from articulation_alignment_worker.sim3 import (
    apply_transform,
    robust_icp_sim3,
)


def _matrix_values(matrix: np.ndarray) -> list[float]:
    return [float(value) for value in matrix.reshape(-1)]


def _geometry_by_state(manifest: dict[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    geometries = manifest.get("geometries")
    if not isinstance(geometries, list):
        raise ValueError("measured-state manifest has no geometries")
    return {
        (str(item["state_id"]), str(item["part_id"])): item
        for item in geometries
        if isinstance(item, dict)
    }


def align(request: dict[str, object], input_root: Path, output_dir: Path) -> None:
    started = time.monotonic()
    measured = read_json(input_root / str(request["measured_states_manifest_path"]))
    by_state = _geometry_by_state(measured)
    reference_state = str(request["reference_state_id"])
    base_part = str(request["base_part_id"])
    reference_item = by_state[(reference_state, base_part)]
    reference_points = load_points(input_root / str(reference_item["measured_point_cloud_path"]))
    diagonal = float(np.linalg.norm(np.ptp(reference_points, axis=0)))
    diagonal = max(diagonal, np.finfo(np.float64).eps)
    acceptance = request["acceptance_configuration"]
    if not isinstance(acceptance, dict):
        raise ValueError("alignment acceptance configuration is invalid")
    transforms = []
    for state_id_value in request["state_ids"]:
        state_id = str(state_id_value)
        if state_id == reference_state:
            matrix = np.eye(4)
            inverse = np.eye(4)
            scale = 1.0
            residuals = np.zeros(len(reference_points))
            validation_residuals = residuals
            correspondence_count = len(reference_points)
        else:
            state_item = by_state[(state_id, base_part)]
            state_points = load_points(input_root / str(state_item["measured_point_cloud_path"]))
            training = np.arange(len(state_points)) % 5 != 0
            heldout = ~training
            fit = robust_icp_sim3(state_points[training], reference_points, with_scale=True)
            matrix, inverse, scale = fit.matrix, fit.inverse, fit.scale
            transformed = apply_transform(state_points, matrix)
            reference_tree_points = reference_points
            from scipy.spatial import cKDTree

            residuals, _ = cKDTree(reference_tree_points).query(transformed, k=1)
            validation_residuals = residuals[heldout] if np.any(heldout) else residuals
            correspondence_count = fit.correspondence_count
        normalized = validation_residuals / diagonal
        median = float(np.median(normalized))
        p90 = float(np.percentile(normalized, 90))
        threshold = float(acceptance["maximum_static_p90_residual_scene_diagonal"])
        heldout_inlier = float(np.mean(normalized <= threshold))
        accepted = (
            correspondence_count >= int(acceptance["minimum_static_correspondences"])
            and median <= float(acceptance["maximum_static_median_residual_scene_diagonal"])
            and p90 <= threshold
            and heldout_inlier >= float(acceptance["minimum_heldout_static_depth_inlier_fraction"])
        )
        if state_id == reference_state:
            accepted = True
        transforms.append(
            {
                "state_id": state_id,
                "matrix_reference_from_state": _matrix_values(matrix),
                "inverse_matrix": _matrix_values(inverse),
                "scale": scale,
                "rotation_determinant": float(np.linalg.det(matrix[:3, :3] / scale)),
                "translation": [float(value) for value in matrix[:3, 3]],
                "fitting_median_residual_scene_diagonal": median,
                "fitting_p90_residual_scene_diagonal": p90,
                "heldout_static_depth_inlier_fraction": heldout_inlier,
                "static_correspondence_count": correspondence_count,
                "excluded_movable_part_ids": request["movable_part_ids"],
                "accepted": accepted,
                "failure_reason": None if accepted else "static alignment gates failed",
            }
        )
    accepted_state_ids = [str(item["state_id"]) for item in transforms if item["accepted"]]
    write_json(
        output_dir / "state_alignment.json",
        {
            "schema_version": "0.2.0",
            "capture_manifest_sha256": request["capture_manifest_sha256"],
            "reference_state_id": reference_state,
            "transforms": transforms,
            "capture_state_count": len(transforms),
            "accepted_alignment_state_ids": accepted_state_ids,
            "aligned_state_count": len(accepted_state_ids),
            "static_evidence_only": True,
            "source_states_unchanged": True,
            "runtime_seconds": time.monotonic() - started,
            "peak_host_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "warnings": [],
        },
    )
    write_preview(
        output_dir / "previews/state_alignment.png",
        "Static-evidence state alignment",
        [
            f"{item['state_id']}: median={item['fitting_median_residual_scene_diagonal']:.5f}"
            for item in transforms
        ],
    )


def estimate_motion_action(
    request: dict[str, object],
    input_root: Path,
    output_dir: Path,
) -> None:
    started = time.monotonic()
    measured = read_json(input_root / str(request["measured_states_manifest_path"]))
    alignment = read_json(input_root / str(request["state_alignment_path"]))
    by_state = _geometry_by_state(measured)
    alignment_by_state = {
        str(item["state_id"]): np.asarray(
            item["matrix_reference_from_state"], dtype=np.float64
        ).reshape(4, 4)
        for item in alignment["transforms"]
        if item["accepted"]
    }
    state_ids = [str(value) for value in request["accepted_state_ids"]]
    reference_state_id = str(request["reference_state_id"])
    if not state_ids or state_ids[0] != reference_state_id:
        raise ValueError("declared reference state must be first in measured-motion evidence")
    movable_parts = request["movable_parts"]
    if not isinstance(movable_parts, list):
        raise ValueError("movable part request is invalid")
    hypotheses = []
    copied_geometries = []
    for item in measured["geometries"]:
        if not isinstance(item, dict) or str(item["state_id"]) not in alignment_by_state:
            continue
        copied = dict(item)
        copied["state_alignment_sha256"] = request["state_alignment_sha256"]
        copied["transformed_to_reference_frame"] = True
        copied_geometries.append(copied)
    configuration = request["motion_configuration"]
    if not isinstance(configuration, dict):
        raise ValueError("motion configuration is invalid")
    for part in movable_parts:
        part_id = str(part["part_id"])
        reference_state = reference_state_id
        reference_points = apply_transform(
            load_points(
                input_root / str(by_state[(reference_state, part_id)]["measured_point_cloud_path"])
            ),
            alignment_by_state[reference_state],
        )
        transforms = [np.eye(4)]
        state_metrics = [(0.0, 1.0, len(reference_points))]
        for state_id in state_ids[1:]:
            points = apply_transform(
                load_points(
                    input_root / str(by_state[(state_id, part_id)]["measured_point_cloud_path"])
                ),
                alignment_by_state[state_id],
            )
            fit = robust_icp_sim3(points, reference_points, with_scale=False)
            transforms.append(fit.matrix)
            diagonal = max(
                float(np.linalg.norm(np.ptp(reference_points, axis=0))),
                np.finfo(np.float64).eps,
            )
            residual = float(np.median(fit.residuals) / diagonal)
            state_metrics.append(
                (
                    residual,
                    float(np.mean(fit.residuals <= 0.03 * diagonal)),
                    len(points),
                )
            )
        part_diagonal = max(
            float(np.linalg.norm(np.ptp(reference_points, axis=0))),
            np.finfo(np.float64).eps,
        )
        joint = (
            JointEstimate(
                joint_type="unknown",
                axis=None,
                pivot=None,
                positions=[0.0],
                orthogonal_residual=None,
                rotation_leakage_degrees=None,
                axis_consistency_degrees=None,
                pivot_residual=None,
            )
            if len(transforms) == 1
            else estimate_joint(
                transforms,
                max_fixed_translation=float(
                    configuration.get("maximum_fixed_translation_part_diagonals", 0.01)
                )
                * part_diagonal,
                max_fixed_rotation_degrees=float(
                    configuration.get("maximum_fixed_rotation_degrees", 2.0)
                ),
                max_prismatic_rotation_degrees=float(
                    configuration.get("maximum_prismatic_rotation_degrees", 5.0)
                ),
                max_prismatic_orthogonal_residual=float(
                    configuration.get("maximum_prismatic_orthogonal_residual", 0.05)
                ),
                max_revolute_axis_error_degrees=float(
                    configuration.get("maximum_revolute_axis_error_degrees", 15.0)
                ),
            )
        )
        pivot_residual_raw = joint.pivot_residual
        pivot_residual_normalized = (
            pivot_residual_raw / part_diagonal if pivot_residual_raw is not None else None
        )
        pivot_limit = float(
            configuration.get("maximum_revolute_pivot_residual_part_diagonals", 0.10)
        )
        pivot_rejected = (
            joint.joint_type == "revolute"
            and pivot_residual_normalized is not None
            and pivot_residual_normalized > pivot_limit
        )
        if pivot_rejected:
            joint = JointEstimate(
                joint_type="unknown",
                axis=None,
                pivot=None,
                positions=[0.0] * len(transforms),
                orthogonal_residual=None,
                rotation_leakage_degrees=None,
                axis_consistency_degrees=None,
                pivot_residual=pivot_residual_raw,
            )
        confidence = 0.25 if joint.joint_type == "unknown" else 0.8
        states = [
            {
                "state_id": state_id,
                "position": float(joint.positions[index]),
                "part_registration_median_residual": state_metrics[index][0],
                "part_coverage": state_metrics[index][1],
                "supporting_point_count": state_metrics[index][2],
                "state_confidence": confidence,
            }
            for index, state_id in enumerate(state_ids)
        ]
        hypotheses.append(
            {
                "joint_id": f"{part_id}_joint",
                "parent_part_id": request["base_part_id"],
                "child_part_id": part_id,
                "joint_type": joint.joint_type,
                "axis": (
                    [float(value) for value in joint.axis] if joint.axis is not None else None
                ),
                "pivot": (
                    [float(value) for value in joint.pivot] if joint.pivot is not None else None
                ),
                "states": states,
                "observed_position_min": (
                    float(min(joint.positions)) if len(state_ids) > 1 else None
                ),
                "observed_position_max": (
                    float(max(joint.positions)) if len(state_ids) > 1 else None
                ),
                "candidate_limit_lower": None,
                "candidate_limit_upper": None,
                "limit_source": "observed_range",
                "orthogonal_residual": joint.orthogonal_residual,
                "rotation_leakage_degrees": joint.rotation_leakage_degrees,
                "axis_consistency_degrees": joint.axis_consistency_degrees,
                "normalization_part_diagonal": part_diagonal,
                "fixed_translation_residual_arbitrary_units": (
                    float(
                        max(
                            np.linalg.norm(matrix[:3, 3] - transforms[0][:3, 3])
                            for matrix in transforms
                        )
                    )
                ),
                "fixed_translation_residual_part_diagonals": (
                    float(
                        max(
                            np.linalg.norm(matrix[:3, 3] - transforms[0][:3, 3])
                            for matrix in transforms
                        )
                    )
                    / part_diagonal
                ),
                "pivot_residual_arbitrary_units": pivot_residual_raw,
                "pivot_residual_part_diagonals": pivot_residual_normalized,
                "confidence": confidence,
                "warnings": (
                    [
                        "single static state cannot establish measured motion; "
                        f"configured joint hint={part.get('expected_joint_hint')}"
                    ]
                    if len(transforms) == 1
                    else (
                        [
                            "revolute hypothesis rejected because normalized pivot "
                            "residual exceeded the configured gate"
                        ]
                        if pivot_rejected
                        else []
                    )
                ),
            }
        )
    valid_measured_motion = any(
        item["joint_type"] in {"fixed", "prismatic", "revolute"} for item in hypotheses
    )
    accepted_count = len(request["accepted_alignment_state_ids"])
    effective_level = (
        "single_state_prior_only"
        if accepted_count < 2 or not valid_measured_motion
        else "two_state_motion_supported"
    )
    write_json(
        output_dir / "measured_motion.json",
        {
            "schema_version": "0.2.0",
            "capture_manifest_sha256": request["capture_manifest_sha256"],
            "state_alignment_sha256": request["state_alignment_sha256"],
            "articulated_object_id": request["articulated_object_id"],
            "reference_state_id": reference_state_id,
            "capture_state_count": request["capture_state_count"],
            "accepted_alignment_state_ids": request["accepted_alignment_state_ids"],
            "effective_motion_evidence_level": effective_level,
            "part_geometries": copied_geometries,
            "joint_hypotheses": hypotheses,
            "base_link_fixed": True,
            "runtime_seconds": time.monotonic() - started,
            "warnings": [],
        },
    )
    write_preview(
        output_dir / "previews/measured_part_motion.png",
        "Measured articulated part motion",
        [f"{item['joint_id']}: {item['joint_type']}" for item in hypotheses],
    )
    write_preview(
        output_dir / "previews/joint_axis_and_pivot.png",
        "Measured joint axis and pivot",
        [f"{item['joint_id']}: axis={item['axis']} pivot={item['pivot']}" for item in hypotheses],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("healthcheck", "align", "estimate-motion"))
    parser.add_argument("--request")
    parser.add_argument("--input-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.action == "healthcheck":
        print(f"articulation_alignment_worker {__version__}: numpy/scipy/trimesh available")
        return 0
    if not args.request or not args.input_root or not args.output_dir:
        parser.error("action requires --request, --input-root, and --output-dir")
    request = read_json(Path(args.request).resolve())
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.action == "align":
        align(request, input_root, output_dir)
    else:
        estimate_motion_action(request, input_root, output_dir)
    return 0
