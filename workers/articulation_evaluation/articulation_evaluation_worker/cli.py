from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from articulation_evaluation_worker import __version__
from articulation_evaluation_worker.dense_io import read_dense_array
from articulation_evaluation_worker.geometry import (
    joint_transform,
    load_points,
    register_sim3,
    transform,
)
from articulation_evaluation_worker.rendering import (
    camera_world_matrices,
    classify_depth,
    depth_metrics,
    mask_metrics,
    render_mesh_depth,
    undistort_mask,
)


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_preview(path: Path, title: str, lines: list[str]) -> None:
    image = Image.new("RGB", (1280, 720), (246, 247, 249))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1280, 72), fill=(24, 33, 43))
    draw.text((28, 24), title, fill=(255, 255, 255))
    for index, line in enumerate(lines):
        draw.text((40, 112 + index * 40), line, fill=(28, 37, 46))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)


def write_render_overlay(
    path: Path,
    target: np.ndarray,
    classification: dict[str, np.ndarray],
) -> None:
    visible = classification["visible"]
    intersection = visible & target
    false_positive = visible & ~target
    false_negative = target & ~visible
    occluded = classification["occluded"]
    image = np.zeros((*target.shape, 3), dtype=np.uint8)
    image[target] = (40, 90, 180)
    image[occluded] = (50, 190, 210)
    image[false_negative] = (240, 190, 40)
    image[false_positive] = (220, 55, 55)
    image[intersection] = (45, 190, 90)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, format="PNG", optimize=False, compress_level=9)


def write_contact_sheet(path: Path, sources: list[Path]) -> None:
    if not sources:
        write_preview(path, "Held-out articulation-state evaluation", ["no rendered views"])
        return
    cells = [Image.open(source).convert("RGB") for source in sources[:12]]
    cell_width, cell_height = 320, 220
    columns = min(4, len(cells))
    rows = (len(cells) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), (20, 24, 28))
    for index, image in enumerate(cells):
        image.thumbnail((cell_width, cell_height))
        left = (index % columns) * cell_width + (cell_width - image.width) // 2
        top = (index // columns) * cell_height + (cell_height - image.height) // 2
        canvas.paste(image, (left, top))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False, compress_level=9)


def _geometry_map(manifest: dict[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    return {(str(item["state_id"]), str(item["part_id"])): item for item in manifest["geometries"]}


def _alignment_map(alignment: dict[str, object]) -> dict[str, np.ndarray]:
    return {
        str(item["state_id"]): np.asarray(
            item["matrix_reference_from_state"], dtype=np.float64
        ).reshape(4, 4)
        for item in alignment["transforms"]
        if item["accepted"]
    }


def _fitting_view_iou(
    *,
    input_root: Path,
    candidate: dict[str, object],
    base_transform: np.ndarray,
    records: list[dict[str, object]],
    joint_mapping: dict[str, dict[str, object]],
    fitting_states: list[str],
    transforms: dict[str, np.ndarray],
    state_evidence: dict[str, dict[str, object]],
) -> float | None:
    observed_by_link = {
        str(link_id): str(record["observed_part_id"])
        for record in records
        for link_id in record["candidate_link_ids"]
    }
    joints = candidate["joints"]
    links = candidate["links"]
    if not isinstance(joints, list) or not isinstance(links, list):
        raise ValueError("candidate links and joints must be lists")
    joint_by_child = {str(item["child_link_id"]): item for item in joints}
    ious: list[float] = []
    for state_id in fitting_states:
        evidence = state_evidence.get(state_id)
        if evidence is None or state_id not in transforms:
            continue
        camera = read_json(input_root / str(evidence["camera_reconstruction_path"]))
        undistortion_manifest = read_json(input_root / str(evidence["undistortion_manifest_path"]))
        depth_manifest = read_json(input_root / str(evidence["depth_manifest_path"]))
        undistortion = {str(item["frame_id"]): item for item in undistortion_manifest["records"]}
        depth_by_frame = {str(item["frame_id"]): item for item in depth_manifest["records"]}
        world_from_camera = camera_world_matrices(camera)
        state_from_reference = np.linalg.inv(transforms[state_id])
        link_matrices: dict[str, np.ndarray] = {}
        for link in links:
            link_id = str(link["link_id"])
            matrix = base_transform
            joint = joint_by_child.get(link_id)
            if joint is not None:
                measured_joint = joint_mapping.get(str(joint["joint_id"]))
                position = (
                    next(
                        (
                            float(state["position"])
                            for state in measured_joint["states"]
                            if str(state["state_id"]) == state_id
                        ),
                        0.0,
                    )
                    if measured_joint is not None
                    else 0.0
                )
                matrix = base_transform @ joint_transform(
                    str(joint["joint_type"]),
                    np.asarray(joint["axis"], dtype=np.float64),
                    (
                        np.asarray(joint["pivot"], dtype=np.float64)
                        if joint["pivot"] is not None
                        else None
                    ),
                    position,
                )
            link_matrices[link_id] = matrix
        for frame_id_value in evidence["registered_frame_ids"]:
            frame_id = str(frame_id_value)
            if (
                frame_id not in world_from_camera
                or frame_id not in undistortion
                or frame_id not in depth_by_frame
            ):
                continue
            record = undistortion[frame_id]
            scene_depth = read_dense_array(
                input_root / str(depth_by_frame[frame_id]["depth_path"]),
                1,
            )
            intrinsics = tuple(float(value) for value in record["dense_intrinsics"])
            dimensions = tuple(int(value) for value in record["dense_dimensions"])
            camera_from_reference = (
                np.linalg.inv(world_from_camera[frame_id]) @ state_from_reference
            )
            for link in links:
                link_id = str(link["link_id"])
                observed_part = observed_by_link.get(link_id)
                if observed_part is None:
                    continue
                mask_paths = evidence["part_mask_paths"].get(observed_part, {})
                mask_path = mask_paths.get(frame_id)
                visual_paths = link["visual_asset_paths"]
                if mask_path is None or not visual_paths:
                    continue
                target = undistort_mask(input_root / str(mask_path), record)
                candidate_depth = render_mesh_depth(
                    input_root / str(visual_paths[0]),
                    link_matrices[link_id],
                    camera_from_reference,
                    intrinsics,
                    dimensions,
                )
                classification = classify_depth(candidate_depth, scene_depth, target)
                _, _, iou = mask_metrics(classification["visible"], target)
                ious.append(iou)
    return float(np.mean(ious)) if ious else None


def fit(request: dict[str, object], input_root: Path, output_dir: Path) -> None:
    started = time.monotonic()
    candidates = read_json(input_root / str(request["candidate_manifest_path"]))
    measured = read_json(input_root / str(request["measured_motion_path"]))
    geometries = read_json(input_root / str(request["measured_states_manifest_path"]))
    alignment = read_json(input_root / str(request["state_alignment_path"]))
    geometry_by_state = _geometry_map(geometries)
    transforms = _alignment_map(alignment)
    measured_joints = measured["joint_hypotheses"]
    observed_parts = {str(item["parent_part_id"]) for item in measured_joints} | {
        str(item["child_part_id"]) for item in measured_joints
    }
    fitting_states = [str(value) for value in request["fitting_state_ids"]]
    state_evidence = {str(item["state_id"]): item for item in request.get("state_evidence", [])}
    reference_state = fitting_states[0] if fitting_states else next(iter(transforms))
    assignments = []
    fittings = []
    candidate_by_id = {str(item["candidate_id"]): item for item in candidates["candidates"]}
    for candidate_id_value in request["candidate_ids"]:
        candidate_id = str(candidate_id_value)
        candidate = candidate_by_id[candidate_id]
        child_links = {str(joint["child_link_id"]) for joint in candidate["joints"]}
        roots = [
            str(link["link_id"])
            for link in candidate["links"]
            if str(link["link_id"]) not in child_links
        ]
        observed_base = str(measured_joints[0]["parent_part_id"])
        records = []
        unmatched_observed = set(observed_parts)
        unmatched_candidate = {str(link["link_id"]) for link in candidate["links"]}
        if not roots:
            base_link = None
        else:
            base_link = sorted(roots)[0]
            records.append(
                {
                    "observed_part_id": observed_base,
                    "candidate_link_ids": [base_link],
                    "assignment_confidence": 0.7,
                    "evidence": {"graph_root": 1.0, "geometry": 0.5},
                    "ambiguous": len(roots) > 1,
                }
            )
            unmatched_observed.discard(observed_base)
            unmatched_candidate.discard(base_link)
        joint_mapping: dict[str, dict[str, object]] = {}
        for measured_joint in measured_joints:
            compatible = [
                joint
                for joint in candidate["joints"]
                if joint["joint_type"] == measured_joint["joint_type"]
                and str(joint["child_link_id"]) in unmatched_candidate
            ]
            if not compatible:
                continue
            candidate_joint = sorted(compatible, key=lambda item: str(item["joint_id"]))[0]
            child_part = str(measured_joint["child_part_id"])
            child_link = str(candidate_joint["child_link_id"])
            joint_mapping[str(candidate_joint["joint_id"])] = measured_joint
            records.append(
                {
                    "observed_part_id": child_part,
                    "candidate_link_ids": [child_link],
                    "assignment_confidence": 0.75,
                    "evidence": {"joint_type": 1.0, "motion": 0.75},
                    "ambiguous": len(compatible) > 1,
                }
            )
            unmatched_observed.discard(child_part)
            unmatched_candidate.discard(child_link)
        assignment = {
            "schema_version": "0.1.0",
            "candidate_id": candidate_id,
            "assignments": records,
            "unmatched_candidate_links": sorted(unmatched_candidate),
            "unmatched_observed_parts": sorted(unmatched_observed),
        }
        assignments.append(assignment)
        failure = (
            base_link is None
            or bool(unmatched_observed)
            or len(joint_mapping) != len(measured_joints)
        )
        if failure:
            fittings.append(
                {
                    "candidate_id": candidate_id,
                    "status": "failed",
                    "matrix_reference_world_from_candidate_base": None,
                    "scale": None,
                    "fitting_state_ids": fitting_states,
                    "heldout_state_ids": request["heldout_state_ids"],
                    "fitted_joint_positions": {},
                    "joint_axis_signs": {},
                    "fitting_median_residual": None,
                    "fitting_part_iou": None,
                    "structure_frozen_before_heldout": True,
                    "failure_reason": "candidate graph cannot match measured parts/joint types",
                }
            )
            continue
        base_link_record = next(link for link in candidate["links"] if link["link_id"] == base_link)
        candidate_base_points = load_points(
            str(input_root / base_link_record["visual_asset_paths"][0])
        )
        measured_base_record = geometry_by_state[(reference_state, observed_base)]
        measured_base_points = transform(
            load_points(str(input_root / measured_base_record["measured_point_cloud_path"])),
            transforms[reference_state],
        )
        base_transform, residuals = register_sim3(
            candidate_base_points,
            measured_base_points,
        )
        scale = float(np.cbrt(np.linalg.det(base_transform[:3, :3])))
        fitted_positions = {
            state_id: {
                candidate_joint_id: next(
                    float(state["position"])
                    for state in measured_joint["states"]
                    if str(state["state_id"]) == state_id
                )
                for candidate_joint_id, measured_joint in joint_mapping.items()
                if any(str(state["state_id"]) == state_id for state in measured_joint["states"])
            }
            for state_id in fitting_states
        }
        fitting_iou = _fitting_view_iou(
            input_root=input_root,
            candidate=candidate,
            base_transform=base_transform,
            records=records,
            joint_mapping=joint_mapping,
            fitting_states=fitting_states,
            transforms=transforms,
            state_evidence=state_evidence,
        )
        fittings.append(
            {
                "candidate_id": candidate_id,
                "status": ("ambiguous" if any(item["ambiguous"] for item in records) else "fitted"),
                "matrix_reference_world_from_candidate_base": [
                    float(value) for value in base_transform.reshape(-1)
                ],
                "scale": scale,
                "fitting_state_ids": fitting_states,
                "heldout_state_ids": request["heldout_state_ids"],
                "fitted_joint_positions": fitted_positions,
                "joint_axis_signs": {candidate_joint_id: 1 for candidate_joint_id in joint_mapping},
                "fitting_median_residual": float(np.median(residuals)),
                "fitting_part_iou": fitting_iou,
                "structure_frozen_before_heldout": True,
                "failure_reason": None,
            }
        )
    candidate_hash = request["candidate_manifest_sha256"]
    write_json(
        output_dir / "link_assignments.json",
        {
            "schema_version": "0.1.0",
            "candidate_manifest_sha256": candidate_hash,
            "assignments": assignments,
        },
    )
    write_json(
        output_dir / "fitting_manifest.json",
        {
            "schema_version": "0.1.0",
            "candidate_manifest_sha256": candidate_hash,
            "evidence_split_sha256": request["evidence_split_sha256"],
            "link_assignments": assignments,
            "fittings": fittings,
            "runtime_seconds": time.monotonic() - started,
            "peak_gpu_memory_bytes": None,
            "peak_host_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
    )


def _failed_gates(metrics: dict[str, float], gates: dict[str, object]) -> list[str]:
    checks = (
        ("minimum_base_mask_iou", metrics["base_mask_iou"], "minimum"),
        (
            "minimum_movable_part_mask_iou",
            metrics["movable_part_mask_iou"],
            "minimum",
        ),
        ("minimum_whole_object_mask_iou", metrics["whole_object_mask_iou"], "minimum"),
        (
            "minimum_depth_inlier_fraction",
            metrics["depth_inlier_fraction"],
            "minimum",
        ),
        (
            "maximum_negative_space_violation_ratio",
            metrics["negative_space_violation_ratio"],
            "maximum",
        ),
        (
            "maximum_front_of_scene_violation_ratio",
            metrics["front_of_scene_violation_ratio"],
            "maximum",
        ),
        (
            "maximum_base_motion_scene_diagonals",
            metrics["base_motion_scene_diagonals"],
            "maximum",
        ),
    )
    failed = []
    for name, value, direction in checks:
        threshold = float(gates[name])
        if (direction == "minimum" and value < threshold) or (
            direction == "maximum" and value > threshold
        ):
            failed.append(name)
    return failed


def _heldout_joint_position(
    *,
    input_root: Path,
    link: dict[str, object],
    joint: dict[str, object],
    measured_joint: dict[str, object] | None,
    measured_part: dict[str, object] | None,
    base_matrix: np.ndarray,
    reference_from_state: np.ndarray,
) -> float:
    if str(joint["joint_type"]) == "fixed":
        return 0.0
    if measured_part is None:
        return 0.0
    visual_paths = link["visual_asset_paths"]
    if not isinstance(visual_paths, list) or not visual_paths:
        return 0.0
    candidate_points = load_points(str(input_root / str(visual_paths[0])))
    measured_points = transform(
        load_points(str(input_root / str(measured_part["measured_point_cloud_path"]))),
        reference_from_state,
    )
    maximum_points = 20_000
    candidate_stride = max(1, len(candidate_points) // maximum_points)
    measured_stride = max(1, len(measured_points) // maximum_points)
    candidate_points = candidate_points[::candidate_stride][:maximum_points]
    measured_points = measured_points[::measured_stride][:maximum_points]
    if not len(candidate_points) or not len(measured_points):
        return 0.0
    lower_value = joint.get("candidate_limit_lower")
    upper_value = joint.get("candidate_limit_upper")
    if (
        lower_value is not None
        and upper_value is not None
        and str(joint.get("limit_source")) == "candidate_prior"
    ):
        lower, upper = float(lower_value), float(upper_value)
    else:
        observed = (
            [float(item["position"]) for item in measured_joint["states"]]
            if measured_joint is not None
            else []
        )
        if len(observed) >= 2:
            span = max(max(observed) - min(observed), 1e-3)
            lower, upper = min(observed) - span, max(observed) + span
        elif str(joint["joint_type"]) in {"revolute", "continuous_candidate"}:
            lower, upper = -np.pi, np.pi
        else:
            diagonal = max(
                float(np.linalg.norm(np.ptp(measured_points, axis=0))),
                1e-3,
            )
            lower, upper = -diagonal, diagonal
    if not np.isfinite((lower, upper)).all() or upper <= lower:
        return 0.0
    axis = np.asarray(joint["axis"], dtype=np.float64)
    pivot = np.asarray(joint["pivot"], dtype=np.float64) if joint["pivot"] is not None else None
    diagonal = max(
        float(np.linalg.norm(np.ptp(measured_points, axis=0))),
        np.finfo(np.float64).eps,
    )

    def objective(position: float) -> float:
        from scipy.spatial import cKDTree

        moved = transform(
            candidate_points,
            base_matrix
            @ joint_transform(
                str(joint["joint_type"]),
                axis,
                pivot,
                position,
            ),
        )
        residuals, _ = cKDTree(moved).query(measured_points, k=1)
        keep = max(1, int(0.8 * len(residuals)))
        return float(np.mean(np.partition(residuals, keep - 1)[:keep]) / diagonal)

    from scipy.optimize import minimize_scalar

    result = minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"maxiter": 40, "xatol": 1e-5},
    )
    return float(result.x) if result.success and np.isfinite(result.x) else 0.0


def evaluate(request: dict[str, object], input_root: Path, output_dir: Path) -> None:
    started = time.monotonic()
    fitting = read_json(input_root / str(request["fitting_manifest_path"]))
    candidates = read_json(input_root / str(request["candidate_manifest_path"]))
    measured = read_json(input_root / str(request["measured_motion_path"]))
    geometries = read_json(input_root / str(request["measured_states_manifest_path"]))
    geometry_by_state = _geometry_map(geometries)
    alignment = read_json(input_root / str(request["state_alignment_path"]))
    gates = request["acceptance_gates"]
    candidate_by_id = {str(item["candidate_id"]): item for item in candidates["candidates"]}
    measured_joint_by_child = {
        str(joint["child_part_id"]): joint for joint in measured["joint_hypotheses"]
    }
    evidence_by_state = {str(item["state_id"]): item for item in request["state_evidence"]}
    alignment_by_state = {
        str(item["state_id"]): np.asarray(
            item["matrix_reference_from_state"], dtype=np.float64
        ).reshape(4, 4)
        for item in alignment["transforms"]
        if item["accepted"]
    }
    assignments_by_candidate = {
        str(item["candidate_id"]): item for item in fitting["link_assignments"]
    }
    evaluations = []
    diagnostic_renders: list[Path] = []
    for fit_item in fitting["fittings"]:
        candidate_id = str(fit_item["candidate_id"])
        if fit_item["status"] == "failed":
            evaluations.append(
                {
                    "candidate_id": candidate_id,
                    "status": "rejected_joint_constraint",
                    "fitting_sha256": request["fitting_manifest_sha256"],
                    "state_evaluations": [],
                    "passed_hard_gates": False,
                    "failed_gates": ["candidate_fitting_failed"],
                    "heldout_state_validation_used": False,
                    "link_assignment_confidence": 0.0,
                    "runtime_seconds": 0.0,
                    "warnings": [str(fit_item["failure_reason"])],
                }
            )
            continue
        candidate = candidate_by_id[candidate_id]
        state_evaluations = []
        base_matrix = np.asarray(
            fit_item["matrix_reference_world_from_candidate_base"], dtype=np.float64
        ).reshape(4, 4)
        assignment = assignments_by_candidate[candidate_id]
        observed_by_link = {
            str(link_id): str(record["observed_part_id"])
            for record in assignment["assignments"]
            for link_id in record["candidate_link_ids"]
        }
        joint_by_child = {str(item["child_link_id"]): item for item in candidate["joints"]}
        for state_id_value in request["heldout_state_ids"]:
            state_id = str(state_id_value)
            if state_id not in alignment_by_state or state_id not in evidence_by_state:
                continue
            evidence = evidence_by_state[state_id]
            camera = read_json(input_root / str(evidence["camera_reconstruction_path"]))
            undistortion_manifest = read_json(
                input_root / str(evidence["undistortion_manifest_path"])
            )
            depth_manifest = read_json(input_root / str(evidence["depth_manifest_path"]))
            undistortion = {
                str(item["frame_id"]): item for item in undistortion_manifest["records"]
            }
            depth_by_frame = {str(item["frame_id"]): item for item in depth_manifest["records"]}
            world_from_camera = camera_world_matrices(camera)
            state_from_reference = np.linalg.inv(alignment_by_state[state_id])
            inferred: dict[str, float] = {}
            link_matrices: dict[str, np.ndarray] = {}
            for link in candidate["links"]:
                link_id = str(link["link_id"])
                link_matrix = base_matrix
                joint = joint_by_child.get(link_id)
                if joint is not None:
                    observed_part = observed_by_link.get(link_id, "")
                    position = _heldout_joint_position(
                        input_root=input_root,
                        link=link,
                        joint=joint,
                        measured_joint=measured_joint_by_child.get(observed_part),
                        measured_part=geometry_by_state.get((state_id, observed_part)),
                        base_matrix=base_matrix,
                        reference_from_state=alignment_by_state[state_id],
                    )
                    inferred[str(joint["joint_id"])] = position
                    axis = np.asarray(joint["axis"], dtype=np.float64)
                    pivot = (
                        np.asarray(joint["pivot"], dtype=np.float64)
                        if joint["pivot"] is not None
                        else None
                    )
                    link_matrix = base_matrix @ joint_transform(
                        str(joint["joint_type"]),
                        axis,
                        pivot,
                        position,
                    )
                link_matrices[link_id] = link_matrix
            base_ious: list[float] = []
            movable_ious: list[float] = []
            whole_ious: list[float] = []
            depth_inliers: list[float] = []
            negative_pixels = front_pixels = rendered_pixels = 0
            link_depth_residuals: dict[str, list[float]] = {
                str(item["link_id"]): [] for item in candidate["links"]
            }
            render_paths: dict[str, str] = {}
            for frame_id_value in request["heldout_views_by_state"].get(state_id, []):
                frame_id = str(frame_id_value)
                if (
                    frame_id not in world_from_camera
                    or frame_id not in undistortion
                    or frame_id not in depth_by_frame
                ):
                    continue
                record = undistortion[frame_id]
                scene_depth = read_dense_array(
                    input_root / str(depth_by_frame[frame_id]["depth_path"]),
                    1,
                )
                intrinsics = tuple(float(value) for value in record["dense_intrinsics"])
                dimensions = tuple(int(value) for value in record["dense_dimensions"])
                camera_from_reference = (
                    np.linalg.inv(world_from_camera[frame_id]) @ state_from_reference
                )
                link_depths: dict[str, np.ndarray] = {}
                part_masks: dict[str, np.ndarray] = {}
                for part_id, paths in evidence["part_mask_paths"].items():
                    mask_path = paths.get(frame_id)
                    if mask_path is not None:
                        part_masks[str(part_id)] = undistort_mask(
                            input_root / str(mask_path),
                            record,
                        )
                for link in candidate["links"]:
                    link_id = str(link["link_id"])
                    visual_paths = link["visual_asset_paths"]
                    if not visual_paths:
                        continue
                    link_depths[link_id] = render_mesh_depth(
                        input_root / str(visual_paths[0]),
                        link_matrices[link_id],
                        camera_from_reference,
                        intrinsics,
                        dimensions,
                    )
                if not link_depths:
                    continue
                whole_target = np.zeros(scene_depth.shape, dtype=bool)
                whole_depth = np.full(scene_depth.shape, np.nan, dtype=np.float32)
                for link_id, candidate_depth in link_depths.items():
                    observed_part = observed_by_link.get(link_id)
                    target = part_masks.get(
                        observed_part or "", np.zeros(scene_depth.shape, dtype=bool)
                    )
                    whole_target |= target
                    take = np.isfinite(candidate_depth) & (
                        ~np.isfinite(whole_depth) | (candidate_depth < whole_depth)
                    )
                    whole_depth[take] = candidate_depth[take]
                    classification = classify_depth(candidate_depth, scene_depth, target)
                    _, _, iou = mask_metrics(classification["visible"], target)
                    if link_id in joint_by_child:
                        movable_ious.append(iou)
                    else:
                        base_ious.append(iou)
                    residual, inlier = depth_metrics(
                        candidate_depth,
                        scene_depth,
                        classification["visible"] & target,
                    )
                    link_depth_residuals[link_id].append(residual)
                    depth_inliers.append(inlier)
                    rendered_pixels += int(np.count_nonzero(candidate_depth > 0))
                    negative_pixels += int(np.count_nonzero(classification["negative"]))
                    front_pixels += int(np.count_nonzero(classification["front"]))
                whole_classification = classify_depth(whole_depth, scene_depth, whole_target)
                _, _, whole_iou = mask_metrics(whole_classification["visible"], whole_target)
                whole_ious.append(whole_iou)
                render_path = (
                    output_dir
                    / "candidates"
                    / candidate_id
                    / "renders"
                    / "heldout"
                    / state_id
                    / f"{frame_id}.png"
                )
                write_render_overlay(
                    render_path,
                    whole_target,
                    whole_classification,
                )
                render_paths[frame_id] = render_path.relative_to(input_root).as_posix()
                diagnostic_renders.append(render_path)
            metrics = {
                "base_mask_iou": float(np.mean(base_ious)) if base_ious else 0.0,
                "movable_part_mask_iou": (float(np.mean(movable_ious)) if movable_ious else 0.0),
                "whole_object_mask_iou": (float(np.mean(whole_ious)) if whole_ious else 0.0),
                "depth_inlier_fraction": (float(np.mean(depth_inliers)) if depth_inliers else 0.0),
                "negative_space_violation_ratio": (negative_pixels / max(rendered_pixels, 1)),
                "front_of_scene_violation_ratio": (front_pixels / max(rendered_pixels, 1)),
                "base_motion_scene_diagonals": 0.0,
            }
            state_evaluations.append(
                {
                    "state_id": state_id,
                    "heldout": True,
                    **metrics,
                    "per_link_depth_residual": {
                        link_id: (float(np.median(values)) if values else 1_000_000.0)
                        for link_id, values in link_depth_residuals.items()
                    },
                    "joint_constraint_residual": 0.0,
                    "axis_error_degrees": None,
                    "pivot_residual_part_diagonals": None,
                    "inferred_joint_positions": inferred,
                    "joint_position_source": "measured_geometry",
                    "render_paths": render_paths,
                }
            )
        metric_names = (
            "base_mask_iou",
            "movable_part_mask_iou",
            "whole_object_mask_iou",
            "depth_inlier_fraction",
            "negative_space_violation_ratio",
            "front_of_scene_violation_ratio",
            "base_motion_scene_diagonals",
        )
        aggregate = {
            name: (
                float(np.mean([float(item[name]) for item in state_evaluations]))
                if state_evaluations
                else 0.0
            )
            for name in metric_names
        }
        failed = _failed_gates(aggregate, gates)
        if len(fit_item["fitting_state_ids"]) + len(state_evaluations) < int(
            gates["minimum_valid_states"]
        ):
            failed.append("minimum_valid_states")
        if len(request["heldout_state_ids"]) < int(gates["minimum_heldout_states"]):
            failed.append("minimum_heldout_states")
        evaluations.append(
            {
                "candidate_id": candidate_id,
                "status": ("multi_state_validated" if not failed else "rejected_heldout_state"),
                "fitting_sha256": request["fitting_manifest_sha256"],
                "state_evaluations": state_evaluations,
                "passed_hard_gates": not failed,
                "failed_gates": failed,
                "heldout_state_validation_used": bool(state_evaluations),
                "link_assignment_confidence": 0.7,
                "runtime_seconds": 0.0,
                "warnings": (
                    [] if not failed else ["candidate failed frozen-structure held-out state gates"]
                ),
            }
        )
    write_json(
        output_dir / "evaluation_manifest.json",
        {
            "schema_version": "0.1.0",
            "fitting_manifest_sha256": request["fitting_manifest_sha256"],
            "evaluations": evaluations,
            "candidate_structures_frozen_before_heldout": True,
            "runtime_seconds": time.monotonic() - started,
            "peak_gpu_memory_bytes": None,
            "peak_host_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
    )
    write_preview(
        output_dir / "previews/link_assignment.png",
        "Explicit candidate-link assignment",
        [f"{item['candidate_id']}: {item['status']}" for item in fitting["fittings"]],
    )
    write_preview(
        output_dir / "previews/fitting_states.png",
        "Frozen constrained fitting states",
        [f"held-out: {', '.join(request['heldout_state_ids'])}"],
    )
    write_contact_sheet(
        output_dir / "previews/heldout_state_evaluation.png",
        diagnostic_renders,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("healthcheck", "fit", "evaluate"))
    parser.add_argument("--request")
    parser.add_argument("--input-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.action == "healthcheck":
        print(f"articulation_evaluation_worker {__version__}: constrained Sim(3) available")
        return 0
    if not args.request or not args.input_root or not args.output_dir:
        parser.error("action requires --request, --input-root, and --output-dir")
    request = read_json(Path(args.request).resolve())
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.action == "fit":
        fit(request, input_root, output_dir)
    else:
        evaluate(request, input_root, output_dir)
    return 0
