from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from collections.abc import Callable
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
from articulation_evaluation_worker.kinematics import (
    prismatic_candidate_q_scale,
    revolute_candidate_q_scale,
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


def write_q_objective_preview(
    path: Path,
    candidate_id: str,
    joint_audits: list[dict[str, object]],
) -> None:
    width, height = 1280, max(720, 640 * len(joint_audits))
    image = Image.new("RGB", (width, height), (246, 247, 249))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 72), fill=(24, 33, 43))
    draw.text((28, 24), f"Held-out q objective: {candidate_id}", fill=(255, 255, 255))
    for audit_index, audit in enumerate(joint_audits):
        samples = audit["samples"]
        if not isinstance(samples, list) or not samples:
            continue
        q_values = [float(item["q"]) for item in samples]
        objectives = [float(item["total_objective"]) for item in samples]
        left, right = 90, width - 50
        top = 110 + audit_index * 640
        bottom = top + 470
        q_min, q_max = min(q_values), max(q_values)
        objective_min, objective_max = min(objectives), max(objectives)
        q_span = max(q_max - q_min, np.finfo(np.float64).eps)
        objective_span = max(
            objective_max - objective_min,
            np.finfo(np.float64).eps,
        )

        def point(
            q_value: float,
            objective: float,
            *,
            plot_left: int = left,
            plot_right: int = right,
            plot_top: int = top,
            plot_bottom: int = bottom,
            q_lower: float = q_min,
            q_range: float = q_span,
            objective_lower: float = objective_min,
            objective_range: float = objective_span,
        ) -> tuple[int, int]:
            x = plot_left + int((q_value - q_lower) / q_range * (plot_right - plot_left))
            y = plot_bottom - int(
                (objective - objective_lower) / objective_range * (plot_bottom - plot_top)
            )
            return x, y

        draw.rectangle((left, top, right, bottom), outline=(110, 120, 130), width=2)
        draw.line(
            [point(q_values[index], objectives[index]) for index in range(len(samples))],
            fill=(42, 105, 170),
            width=3,
        )
        markers = (
            ("legacy", audit.get("legacy_optimizer_q"), (215, 85, 45)),
            ("grid", audit["grid_global_minimum_q"], (125, 90, 190)),
            ("selected", audit["selected_q"], (25, 150, 80)),
        )
        for label, value, color in markers:
            if value is None:
                continue
            marker_x = point(float(value), objective_min)[0]
            draw.line((marker_x, top, marker_x, bottom), fill=color, width=2)
            draw.text((marker_x + 4, top + 4), label, fill=color)
        semantic = audit["semantic_ordering"]
        draw.text(
            (left, bottom + 18),
            (
                f"{audit['joint_id']} [{audit['lower_bound']:.6g}, "
                f"{audit['upper_bound']:.6g}] "
                f"selected={audit['selected_q']:.6g} "
                f"class={audit['classification']}"
            ),
            fill=(28, 37, 46),
        )
        draw.text(
            (left, bottom + 44),
            (
                f"semantic ordering={semantic['direction']} "
                f"consistent={semantic['ordering_consistent']} "
                f"local minima={len(audit['all_local_minima'])}"
            ),
            fill=(28, 37, 46),
        )
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


def _stable_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _diagonal(points: np.ndarray) -> float:
    return max(float(np.linalg.norm(np.ptp(points, axis=0))), np.finfo(np.float64).eps)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError("cannot normalize a zero or non-finite articulation vector")
    return vector / norm


def _axis_angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.degrees(
            np.arccos(np.clip(abs(float(np.dot(_normalize(left), _normalize(right)))), -1.0, 1.0))
        )
    )


def _rotation_axis_and_signed_angle(
    rotation: np.ndarray,
    reference_axis: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    skew = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    sine_vector = 0.5 * skew
    sine = float(np.dot(_normalize(reference_axis), sine_vector))
    angle = float(np.arctan2(sine, cosine))
    norm = float(np.linalg.norm(sine_vector))
    if norm <= 1e-8:
        return None, angle
    axis = sine_vector / norm
    if float(np.dot(axis, reference_axis)) < 0:
        axis = -axis
    return axis, angle


def _bounded_axis_refinement(
    candidate_axis_world: np.ndarray,
    measured_axis_world: np.ndarray,
    maximum_degrees: float,
) -> tuple[np.ndarray, int, float]:
    candidate = _normalize(candidate_axis_world)
    measured = _normalize(measured_axis_world)
    sign = 1 if float(np.dot(candidate, measured)) >= 0 else -1
    signed_candidate = sign * candidate
    angle = _axis_angle_degrees(signed_candidate, measured)
    if angle <= 1e-9:
        return signed_candidate, sign, 0.0
    fraction = min(1.0, maximum_degrees / angle)
    refined = _normalize((1.0 - fraction) * signed_candidate + fraction * measured)
    return refined, sign, min(angle, maximum_degrees)


def _trimmed_surface_residual(source: np.ndarray, target: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(source).query(target, k=1)
    keep = max(1, int(0.8 * len(distances)))
    return float(np.median(np.partition(distances, keep - 1)[:keep]))


def _candidate_link_points(
    input_root: Path,
    candidate: dict[str, object],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for link in candidate["links"]:
        paths = link["visual_asset_paths"]
        if not isinstance(paths, list) or not paths:
            continue
        path = str(paths[0])
        asset_to_candidate = _visual_asset_to_candidate_matrix(input_root, link, path)
        result[str(link["link_id"])] = transform(
            load_points(str(input_root / path)),
            asset_to_candidate,
        )
    return result


def _visual_asset_to_candidate_matrix(
    input_root: Path,
    link: dict[str, object],
    path: str,
) -> np.ndarray:
    spaces = link.get("visual_asset_spaces")
    transforms = link.get("visual_asset_transforms_candidate_base")
    if not isinstance(spaces, dict) or not isinstance(transforms, dict):
        raise ValueError("candidate link omits explicit visual asset-space metadata")
    if path not in spaces or path not in transforms:
        raise ValueError(f"candidate visual {path!r} has no explicit asset-space record")
    hashes = link.get("visual_asset_hashes")
    if not isinstance(hashes, dict) or path not in hashes:
        raise ValueError(f"candidate visual {path!r} has no declared content hash")
    if sha256(input_root / path) != str(hashes[path]):
        raise ValueError(f"candidate visual {path!r} content hash mismatch")
    space = str(spaces[path])
    if space not in {"reference_world", "candidate_base", "link_local"}:
        raise ValueError(f"candidate visual {path!r} has unsupported asset space {space!r}")
    matrix = np.asarray(transforms[path], dtype=np.float64).reshape(4, 4)
    if not np.isfinite(matrix).all() or not np.allclose(
        matrix[3],
        np.array([0.0, 0.0, 0.0, 1.0]),
        atol=1e-8,
    ):
        raise ValueError(f"candidate visual {path!r} has an invalid asset transform")
    if space == "reference_world" and not np.allclose(matrix, np.eye(4), atol=1e-8):
        raise ValueError(
            f"reference-world candidate visual {path!r} requires an identity transform"
        )
    if space == "candidate_base" and not np.allclose(matrix, np.eye(4), atol=1e-8):
        raise ValueError("candidate-base visual asset must use an identity transform")
    return matrix


def _refine_pivot(
    *,
    candidate_pivot_local: np.ndarray,
    measured_pivot_world: np.ndarray,
    base_transform: np.ndarray,
    maximum_world_distance: float,
) -> tuple[np.ndarray, float, float]:
    candidate_world = transform(candidate_pivot_local[None, :], base_transform)[0]
    delta = measured_pivot_world - candidate_world
    raw_distance = float(np.linalg.norm(delta))
    if raw_distance > maximum_world_distance > 0:
        delta *= maximum_world_distance / raw_distance
    refined_world = candidate_world + delta
    refined_local = transform(refined_world[None, :], np.linalg.inv(base_transform))[0]
    return refined_local, float(np.linalg.norm(delta)), raw_distance


def _fit_q_offset(
    *,
    input_root: Path,
    evidence_state_ids: list[str],
    measured_positions: dict[str, float],
    geometry_by_state: dict[tuple[str, str], dict[str, object]],
    child_part: str,
    state_transforms: dict[str, np.ndarray],
    link_points: np.ndarray,
    base_transform: np.ndarray,
    joint_type: str,
    fitted_axis: np.ndarray,
    fitted_pivot: np.ndarray | None,
    q_scale: float,
    offset_bound: float,
) -> float | None:
    from scipy.optimize import minimize_scalar

    def objective(candidate_offset: float) -> float:
        values = []
        for state_id in evidence_state_ids:
            measured_record = geometry_by_state[(state_id, child_part)]
            measured_points = transform(
                load_points(str(input_root / measured_record["measured_point_cloud_path"])),
                state_transforms[state_id],
            )
            moved = transform(
                link_points,
                base_transform
                @ joint_transform(
                    joint_type,
                    fitted_axis,
                    fitted_pivot,
                    candidate_offset + q_scale * measured_positions[state_id],
                ),
            )
            values.append(_trimmed_surface_residual(moved, measured_points))
        return float(np.median(values))

    result = minimize_scalar(
        objective,
        bounds=(-offset_bound, offset_bound),
        method="bounded",
        options={"maxiter": 40, "xatol": 1e-5},
    )
    return float(result.x) if result.success and np.isfinite(result.x) else None


def _fitting_view_iou(
    *,
    input_root: Path,
    candidate: dict[str, object],
    base_transform: np.ndarray,
    records: list[dict[str, object]],
    fitted_joints: list[dict[str, object]],
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
    fitted_by_candidate = {str(item["candidate_joint_id"]): item for item in fitted_joints}
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
                fitted = fitted_by_candidate.get(str(joint["joint_id"]))
                if fitted is None:
                    continue
                position = float(fitted["fitting_state_q"].get(state_id, 0.0))
                matrix = base_transform @ joint_transform(
                    str(joint["joint_type"]),
                    np.asarray(fitted["fitted_axis"], dtype=np.float64),
                    (
                        np.asarray(fitted["fitted_pivot"], dtype=np.float64)
                        if fitted["fitted_pivot"] is not None
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
                    link_matrices[link_id]
                    @ _visual_asset_to_candidate_matrix(
                        input_root,
                        link,
                        str(visual_paths[0]),
                    ),
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
    generation_states = [str(value) for value in request["generation_state_ids"]]
    fitting_states = [str(value) for value in request["fitting_state_ids"]]
    structure_states = [*generation_states, *fitting_states]
    state_evidence = {str(item["state_id"]): item for item in request.get("state_evidence", [])}
    reference_state = str(request["reference_state_id"])
    if reference_state not in transforms or reference_state not in structure_states:
        raise ValueError("declared reference state must be accepted and materialized for fitting")
    configuration = request["fitting_configuration"]
    if not isinstance(configuration, dict):
        raise ValueError("articulation fitting configuration is invalid")
    ambiguity_margin = float(configuration["link_assignment_ambiguity_margin"])
    maximum_axis_refinement = float(configuration["maximum_axis_refinement_degrees"])
    maximum_pivot_refinement = float(configuration["maximum_pivot_refinement_part_diagonals"])
    assignments = []
    fittings = []
    candidate_by_id = {str(item["candidate_id"]): item for item in candidates["candidates"]}
    for candidate_id_value in request["candidate_ids"]:
        candidate_id = str(candidate_id_value)
        candidate = candidate_by_id[candidate_id]
        candidate_points_by_link = _candidate_link_points(input_root, candidate)
        child_links = {str(joint["child_link_id"]) for joint in candidate["joints"]}
        roots = [
            str(link["link_id"])
            for link in candidate["links"]
            if str(link["link_id"]) not in child_links
        ]
        observed_base = str(measured_joints[0]["parent_part_id"])
        base_state_points = [
            transform(
                load_points(
                    str(
                        input_root
                        / geometry_by_state[(state_id, observed_base)]["measured_point_cloud_path"]
                    )
                ),
                transforms[state_id],
            )
            for state_id in structure_states
            if (state_id, observed_base) in geometry_by_state
        ]
        if not base_state_points:
            raise ValueError("no accepted fitting-state base geometry is available")
        measured_base_points = np.concatenate(base_state_points, axis=0)
        scene_diagonal = _diagonal(measured_base_points)
        root_scores: list[tuple[float, str, np.ndarray, np.ndarray]] = []
        for root_link_id in roots:
            candidate_base_points = candidate_points_by_link.get(root_link_id)
            if candidate_base_points is None or len(candidate_base_points) < 3:
                continue
            base_transform, residuals = register_sim3(
                candidate_base_points,
                measured_base_points,
            )
            score = float(np.median(residuals) / scene_diagonal)
            root_scores.append((score, root_link_id, base_transform, residuals))
        root_scores.sort(key=lambda item: (item[0], item[1]))
        records = []
        unmatched_observed = set(observed_parts)
        unmatched_candidate = {str(link["link_id"]) for link in candidate["links"]}
        ambiguity_reasons: list[str] = []
        if not root_scores:
            base_link = None
            base_transform = None
            base_residuals = np.asarray([], dtype=np.float64)
        else:
            base_score, base_link, base_transform, base_residuals = root_scores[0]
            base_ambiguous = (
                len(root_scores) > 1 and root_scores[1][0] - base_score <= ambiguity_margin
            )
            if base_ambiguous:
                ambiguity_reasons.append("multiple candidate base links have equivalent costs")
            records.append(
                {
                    "observed_part_id": observed_base,
                    "candidate_link_ids": [base_link],
                    "assignment_confidence": max(0.0, min(1.0, 1.0 - base_score)),
                    "evidence": {
                        "graph_role_cost": 0.0,
                        "multi_state_geometry_residual": base_score,
                    },
                    "ambiguous": base_ambiguous,
                }
            )
            unmatched_observed.discard(observed_base)
            unmatched_candidate.discard(base_link)
        fitted_joints: list[dict[str, object]] = []
        candidate_joint_by_id = {str(item["joint_id"]): item for item in candidate["joints"]}
        for measured_joint in measured_joints:
            if base_transform is None:
                break
            measured_joint_type = str(measured_joint["joint_type"])
            compatible = [
                joint
                for joint in candidate["joints"]
                if (
                    str(joint["joint_type"]) == measured_joint_type
                    or (
                        measured_joint_type == "revolute"
                        and str(joint["joint_type"]) == "continuous_candidate"
                    )
                )
                and str(joint["child_link_id"]) in unmatched_candidate
            ]
            if not compatible:
                continue
            child_part = str(measured_joint["child_part_id"])
            measured_child_record = geometry_by_state[(reference_state, child_part)]
            measured_child_points = transform(
                load_points(str(input_root / measured_child_record["measured_point_cloud_path"])),
                transforms[reference_state],
            )
            part_diagonal = _diagonal(measured_child_points)
            measured_axis = (
                np.asarray(measured_joint["axis"], dtype=np.float64)
                if measured_joint["axis"] is not None
                else None
            )
            measured_range = (
                float(measured_joint["observed_position_max"])
                - float(measured_joint["observed_position_min"])
                if measured_joint["observed_position_min"] is not None
                and measured_joint["observed_position_max"] is not None
                else 0.0
            )
            candidate_costs: list[tuple[float, str, dict[str, float]]] = []
            scale = float(np.cbrt(np.linalg.det(base_transform[:3, :3])))
            rotation = base_transform[:3, :3] / scale
            measured_size_ratio = part_diagonal / scene_diagonal
            measured_centroid = np.median(measured_child_points, axis=0)
            for candidate_joint in compatible:
                joint_id = str(candidate_joint["joint_id"])
                child_link = str(candidate_joint["child_link_id"])
                link_points = candidate_points_by_link.get(child_link)
                if link_points is None or len(link_points) < 3:
                    continue
                candidate_size_ratio = _diagonal(link_points) / max(
                    _diagonal(candidate_points_by_link[base_link]),
                    np.finfo(np.float64).eps,
                )
                size_cost = abs(
                    np.log(max(candidate_size_ratio, 1e-9) / max(measured_size_ratio, 1e-9))
                )
                axis_cost = 0.0
                if measured_axis is not None:
                    candidate_axis_world = rotation @ np.asarray(
                        candidate_joint["axis"], dtype=np.float64
                    )
                    axis_cost = (
                        _axis_angle_degrees(
                            candidate_axis_world,
                            measured_axis,
                        )
                        / 180.0
                    )
                candidate_centroid = np.median(
                    transform(link_points, base_transform),
                    axis=0,
                )
                placement_cost = float(
                    np.linalg.norm(candidate_centroid - measured_centroid) / scene_diagonal
                )
                candidate_range = 0.0
                if (
                    candidate_joint["candidate_limit_lower"] is not None
                    and candidate_joint["candidate_limit_upper"] is not None
                ):
                    candidate_range = float(candidate_joint["candidate_limit_upper"]) - float(
                        candidate_joint["candidate_limit_lower"]
                    )
                    if measured_joint_type == "prismatic":
                        candidate_range *= scale
                range_cost = (
                    abs(np.log(max(abs(candidate_range), 1e-9) / max(abs(measured_range), 1e-9)))
                    if measured_range > 1e-9 and candidate_range > 1e-9
                    else 0.0
                )
                measured_positions = {
                    str(state["state_id"]): float(state["position"])
                    for state in measured_joint["states"]
                }
                candidate_axis = np.asarray(candidate_joint["axis"], dtype=np.float64)
                if (
                    measured_axis is not None
                    and float(np.dot(rotation @ candidate_axis, measured_axis)) < 0
                ):
                    candidate_axis = -candidate_axis
                candidate_pivot = (
                    np.asarray(candidate_joint["pivot"], dtype=np.float64)
                    if candidate_joint["pivot"] is not None
                    else None
                )
                multi_state_residuals: list[float] = []
                for state_id in structure_states:
                    measured_record = geometry_by_state.get((state_id, child_part))
                    if measured_record is None or state_id not in measured_positions:
                        continue
                    state_points = transform(
                        load_points(str(input_root / measured_record["measured_point_cloud_path"])),
                        transforms[state_id],
                    )
                    measured_position = measured_positions[state_id]
                    candidate_position = (
                        measured_position / scale
                        if measured_joint_type == "prismatic"
                        else measured_position
                    )
                    moved_link_points = transform(
                        link_points,
                        base_transform
                        @ joint_transform(
                            str(candidate_joint["joint_type"]),
                            candidate_axis,
                            candidate_pivot,
                            candidate_position,
                        ),
                    )
                    multi_state_residuals.append(
                        _trimmed_surface_residual(
                            moved_link_points,
                            state_points,
                        )
                        / part_diagonal
                    )
                geometry_cost = (
                    float(np.median(multi_state_residuals))
                    if multi_state_residuals
                    else float("inf")
                )
                if not np.isfinite(geometry_cost):
                    continue
                terms = {
                    "joint_type_cost": 0.0,
                    "graph_role_cost": 0.0,
                    "part_size_ratio_cost": float(size_cost),
                    "observed_axis_cost": float(axis_cost),
                    "observed_motion_range_cost": float(range_cost),
                    "relative_parent_child_placement_cost": placement_cost,
                    "multi_state_geometry_residual": geometry_cost,
                }
                total = (
                    0.15 * size_cost
                    + 0.20 * axis_cost
                    + 0.10 * range_cost
                    + 0.20 * placement_cost
                    + 0.35 * geometry_cost
                )
                candidate_costs.append((float(total), joint_id, terms))
            candidate_costs.sort(key=lambda item: (item[0], item[1]))
            if not candidate_costs:
                continue
            best_cost, candidate_joint_id, evidence = candidate_costs[0]
            candidate_joint = candidate_joint_by_id[candidate_joint_id]
            child_link = str(candidate_joint["child_link_id"])
            assignment_ambiguous = (
                len(candidate_costs) > 1 and candidate_costs[1][0] - best_cost <= ambiguity_margin
            )
            if assignment_ambiguous:
                ambiguity_reasons.append(
                    f"joint {measured_joint['joint_id']} has multiple assignments "
                    "within configured margin"
                )
            records.append(
                {
                    "observed_part_id": child_part,
                    "candidate_link_ids": [child_link],
                    "assignment_confidence": max(0.0, min(1.0, 1.0 - best_cost)),
                    "evidence": evidence,
                    "ambiguous": assignment_ambiguous,
                }
            )
            unmatched_observed.discard(child_part)
            unmatched_candidate.discard(child_link)
            candidate_axis = np.asarray(candidate_joint["axis"], dtype=np.float64)
            candidate_axis_world = rotation @ candidate_axis
            if measured_axis is None:
                refined_axis_world = _normalize(candidate_axis_world)
                axis_sign = 1
                axis_refinement = 0.0
            else:
                refined_axis_world, axis_sign, axis_refinement = _bounded_axis_refinement(
                    candidate_axis_world,
                    measured_axis,
                    maximum_axis_refinement,
                )
            refined_axis_local = _normalize(rotation.T @ refined_axis_world)
            fitted_pivot = None
            pivot_refinement_raw = None
            pivot_refinement_normalized = None
            if measured_joint_type in {"revolute", "continuous_candidate"}:
                candidate_pivot = np.asarray(
                    candidate_joint["pivot"],
                    dtype=np.float64,
                )
                measured_pivot = np.asarray(measured_joint["pivot"], dtype=np.float64)
                fitted_pivot, pivot_refinement_raw, _ = _refine_pivot(
                    candidate_pivot_local=candidate_pivot,
                    measured_pivot_world=measured_pivot,
                    base_transform=base_transform,
                    maximum_world_distance=maximum_pivot_refinement * part_diagonal,
                )
                pivot_refinement_normalized = pivot_refinement_raw / part_diagonal
            if measured_joint_type == "prismatic":
                q_scale = prismatic_candidate_q_scale(scale)
            elif measured_joint_type in {"revolute", "continuous_candidate"}:
                q_scale = revolute_candidate_q_scale()
            else:
                q_scale = 0.0
            link_points = candidate_points_by_link[child_link]
            measured_positions = {
                str(state["state_id"]): float(state["position"])
                for state in measured_joint["states"]
            }
            q_offset_evidence = [
                state_id
                for state_id in structure_states
                if state_id in measured_positions
                and geometry_by_state.get((state_id, child_part)) is not None
            ]
            q_offset = 0.0
            q_offset_fitted = False
            if len(q_offset_evidence) >= 2 and q_scale != 0.0:
                if measured_joint_type == "prismatic":
                    measured_span = max(
                        measured_positions[state_id] for state_id in q_offset_evidence
                    ) - min(measured_positions[state_id] for state_id in q_offset_evidence)
                    offset_bound = max(abs(q_scale * measured_span), part_diagonal / scale, 1e-3)
                else:
                    offset_bound = np.pi
                fitted_offset = _fit_q_offset(
                    input_root=input_root,
                    evidence_state_ids=q_offset_evidence,
                    measured_positions=measured_positions,
                    geometry_by_state=geometry_by_state,
                    child_part=child_part,
                    state_transforms=transforms,
                    link_points=link_points,
                    base_transform=base_transform,
                    joint_type=measured_joint_type,
                    fitted_axis=refined_axis_local,
                    fitted_pivot=fitted_pivot,
                    q_scale=q_scale,
                    offset_bound=offset_bound,
                )
                if fitted_offset is not None:
                    q_offset = fitted_offset
                    q_offset_fitted = True
            fitting_q = {
                state_id: q_offset + q_scale * measured_positions[state_id]
                for state_id in q_offset_evidence
            }
            residual_values: list[float] = []
            for state_id, q_value in fitting_q.items():
                measured_record = geometry_by_state.get((state_id, child_part))
                if measured_record is None:
                    continue
                measured_points = transform(
                    load_points(str(input_root / measured_record["measured_point_cloud_path"])),
                    transforms[state_id],
                )
                moved = transform(
                    link_points,
                    base_transform
                    @ joint_transform(
                        measured_joint_type,
                        refined_axis_local,
                        fitted_pivot,
                        q_value,
                    ),
                )
                residual_values.append(_trimmed_surface_residual(moved, measured_points))
            fit_residual = float(np.median(residual_values)) if residual_values else float("inf")
            if not np.isfinite(fit_residual):
                ambiguity_reasons.append(
                    f"joint {candidate_joint_id} has no finite multi-state fit residual"
                )
                fit_residual = 1_000_000.0
            fitted_joints.append(
                {
                    "candidate_joint_id": candidate_joint_id,
                    "measured_joint_id": measured_joint["joint_id"],
                    "parent_observed_part_id": measured_joint["parent_part_id"],
                    "child_observed_part_id": child_part,
                    "joint_type": measured_joint_type,
                    "fitted_axis": [float(value) for value in refined_axis_local],
                    "fitted_pivot": (
                        [float(value) for value in fitted_pivot]
                        if fitted_pivot is not None
                        else None
                    ),
                    "axis_sign": axis_sign,
                    "axis_convention": "oriented_toward_measured_axis",
                    "axis_sign_role": "native_axis_flip_provenance_only",
                    "q_scale": q_scale,
                    "q_scale_convention": "candidate_q_per_measured_q",
                    "q_offset": q_offset,
                    "q_offset_fitted": q_offset_fitted,
                    "q_offset_evidence_state_ids": (q_offset_evidence if q_offset_fitted else []),
                    "fitting_state_q": fitting_q,
                    "axis_refinement_degrees": axis_refinement,
                    "pivot_refinement_arbitrary_units": pivot_refinement_raw,
                    "pivot_refinement_part_diagonals": pivot_refinement_normalized,
                    "fitting_residual_arbitrary_units": fit_residual,
                    "fitting_residual_part_diagonals": fit_residual / part_diagonal,
                }
            )
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
            or len(fitted_joints) != len(measured_joints)
            or base_transform is None
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
                    "fitted_model": None,
                    "fitted_model_sha256": None,
                    "structure_frozen_before_heldout": True,
                    "failure_reason": "candidate graph cannot match measured parts/joint types",
                }
            )
            continue
        scale = float(np.cbrt(np.linalg.det(base_transform[:3, :3])))
        fitted_positions = {
            state_id: {
                str(item["candidate_joint_id"]): float(item["fitting_state_q"][state_id])
                for item in fitted_joints
                if state_id in item["fitting_state_q"]
            }
            for state_id in structure_states
        }
        fitting_iou = _fitting_view_iou(
            input_root=input_root,
            candidate=candidate,
            base_transform=base_transform,
            records=records,
            fitted_joints=fitted_joints,
            fitting_states=fitting_states,
            transforms=transforms,
            state_evidence=state_evidence,
        )
        fitted_model = {
            "schema_version": "0.1.0",
            "candidate_id": candidate_id,
            "matrix_reference_world_from_candidate_base": [
                float(value) for value in base_transform.reshape(-1)
            ],
            "scale": scale,
            "link_assignments": records,
            "fitted_joints": fitted_joints,
            "generation_state_ids": generation_states,
            "fitting_state_ids": fitting_states,
            "heldout_state_ids": request["heldout_state_ids"],
            "fit_residual_arbitrary_units": float(np.median(base_residuals)),
            "fit_residual_scene_diagonals": (float(np.median(base_residuals)) / scene_diagonal),
            "ambiguity_reasons": ambiguity_reasons,
        }
        fittings.append(
            {
                "candidate_id": candidate_id,
                "status": ("ambiguous" if ambiguity_reasons else "fitted"),
                "matrix_reference_world_from_candidate_base": [
                    float(value) for value in base_transform.reshape(-1)
                ],
                "scale": scale,
                "fitting_state_ids": fitting_states,
                "heldout_state_ids": request["heldout_state_ids"],
                "fitted_joint_positions": fitted_positions,
                "joint_axis_signs": {
                    str(item["candidate_joint_id"]): int(item["axis_sign"])
                    for item in fitted_joints
                },
                "fitting_median_residual": float(np.median(base_residuals)),
                "fitting_part_iou": fitting_iou,
                "fitted_model": fitted_model,
                "fitted_model_sha256": _stable_hash(fitted_model),
                "structure_frozen_before_heldout": True,
                "failure_reason": ("; ".join(ambiguity_reasons) if ambiguity_reasons else None),
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


def _failed_gates(
    metrics: dict[str, float | None],
    gates: dict[str, object],
    *,
    joint_types: set[str],
    minimum_usable_views: int,
) -> list[str]:
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
        if value is None or not np.isfinite(value):
            failed.append(f"{name}_unavailable")
            continue
        threshold = float(gates[name])
        if (direction == "minimum" and value < threshold) or (
            direction == "maximum" and value > threshold
        ):
            failed.append(name)
    for metric_name, gate_name in (
        ("usable_heldout_view_count", "minimum_usable_heldout_views"),
        ("rendered_heldout_view_count", "minimum_rendered_heldout_views"),
        ("views_with_target_masks", "minimum_target_mask_heldout_views"),
        ("views_with_valid_depth", "minimum_valid_depth_heldout_views"),
    ):
        if int(metrics.get(metric_name) or 0) < minimum_usable_views:
            failed.append(gate_name)
    if "prismatic" in joint_types:
        for name, gate_name in (
            ("prismatic_orthogonal_residual", "maximum_prismatic_orthogonal_residual"),
            (
                "prismatic_rotation_leakage_degrees",
                "maximum_prismatic_rotation_degrees",
            ),
        ):
            value = metrics.get(name)
            if value is None or not np.isfinite(value):
                failed.append(f"{gate_name}_unavailable")
            elif value > float(gates[gate_name]):
                failed.append(gate_name)
    if joint_types & {"revolute", "continuous_candidate"}:
        for name, gate_name in (
            ("axis_error_degrees", "maximum_revolute_axis_error_degrees"),
            (
                "pivot_residual_part_diagonals",
                "maximum_revolute_pivot_residual_part_diagonals",
            ),
        ):
            value = metrics.get(name)
            if value is None or not np.isfinite(value):
                failed.append(f"{gate_name}_unavailable")
            elif value > float(gates[gate_name]):
                failed.append(gate_name)
    return failed


def _deterministic_global_q_search(
    objective: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    grid_sample_count: int = 401,
) -> dict[str, object]:
    if grid_sample_count < 401:
        raise ValueError("held-out q global search requires at least 401 grid samples")
    if not np.isfinite((lower, upper)).all() or upper <= lower:
        raise ValueError("held-out q bounds must be finite and ordered")
    from scipy.optimize import minimize_scalar

    legacy = minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"maxiter": 40, "xatol": 1e-5},
    )
    legacy_success = bool(legacy.success and np.isfinite(legacy.x) and np.isfinite(legacy.fun))
    q_values = np.linspace(lower, upper, grid_sample_count, dtype=np.float64)
    objective_values = np.asarray(
        [objective(float(value)) for value in q_values],
        dtype=np.float64,
    )
    if not np.isfinite(objective_values).all():
        raise RuntimeError("held-out q objective grid contains non-finite values")
    grid_index = int(np.argmin(objective_values))
    local_indices = [
        index
        for index, value in enumerate(objective_values)
        if (index == 0 or value <= objective_values[index - 1])
        and (index == len(objective_values) - 1 or value <= objective_values[index + 1])
    ]
    local_minima: list[dict[str, object]] = []
    for index in local_indices:
        if index in {0, len(q_values) - 1}:
            local_minima.append(
                {
                    "q": float(q_values[index]),
                    "total_objective": float(objective_values[index]),
                    "source": "grid_endpoint",
                }
            )
            continue
        refined = minimize_scalar(
            objective,
            bounds=(float(q_values[index - 1]), float(q_values[index + 1])),
            method="bounded",
            options={"maxiter": 40, "xatol": 1e-7},
        )
        if refined.success and np.isfinite((refined.x, refined.fun)).all():
            local_minima.append(
                {
                    "q": float(refined.x),
                    "total_objective": float(refined.fun),
                    "source": "locally_refined",
                }
            )
        else:
            local_minima.append(
                {
                    "q": float(q_values[index]),
                    "total_objective": float(objective_values[index]),
                    "source": "grid_endpoint",
                }
            )
    if not local_minima:
        raise RuntimeError("held-out q global search found no finite local minimum")
    local_minima.sort(
        key=lambda item: (
            float(item["total_objective"]),
            float(item["q"]),
        )
    )
    selected = local_minima[0]
    selected_objective = float(selected["total_objective"])
    verification_tolerance = 1e-6 * max(1.0, abs(selected_objective))
    legacy_global = bool(
        legacy_success and float(legacy.fun) <= selected_objective + verification_tolerance
    )
    deterministic_global = bool(
        selected_objective <= float(objective_values[grid_index]) + verification_tolerance
    )
    samples = [
        {
            "q": float(q_value),
            "total_objective": float(objective_value),
        }
        for q_value, objective_value in zip(q_values, objective_values, strict=True)
    ]
    return {
        "grid_sample_count": grid_sample_count,
        "samples": samples,
        "legacy_optimizer_success": legacy_success,
        "legacy_optimizer_q": float(legacy.x) if legacy_success else None,
        "legacy_optimizer_objective": float(legacy.fun) if legacy_success else None,
        "legacy_optimizer_matches_global_minimum": legacy_global,
        "grid_global_minimum_q": float(q_values[grid_index]),
        "grid_global_minimum_objective": float(objective_values[grid_index]),
        "refined_global_minimum_q": float(selected["q"]),
        "refined_global_minimum_objective": selected_objective,
        "all_local_minima": sorted(
            local_minima,
            key=lambda item: float(item["q"]),
        ),
        "optimizer_global_minimum_verified": deterministic_global,
    }


def _semantic_q_ordering(
    *,
    fitted_joint: dict[str, object],
    heldout_state_id: str,
    heldout_q: float,
    semantic_state_labels: dict[str, str],
    local_minima: list[dict[str, object]],
) -> dict[str, object]:
    candidate_q_by_state = {
        str(state_id): float(value) for state_id, value in fitted_joint["fitting_state_q"].items()
    }
    candidate_q_by_state[heldout_state_id] = heldout_q
    q_scale = float(fitted_joint.get("q_scale", 1.0))
    q_offset = float(fitted_joint.get("q_offset", 0.0))
    measured_q_by_state = (
        {state_id: (value - q_offset) / q_scale for state_id, value in candidate_q_by_state.items()}
        if abs(q_scale) > 1e-12
        else {}
    )
    state_for_role: dict[str, str] = {}
    for state_id, label in semantic_state_labels.items():
        normalized = label.lower().replace("-", "_").replace(" ", "_")
        if normalized in {"closed", "half_open", "open"}:
            state_for_role[normalized] = state_id
    ordered_ids = [
        state_for_role[role] for role in ("closed", "half_open", "open") if role in state_for_role
    ]
    direction = "unavailable"
    ordering_consistent: bool | None = None
    if len(ordered_ids) == 3 and all(state_id in measured_q_by_state for state_id in ordered_ids):
        values = [measured_q_by_state[state_id] for state_id in ordered_ids]
        first_delta, second_delta = values[1] - values[0], values[2] - values[1]
        tolerance = 1e-8 * max(1.0, *(abs(value) for value in values))
        if first_delta > tolerance and second_delta > tolerance:
            direction = "increasing"
            ordering_consistent = True
        elif first_delta < -tolerance and second_delta < -tolerance:
            direction = "decreasing"
            ordering_consistent = True
        else:
            direction = "inconsistent"
            ordering_consistent = False
    sorted_states = sorted(
        measured_q_by_state,
        key=lambda state_id: (measured_q_by_state[state_id], state_id),
    )
    objective_gap = None
    minima_by_objective = sorted(float(item["total_objective"]) for item in local_minima)
    if len(minima_by_objective) >= 2:
        objective_gap = max(0.0, minima_by_objective[1] - minima_by_objective[0])
    return {
        "expected_semantic_ordering": (
            "closed -> half_open -> open, monotonic in either canonical sign"
        ),
        "candidate_q_by_state": candidate_q_by_state,
        "measured_q_by_state": measured_q_by_state,
        "observed_ordering": sorted_states,
        "direction": direction,
        "ordering_consistent": ordering_consistent,
        "objective_gap_to_second_minimum": objective_gap,
    }


def _heldout_joint_position_with_audit(
    *,
    input_root: Path,
    link: dict[str, object],
    joint: dict[str, object],
    fitted_joint: dict[str, object],
    measured_joint: dict[str, object] | None,
    measured_part: dict[str, object] | None,
    base_matrix: np.ndarray,
    reference_from_state: np.ndarray,
    heldout_state_id: str,
    semantic_state_labels: dict[str, str],
) -> tuple[float | None, float | None, dict[str, object] | None]:
    if str(joint["joint_type"]) == "fixed":
        return 0.0, 0.0, None
    if measured_part is None:
        return None, None, None
    visual_paths = link["visual_asset_paths"]
    if not isinstance(visual_paths, list) or not visual_paths:
        return None, None, None
    visual_path = str(visual_paths[0])
    candidate_points = transform(
        load_points(str(input_root / visual_path)),
        _visual_asset_to_candidate_matrix(input_root, link, visual_path),
    )
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
        return None, None, None
    lower_value = joint.get("candidate_limit_lower")
    upper_value = joint.get("candidate_limit_upper")
    if (
        lower_value is not None
        and upper_value is not None
        and str(joint.get("limit_source")) == "candidate_prior"
    ):
        lower, upper = float(lower_value), float(upper_value)
    else:
        observed = list(fitted_joint["fitting_state_q"].values())
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
        return None, None, None
    axis = np.asarray(fitted_joint["fitted_axis"], dtype=np.float64)
    pivot = (
        np.asarray(fitted_joint["fitted_pivot"], dtype=np.float64)
        if fitted_joint["fitted_pivot"] is not None
        else None
    )
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

    search = _deterministic_global_q_search(
        objective,
        lower,
        upper,
        grid_sample_count=401,
    )
    selected_q = float(search["refined_global_minimum_q"])
    selected_objective = float(search["refined_global_minimum_objective"])
    samples = [
        {
            "q": float(item["q"]),
            "measured_point_to_link_residual": float(float(item["total_objective"]) * diagonal),
            "mask_loss": None,
            "depth_residual": None,
            "negative_space_penalty": None,
            "front_of_scene_penalty": None,
            "total_objective": float(item["total_objective"]),
            "usable_view_count": None,
        }
        for item in search["samples"]
    ]
    semantic_ordering = _semantic_q_ordering(
        fitted_joint=fitted_joint,
        heldout_state_id=heldout_state_id,
        heldout_q=selected_q,
        semantic_state_labels=semantic_state_labels,
        local_minima=search["all_local_minima"],
    )
    objective_values = sorted(float(item["total_objective"]) for item in search["all_local_minima"])
    ambiguous = bool(
        len(objective_values) >= 2
        and objective_values[1] - objective_values[0] <= 1e-4 * max(1.0, objective_values[0])
    )
    if not bool(search["legacy_optimizer_matches_global_minimum"]):
        classification = "legacy_optimizer_failure"
    elif ambiguous:
        classification = "symmetric_or_multimodal_ambiguity"
    elif semantic_ordering["ordering_consistent"] is False:
        classification = "heldout_motion_inconsistent"
    else:
        classification = "global_minimum_verified"
    diagnostics = [
        "q objective uses fitting-frozen candidate geometry and held-out partial measured points",
        "mask, depth, negative-space, and front-of-scene metrics are evaluated after q freezes",
    ]
    if semantic_ordering["ordering_consistent"] is False:
        diagnostics.append("semantic closed/half_open/open q ordering is inconsistent")
    if selected_objective > 0.05:
        diagnostics.append("candidate geometry has a material held-out measured-surface residual")
    audit = {
        "state_id": heldout_state_id,
        "joint_id": str(joint["joint_id"]),
        "joint_type": str(joint["joint_type"]),
        "lower_bound": lower,
        "upper_bound": upper,
        "candidate_limit_source": (
            str(joint.get("limit_source")) if joint.get("limit_source") is not None else None
        ),
        "grid_sample_count": search["grid_sample_count"],
        "legacy_optimizer_success": search["legacy_optimizer_success"],
        "legacy_optimizer_q": search["legacy_optimizer_q"],
        "legacy_optimizer_objective": search["legacy_optimizer_objective"],
        "legacy_optimizer_matches_global_minimum": search[
            "legacy_optimizer_matches_global_minimum"
        ],
        "grid_global_minimum_q": search["grid_global_minimum_q"],
        "grid_global_minimum_objective": search["grid_global_minimum_objective"],
        "refined_global_minimum_q": selected_q,
        "refined_global_minimum_objective": selected_objective,
        "selected_q": selected_q,
        "selected_residual_arbitrary_units": selected_objective * diagonal,
        "all_local_minima": search["all_local_minima"],
        "fitting_state_q": {
            str(state_id): float(value)
            for state_id, value in fitted_joint["fitting_state_q"].items()
        },
        "component_availability": {
            "measured_point_to_link_residual": True,
            "mask_loss": False,
            "depth_residual": False,
            "negative_space_penalty": False,
            "front_of_scene_penalty": False,
            "usable_view_count": False,
        },
        "samples": samples,
        "optimizer_global_minimum_verified": search["optimizer_global_minimum_verified"],
        "classification": classification,
        "inconsistency_diagnostics": diagnostics,
        "semantic_ordering": semantic_ordering,
    }
    return selected_q, selected_objective * diagonal, audit


def _heldout_joint_position(
    *,
    input_root: Path,
    link: dict[str, object],
    joint: dict[str, object],
    fitted_joint: dict[str, object],
    measured_joint: dict[str, object] | None,
    measured_part: dict[str, object] | None,
    base_matrix: np.ndarray,
    reference_from_state: np.ndarray,
) -> tuple[float | None, float | None]:
    position, residual, _ = _heldout_joint_position_with_audit(
        input_root=input_root,
        link=link,
        joint=joint,
        fitted_joint=fitted_joint,
        measured_joint=measured_joint,
        measured_part=measured_part,
        base_matrix=base_matrix,
        reference_from_state=reference_from_state,
        heldout_state_id="heldout",
        semantic_state_labels={},
    )
    return position, residual


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
    reference_state_id = str(request["reference_state_id"])
    accepted_alignment_state_ids = [str(value) for value in request["accepted_alignment_state_ids"]]
    evaluations = []
    diagnostic_renders: list[Path] = []
    for fit_item in fitting["fittings"]:
        candidate_id = str(fit_item["candidate_id"])
        if fit_item["status"] != "fitted":
            ambiguous = fit_item["status"] == "ambiguous"
            evaluations.append(
                {
                    "candidate_id": candidate_id,
                    "status": (
                        "ambiguous_link_assignment" if ambiguous else "rejected_joint_constraint"
                    ),
                    "fitting_sha256": request["fitting_manifest_sha256"],
                    "candidate_sha256": request["candidate_sha256_by_id"][candidate_id],
                    "fitted_model_sha256": request["fitted_model_sha256_by_id"][candidate_id],
                    "link_assignment_sha256": request["link_assignment_sha256_by_id"][candidate_id],
                    "heldout_evidence_sha256": request["heldout_evidence_sha256"],
                    "state_evaluations": [],
                    "passed_hard_gates": False,
                    "failed_gates": [
                        ("ambiguous_link_assignment" if ambiguous else "candidate_fitting_failed")
                    ],
                    "heldout_state_validation_used": False,
                    "capture_state_count": request["capture_state_count"],
                    "accepted_alignment_state_ids": accepted_alignment_state_ids,
                    "selected_candidate_validation_level": measured[
                        "effective_motion_evidence_level"
                    ],
                    "link_assignment_confidence": 0.0,
                    "heldout_q_objective_path": None,
                    "heldout_q_objective_sha256": None,
                    "heldout_q_objective_preview_path": None,
                    "heldout_q_objective_preview_sha256": None,
                    "runtime_seconds": 0.0,
                    "warnings": [str(fit_item.get("failure_reason") or "candidate not fitted")],
                }
            )
            continue
        candidate = candidate_by_id[candidate_id]
        fitted_model = fit_item["fitted_model"]
        if not isinstance(fitted_model, dict):
            raise ValueError("successful fitting omitted typed fitted kinematic model")
        state_evaluations = []
        candidate_q_audits: list[dict[str, object]] = []
        base_matrix = np.asarray(
            fitted_model["matrix_reference_world_from_candidate_base"],
            dtype=np.float64,
        ).reshape(4, 4)
        assignment = assignments_by_candidate[candidate_id]
        observed_by_link = {
            str(link_id): str(record["observed_part_id"])
            for record in assignment["assignments"]
            for link_id in record["candidate_link_ids"]
        }
        assignment_records = {
            str(record["observed_part_id"]): record for record in assignment["assignments"]
        }
        joint_by_child = {str(item["child_link_id"]): item for item in candidate["joints"]}
        fitted_joint_by_candidate = {
            str(item["candidate_joint_id"]): item for item in fitted_model["fitted_joints"]
        }
        candidate_link_by_id = {str(item["link_id"]): item for item in candidate["links"]}
        base_observed_part = next(
            (str(item["parent_observed_part_id"]) for item in fitted_model["fitted_joints"]),
            next(iter(assignment_records), ""),
        )
        base_assignment = assignment_records.get(base_observed_part)
        base_candidate_link = (
            str(base_assignment["candidate_link_ids"][0])
            if base_assignment is not None and base_assignment["candidate_link_ids"]
            else ""
        )
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
            heldout_q_fit_residuals: dict[str, float] = {}
            link_matrices: dict[str, np.ndarray] = {}
            for link in candidate["links"]:
                link_id = str(link["link_id"])
                link_matrix = base_matrix
                joint = joint_by_child.get(link_id)
                if joint is not None:
                    fitted_joint = fitted_joint_by_candidate.get(str(joint["joint_id"]))
                    if fitted_joint is None:
                        continue
                    observed_part = observed_by_link.get(link_id, "")
                    position, q_fit_residual, q_audit = _heldout_joint_position_with_audit(
                        input_root=input_root,
                        link=link,
                        joint=joint,
                        fitted_joint=fitted_joint,
                        measured_joint=measured_joint_by_child.get(observed_part),
                        measured_part=geometry_by_state.get((state_id, observed_part)),
                        base_matrix=base_matrix,
                        reference_from_state=alignment_by_state[state_id],
                        heldout_state_id=state_id,
                        semantic_state_labels={
                            str(key): str(value)
                            for key, value in request["semantic_state_labels"].items()
                        },
                    )
                    if position is None:
                        continue
                    if q_audit is not None:
                        candidate_q_audits.append(q_audit)
                    inferred[str(joint["joint_id"])] = position
                    if q_fit_residual is not None:
                        heldout_q_fit_residuals[str(joint["joint_id"])] = q_fit_residual
                    axis = np.asarray(fitted_joint["fitted_axis"], dtype=np.float64)
                    pivot = (
                        np.asarray(fitted_joint["fitted_pivot"], dtype=np.float64)
                        if fitted_joint["fitted_pivot"] is not None
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
            view_evaluations: list[dict[str, object]] = []
            requested_views = [
                str(value) for value in request["heldout_views_by_state"].get(state_id, [])
            ]
            usable_views = rendered_views = target_views = valid_depth_views = 0
            required_link_ids = sorted(observed_by_link)
            for frame_id in requested_views:
                view_record: dict[str, object] = {
                    "frame_id": frame_id,
                    "camera_reconstruction_sha256": evidence["camera_reconstruction_sha256"],
                    "depth_path": None,
                    "depth_sha256": None,
                    "valid_depth": False,
                    "target_mask_paths": {},
                    "target_mask_hashes": {},
                    "target_masks_complete": False,
                    "required_link_ids": required_link_ids,
                    "rendered_link_ids": [],
                    "missing_link_ids": required_link_ids,
                    "usable": False,
                    "failure_reasons": [],
                    "render_path": None,
                    "render_sha256": None,
                    "raw_candidate_pixel_count": 0,
                    "visible_candidate_pixel_count": 0,
                    "target_mask_pixel_count": 0,
                }
                if (
                    frame_id not in world_from_camera
                    or frame_id not in undistortion
                    or frame_id not in depth_by_frame
                ):
                    view_record["failure_reasons"] = ["camera_or_dense_record_unavailable"]
                    view_evaluations.append(view_record)
                    continue
                record = undistortion[frame_id]
                depth_path = str(depth_by_frame[frame_id]["depth_path"])
                view_record["depth_path"] = depth_path
                view_record["depth_sha256"] = sha256(input_root / depth_path)
                scene_depth = read_dense_array(
                    input_root / depth_path,
                    1,
                )
                if not np.any(np.isfinite(scene_depth) & (scene_depth > 0)):
                    view_record["failure_reasons"] = ["valid_dense_depth_unavailable"]
                    view_evaluations.append(view_record)
                    continue
                valid_depth_views += 1
                view_record["valid_depth"] = True
                intrinsics = tuple(float(value) for value in record["dense_intrinsics"])
                dimensions = tuple(int(value) for value in record["dense_dimensions"])
                camera_from_reference = (
                    np.linalg.inv(world_from_camera[frame_id]) @ state_from_reference
                )
                link_depths: dict[str, np.ndarray] = {}
                part_masks: dict[str, np.ndarray] = {}
                target_mask_paths: dict[str, str] = {}
                for part_id, paths in evidence["part_mask_paths"].items():
                    mask_path = paths.get(frame_id)
                    if mask_path is not None:
                        target_mask_paths[str(part_id)] = str(mask_path)
                        part_masks[str(part_id)] = undistort_mask(
                            input_root / str(mask_path),
                            record,
                        )
                view_record["target_mask_paths"] = target_mask_paths
                view_record["target_mask_hashes"] = {
                    part_id: sha256(input_root / path)
                    for part_id, path in target_mask_paths.items()
                }
                if not part_masks or not any(np.any(mask) for mask in part_masks.values()):
                    view_record["failure_reasons"] = ["target_part_mask_unavailable"]
                    view_evaluations.append(view_record)
                    continue
                missing_target_parts = sorted(
                    {
                        observed_by_link[link_id]
                        for link_id in required_link_ids
                        if observed_by_link[link_id] not in part_masks
                        or not np.any(part_masks[observed_by_link[link_id]])
                    }
                )
                if missing_target_parts:
                    view_record["failure_reasons"] = [
                        "required_target_part_masks_unavailable:" + ",".join(missing_target_parts)
                    ]
                    view_evaluations.append(view_record)
                    continue
                view_record["target_masks_complete"] = True
                target_views += 1
                for link in candidate["links"]:
                    link_id = str(link["link_id"])
                    if link_id not in required_link_ids or link_id not in link_matrices:
                        continue
                    visual_paths = link["visual_asset_paths"]
                    if not visual_paths:
                        continue
                    link_depths[link_id] = render_mesh_depth(
                        input_root / str(visual_paths[0]),
                        link_matrices[link_id]
                        @ _visual_asset_to_candidate_matrix(
                            input_root,
                            link,
                            str(visual_paths[0]),
                        ),
                        camera_from_reference,
                        intrinsics,
                        dimensions,
                    )
                rendered_link_ids = sorted(
                    link_id
                    for link_id, value in link_depths.items()
                    if np.any(np.isfinite(value) & (value > 0))
                )
                missing_link_ids = sorted(set(required_link_ids) - set(rendered_link_ids))
                view_record["rendered_link_ids"] = rendered_link_ids
                view_record["missing_link_ids"] = missing_link_ids
                if missing_link_ids:
                    view_record["failure_reasons"] = [
                        "required_candidate_links_not_rendered:" + ",".join(missing_link_ids)
                    ]
                    view_evaluations.append(view_record)
                    continue
                rendered_views += 1
                usable_views += 1
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
                view_record.update(
                    {
                        "usable": True,
                        "failure_reasons": [],
                        "render_path": render_paths[frame_id],
                        "render_sha256": sha256(render_path),
                        "raw_candidate_pixel_count": int(
                            np.count_nonzero(np.isfinite(whole_depth) & (whole_depth > 0))
                        ),
                        "visible_candidate_pixel_count": int(
                            np.count_nonzero(whole_classification["visible"])
                        ),
                        "target_mask_pixel_count": int(np.count_nonzero(whole_target)),
                    }
                )
                view_evaluations.append(view_record)
                diagnostic_renders.append(render_path)
            base_state_geometry = geometry_by_state.get((state_id, base_observed_part))
            reference_base_geometry = geometry_by_state.get(
                (reference_state_id, base_observed_part)
            )
            base_link = candidate_link_by_id.get(base_candidate_link)
            base_point_residual_raw: float | None = None
            base_motion_raw: float | None = None
            scene_diagonal: float | None = None
            if (
                base_state_geometry is not None
                and reference_base_geometry is not None
                and base_link is not None
                and base_link["visual_asset_paths"]
            ):
                heldout_base_points = transform(
                    load_points(
                        str(input_root / str(base_state_geometry["measured_point_cloud_path"]))
                    ),
                    alignment_by_state[state_id],
                )
                reference_base_points = transform(
                    load_points(
                        str(input_root / str(reference_base_geometry["measured_point_cloud_path"]))
                    ),
                    alignment_by_state[reference_state_id],
                )
                candidate_base_points = transform(
                    load_points(str(input_root / str(base_link["visual_asset_paths"][0]))),
                    base_matrix
                    @ _visual_asset_to_candidate_matrix(
                        input_root,
                        base_link,
                        str(base_link["visual_asset_paths"][0]),
                    ),
                )
                scene_diagonal = _diagonal(reference_base_points)
                base_point_residual_raw = _trimmed_surface_residual(
                    candidate_base_points,
                    heldout_base_points,
                )
                base_motion_raw = float(
                    np.linalg.norm(
                        np.median(heldout_base_points, axis=0)
                        - np.median(reference_base_points, axis=0)
                    )
                )
            movable_residuals_raw: list[float] = []
            prismatic_orthogonal: list[float] = []
            prismatic_rotation: list[float] = []
            q_residuals: list[float] = []
            axis_errors: list[float] = []
            pivot_residuals: list[float] = []
            part_diagonals: list[float] = []
            base_scale = float(np.cbrt(np.linalg.det(base_matrix[:3, :3])))
            base_rotation = base_matrix[:3, :3] / base_scale
            for fitted_joint in fitted_model["fitted_joints"]:
                candidate_joint_id = str(fitted_joint["candidate_joint_id"])
                candidate_joint = next(
                    item
                    for item in candidate["joints"]
                    if str(item["joint_id"]) == candidate_joint_id
                )
                child_part = str(fitted_joint["child_observed_part_id"])
                child_link_id = str(candidate_joint["child_link_id"])
                child_geometry = geometry_by_state.get((state_id, child_part))
                reference_child_geometry = geometry_by_state.get((reference_state_id, child_part))
                link = candidate_link_by_id.get(child_link_id)
                q_value = inferred.get(candidate_joint_id)
                if (
                    child_geometry is None
                    or reference_child_geometry is None
                    or link is None
                    or not link["visual_asset_paths"]
                    or q_value is None
                ):
                    continue
                heldout_points = transform(
                    load_points(str(input_root / str(child_geometry["measured_point_cloud_path"]))),
                    alignment_by_state[state_id],
                )
                reference_points = transform(
                    load_points(
                        str(input_root / str(reference_child_geometry["measured_point_cloud_path"]))
                    ),
                    alignment_by_state[reference_state_id],
                )
                part_diagonal = _diagonal(reference_points)
                part_diagonals.append(part_diagonal)
                link_points = transform(
                    load_points(str(input_root / str(link["visual_asset_paths"][0]))),
                    link_matrices[child_link_id]
                    @ _visual_asset_to_candidate_matrix(
                        input_root,
                        link,
                        str(link["visual_asset_paths"][0]),
                    ),
                )
                raw_residual = _trimmed_surface_residual(link_points, heldout_points)
                movable_residuals_raw.append(raw_residual)
                fitted_axis_world = _normalize(
                    base_rotation @ np.asarray(fitted_joint["fitted_axis"], dtype=np.float64)
                )
                displacement = np.median(heldout_points, axis=0) - np.median(
                    reference_points, axis=0
                )
                measured_q = float(np.dot(displacement, fitted_axis_world))
                q_scale = float(fitted_joint["q_scale"])
                q_offset = float(fitted_joint["q_offset"])
                inferred_measured_q = (
                    (q_value - q_offset) / q_scale if abs(q_scale) > 1e-12 else 0.0
                )
                joint_type = str(fitted_joint["joint_type"])
                if joint_type == "prismatic":
                    orthogonal = displacement - measured_q * fitted_axis_world
                    prismatic_orthogonal.append(float(np.linalg.norm(orthogonal) / part_diagonal))
                    relative_fit, _ = register_sim3(
                        reference_points,
                        heldout_points,
                    )
                    relative_scale = float(np.cbrt(np.linalg.det(relative_fit[:3, :3])))
                    relative_rotation = relative_fit[:3, :3] / relative_scale
                    rotation_angle = float(
                        np.degrees(
                            np.arccos(
                                np.clip(
                                    (np.trace(relative_rotation) - 1.0) / 2.0,
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                    )
                    prismatic_rotation.append(rotation_angle)
                    q_residuals.append(abs(inferred_measured_q - measured_q))
                elif joint_type in {"revolute", "continuous_candidate"}:
                    relative_fit, _ = register_sim3(
                        reference_points,
                        heldout_points,
                    )
                    relative_scale = float(np.cbrt(np.linalg.det(relative_fit[:3, :3])))
                    relative_rotation = relative_fit[:3, :3] / relative_scale
                    heldout_axis, heldout_angle = _rotation_axis_and_signed_angle(
                        relative_rotation,
                        fitted_axis_world,
                    )
                    if heldout_axis is not None:
                        axis_errors.append(_axis_angle_degrees(fitted_axis_world, heldout_axis))
                    if fitted_joint["fitted_pivot"] is not None:
                        fitted_pivot_world = transform(
                            np.asarray(fitted_joint["fitted_pivot"], dtype=np.float64)[None, :],
                            base_matrix,
                        )[0]
                        rigid_translation = np.median(heldout_points, axis=0) - (
                            relative_rotation @ np.median(reference_points, axis=0)
                        )
                        pivot_after_observed_motion = (
                            relative_rotation @ fitted_pivot_world + rigid_translation
                        )
                        pivot_residuals.append(
                            float(
                                np.linalg.norm(pivot_after_observed_motion - fitted_pivot_world)
                                / part_diagonal
                            )
                        )
                    q_residuals.append(abs(inferred_measured_q - heldout_angle))
            base_depth_values = link_depth_residuals.get(base_candidate_link, [])
            metrics: dict[str, float | None] = {
                "base_mask_iou": float(np.mean(base_ious)) if base_ious else None,
                "movable_part_mask_iou": (float(np.mean(movable_ious)) if movable_ious else None),
                "whole_object_mask_iou": (float(np.mean(whole_ious)) if whole_ious else None),
                "depth_inlier_fraction": (float(np.mean(depth_inliers)) if depth_inliers else None),
                "negative_space_violation_ratio": (
                    negative_pixels / rendered_pixels if rendered_pixels else None
                ),
                "front_of_scene_violation_ratio": (
                    front_pixels / rendered_pixels if rendered_pixels else None
                ),
                "base_motion_scene_diagonals": (
                    base_motion_raw / scene_diagonal
                    if base_motion_raw is not None and scene_diagonal is not None
                    else None
                ),
                "usable_heldout_view_count": float(usable_views),
                "prismatic_orthogonal_residual": (
                    float(np.median(prismatic_orthogonal)) if prismatic_orthogonal else None
                ),
                "prismatic_rotation_leakage_degrees": (
                    float(np.median(prismatic_rotation)) if prismatic_rotation else None
                ),
                "axis_error_degrees": (float(np.median(axis_errors)) if axis_errors else None),
                "pivot_residual_part_diagonals": (
                    float(np.median(pivot_residuals)) if pivot_residuals else None
                ),
            }
            state_evaluations.append(
                {
                    "state_id": state_id,
                    "heldout": True,
                    "requested_heldout_view_count": len(requested_views),
                    "usable_heldout_view_count": usable_views,
                    "rendered_heldout_view_count": rendered_views,
                    "views_with_target_masks": target_views,
                    "views_with_valid_depth": valid_depth_views,
                    "base_mask_iou": metrics["base_mask_iou"],
                    "movable_part_mask_iou": metrics["movable_part_mask_iou"],
                    "whole_object_mask_iou": metrics["whole_object_mask_iou"],
                    "per_link_depth_residual": {
                        link_id: (float(np.median(values)) if values else None)
                        for link_id, values in link_depth_residuals.items()
                    },
                    "base_depth_residual": (
                        float(np.median(base_depth_values)) if base_depth_values else None
                    ),
                    "depth_inlier_fraction": metrics["depth_inlier_fraction"],
                    "negative_space_violation_ratio": metrics["negative_space_violation_ratio"],
                    "front_of_scene_violation_ratio": metrics["front_of_scene_violation_ratio"],
                    "scene_diagonal_arbitrary_units": scene_diagonal,
                    "base_point_residual_arbitrary_units": base_point_residual_raw,
                    "base_point_residual_scene_diagonals": (
                        base_point_residual_raw / scene_diagonal
                        if base_point_residual_raw is not None and scene_diagonal is not None
                        else None
                    ),
                    "base_motion_arbitrary_units": base_motion_raw,
                    "base_motion_scene_diagonals": metrics["base_motion_scene_diagonals"],
                    "movable_point_residual_arbitrary_units": (
                        float(np.median(movable_residuals_raw)) if movable_residuals_raw else None
                    ),
                    "joint_constraint_residual": (
                        float(np.median(list(heldout_q_fit_residuals.values())))
                        / float(np.median(part_diagonals))
                        if heldout_q_fit_residuals and part_diagonals
                        else None
                    ),
                    "prismatic_orthogonal_residual": metrics["prismatic_orthogonal_residual"],
                    "prismatic_rotation_leakage_degrees": metrics[
                        "prismatic_rotation_leakage_degrees"
                    ],
                    "joint_q_residual": (float(np.median(q_residuals)) if q_residuals else None),
                    "axis_error_degrees": metrics["axis_error_degrees"],
                    "pivot_residual_part_diagonals": metrics["pivot_residual_part_diagonals"],
                    "inferred_joint_positions": inferred,
                    "joint_position_source": "measured_geometry",
                    "render_paths": render_paths,
                    "view_evaluations": view_evaluations,
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
            "prismatic_orthogonal_residual",
            "prismatic_rotation_leakage_degrees",
            "axis_error_degrees",
            "pivot_residual_part_diagonals",
        )
        aggregate = {
            name: (
                float(
                    np.mean(
                        [float(item[name]) for item in state_evaluations if item[name] is not None]
                    )
                )
                if any(item[name] is not None for item in state_evaluations)
                else None
            )
            for name in metric_names
        }
        aggregate["usable_heldout_view_count"] = (
            float(min(item["usable_heldout_view_count"] for item in state_evaluations))
            if state_evaluations
            else 0.0
        )
        for count_name in (
            "rendered_heldout_view_count",
            "views_with_target_masks",
            "views_with_valid_depth",
        ):
            aggregate[count_name] = (
                float(min(item[count_name] for item in state_evaluations))
                if state_evaluations
                else 0.0
            )
        joint_types = {str(item["joint_type"]) for item in fitted_model["fitted_joints"]}
        failed = _failed_gates(
            aggregate,
            gates,
            joint_types=joint_types,
            minimum_usable_views=int(gates["minimum_usable_heldout_views"]),
        )
        evaluated_state_ids = {str(item["state_id"]) for item in state_evaluations}
        distinct_valid_states = (
            set(str(value) for value in request["generation_state_ids"])
            | set(str(value) for value in request["fitting_state_ids"])
            | evaluated_state_ids
        ) & set(accepted_alignment_state_ids)
        if len(distinct_valid_states) < int(gates["minimum_valid_states"]):
            failed.append("minimum_valid_states")
        if len(evaluated_state_ids) < int(gates["minimum_heldout_states"]):
            failed.append("minimum_heldout_states")
        failed = sorted(set(failed))
        heldout_ran = bool(state_evaluations)
        validation_level = (
            "multi_state_heldout_validated"
            if not failed and len(accepted_alignment_state_ids) >= 3 and heldout_ran
            else (
                "multi_state_heldout_available"
                if len(accepted_alignment_state_ids) >= 3 and heldout_ran
                else measured["effective_motion_evidence_level"]
            )
        )
        assignment_confidences = [
            float(record["assignment_confidence"]) for record in assignment["assignments"]
        ]
        safe_candidate_id = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in candidate_id
        )
        q_objective_relative = None
        q_objective_sha256 = None
        q_preview_relative = None
        q_preview_sha256 = None
        if candidate_q_audits:
            q_objective_relative = (
                f"reconstruction/articulation/raw/open_q_objective_{safe_candidate_id}.json"
            )
            q_preview_relative = (
                f"reconstruction/articulation/previews/open_q_objective_{safe_candidate_id}.png"
            )
            q_objective_path = output_dir / "raw" / f"open_q_objective_{safe_candidate_id}.json"
            q_preview_path = output_dir / "previews" / f"open_q_objective_{safe_candidate_id}.png"
            write_json(
                q_objective_path,
                {
                    "schema_version": "0.1.0",
                    "candidate_id": candidate_id,
                    "objective_definition": (
                        "trimmed_measured_point_to_candidate_link_distance_normalized_by_part_diagonal"
                    ),
                    "trim_fraction": 0.8,
                    "candidate_structure_frozen": True,
                    "grid_sample_count": 401,
                    "joint_audits": candidate_q_audits,
                },
            )
            write_q_objective_preview(
                q_preview_path,
                candidate_id,
                candidate_q_audits,
            )
            q_objective_sha256 = sha256(q_objective_path)
            q_preview_sha256 = sha256(q_preview_path)
        evaluations.append(
            {
                "candidate_id": candidate_id,
                "status": ("multi_state_validated" if not failed else "rejected_heldout_state"),
                "fitting_sha256": request["fitting_manifest_sha256"],
                "candidate_sha256": request["candidate_sha256_by_id"][candidate_id],
                "fitted_model_sha256": request["fitted_model_sha256_by_id"][candidate_id],
                "link_assignment_sha256": request["link_assignment_sha256_by_id"][candidate_id],
                "heldout_evidence_sha256": request["heldout_evidence_sha256"],
                "state_evaluations": state_evaluations,
                "passed_hard_gates": not failed,
                "failed_gates": failed,
                "heldout_state_validation_used": bool(state_evaluations),
                "capture_state_count": request["capture_state_count"],
                "accepted_alignment_state_ids": accepted_alignment_state_ids,
                "selected_candidate_validation_level": validation_level,
                "link_assignment_confidence": (
                    min(assignment_confidences) if assignment_confidences else 0.0
                ),
                "heldout_q_objective_path": q_objective_relative,
                "heldout_q_objective_sha256": q_objective_sha256,
                "heldout_q_objective_preview_path": q_preview_relative,
                "heldout_q_objective_preview_sha256": q_preview_sha256,
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
            "request_sha256": request["request_sha256"],
            "fitting_manifest_sha256": request["fitting_manifest_sha256"],
            "link_assignments_sha256": request["link_assignments_sha256"],
            "candidate_manifest_sha256": request["candidate_manifest_sha256"],
            "evidence_split_sha256": request["evidence_split_sha256"],
            "measured_states_manifest_sha256": request["measured_states_manifest_sha256"],
            "state_alignment_sha256": request["state_alignment_sha256"],
            "measured_motion_sha256": request["measured_motion_sha256"],
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
    request_path = Path(args.request).resolve()
    request = read_json(request_path)
    request["request_sha256"] = sha256(request_path)
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.action == "fit":
        fit(request, input_root, output_dir)
    else:
        evaluate(request, input_root, output_dir)
    return 0
