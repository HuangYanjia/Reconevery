from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

IDENTITY = [
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_preview(path: Path, title: str) -> None:
    image = Image.new("RGB", (640, 360), (242, 244, 247))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 640, 54), fill=(24, 32, 40))
    draw.text((18, 18), title, fill=(255, 255, 255))
    draw.line((80, 260, 560, 100), fill=(35, 116, 165), width=5)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)


def write_mesh(path: Path, offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "ply\nformat ascii 1.0\n"
        "element vertex 8\nproperty float x\nproperty float y\nproperty float z\n"
        "element face 12\nproperty list uchar int vertex_indices\nend_header\n"
        f"{offset} 0 0\n{offset + 1} 0 0\n{offset + 1} 1 0\n{offset} 1 0\n"
        f"{offset} 0 1\n{offset + 1} 0 1\n{offset + 1} 1 1\n{offset} 1 1\n"
        "3 0 1 2\n3 0 2 3\n3 4 6 5\n3 4 7 6\n"
        "3 0 4 5\n3 0 5 1\n3 1 5 6\n3 1 6 2\n"
        "3 2 6 7\n3 2 7 3\n3 3 7 4\n3 3 4 0\n",
        encoding="ascii",
    )


def maybe_fail(request: dict[str, object]) -> None:
    mode = request.get("fake_mode", "success")
    if mode == "timeout":
        time.sleep(60)
    if mode == "oom":
        print("CUDA out of memory", file=sys.stderr)
        raise SystemExit(9)
    if mode == "backend_failure":
        print("fake articulated backend failure", file=sys.stderr)
        raise SystemExit(7)


def align(request: dict[str, object], input_root: Path, output_dir: Path) -> None:
    maybe_fail(request)
    capture = json.loads(
        (input_root / str(request["capture_manifest_path"])).read_text(encoding="utf-8")
    )
    mode = request.get("fake_mode", "success")
    transforms = []
    for state in capture["states"]:
        reference = state["state_id"] == capture["reference_state_id"]
        accepted = mode != "state_alignment_failure" or reference
        transforms.append(
            {
                "state_id": state["state_id"],
                "matrix_reference_from_state": IDENTITY,
                "inverse_matrix": IDENTITY,
                "scale": 1.0,
                "rotation_determinant": 1.0,
                "translation": [0.0, 0.0, 0.0],
                "fitting_median_residual_scene_diagonal": (0.0 if reference else 0.004),
                "fitting_p90_residual_scene_diagonal": 0.009,
                "heldout_static_depth_inlier_fraction": 0.92,
                "static_correspondence_count": 800,
                "excluded_movable_part_ids": request["movable_part_ids"],
                "accepted": accepted,
                "failure_reason": None if accepted else "fake state alignment failure",
            }
        )
    write_json(
        output_dir / "state_alignment.json",
        {
            "schema_version": "0.2.0",
            "capture_manifest_sha256": request["capture_manifest_sha256"],
            "reference_state_id": capture["reference_state_id"],
            "capture_state_count": len(capture["states"]),
            "accepted_alignment_state_ids": [
                item["state_id"] for item in transforms if item["accepted"]
            ],
            "aligned_state_count": sum(bool(item["accepted"]) for item in transforms),
            "transforms": transforms,
            "static_evidence_only": True,
            "source_states_unchanged": True,
            "runtime_seconds": 0.02,
            "peak_host_memory_bytes": 1024,
            "warnings": [],
        },
    )
    write_preview(output_dir / "previews/state_alignment.png", "Fake state alignment")


def estimate_motion(
    request: dict[str, object],
    input_root: Path,
    output_dir: Path,
) -> None:
    maybe_fail(request)
    capture = json.loads(
        (input_root / str(request["capture_manifest_path"])).read_text(encoding="utf-8")
    )
    geometry = json.loads(
        (input_root / str(request["measured_states_manifest_path"])).read_text(encoding="utf-8")
    )
    accepted = set(request["accepted_state_ids"])
    states_in_scope = [state for state in capture["states"] if state["state_id"] in accepted]
    state_count = len(states_in_scope)
    if state_count == 1:
        joint_type = "unknown"
        axis = None
        states = [
            {
                "state_id": states_in_scope[0]["state_id"],
                "position": 0.0,
                "part_registration_median_residual": 0.0,
                "part_coverage": 0.8,
                "supporting_point_count": 8,
                "state_confidence": 0.3,
            }
        ]
        position_min = None
        position_max = None
    else:
        joint_type = "prismatic"
        axis = [1.0, 0.0, 0.0]
        states = [
            {
                "state_id": state["state_id"],
                "position": index * 0.35,
                "part_registration_median_residual": 0.005,
                "part_coverage": 0.9,
                "supporting_point_count": 8,
                "state_confidence": 0.95,
            }
            for index, state in enumerate(states_in_scope)
        ]
        position_min = 0.0
        position_max = (state_count - 1) * 0.35
    geometries = []
    for item in geometry["geometries"]:
        copied = dict(item)
        copied["state_alignment_sha256"] = request["state_alignment_sha256"]
        copied["transformed_to_reference_frame"] = True
        geometries.append(copied)
    write_json(
        output_dir / "measured_motion.json",
        {
            "schema_version": "0.2.0",
            "capture_manifest_sha256": request["capture_manifest_sha256"],
            "state_alignment_sha256": request["state_alignment_sha256"],
            "articulated_object_id": request["articulated_object_id"],
            "reference_state_id": request["reference_state_id"],
            "capture_state_count": request["capture_state_count"],
            "accepted_alignment_state_ids": request["accepted_alignment_state_ids"],
            "effective_motion_evidence_level": (
                "two_state_motion_supported" if state_count >= 2 else "single_state_prior_only"
            ),
            "part_geometries": geometries,
            "joint_hypotheses": [
                {
                    "joint_id": "drawer_joint",
                    "parent_part_id": request["base_part_id"],
                    "child_part_id": request["movable_parts"][0]["part_id"],
                    "joint_type": joint_type,
                    "axis": axis,
                    "pivot": None,
                    "states": states,
                    "observed_position_min": position_min,
                    "observed_position_max": position_max,
                    "candidate_limit_lower": None,
                    "candidate_limit_upper": None,
                    "limit_source": "observed_range",
                    "orthogonal_residual": 0.005 if state_count > 1 else None,
                    "rotation_leakage_degrees": 0.2 if state_count > 1 else None,
                    "axis_consistency_degrees": None,
                    "normalization_part_diagonal": 1.0,
                    "fixed_translation_residual_arbitrary_units": (
                        0.0 if state_count > 1 else None
                    ),
                    "fixed_translation_residual_part_diagonals": (0.0 if state_count > 1 else None),
                    "pivot_residual_arbitrary_units": None,
                    "pivot_residual_part_diagonals": None,
                    "confidence": 0.95 if state_count > 1 else 0.3,
                    "warnings": [],
                }
            ],
            "base_link_fixed": True,
            "runtime_seconds": 0.03,
            "warnings": [],
        },
    )
    write_preview(output_dir / "previews/measured_part_motion.png", "Fake measured motion")
    write_preview(output_dir / "previews/joint_axis_and_pivot.png", "Fake joint axis")


def retrieve(request: dict[str, object], output_dir: Path) -> None:
    maybe_fail(request)
    family = str(request["source_family"])
    run_root = output_dir.parents[1]
    candidates = []
    for index in range(min(2, int(request["maximum_candidates"]))):
        candidate_id = f"{request['articulated_object_id']}__{family}__{index:02d}"
        candidate_root = output_dir / "candidates" / candidate_id
        base_path = candidate_root / "retrieved/base.ply"
        drawer_path = candidate_root / "retrieved/drawer.ply"
        write_mesh(base_path, 0.0)
        write_mesh(drawer_path, 0.2)
        base_relative = base_path.relative_to(run_root).as_posix()
        drawer_relative = drawer_path.relative_to(run_root).as_posix()
        bundle_path = candidate_root / "candidate.json"
        license_record = {
            "source_family": family,
            "code_license": "fake fixture",
            "checkpoint_license": "not_applicable",
            "dependency_licenses": {},
            "asset_license": "research fixture only",
            "training_data_notes": [],
            "commercial_review_status": "research_only",
            "research_evaluation_allowed": True,
            "production_selectable": False,
        }
        write_json(
            bundle_path,
            {
                "candidate_id": candidate_id,
                "articulated_object_id": request["articulated_object_id"],
                "source_family": family,
                "source_asset_id": f"{family}_asset_{index:03d}",
                "links": [
                    {
                        "link_id": "cabinet_body",
                        "name": "cabinet body",
                        "visual_asset_paths": [base_relative],
                        "visual_asset_hashes": {base_relative: sha256(base_path)},
                        "visual_asset_spaces": {base_relative: "candidate_base"},
                        "visual_asset_transforms_candidate_base": {base_relative: IDENTITY},
                        "native_bounds_min": [0.0, 0.0, 0.0],
                        "native_bounds_max": [1.0, 1.0, 1.0],
                    },
                    {
                        "link_id": "drawer_0001",
                        "name": "drawer",
                        "visual_asset_paths": [drawer_relative],
                        "visual_asset_hashes": {drawer_relative: sha256(drawer_path)},
                        "visual_asset_spaces": {drawer_relative: "candidate_base"},
                        "visual_asset_transforms_candidate_base": {drawer_relative: IDENTITY},
                        "native_bounds_min": [0.2, 0.0, 0.0],
                        "native_bounds_max": [1.2, 1.0, 1.0],
                    },
                ],
                "joints": [
                    {
                        "joint_id": "drawer_joint",
                        "parent_link_id": "cabinet_body",
                        "child_link_id": "drawer_0001",
                        "joint_type": "prismatic",
                        "axis": [1.0, 0.0, 0.0],
                        "pivot": None,
                        "candidate_limit_lower": 0.0,
                        "candidate_limit_upper": 1.0,
                        "limit_source": "candidate_prior",
                    }
                ],
                "states": [],
                "native_coordinate_convention": "fake +Z-up retrieved frame",
                "native_units": "arbitrary_units",
                "native_output_paths": [],
                "native_output_hashes": {},
                "license_record": license_record,
                "production_selectable": False,
                "provenance": {
                    "adapter_name": f"{family}_retrieval",
                    "adapter_version": "0.1.0",
                    "configuration": {},
                    "input_artifact_paths": [],
                    "output_artifact_paths": [base_relative, drawer_relative],
                    "timestamp": "2026-01-01T00:00:00Z",
                    "confidence": {
                        "score": 0.9 - index * 0.1,
                        "method": "fake_articulated_retrieval",
                        "notes": None,
                    },
                    "source": "retrieved",
                },
                "warnings": [],
            },
        )
        bundle_relative = bundle_path.relative_to(run_root).as_posix()
        candidates.append(
            {
                "candidate_id": candidate_id,
                "source_family": family,
                "source_asset_id": f"{family}_asset_{index:03d}",
                "retrieval_score": 0.9 - index * 0.1,
                "evidence_terms": {
                    "semantic_category": 1.0,
                    "joint_type": 0.9,
                    "part_count": 0.8,
                    "rgb_appearance": 0.0,
                },
                "production_selectable": False,
                "candidate_bundle_path": bundle_relative,
                "candidate_bundle_sha256": sha256(bundle_path),
                "visual_asset_paths": [base_relative, drawer_relative],
                "visual_asset_hashes": {
                    base_relative: sha256(base_path),
                    drawer_relative: sha256(drawer_path),
                },
            }
        )
    write_json(
        output_dir / Path(str(request["output_path"])).name,
        {
            "schema_version": "0.1.0",
            "articulated_object_id": request["articulated_object_id"],
            "measured_motion_sha256": request["measured_motion_sha256"],
            "candidates": candidates,
            "artvip_index_sha256": (request["asset_index_sha256"] if family == "artvip" else None),
            "partnet_index_sha256": (
                request["asset_index_sha256"] if family == "partnet_mobility" else None
            ),
            "runtime_seconds": 0.01,
            "warnings": [],
        },
    )


def generate(
    request: dict[str, object],
    input_root: Path,
    output_dir: Path,
    request_path: Path,
) -> None:
    maybe_fail(request)
    candidates = []
    workers = []
    for item in request["requests"]:
        candidate_id = item["candidate_id"]
        base_path = output_dir / "candidates" / candidate_id / "links/base.ply"
        drawer_path = output_dir / "candidates" / candidate_id / "links/drawer.ply"
        write_mesh(base_path, 0.0)
        write_mesh(drawer_path, 0.2)
        base_relative = base_path.relative_to(input_root).as_posix()
        drawer_relative = drawer_path.relative_to(input_root).as_posix()
        license_record = item["source_license"]
        license_record["source_family"] = "particulate"
        license_record["production_selectable"] = False
        license_record["commercial_review_status"] = "research_only"
        candidate = {
            "candidate_id": candidate_id,
            "articulated_object_id": item["articulated_object_id"],
            "source_family": "particulate",
            "source_asset_id": item["source_backend"],
            "links": [
                {
                    "link_id": "cabinet_body",
                    "name": "cabinet body",
                    "visual_asset_paths": [base_relative],
                    "visual_asset_hashes": {base_relative: sha256(base_path)},
                    "visual_asset_spaces": {base_relative: "candidate_base"},
                    "visual_asset_transforms_candidate_base": {base_relative: IDENTITY},
                    "native_bounds_min": [0.0, 0.0, 0.0],
                    "native_bounds_max": [1.0, 1.0, 1.0],
                },
                {
                    "link_id": "drawer_0001",
                    "name": "drawer",
                    "visual_asset_paths": [drawer_relative],
                    "visual_asset_hashes": {drawer_relative: sha256(drawer_path)},
                    "visual_asset_spaces": {drawer_relative: "candidate_base"},
                    "visual_asset_transforms_candidate_base": {drawer_relative: IDENTITY},
                    "native_bounds_min": [0.2, 0.0, 0.0],
                    "native_bounds_max": [1.2, 1.0, 1.0],
                },
            ],
            "joints": [
                {
                    "joint_id": "drawer_joint",
                    "parent_link_id": "cabinet_body",
                    "child_link_id": "drawer_0001",
                    "joint_type": "prismatic",
                    "axis": [1.0, 0.0, 0.0],
                    "pivot": None,
                    "candidate_limit_lower": 0.0,
                    "candidate_limit_upper": 1.0,
                    "limit_source": "candidate_prior",
                }
            ],
            "states": [],
            "native_coordinate_convention": "+Z up Particulate working frame",
            "native_units": "normalized_arbitrary_units",
            "working_transform_source_to_particulate": IDENTITY,
            "working_transform_particulate_to_source": IDENTITY,
            "working_frame_hypothesis": item["working_frame_hypothesis"],
            "working_frame_hypotheses_evaluated": item["hypotheses_evaluated"],
            "working_frame_selection_evidence": item["hypothesis_selection_evidence"],
            "license_record": license_record,
            "production_selectable": False,
            "provenance": {
                "adapter_name": "particulate_candidates",
                "adapter_version": "0.1.0",
                "configuration": {},
                "input_artifact_paths": [item["source_mesh_path"]],
                "output_artifact_paths": [base_relative, drawer_relative],
                "timestamp": "2026-01-01T00:00:00Z",
                "confidence": {
                    "score": 0.8,
                    "method": "fake_particulate_candidate",
                    "notes": None,
                },
                "source": "generated",
            },
            "warnings": [],
        }
        candidates.append(candidate)
        workers.append(
            {
                "schema_version": "0.1.0",
                "worker_version": "0.1.0",
                "request_sha256": sha256(request_path),
                "official_repository": item["official_repository"],
                "official_code_commit": item["official_code_commit"],
                "checkpoint_repository": item["checkpoint_repository"],
                "checkpoint_revision": item["checkpoint_revision"],
                "checkpoint_hashes": item["checkpoint_hashes"],
                "runtime_model_hashes": item["runtime_model_hashes"],
                "runtime_seconds": 0.02,
                "peak_gpu_memory_bytes": 0,
                "peak_host_memory_bytes": 2048,
                "warnings": [],
            }
        )
    write_json(
        output_dir / "candidate_manifest.json",
        {
            "schema_version": "0.1.0",
            "measured_motion_sha256": request["measured_motion_sha256"],
            "retrieval_manifest_sha256": request["retrieval_manifest_sha256"],
            "candidates": candidates,
            "worker_manifests": workers,
            "failed_candidate_ids": [],
            "runtime_seconds": 0.02,
            "warnings": [],
        },
    )


def fit(request: dict[str, object], output_dir: Path) -> None:
    maybe_fail(request)
    assignments = []
    fittings = []
    failed = request.get("fake_mode") == "wrong_joint_type"
    for candidate_id in request["candidate_ids"]:
        assignment = {
            "candidate_id": candidate_id,
            "assignments": [
                {
                    "observed_part_id": "cabinet_body",
                    "candidate_link_ids": ["cabinet_body"],
                    "assignment_confidence": 0.98,
                    "evidence": {"semantic": 1.0, "geometry": 0.95},
                    "ambiguous": False,
                },
                {
                    "observed_part_id": "drawer",
                    "candidate_link_ids": ["drawer_0001"],
                    "assignment_confidence": 0.97,
                    "evidence": {"semantic": 1.0, "motion": 0.94},
                    "ambiguous": False,
                },
            ],
            "unmatched_candidate_links": [],
            "unmatched_observed_parts": [],
        }
        assignments.append(assignment)
        structure_states = list(
            dict.fromkeys([*request["generation_state_ids"], *request["fitting_state_ids"]])
        )
        fitted_model = {
            "schema_version": "0.1.0",
            "candidate_id": candidate_id,
            "matrix_reference_world_from_candidate_base": IDENTITY,
            "scale": 1.0,
            "link_assignments": assignment["assignments"],
            "fitted_joints": [
                {
                    "candidate_joint_id": "drawer_joint",
                    "measured_joint_id": "drawer_joint",
                    "parent_observed_part_id": "cabinet_body",
                    "child_observed_part_id": "drawer",
                    "joint_type": "prismatic",
                    "fitted_axis": [1.0, 0.0, 0.0],
                    "fitted_pivot": None,
                    "axis_sign": 1,
                    "axis_convention": "oriented_toward_measured_axis",
                    "axis_sign_role": "native_axis_flip_provenance_only",
                    "q_scale": 1.0,
                    "q_scale_convention": "candidate_q_per_measured_q",
                    "q_offset": 0.0,
                    "q_offset_fitted": False,
                    "q_offset_evidence_state_ids": [],
                    "fitting_state_q": {
                        state_id: index * 0.35 for index, state_id in enumerate(structure_states)
                    },
                    "axis_refinement_degrees": 0.0,
                    "pivot_refinement_arbitrary_units": None,
                    "pivot_refinement_part_diagonals": None,
                    "fitting_residual_arbitrary_units": 0.01,
                    "fitting_residual_part_diagonals": 0.01,
                }
            ],
            "generation_state_ids": request["generation_state_ids"],
            "fitting_state_ids": request["fitting_state_ids"],
            "heldout_state_ids": request["heldout_state_ids"],
            "fit_residual_arbitrary_units": 0.01,
            "fit_residual_scene_diagonals": 0.01,
            "ambiguity_reasons": [],
        }
        fittings.append(
            {
                "candidate_id": candidate_id,
                "status": "failed" if failed else "fitted",
                "matrix_reference_world_from_candidate_base": None if failed else IDENTITY,
                "scale": None if failed else 1.0,
                "fitting_state_ids": request["fitting_state_ids"],
                "heldout_state_ids": request["heldout_state_ids"],
                "fitted_joint_positions": {
                    state_id: {"drawer_joint": 0.35} for state_id in request["fitting_state_ids"]
                },
                "joint_axis_signs": {"drawer_joint": 1},
                "fitting_median_residual": None if failed else 0.01,
                "fitting_part_iou": None if failed else 0.82,
                "fitted_model": None if failed else fitted_model,
                "fitted_model_sha256": None if failed else stable_hash(fitted_model),
                "structure_frozen_before_heldout": True,
                "failure_reason": "wrong joint type" if failed else None,
            }
        )
    write_json(
        output_dir / "link_assignments.json",
        {
            "schema_version": "0.1.0",
            "candidate_manifest_sha256": request["candidate_manifest_sha256"],
            "assignments": assignments,
        },
    )
    write_json(
        output_dir / "fitting_manifest.json",
        {
            "schema_version": "0.1.0",
            "candidate_manifest_sha256": request["candidate_manifest_sha256"],
            "evidence_split_sha256": request["evidence_split_sha256"],
            "link_assignments": assignments,
            "fittings": fittings,
            "runtime_seconds": 0.04,
            "peak_gpu_memory_bytes": 0,
            "peak_host_memory_bytes": 4096,
        },
    )


def evaluate(request: dict[str, object], input_root: Path, output_dir: Path) -> None:
    maybe_fail(request)
    reject = request.get("fake_mode") in {"heldout_state_failure", "wrong_joint_type"}
    missing_views = request.get("fake_mode") == "missing_heldout_views"
    missing_target = request.get("fake_mode") == "no_target_mask"
    evaluations = []
    for candidate_id in request["frozen_candidate_ids"]:
        state_evaluations = []
        for state_id in request["heldout_state_ids"]:
            evidence = next(
                item for item in request["state_evidence"] if item["state_id"] == state_id
            )
            depth_manifest = json.loads(
                (input_root / evidence["depth_manifest_path"]).read_text(encoding="utf-8")
            )
            depth_by_frame = {str(item["frame_id"]): item for item in depth_manifest["records"]}
            requested_frames = list(request["heldout_views_by_state"].get(state_id, [])) or [
                "frame_000001"
            ]
            view_evaluations = []
            render_paths = {}
            for frame_id in requested_frames:
                render_path = (
                    output_dir
                    / "candidates"
                    / candidate_id
                    / "renders"
                    / "heldout"
                    / state_id
                    / f"{frame_id}.png"
                )
                usable = not missing_views and not missing_target
                if usable:
                    write_preview(render_path, f"{candidate_id} {state_id} {frame_id}")
                    relative_render = render_path.relative_to(output_dir.parent.parent).as_posix()
                    render_paths[frame_id] = relative_render
                else:
                    relative_render = None
                masks = (
                    {}
                    if missing_target
                    else {
                        str(part_id): str(paths[frame_id])
                        for part_id, paths in evidence["part_mask_paths"].items()
                        if frame_id in paths
                    }
                )
                depth_path = str(depth_by_frame[frame_id]["depth_path"])
                view_evaluations.append(
                    {
                        "frame_id": frame_id,
                        "camera_reconstruction_sha256": evidence["camera_reconstruction_sha256"],
                        "depth_path": depth_path,
                        "depth_sha256": sha256(input_root / depth_path),
                        "valid_depth": True,
                        "target_mask_paths": masks,
                        "target_mask_hashes": {
                            part_id: sha256(input_root / path) for part_id, path in masks.items()
                        },
                        "target_masks_complete": not missing_target,
                        "required_link_ids": ["cabinet_body", "drawer_0001"],
                        "rendered_link_ids": (["cabinet_body", "drawer_0001"] if usable else []),
                        "missing_link_ids": ([] if usable else ["cabinet_body", "drawer_0001"]),
                        "usable": usable,
                        "failure_reasons": (
                            []
                            if usable
                            else [
                                "target_part_mask_unavailable"
                                if missing_target
                                else "heldout_render_unavailable"
                            ]
                        ),
                        "render_path": relative_render,
                        "render_sha256": sha256(render_path) if usable else None,
                        "raw_candidate_pixel_count": 128 if usable else 0,
                        "visible_candidate_pixel_count": 96 if usable else 0,
                        "target_mask_pixel_count": 100 if not missing_target else 0,
                    }
                )
            state_evaluations.append(
                {
                    "state_id": state_id,
                    "heldout": True,
                    "requested_heldout_view_count": len(requested_frames),
                    "usable_heldout_view_count": sum(item["usable"] for item in view_evaluations),
                    "rendered_heldout_view_count": len(render_paths),
                    "views_with_target_masks": sum(
                        bool(item["target_mask_paths"]) for item in view_evaluations
                    ),
                    "views_with_valid_depth": len(view_evaluations),
                    "base_mask_iou": 0.78 if not reject else 0.30,
                    "movable_part_mask_iou": 0.74 if not reject else 0.10,
                    "whole_object_mask_iou": 0.79 if not reject else 0.25,
                    "per_link_depth_residual": {
                        "cabinet_body": 0.01,
                        "drawer_0001": 0.015 if not reject else 0.2,
                    },
                    "depth_inlier_fraction": 0.86 if not reject else 0.2,
                    "negative_space_violation_ratio": 0.03 if not reject else 0.3,
                    "front_of_scene_violation_ratio": 0.02 if not reject else 0.2,
                    "scene_diagonal_arbitrary_units": 1.0,
                    "base_point_residual_arbitrary_units": 0.003,
                    "base_point_residual_scene_diagonals": 0.003,
                    "base_motion_arbitrary_units": 0.003,
                    "base_motion_scene_diagonals": 0.003,
                    "movable_point_residual_arbitrary_units": (0.01 if not reject else 0.2),
                    "joint_constraint_residual": 0.01 if not reject else 0.2,
                    "prismatic_orthogonal_residual": 0.01 if not reject else 0.2,
                    "prismatic_rotation_leakage_degrees": (0.5 if not reject else 10.0),
                    "joint_q_residual": 0.01 if not reject else 0.2,
                    "axis_error_degrees": 2.0,
                    "pivot_residual_part_diagonals": None,
                    "inferred_joint_positions": {"drawer_joint": 0.7},
                    "joint_position_source": "measured_geometry",
                    "render_paths": render_paths,
                    "view_evaluations": view_evaluations,
                }
            )
        passed = not reject and not missing_views and not missing_target and bool(state_evaluations)
        safe_candidate_id = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in candidate_id
        )
        q_objective_path = output_dir / "raw" / f"open_q_objective_{safe_candidate_id}.json"
        q_preview_path = output_dir / "previews" / f"open_q_objective_{safe_candidate_id}.png"
        heldout_state_id = (
            str(request["heldout_state_ids"][0])
            if request["heldout_state_ids"]
            else "heldout_unavailable"
        )
        q_samples = []
        for index in range(401):
            q_value = index / 400
            objective = (q_value - 0.7) ** 2
            q_samples.append(
                {
                    "q": q_value,
                    "measured_point_to_link_residual": objective,
                    "mask_loss": None,
                    "depth_residual": None,
                    "negative_space_penalty": None,
                    "front_of_scene_penalty": None,
                    "total_objective": objective,
                    "usable_view_count": None,
                }
            )
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
                "joint_audits": [
                    {
                        "state_id": heldout_state_id,
                        "joint_id": "drawer_joint",
                        "joint_type": "prismatic",
                        "lower_bound": 0.0,
                        "upper_bound": 1.0,
                        "candidate_limit_source": "candidate_prior",
                        "grid_sample_count": 401,
                        "legacy_optimizer_success": True,
                        "legacy_optimizer_q": 0.7,
                        "legacy_optimizer_objective": 0.0,
                        "legacy_optimizer_matches_global_minimum": True,
                        "grid_global_minimum_q": 0.7,
                        "grid_global_minimum_objective": 0.0,
                        "refined_global_minimum_q": 0.7,
                        "refined_global_minimum_objective": 0.0,
                        "selected_q": 0.7,
                        "selected_residual_arbitrary_units": 0.0,
                        "all_local_minima": [
                            {
                                "q": 0.7,
                                "total_objective": 0.0,
                                "source": "locally_refined",
                            }
                        ],
                        "fitting_state_q": {
                            "state_000_closed": 0.0,
                            "state_001_half_open": 0.5,
                        },
                        "component_availability": {
                            "measured_point_to_link_residual": True,
                            "mask_loss": False,
                            "depth_residual": False,
                            "negative_space_penalty": False,
                            "front_of_scene_penalty": False,
                            "usable_view_count": False,
                        },
                        "samples": q_samples,
                        "optimizer_global_minimum_verified": True,
                        "classification": "global_minimum_verified",
                        "inconsistency_diagnostics": [
                            (
                                "q objective uses fitting-frozen candidate geometry and "
                                "held-out partial measured points"
                            )
                        ],
                        "semantic_ordering": {
                            "expected_semantic_ordering": (
                                "closed -> half_open -> open, monotonic in either canonical sign"
                            ),
                            "candidate_q_by_state": {
                                "state_000_closed": 0.0,
                                "state_001_half_open": 0.5,
                                heldout_state_id: 0.7,
                            },
                            "measured_q_by_state": {
                                "state_000_closed": 0.0,
                                "state_001_half_open": 0.5,
                                heldout_state_id: 0.7,
                            },
                            "observed_ordering": [
                                "state_000_closed",
                                "state_001_half_open",
                                heldout_state_id,
                            ],
                            "direction": "increasing",
                            "ordering_consistent": True,
                            "objective_gap_to_second_minimum": None,
                        },
                    }
                ],
            },
        )
        write_preview(q_preview_path, f"{candidate_id} held-out q objective")
        evaluations.append(
            {
                "candidate_id": candidate_id,
                "status": ("multi_state_validated" if passed else "rejected_heldout_state"),
                "fitting_sha256": request["fitting_manifest_sha256"],
                "candidate_sha256": request["candidate_sha256_by_id"][candidate_id],
                "fitted_model_sha256": request["fitted_model_sha256_by_id"][candidate_id],
                "link_assignment_sha256": request["link_assignment_sha256_by_id"][candidate_id],
                "heldout_evidence_sha256": request["heldout_evidence_sha256"],
                "state_evaluations": state_evaluations,
                "passed_hard_gates": passed,
                "failed_gates": (
                    ["minimum_usable_heldout_views"]
                    if missing_views
                    else ["target_part_mask_unavailable"]
                    if missing_target
                    else ["minimum_movable_part_mask_iou"]
                    if reject
                    else ([] if state_evaluations else ["minimum_heldout_states"])
                ),
                "heldout_state_validation_used": bool(state_evaluations),
                "capture_state_count": request["capture_state_count"],
                "accepted_alignment_state_ids": request["accepted_alignment_state_ids"],
                "selected_candidate_validation_level": (
                    "multi_state_heldout_validated"
                    if passed
                    else (
                        "multi_state_heldout_available"
                        if len(request["accepted_alignment_state_ids"]) >= 3
                        else "two_state_motion_supported"
                    )
                ),
                "link_assignment_confidence": 0.97,
                "heldout_q_objective_path": (
                    f"reconstruction/articulation/raw/open_q_objective_{safe_candidate_id}.json"
                ),
                "heldout_q_objective_sha256": sha256(q_objective_path),
                "heldout_q_objective_preview_path": (
                    f"reconstruction/articulation/previews/open_q_objective_{safe_candidate_id}.png"
                ),
                "heldout_q_objective_preview_sha256": sha256(q_preview_path),
                "runtime_seconds": 0.03,
                "warnings": [],
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
            "runtime_seconds": 0.03,
            "peak_gpu_memory_bytes": 0,
            "peak_host_memory_bytes": 4096,
        },
    )
    write_preview(output_dir / "previews/link_assignment.png", "Fake link assignment")
    write_preview(output_dir / "previews/fitting_states.png", "Fake fitting states")
    write_preview(
        output_dir / "previews/heldout_state_evaluation.png",
        "Fake held-out state evaluation",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action")
    parser.add_argument("--request", required=True)
    parser.add_argument("--input-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    request_path = Path(args.request).resolve()
    if args.action == "healthcheck":
        print("ok: fake articulation worker")
        return 0
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["request_sha256"] = sha256(request_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.action == "align":
        align(request, input_root, output_dir)
    elif args.action == "estimate-motion":
        estimate_motion(request, input_root, output_dir)
    elif args.action == "retrieve":
        retrieve(request, output_dir)
    elif args.action == "generate":
        generate(request, input_root, output_dir, request_path)
    elif args.action == "fit":
        fit(request, output_dir)
    elif args.action == "evaluate":
        evaluate(request, input_root, output_dir)
    else:
        raise ValueError(f"unsupported fake articulation action: {args.action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
