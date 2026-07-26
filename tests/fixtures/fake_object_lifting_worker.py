#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import struct
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_face_ids(
    path: Path,
    values: list[int],
    *,
    mesh_hash: str,
    relative_path: str,
    corrupt: bool = False,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(struct.pack("<I", value) for value in values)
    path.write_bytes(payload + (b"x" if corrupt else b""))
    return {
        "relative_path": relative_path,
        "dtype": "uint32",
        "byte_order": "little",
        "count": len(values),
        "global_mesh_sha256": mesh_hash,
        "minimum_face_id": min(values) if values else None,
        "maximum_face_id": max(values) if values else None,
        "content_sha256": sha256_file(path),
    }


CUBE_VERTICES = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
]
CUBE_FACES = [
    (0, 1, 2),
    (0, 2, 3),
    (4, 6, 5),
    (4, 7, 6),
    (0, 4, 5),
    (0, 5, 1),
    (1, 5, 6),
    (1, 6, 2),
    (2, 6, 7),
    (2, 7, 3),
    (3, 7, 4),
    (3, 4, 0),
]


def write_surface(path: Path, face_ids: list[int], *, nonfinite: bool = False) -> None:
    used = sorted({vertex for face_id in face_ids for vertex in CUBE_FACES[face_id % 12]})
    mapping = {old: new for new, old in enumerate(used)}
    vertices = [CUBE_VERTICES[index] for index in used]
    if nonfinite and vertices:
        vertices[0] = (math.nan, vertices[0][1], vertices[0][2])
    faces = [tuple(mapping[index] for index in CUBE_FACES[face_id % 12]) for face_id in face_ids]
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
        *[f"{x} {y} {z}" for x, y, z in vertices],
        *[f"3 {a} {b} {c}" for a, b, c in faces],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_points(path: Path, face_ids: list[int]) -> None:
    points = [
        tuple(
            sum(CUBE_VERTICES[index][axis] for index in CUBE_FACES[face_id % 12]) / 3
            for axis in range(3)
        )
        for face_id in face_ids
    ]
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
        *[f"{x} {y} {z}" for x, y, z in points],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def preview(path: Path, title: str, color: str) -> None:
    image = Image.new("RGB", (480, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 48, 450, 260), outline="#30343b", width=2)
    draw.polygon([(110, 220), (235, 70), (385, 210)], fill=color, outline="#20252b")
    draw.text((30, 18), title, fill="#111111")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", compress_level=6, optimize=False)


def provenance(object_id: str, outputs: list[str]) -> dict[str, object]:
    return {
        "adapter_name": "object_surface_lifting",
        "adapter_version": "0.1.1",
        "configuration": {"backend": "fake"},
        "input_artifact_paths": [
            "camera/reconstruction.json",
            "observations/object_tracks.json",
            "reconstruction/global/mesh.ply",
        ],
        "output_artifact_paths": outputs,
        "timestamp": "2024-01-01T00:00:00Z",
        "confidence": {
            "score": 0.76,
            "method": "observation_supported_surface_formula",
            "notes": f"fake deterministic evidence for {object_id}",
        },
        "source": "fused",
    }


def healthcheck(config_path: Path) -> int:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "available": True,
                "backend": payload.get("backend", "fake"),
                "worker_version": "0.1.1",
            },
            sort_keys=True,
        )
    )
    return 0


def infer(request_path: Path, input_root: Path, output_dir: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    mode = request["surface_extraction_configuration"].get("fake_mode", "success")
    forbidden = [
        "camera/colmap/database.db",
        "camera/colmap/sparse",
        "camera/colmap/logs",
        "observations/raw",
        "reconstruction/global/raw",
    ]
    visible_forbidden = [path for path in forbidden if (input_root / path).exists()]
    if visible_forbidden:
        print(
            f"worker input isolation failed; forbidden paths are visible: {visible_forbidden}",
            file=sys.stderr,
        )
        return 21
    if mode == "nonzero_exit":
        print("simulated object-lifting failure", file=sys.stderr)
        return 17
    if mode == "oom":
        print("CUDA out of memory during rasterization", file=sys.stderr)
        return 18
    if mode == "rasterizer_failure":
        print("nvdiffrast rasterizer initialization failed", file=sys.stderr)
        return 19
    if mode == "unsupported_camera":
        print("unsupported camera model FOV", file=sys.stderr)
        return 20
    if mode == "timeout":
        time.sleep(120)
    if mode == "interruption":
        os.kill(os.getpid(), signal.SIGINT)

    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "modify_upstream":
        with (input_root / request["global_mesh_path"]).open("ab") as file:
            file.write(b"\n# modified attempt-local copy\n")
    camera = json.loads((input_root / request["camera_reconstruction_path"]).read_text())
    mesh_hash = request["global_mesh_sha256"]
    if mode == "wrong_mesh_hash":
        mesh_hash = "0" * 64
    segmentation_hash = request["segmentation_tracking_sha256"]
    if mode == "wrong_segmentation_hash":
        segmentation_hash = "1" * 64
    camera_hash = request["camera_reconstruction_sha256"]
    if mode == "wrong_camera_hash":
        camera_hash = "2" * 64
    convention = request["coordinate_convention"]
    if mode == "coordinate_mismatch":
        convention = dict(convention)
        convention["world_frame"] = "canonical_x_forward_y_left_z_up"
    global_face_count = int(request["rasterization_configuration"]["global_face_count"])
    registered = set(request["registered_frame_ids"])
    intrinsics = camera["intrinsics"]
    source_model = camera["model"]

    hypotheses = []
    accepted_by_object: dict[str, int] = {}
    ambiguous_by_object: dict[str, int] = {}
    all_accepted: set[int] = set()
    raw_paths: list[str] = []
    supports: list[float] = []
    reprojections: list[float] = []
    for index, track in enumerate(request["object_tracks"]):
        object_id = track["object_id"]
        object_root = output_dir / "objects" / object_id
        object_root.mkdir(parents=True, exist_ok=True)
        unresolved = mode == "unresolved" or not registered.intersection(
            track["mask_paths_by_frame"]
        )
        available_faces = min(global_face_count, 12)
        face_ids = (
            [] if unresolved else [index * 2 % available_faces, (index * 2 + 1) % available_faces]
        )
        face_ids = sorted(set(face_ids))
        ambiguous_ids = (
            [((index * 2 + 2) % min(global_face_count, 12))]
            if mode == "ambiguous" and not unresolved
            else []
        )
        accepted_path = object_root / "accepted_face_ids.bin"
        ambiguous_path = object_root / "ambiguous_face_ids.bin"
        accepted_rel = f"reconstruction/object_surfaces/objects/{object_id}/accepted_face_ids.bin"
        ambiguous_rel = f"reconstruction/object_surfaces/objects/{object_id}/ambiguous_face_ids.bin"
        accepted_manifest = write_face_ids(
            accepted_path,
            face_ids,
            mesh_hash=request["global_mesh_sha256"],
            relative_path=accepted_rel,
            corrupt=mode == "corrupt_face_array",
        )
        ambiguous_manifest = write_face_ids(
            ambiguous_path,
            ambiguous_ids,
            mesh_hash=request["global_mesh_sha256"],
            relative_path=ambiguous_rel,
        )
        evidence_path = object_root / "face_evidence.npz"
        evidence_path.write_bytes(b"PK\x03\x04fake-deterministic-face-evidence\n")
        evidence_rel = f"reconstruction/object_surfaces/objects/{object_id}/face_evidence.npz"
        surface_rel = None
        points_rel = None
        vertex_count = 0
        if face_ids:
            surface_path = object_root / "surface_mesh.ply"
            points_path = object_root / "surface_points.ply"
            write_surface(surface_path, face_ids, nonfinite=mode == "nonfinite_surface")
            write_points(points_path, face_ids)
            surface_rel = f"reconstruction/object_surfaces/objects/{object_id}/surface_mesh.ply"
            points_rel = f"reconstruction/object_surfaces/objects/{object_id}/surface_points.ply"
            vertex_count = len(
                {vertex for face_id in face_ids for vertex in CUBE_FACES[face_id % 12]}
            )
        registered_frames = [
            frame_id
            for frame_id in request["master_frame_order"]
            if frame_id in registered and frame_id in track["mask_paths_by_frame"]
        ]
        observations = []
        for frame_id in registered_frames:
            mask_path = input_root / track["mask_paths_by_frame"][frame_id]
            with Image.open(mask_path) as mask:
                width, height = mask.size
                mask_area = sum(pixel != 0 for pixel in mask.convert("L").getdata())
            undistortion_hash = stable_hash([source_model, intrinsics, width, height, frame_id])
            observations.append(
                {
                    "frame_id": frame_id,
                    "registered": True,
                    "source_camera_model": source_model,
                    "source_distortion": intrinsics.get("distortion", []),
                    "undistorted_width": width,
                    "undistorted_height": height,
                    "undistorted_intrinsics": intrinsics,
                    "undistortion_map_hash": undistortion_hash,
                    "visible_face_count": min(global_face_count, 12),
                    "supporting_face_count": len(face_ids),
                    "iou": 0.72 if face_ids else 0.0,
                    "precision": 0.80 if face_ids else 0.0,
                    "recall": 0.76 if face_ids else 0.0,
                    "rendered_area_pixels": mask_area if face_ids else 0,
                    "mask_area_pixels": mask_area,
                    "false_positive_area_pixels": max(0, mask_area // 10 if face_ids else 0),
                    "false_negative_area_pixels": max(0, mask_area // 8 if face_ids else mask_area),
                }
            )
        support = 0.78 if face_ids else 0.0
        reprojection = 0.72 if face_ids else 0.0
        status = "unresolved" if unresolved else ("ambiguous" if ambiguous_ids else "accepted")
        outputs = [
            accepted_rel,
            ambiguous_rel,
            evidence_rel,
            *([surface_rel, points_rel] if surface_rel and points_rel else []),
        ]
        hypothesis = {
            "object_id": object_id,
            "semantic_label": track["semantic_label"],
            "prompt_id": track["prompt_id"],
            "asset_type_hint": track.get("asset_type_hint"),
            "status": status,
            "unresolved_reason": ("insufficient_multiview_surface_support" if unresolved else None),
            "source_track_path": "observations/object_tracks.json",
            "source_mask_paths": list(track["mask_paths_by_frame"].values()),
            "supporting_frame_ids": list(track["mask_paths_by_frame"]),
            "supporting_registered_frame_ids": registered_frames,
            "global_mesh_path": "reconstruction/global/mesh.ply",
            "global_mesh_sha256": mesh_hash,
            "global_face_count": global_face_count,
            "accepted_global_face_ids": accepted_manifest,
            "ambiguous_global_face_ids": ambiguous_manifest,
            "face_evidence_path": evidence_rel,
            "face_evidence_sha256": sha256_file(evidence_path),
            "face_evidence_arrays": [
                {
                    "name": "global_face_ids",
                    "shape": [len(face_ids)],
                    "dtype": "uint32",
                    "content_sha256": hashlib.sha256(
                        b"".join(struct.pack("<I", value) for value in face_ids)
                    ).hexdigest(),
                }
            ],
            "surface_mesh_path": surface_rel,
            "surface_point_cloud_path": points_rel,
            "surface_visual_glb_path": None,
            "vertex_count": vertex_count,
            "face_count": len(face_ids),
            "component_count": 1 if face_ids else 0,
            "exact_component_count": 1 if face_ids else 0,
            "seam_aware_component_count": 1 if face_ids else 0,
            "potential_chunk_seam_merges": 0,
            "components": (
                [
                    {
                        "component_id": "component_0001",
                        "face_count": len(face_ids),
                        "surface_area_arbitrary_units_squared": float(len(face_ids)) * 0.5,
                        "relative_face_ratio": 1.0,
                        "relative_surface_area": 1.0,
                        "retained": True,
                        "removal_reason": None,
                    }
                ]
                if face_ids
                else []
            ),
            "bbox_min": [0.0, 0.0, 0.0] if face_ids else None,
            "bbox_max": [1.0, 1.0, 1.0] if face_ids else None,
            "bbox_extent": [1.0, 1.0, 1.0] if face_ids else None,
            "centroid": [0.5, 0.5, 0.5] if face_ids else None,
            "mean_face_support_score": support,
            "median_face_support_score": support,
            "supporting_view_count": len(registered_frames),
            "median_reprojection_iou": reprojection,
            "mean_reprojection_iou": reprojection,
            "track_coverage": track["track_coverage"],
            "association_precision": 0.80 if face_ids else 0.0,
            "mask_recall": 0.76 if face_ids else 0.0,
            "reprojection_iou": reprojection,
            "multiview_support": 1.0 if face_ids else 0.0,
            "surface_connectedness": 1.0 if face_ids else 0.0,
            "observed_surface_coverage": 0.76 if face_ids else 0.0,
            "association_confidence": 0.78 if face_ids else 0.0,
            "completeness_confidence": 0.0,
            "observation_support": observations,
            "geometry_status": "partial_observation_supported",
            "completion_status": "not_completed",
            "hidden_surface_completion": "not_implemented",
            "sim_ready": False,
            "metric_scale_known": False,
            "canonical_gravity_alignment_known": False,
            "coordinate_convention": convention,
            "scale_status": "scale_ambiguous",
            "confidence": {
                "score": 0.76 if face_ids else 0.0,
                "method": "observation_supported_surface_formula",
                "notes": "Does not represent hidden-surface or physical accuracy",
            },
            "provenance": provenance(object_id, [path for path in outputs if path]),
            "warnings": ([] if face_ids else ["No face passed multi-view support thresholds"]),
        }
        hypotheses.append(hypothesis)
        accepted_by_object[object_id] = len(face_ids)
        ambiguous_by_object[object_id] = len(ambiguous_ids)
        all_accepted.update(face_ids)
        supports.append(support)
        reprojections.append(reprojection)
        object_preview = output_dir / "previews" / "objects" / f"{object_id}.png"
        preview(object_preview, f"{object_id} partial surface", "#56b4e9")
        raw_paths.extend(path for path in outputs if path)
        raw_paths.append(f"reconstruction/object_surfaces/previews/objects/{object_id}.png")

    preview_specs = {
        "global_face_assignment": ("Global face assignment", "#4daf4a"),
        "object_surface_contact_sheet": ("Object partial surfaces", "#377eb8"),
        "reprojection_contact_sheet": ("Mask / render reprojection", "#984ea3"),
        "conflict_heatmap": ("Face conflicts and overlaps", "#ffbf00"),
        "global_mesh_depth_contact_sheet": ("Global mesh depth", "#4f81bd"),
        "global_mesh_edge_overlay": ("Global mesh edges", "#00a6a6"),
        "sparse_point_vs_mesh_depth": ("Sparse vs mesh depth", "#8c6bb1"),
        "surface_sample_fusion": ("Surface sample fusion", "#2ca25f"),
    }
    for name, (title, color) in preview_specs.items():
        preview(output_dir / "previews" / f"{name}.png", title, color)
    conflicts = []
    if mode == "different_label_overlap" and len(hypotheses) >= 2:
        conflicts.append(
            {
                "conflict_type": "different_semantic_label",
                "object_ids": [hypotheses[0]["object_id"], hypotheses[1]["object_id"]],
                "face_count": 1,
                "resolution": "multi_label_retained",
            }
        )
    assigned = len(all_accepted)
    partition = {
        "global_face_count": global_face_count,
        "unassigned_face_count": max(0, global_face_count - assigned),
        "exactly_one_object_face_count": assigned,
        "multi_label_face_count": sum(
            conflict["face_count"]
            for conflict in conflicts
            if conflict["conflict_type"] == "different_semantic_label"
        ),
        "same_class_conflict_face_count": 0,
        "assigned_face_count_by_object": accepted_by_object,
        "ambiguous_face_count_by_object": ambiguous_by_object,
        "unassigned_face_ratio": max(0.0, (global_face_count - assigned) / global_face_count),
    }
    evidence_manifest = {
        "schema_version": "0.1.0",
        "request_path": "reconstruction/object_surfaces/request.json",
        "worker_manifest_path": "reconstruction/object_surfaces/worker_manifest.json",
        "diagnostics_path": "reconstruction/object_surfaces/diagnostics.json",
        "preview_manifest_path": "reconstruction/object_surfaces/preview_manifest.json",
        "scene_ir_path": "scene_ir/phase4_scene.json",
        "manifest_sha256": request["manifest_sha256"],
        "frame_sequence_digest": request["frame_sequence_digest"],
        "camera_reconstruction_sha256": camera_hash,
        "segmentation_tracking_sha256": segmentation_hash,
        "global_reconstruction_sha256": request["global_reconstruction_sha256"],
        "global_mesh_sha256": mesh_hash,
        "alignment_policy": request.get("alignment_policy", "none"),
        "alignment_sha256": request.get("alignment_sha256"),
        "alignment_status": request.get("alignment_status"),
        "alignment_accepted": request.get("alignment_accepted", False),
        "coordinate_convention": convention,
        "scale_status": "scale_ambiguous",
        "geometry_status": "partial_observation_supported",
        "hidden_surface_completion": "not_implemented",
        "sim_ready": False,
        "metric_scale_known": False,
        "canonical_gravity_alignment_known": False,
        "hypotheses": hypotheses,
        "partition": partition,
        "conflicts": conflicts,
        "provenance": provenance(
            "scene",
            ["reconstruction/object_surfaces/evidence_manifest.json"],
        ),
        "warnings": [],
    }
    write_json(output_dir / "evidence_manifest.json", evidence_manifest)
    object_preview_paths = {
        item["object_id"]: (
            f"reconstruction/object_surfaces/previews/objects/{item['object_id']}.png"
        )
        for item in hypotheses
    }
    write_json(
        output_dir / "preview_manifest.json",
        {
            "global_face_assignment_path": (
                "reconstruction/object_surfaces/previews/global_face_assignment.png"
            ),
            "object_surface_contact_sheet_path": (
                "reconstruction/object_surfaces/previews/object_surface_contact_sheet.png"
            ),
            "reprojection_contact_sheet_path": (
                "reconstruction/object_surfaces/previews/reprojection_contact_sheet.png"
            ),
            "conflict_heatmap_path": (
                "reconstruction/object_surfaces/previews/conflict_heatmap.png"
            ),
            "global_mesh_depth_contact_sheet_path": (
                "reconstruction/object_surfaces/previews/global_mesh_depth_contact_sheet.png"
            ),
            "global_mesh_edge_overlay_path": (
                "reconstruction/object_surfaces/previews/global_mesh_edge_overlay.png"
            ),
            "sparse_point_vs_mesh_depth_path": (
                "reconstruction/object_surfaces/previews/sparse_point_vs_mesh_depth.png"
            ),
            "surface_sample_fusion_path": (
                "reconstruction/object_surfaces/previews/surface_sample_fusion.png"
            ),
            "object_preview_paths": object_preview_paths,
        },
    )
    comparison_metrics = []
    for hypothesis in hypotheses:
        for method in ("exact_face_vote_v1", "surface_sample_fusion_v2"):
            comparison_metrics.append(
                {
                    "object_id": hypothesis["object_id"],
                    "method": method,
                    "accepted_faces": hypothesis["face_count"],
                    "ambiguous_faces": hypothesis["ambiguous_global_face_ids"]["count"],
                    "component_count": hypothesis["component_count"],
                    "surface_area_arbitrary_units_squared": float(hypothesis["face_count"]) * 0.5,
                    "reprojection_iou": hypothesis["mean_reprojection_iou"],
                    "precision": hypothesis["association_precision"],
                    "recall": hypothesis["mask_recall"],
                    "supporting_views": hypothesis["supporting_view_count"],
                    "runtime_seconds": 0.02,
                    "peak_gpu_memory_bytes": None,
                }
            )
    write_json(
        output_dir / "method_comparison.json",
        {
            "schema_version": "0.1.0",
            "selected_method": request["lifting_method"],
            "metrics": comparison_metrics,
            "conclusion": "Fake methods agree deterministically.",
            "warnings": [],
        },
    )
    alignment_frames = [
        {
            "frame_id": frame_id,
            "mesh_pixel_coverage": 0.75,
            "depth_finite_ratio": 0.75,
            "visible_global_face_count": min(global_face_count, 12),
            "depth_percentiles": {"p05": 1.0, "p50": 2.0, "p95": 3.0},
            "sparse_observation_count": 3,
            "normalized_depth_residual_median": 0.02,
            "normalized_depth_residual_p90": 0.04,
            "depth_inlier_fraction": 1.0,
        }
        for frame_id in request["registered_frame_ids"]
    ]
    write_json(
        output_dir / "camera_mesh_alignment.json",
        {
            "schema_version": "0.1.0",
            "frame_sequence_digest": request["frame_sequence_digest"],
            "camera_reconstruction_sha256": camera_hash,
            "global_mesh_sha256": mesh_hash,
            "frames": alignment_frames,
            "mesh_pixel_coverage_mean": 0.75,
            "sparse_depth_residual_median": 0.02,
            "sparse_depth_residual_p90": 0.04,
            "sparse_depth_inlier_fraction": 1.0,
            "alignment_sufficient_for_lifting": True,
            "diagnosis": "Fake camera and mesh are aligned.",
            "warnings": [],
        },
    )
    accepted_objects = sum(item["status"] == "accepted" for item in hypotheses)
    ambiguous_objects = sum(item["status"] == "ambiguous" for item in hypotheses)
    unresolved_objects = sum(item["status"] == "unresolved" for item in hypotheses)
    write_json(
        output_dir / "diagnostics.json",
        {
            "schema_version": "0.1.0",
            "track_count": len(hypotheses),
            "accepted_object_count": accepted_objects,
            "partial_object_count": 0,
            "ambiguous_object_count": ambiguous_objects,
            "unresolved_object_count": unresolved_objects,
            "global_vertex_count": int(
                request["rasterization_configuration"]["global_vertex_count"]
            ),
            "global_face_count": global_face_count,
            "processed_camera_count": len(request["registered_frame_ids"]),
            "canonical_mask_count": sum(
                len(track["mask_paths_by_frame"]) for track in request["object_tracks"]
            ),
            "accepted_face_count": sum(accepted_by_object.values()),
            "ambiguous_face_count": sum(ambiguous_by_object.values()),
            "same_class_conflict_count": 0,
            "different_label_overlap_count": len(conflicts),
            "unassigned_face_ratio": partition["unassigned_face_ratio"],
            "mean_face_support": sum(supports) / max(len(supports), 1),
            "median_face_support": sorted(supports)[len(supports) // 2] if supports else 0.0,
            "mean_reprojection_iou": sum(reprojections) / max(len(reprojections), 1),
            "median_reprojection_iou": (
                sorted(reprojections)[len(reprojections) // 2] if reprojections else 0.0
            ),
            "alignment_sufficient_for_lifting": True,
            "diagnosed_bottleneck": "mixed_or_inconclusive",
            "runtime_seconds": 0.25,
            "peak_gpu_memory_bytes": None,
            "timings_seconds": {
                "mesh_load": 0.01,
                "rasterization": 0.05,
                "evidence_accumulation": 0.05,
                "surface_extraction": 0.04,
                "preview": 0.10,
            },
            "warnings": [],
        },
    )
    raw_paths.extend(
        [
            "reconstruction/object_surfaces/evidence_manifest.json",
            "reconstruction/object_surfaces/diagnostics.json",
            "reconstruction/object_surfaces/method_comparison.json",
            "reconstruction/object_surfaces/camera_mesh_alignment.json",
            "reconstruction/object_surfaces/preview_manifest.json",
            *[f"reconstruction/object_surfaces/previews/{name}.png" for name in preview_specs],
        ]
    )
    worker_manifest = {
        "schema_version": "0.1.0",
        "worker_version": "0.1.1",
        "backend": "fake",
        "python_version": sys.version.split()[0],
        "torch_version": None,
        "cuda_version": None,
        "nvdiffrast_version": None,
        "device": "cpu",
        "device_name": None,
        "request_sha256": sha256_file(request_path),
        "manifest_sha256": request["manifest_sha256"],
        "frame_sequence_digest": request["frame_sequence_digest"],
        "camera_reconstruction_sha256": camera_hash,
        "segmentation_tracking_sha256": segmentation_hash,
        "global_reconstruction_sha256": request["global_reconstruction_sha256"],
        "global_mesh_sha256": mesh_hash,
        "alignment_policy": request.get("alignment_policy", "none"),
        "alignment_sha256": request.get("alignment_sha256"),
        "alignment_status": request.get("alignment_status"),
        "alignment_accepted": request.get("alignment_accepted", False),
        "processed_registered_frame_ids": request["registered_frame_ids"],
        "global_vertex_count": int(request["rasterization_configuration"]["global_vertex_count"]),
        "global_face_count": global_face_count,
        "lifting_method": request["lifting_method"],
        "median_global_edge_length": 1.0,
        "sample_voxel_edge_length": float(
            request["surface_sample_configuration"]["sample_voxel_edge_multiplier"]
        ),
        "fused_sample_cell_count": sum(int(item["face_count"]) for item in hypotheses),
        "processed_face_count_by_frame": {
            frame_id: min(global_face_count, 12) for frame_id in request["registered_frame_ids"]
        },
        "culled_face_count_by_frame": {
            frame_id: max(0, global_face_count - min(global_face_count, 12))
            for frame_id in request["registered_frame_ids"]
        },
        "mesh_load_seconds": 0.01,
        "rasterization_seconds": 0.05,
        "evidence_accumulation_seconds": 0.05,
        "surface_extraction_seconds": 0.04,
        "preview_seconds": 0.10,
        "runtime_seconds": 0.25,
        "peak_gpu_memory_bytes": None,
        "peak_host_memory_bytes": 8_388_608,
        "raw_output_paths": sorted(set(raw_paths)),
        "warnings": [],
    }
    if mode == "malformed_manifest":
        write_json(output_dir / "worker_manifest.json", {"schema_version": "0.1.0"})
    elif mode == "path_escape":
        worker_manifest["raw_output_paths"].append("../escape")
        write_json(output_dir / "worker_manifest.json", worker_manifest)
    else:
        write_json(output_dir / "worker_manifest.json", worker_manifest)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    health = subparsers.add_parser("healthcheck")
    health.add_argument("--config", type=Path, required=True)
    inference = subparsers.add_parser("infer")
    inference.add_argument("--request", type=Path, required=True)
    inference.add_argument("--input-root", type=Path, required=True)
    inference.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "healthcheck":
        return healthcheck(args.config)
    return infer(args.request, args.input_root, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
