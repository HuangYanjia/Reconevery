from __future__ import annotations

import hashlib
import json
import platform
import resource
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from object_lifting_worker.distortion import undistort_binary_mask
from object_lifting_worker.face_evidence import (
    FaceStatistics,
    accumulate_positive,
    accumulate_visibility_and_negative,
    score_faces,
    write_evidence_npz,
)
from object_lifting_worker.mask_processing import MaskRegions, preprocess_mask
from object_lifting_worker.previews import (
    add_title,
    annotated_tile,
    assignment_image,
    contact_sheet,
)
from object_lifting_worker.rasterization import NvdiffrastRasterizer, RasterResult
from object_lifting_worker.schema import WorkerRequest, load_request
from object_lifting_worker.surface_extraction import (
    extract_surface_assets,
    filter_components,
    write_face_ids,
)
from object_lifting_worker.version import __version__


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_hash(root: Path, relative_path: str, expected: str) -> Path:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"required object-lifting input is missing: {relative_path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"object-lifting input hash mismatch for {relative_path}: "
            f"expected {expected}, got {actual}"
        )
    return path


@dataclass
class FrameObjectData:
    regions: MaskRegions
    mask: Any
    map_hash: str
    intrinsics: dict[str, Any]
    score: float


@dataclass
class FrameData:
    frame_id: str
    frame_index: int
    pose: dict[str, Any]
    raster: RasterResult
    objects: dict[str, FrameObjectData]


def _load_inputs(
    request: WorkerRequest,
    input_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    manifest_path = require_hash(input_root, request.manifest_path, request.manifest_sha256)
    camera_path = require_hash(
        input_root,
        request.camera_reconstruction_path,
        request.camera_reconstruction_sha256,
    )
    tracks_path = require_hash(
        input_root,
        request.segmentation_tracking_path,
        request.segmentation_tracking_sha256,
    )
    require_hash(
        input_root,
        request.global_reconstruction_path,
        request.global_reconstruction_sha256,
    )
    mesh_path = require_hash(input_root, request.global_mesh_path, request.global_mesh_sha256)
    require_hash(input_root, request.global_scene_glb_path, request.global_scene_glb_sha256)
    return (
        json.loads(manifest_path.read_text(encoding="utf-8")),
        json.loads(camera_path.read_text(encoding="utf-8")),
        json.loads(tracks_path.read_text(encoding="utf-8")),
        mesh_path,
    )


def _load_mesh(path: Path) -> tuple[Any, Any]:
    import numpy as np
    import trimesh

    loaded = trimesh.load_mesh(path, process=False, maintain_order=True)
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) != 1:
            raise ValueError("global PLY unexpectedly contains multiple geometries")
        loaded = next(iter(loaded.geometry.values()))
    vertices = np.asarray(loaded.vertices, dtype=np.float32)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if not len(vertices) or not len(faces):
        raise ValueError("global mesh is empty")
    if not np.isfinite(vertices).all():
        raise ValueError("global mesh contains non-finite coordinates")
    return vertices, faces


def _load_mask(path: Path, width: int, height: int) -> Any:
    import numpy as np

    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8)
    if mask.shape != (height, width):
        raise ValueError(
            f"canonical mask {path} dimensions {mask.shape[::-1]} "
            f"do not match camera {(width, height)}"
        )
    unique = set(np.unique(mask).tolist())
    if not unique <= {0, 255}:
        raise ValueError(f"canonical mask {path} is not binary: {sorted(unique)}")
    if not mask.any():
        raise ValueError(f"canonical mask {path} is empty")
    return mask


def _prepare_frames(
    request: WorkerRequest,
    input_root: Path,
    camera: dict[str, Any],
    rasterizer: NvdiffrastRasterizer,
) -> list[FrameData]:
    poses = {item["frame_id"]: item for item in camera["poses"]}
    intrinsics = camera["intrinsics"]
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    camera_model = str(camera["model"])
    track_by_id = {track.object_id: track for track in request.object_tracks}
    frame_data: list[FrameData] = []
    scale = float(request.rasterization_configuration["raster_scale"])
    processing = request.mask_processing_configuration
    radii = {
        "core_erosion_pixels": max(
            0, int(round(int(processing["mask_core_erosion_pixels"]) * scale))
        ),
        "boundary_width_pixels": max(
            0, int(round(int(processing["mask_boundary_width_pixels"]) * scale))
        ),
        "exclusion_dilation_pixels": max(
            0, int(round(int(processing["mask_exclusion_dilation_pixels"]) * scale))
        ),
    }
    for frame_index, frame_id in enumerate(request.master_frame_order):
        if frame_id not in poses:
            continue
        object_ids = [
            object_id
            for object_id, track in track_by_id.items()
            if frame_id in track.mask_paths_by_frame
        ]
        if not object_ids:
            continue
        objects: dict[str, FrameObjectData] = {}
        undistorted_intrinsics: dict[str, Any] | None = None
        for object_id in object_ids:
            track = track_by_id[object_id]
            mask_path = input_root / track.mask_paths_by_frame[frame_id]
            source_mask = _load_mask(mask_path, width, height)
            undistorted = undistort_binary_mask(
                source_mask,
                camera_model=camera_model,
                intrinsics=intrinsics,
                raster_scale=scale,
            )
            if (
                undistorted_intrinsics is not None
                and undistorted.intrinsics != undistorted_intrinsics
            ):
                raise ValueError("undistorted intrinsics changed within one frame")
            undistorted_intrinsics = undistorted.intrinsics
            objects[object_id] = FrameObjectData(
                regions=preprocess_mask(undistorted.mask, **radii),
                mask=undistorted.mask > 0,
                map_hash=undistorted.map_hash,
                intrinsics=undistorted.intrinsics,
                score=float(track.frame_scores[frame_id]),
            )
        assert undistorted_intrinsics is not None
        raster = rasterizer.rasterize(poses[frame_id], undistorted_intrinsics)
        frame_data.append(
            FrameData(
                frame_id=frame_id,
                frame_index=frame_index,
                pose=poses[frame_id],
                raster=raster,
                objects=objects,
            )
        )
    if not frame_data:
        raise ValueError("all registered frames are unusable for object lifting")
    return frame_data


def _accumulate(
    request: WorkerRequest,
    frames: list[FrameData],
) -> tuple[
    dict[str, dict[int, FaceStatistics]],
    dict[str, list[int]],
    dict[str, list[int]],
]:
    stats_by_object = {track.object_id: {} for track in request.object_tracks}
    for frame in frames:
        for object_id, item in frame.objects.items():
            accumulate_positive(
                stats_by_object[object_id],
                face_ids=frame.raster.face_ids,
                depth=frame.raster.depth,
                core=item.regions.core,
                boundary=item.regions.boundary,
                frame_index=frame.frame_index,
                frame_score=item.score,
            )
    for frame in frames:
        for object_id, item in frame.objects.items():
            accumulate_visibility_and_negative(
                stats_by_object[object_id],
                face_ids=frame.raster.face_ids,
                exterior=item.regions.exterior,
                frame_score=item.score,
            )
    accepted: dict[str, list[int]] = {}
    ambiguous: dict[str, list[int]] = {}
    for object_id, stats in stats_by_object.items():
        accepted[object_id], ambiguous[object_id] = score_faces(
            stats,
            configuration=request.face_evidence_configuration,
        )
    return stats_by_object, accepted, ambiguous


def _resolve_overlaps(
    request: WorkerRequest,
    stats_by_object: dict[str, dict[int, FaceStatistics]],
    accepted: dict[str, list[int]],
    ambiguous: dict[str, list[int]],
) -> tuple[list[dict[str, object]], set[int], set[int]]:
    tracks = {track.object_id: track for track in request.object_tracks}
    conflicts: list[dict[str, object]] = []
    same_class_conflict_faces: set[int] = set()
    different_label_overlap_faces: set[int] = set()
    by_label: dict[str, list[str]] = {}
    for track in request.object_tracks:
        by_label.setdefault(track.semantic_label.strip().lower(), []).append(track.object_id)
    margin = float(request.face_evidence_configuration["instance_score_margin"])
    for object_ids in by_label.values():
        face_owners: dict[int, list[str]] = {}
        for object_id in object_ids:
            for face_id in accepted[object_id]:
                face_owners.setdefault(face_id, []).append(object_id)
        for face_id, owners in face_owners.items():
            if len(owners) < 2:
                continue
            ranked = sorted(
                owners,
                key=lambda object_id: (
                    -stats_by_object[object_id][face_id].support_score,
                    object_id,
                ),
            )
            best = stats_by_object[ranked[0]][face_id].support_score
            second = stats_by_object[ranked[1]][face_id].support_score
            same_class_conflict_faces.add(face_id)
            if best - second >= margin:
                for loser in ranked[1:]:
                    accepted[loser].remove(face_id)
                    ambiguous[loser].append(face_id)
                resolution = "winner_by_support"
            else:
                for owner in ranked:
                    accepted[owner].remove(face_id)
                    ambiguous[owner].append(face_id)
                resolution = "ambiguous_below_margin"
            conflicts.append(
                {
                    "conflict_type": "same_class_instance",
                    "object_ids": sorted(owners),
                    "face_count": 1,
                    "resolution": resolution,
                }
            )
    labels = sorted(by_label)
    for left_index, left_label in enumerate(labels):
        left_faces = set().union(*(set(accepted[item]) for item in by_label[left_label]))
        for right_label in labels[left_index + 1 :]:
            right_faces = set().union(*(set(accepted[item]) for item in by_label[right_label]))
            overlap = left_faces & right_faces
            if overlap:
                different_label_overlap_faces.update(overlap)
                owners = sorted(
                    [
                        object_id
                        for object_id, track in tracks.items()
                        if track.semantic_label.strip().lower() in {left_label, right_label}
                        and set(accepted[object_id]) & overlap
                    ]
                )
                conflicts.append(
                    {
                        "conflict_type": "different_semantic_label",
                        "object_ids": owners,
                        "face_count": len(overlap),
                        "resolution": "multi_label_retained",
                    }
                )
    for object_id in accepted:
        accepted[object_id] = sorted(set(accepted[object_id]))
        ambiguous[object_id] = sorted(set(ambiguous[object_id]) - set(accepted[object_id]))
    return conflicts, same_class_conflict_faces, different_label_overlap_faces


def _metrics(mask: Any, rendered: Any) -> dict[str, float | int]:
    import numpy as np

    mask = mask.astype(bool)
    rendered = rendered.astype(bool)
    intersection = int(np.logical_and(mask, rendered).sum())
    union = int(np.logical_or(mask, rendered).sum())
    rendered_area = int(rendered.sum())
    mask_area = int(mask.sum())
    return {
        "iou": intersection / union if union else 0.0,
        "precision": intersection / rendered_area if rendered_area else 0.0,
        "recall": intersection / mask_area if mask_area else 0.0,
        "rendered_area_pixels": rendered_area,
        "mask_area_pixels": mask_area,
        "false_positive_area_pixels": int(np.logical_and(rendered, ~mask).sum()),
        "false_negative_area_pixels": int(np.logical_and(mask, ~rendered).sum()),
    }


def _provenance(
    request: WorkerRequest,
    timestamp: str,
    outputs: list[str],
    confidence: float,
) -> dict[str, object]:
    return {
        "adapter_name": "object_surface_lifting",
        "adapter_version": "0.1.0",
        "configuration": {
            "rasterization": request.rasterization_configuration,
            "mask_processing": request.mask_processing_configuration,
            "face_evidence": request.face_evidence_configuration,
            "surface_extraction": {
                key: value
                for key, value in request.surface_extraction_configuration.items()
                if key != "fake_mode"
            },
        },
        "input_artifact_paths": [
            request.camera_reconstruction_path,
            request.segmentation_tracking_path,
            request.global_reconstruction_path,
            request.global_mesh_path,
        ],
        "output_artifact_paths": outputs,
        "timestamp": timestamp,
        "confidence": {
            "score": confidence,
            "method": "observation_supported_surface_formula",
            "notes": "Confidence excludes hidden shape, materials, physics, and metric scale",
        },
        "source": "fused",
    }


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def run_inference(
    request_path: Path,
    input_root: Path,
    output_dir: Path,
) -> None:
    import numpy as np
    import torch

    total_start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    request = load_request(request_path)
    manifest, camera, _tracks, mesh_path = _load_inputs(request, input_root)
    load_start = time.monotonic()
    vertices, faces = _load_mesh(mesh_path)
    mesh_load_seconds = time.monotonic() - load_start
    expected_vertices = int(request.rasterization_configuration["global_vertex_count"])
    expected_faces = int(request.rasterization_configuration["global_face_count"])
    if len(vertices) != expected_vertices or len(faces) != expected_faces:
        raise ValueError(
            "global mesh topology does not match Phase 3 metadata: "
            f"{len(vertices)}/{len(faces)} vs {expected_vertices}/{expected_faces}"
        )
    rasterizer = NvdiffrastRasterizer(
        vertices,
        faces,
        face_chunk_size=int(request.rasterization_configuration["face_chunk_size"]),
    )
    raster_start = time.monotonic()
    frames = _prepare_frames(request, input_root, camera, rasterizer)
    rasterization_seconds = time.monotonic() - raster_start
    evidence_start = time.monotonic()
    stats_by_object, accepted, ambiguous = _accumulate(request, frames)
    conflicts, same_conflict_faces, different_overlap_faces = _resolve_overlaps(
        request,
        stats_by_object,
        accepted,
        ambiguous,
    )
    evidence_accumulation_seconds = time.monotonic() - evidence_start

    extraction_start = time.monotonic()
    hypotheses = []
    accepted_sets: dict[str, set[int]] = {}
    ambiguous_faces_all: set[int] = set()
    object_tiles: list[Image.Image] = []
    reprojection_tiles: list[Image.Image] = []
    object_preview_paths: dict[str, str] = {}
    confidence_by_object: dict[str, float] = {}
    raw_paths: list[str] = []
    all_supports: list[float] = []
    all_ious: list[float] = []
    timestamp = str(manifest["provenance"]["timestamp"])
    tracks_by_id = {track.object_id: track for track in request.object_tracks}
    camera_model = str(camera["model"])
    camera_distortion = list(camera["intrinsics"].get("distortion", []))
    component_config = request.surface_extraction_configuration
    for object_id in sorted(tracks_by_id):
        track = tracks_by_id[object_id]
        retained, components = filter_components(
            faces,
            accepted[object_id],
            min_faces=int(component_config["min_component_faces"]),
            min_relative_area=float(component_config["min_relative_component_area"]),
        )
        removed = set(accepted[object_id]) - set(retained)
        accepted[object_id] = retained
        ambiguous[object_id] = sorted((set(ambiguous[object_id]) | removed) - set(retained))
        accepted_sets[object_id] = set(retained)
        ambiguous_faces_all.update(ambiguous[object_id])
        object_root = output_dir / "objects" / object_id
        object_root.mkdir(parents=True, exist_ok=True)
        accepted_rel = f"reconstruction/object_surfaces/objects/{object_id}/accepted_face_ids.bin"
        ambiguous_rel = f"reconstruction/object_surfaces/objects/{object_id}/ambiguous_face_ids.bin"
        accepted_manifest = write_face_ids(
            object_root / "accepted_face_ids.bin",
            retained,
            global_mesh_sha256=request.global_mesh_sha256,
            relative_path=accepted_rel,
        )
        ambiguous_manifest = write_face_ids(
            object_root / "ambiguous_face_ids.bin",
            ambiguous[object_id],
            global_mesh_sha256=request.global_mesh_sha256,
            relative_path=ambiguous_rel,
        )
        evidence_rel = f"reconstruction/object_surfaces/objects/{object_id}/face_evidence.npz"
        evidence_path = object_root / "face_evidence.npz"
        array_records = write_evidence_npz(
            evidence_path,
            stats_by_object[object_id],
        )
        surface_rel: str | None = None
        points_rel: str | None = None
        surface_stats = {
            "vertex_count": 0,
            "face_count": 0,
            "bbox_min": None,
            "bbox_max": None,
            "bbox_extent": None,
            "centroid": None,
        }
        if retained:
            surface_rel = f"reconstruction/object_surfaces/objects/{object_id}/surface_mesh.ply"
            points_rel = f"reconstruction/object_surfaces/objects/{object_id}/surface_points.ply"
            surface_stats = extract_surface_assets(
                vertices,
                faces,
                retained,
                mesh_path=object_root / "surface_mesh.ply",
                points_path=object_root / "surface_points.ply",
            )
        observation_support = []
        object_reprojection_tiles: list[tuple[float, Image.Image]] = []
        object_ious: list[float] = []
        supporting_registered: list[str] = []
        for frame in frames:
            item = frame.objects.get(object_id)
            if item is None:
                continue
            rendered = np.isin(
                frame.raster.face_ids,
                np.asarray(retained, dtype=np.int64),
            )
            metrics = _metrics(item.mask, rendered)
            object_ious.append(float(metrics["iou"]))
            if int(metrics["rendered_area_pixels"]) > 0:
                supporting_registered.append(frame.frame_id)
            observation_support.append(
                {
                    "frame_id": frame.frame_id,
                    "registered": True,
                    "source_camera_model": camera_model,
                    "source_distortion": camera_distortion,
                    "undistorted_width": int(item.intrinsics["width"]),
                    "undistorted_height": int(item.intrinsics["height"]),
                    "undistorted_intrinsics": item.intrinsics,
                    "undistortion_map_hash": item.map_hash,
                    "visible_face_count": int(
                        np.unique(frame.raster.face_ids[frame.raster.valid]).size
                    ),
                    "supporting_face_count": int(
                        np.unique(frame.raster.face_ids[rendered & frame.raster.valid]).size
                    ),
                    **metrics,
                }
            )
            object_reprojection_tiles.append(
                (
                    float(metrics["iou"]),
                    annotated_tile(
                        item.mask,
                        rendered,
                        title=f"{object_id} / {frame.frame_id}",
                        iou=float(metrics["iou"]),
                    ),
                )
            )
        mean_support = (
            statistics.mean(
                stats_by_object[object_id][face_id].support_score for face_id in retained
            )
            if retained
            else 0.0
        )
        median_support = (
            statistics.median(
                stats_by_object[object_id][face_id].support_score for face_id in retained
            )
            if retained
            else 0.0
        )
        retained_components = [item for item in components if item["retained"]]
        largest_ratio = max(
            (float(item["relative_face_ratio"]) for item in retained_components),
            default=0.0,
        )
        mean_iou = statistics.mean(object_ious) if object_ious else 0.0
        median_iou = _median(object_ious)
        view_norm = min(
            1.0,
            len(set(supporting_registered))
            / max(2 * int(request.face_evidence_configuration["min_supporting_views"]), 1),
        )
        object_conflicts = sum(
            int(conflict["face_count"])
            for conflict in conflicts
            if object_id in conflict["object_ids"]
        )
        conflict_penalty = min(
            0.2,
            0.2 * object_conflicts / max(len(retained), 1),
        )
        confidence = max(
            0.0,
            min(
                1.0,
                0.30 * mean_support
                + 0.25 * median_iou
                + 0.20 * view_norm
                + 0.15 * track.track_coverage
                + 0.10 * largest_ratio
                - conflict_penalty,
            ),
        )
        confidence_by_object[object_id] = confidence
        unresolved = not retained
        status = (
            "unresolved" if unresolved else ("ambiguous" if ambiguous[object_id] else "accepted")
        )
        outputs = [accepted_rel, ambiguous_rel, evidence_rel]
        if surface_rel and points_rel:
            outputs.extend([surface_rel, points_rel])
        provenance = _provenance(request, timestamp, outputs, confidence)
        hypothesis = {
            "object_id": object_id,
            "semantic_label": track.semantic_label,
            "prompt_id": track.prompt_id,
            "asset_type_hint": track.asset_type_hint,
            "status": status,
            "unresolved_reason": ("insufficient_multiview_surface_support" if unresolved else None),
            "source_track_path": request.segmentation_tracking_path,
            "source_mask_paths": list(track.mask_paths_by_frame.values()),
            "supporting_frame_ids": [
                frame_id
                for frame_id in request.master_frame_order
                if frame_id in track.mask_paths_by_frame
            ],
            "supporting_registered_frame_ids": sorted(
                set(supporting_registered),
                key=request.master_frame_order.index,
            ),
            "global_mesh_path": request.global_mesh_path,
            "global_mesh_sha256": request.global_mesh_sha256,
            "global_face_count": len(faces),
            "accepted_global_face_ids": accepted_manifest,
            "ambiguous_global_face_ids": ambiguous_manifest,
            "face_evidence_path": evidence_rel,
            "face_evidence_sha256": sha256_file(evidence_path),
            "face_evidence_arrays": array_records,
            "surface_mesh_path": surface_rel,
            "surface_point_cloud_path": points_rel,
            "surface_visual_glb_path": None,
            **surface_stats,
            "component_count": len(retained_components),
            "components": components,
            "mean_face_support_score": mean_support,
            "median_face_support_score": median_support,
            "supporting_view_count": len(set(supporting_registered)),
            "median_reprojection_iou": median_iou,
            "mean_reprojection_iou": mean_iou,
            "track_coverage": track.track_coverage,
            "observation_support": observation_support,
            "geometry_status": "partial_observation_supported",
            "completion_status": "not_completed",
            "hidden_surface_completion": "not_implemented",
            "sim_ready": False,
            "metric_scale_known": False,
            "canonical_gravity_alignment_known": False,
            "coordinate_convention": request.coordinate_convention,
            "scale_status": "scale_ambiguous",
            "confidence": provenance["confidence"],
            "provenance": provenance,
            "warnings": (
                ["No global face passed multi-view support thresholds"] if unresolved else []
            ),
        }
        hypotheses.append(hypothesis)
        all_supports.extend(
            stats_by_object[object_id][face_id].support_score for face_id in retained
        )
        all_ious.extend(object_ious)
        if object_reprojection_tiles:
            ranked_tiles = [
                image
                for _, image in sorted(
                    object_reprojection_tiles,
                    key=lambda item: -item[0],
                )[:3]
            ]
            object_preview_path = output_dir / "previews" / "objects" / f"{object_id}.png"
            object_preview = add_title(
                ranked_tiles[0],
                f"{object_id} partial surface, confidence={confidence:.3f}",
            )
            object_preview_path.parent.mkdir(parents=True, exist_ok=True)
            object_preview.save(
                object_preview_path,
                format="PNG",
                compress_level=6,
                optimize=False,
            )
            object_preview_paths[object_id] = (
                f"reconstruction/object_surfaces/previews/objects/{object_id}.png"
            )
            object_tiles.append(object_preview)
            reprojection_tiles.extend(ranked_tiles)
        raw_paths.extend(outputs)
    surface_extraction_seconds = time.monotonic() - extraction_start

    preview_start = time.monotonic()
    first_raster = frames[0].raster.face_ids
    assignment = assignment_image(
        first_raster,
        accepted_sets,
        ambiguous_faces_all,
        same_conflict_faces,
    )
    assignment = add_title(
        assignment,
        "Original global face IDs: objects / gray unassigned / yellow ambiguous",
    )
    preview_root = output_dir / "previews"
    preview_root.mkdir(parents=True, exist_ok=True)
    assignment.save(
        preview_root / "global_face_assignment.png",
        format="PNG",
        compress_level=6,
        optimize=False,
    )
    contact_sheet(
        object_tiles,
        preview_root / "object_surface_contact_sheet.png",
        columns=2,
    )
    contact_sheet(
        reprojection_tiles[:12],
        preview_root / "reprojection_contact_sheet.png",
        columns=3,
    )
    conflict = assignment_image(
        first_raster,
        {},
        different_overlap_faces,
        same_conflict_faces,
    )
    add_title(
        conflict,
        "Red: same-class conflict; yellow: different-label overlap",
    ).save(
        preview_root / "conflict_heatmap.png",
        format="PNG",
        compress_level=6,
        optimize=False,
    )
    preview_seconds = time.monotonic() - preview_start

    assignment_counts: dict[int, int] = {}
    for values in accepted_sets.values():
        for face_id in values:
            assignment_counts[face_id] = assignment_counts.get(face_id, 0) + 1
    assigned_faces = set(assignment_counts)
    partition = {
        "global_face_count": len(faces),
        "unassigned_face_count": len(faces) - len(assigned_faces),
        "exactly_one_object_face_count": sum(count == 1 for count in assignment_counts.values()),
        "multi_label_face_count": len(different_overlap_faces),
        "same_class_conflict_face_count": len(same_conflict_faces),
        "assigned_face_count_by_object": {
            object_id: len(values) for object_id, values in accepted_sets.items()
        },
        "ambiguous_face_count_by_object": {
            object_id: len(ambiguous[object_id]) for object_id in sorted(ambiguous)
        },
        "unassigned_face_ratio": 1.0 - len(assigned_faces) / len(faces),
    }
    scene_confidence = (
        statistics.mean(confidence_by_object.values()) if confidence_by_object else 0.0
    )
    evidence_manifest = {
        "schema_version": "0.1.0",
        "request_path": "reconstruction/object_surfaces/request.json",
        "worker_manifest_path": "reconstruction/object_surfaces/worker_manifest.json",
        "diagnostics_path": "reconstruction/object_surfaces/diagnostics.json",
        "preview_manifest_path": "reconstruction/object_surfaces/preview_manifest.json",
        "scene_ir_path": "scene_ir/phase4_scene.json",
        "manifest_sha256": request.manifest_sha256,
        "frame_sequence_digest": request.frame_sequence_digest,
        "camera_reconstruction_sha256": request.camera_reconstruction_sha256,
        "segmentation_tracking_sha256": request.segmentation_tracking_sha256,
        "global_reconstruction_sha256": request.global_reconstruction_sha256,
        "global_mesh_sha256": request.global_mesh_sha256,
        "coordinate_convention": request.coordinate_convention,
        "scale_status": "scale_ambiguous",
        "geometry_status": "partial_observation_supported",
        "hidden_surface_completion": "not_implemented",
        "sim_ready": False,
        "metric_scale_known": False,
        "canonical_gravity_alignment_known": False,
        "hypotheses": hypotheses,
        "partition": partition,
        "conflicts": conflicts,
        "provenance": _provenance(
            request,
            timestamp,
            ["reconstruction/object_surfaces/evidence_manifest.json"],
            scene_confidence,
        ),
        "warnings": [],
    }
    write_json(output_dir / "evidence_manifest.json", evidence_manifest)
    preview_manifest = {
        "global_face_assignment_path": (
            "reconstruction/object_surfaces/previews/global_face_assignment.png"
        ),
        "object_surface_contact_sheet_path": (
            "reconstruction/object_surfaces/previews/object_surface_contact_sheet.png"
        ),
        "reprojection_contact_sheet_path": (
            "reconstruction/object_surfaces/previews/reprojection_contact_sheet.png"
        ),
        "conflict_heatmap_path": ("reconstruction/object_surfaces/previews/conflict_heatmap.png"),
        "object_preview_paths": object_preview_paths,
    }
    write_json(output_dir / "preview_manifest.json", preview_manifest)
    total_runtime = time.monotonic() - total_start
    diagnostics = {
        "schema_version": "0.1.0",
        "track_count": len(hypotheses),
        "accepted_object_count": sum(item["status"] == "accepted" for item in hypotheses),
        "ambiguous_object_count": sum(item["status"] == "ambiguous" for item in hypotheses),
        "unresolved_object_count": sum(item["status"] == "unresolved" for item in hypotheses),
        "global_vertex_count": len(vertices),
        "global_face_count": len(faces),
        "processed_camera_count": len(frames),
        "canonical_mask_count": sum(len(frame.objects) for frame in frames),
        "accepted_face_count": sum(len(values) for values in accepted_sets.values()),
        "ambiguous_face_count": sum(len(values) for values in ambiguous.values()),
        "same_class_conflict_count": len(same_conflict_faces),
        "different_label_overlap_count": len(different_overlap_faces),
        "unassigned_face_ratio": partition["unassigned_face_ratio"],
        "mean_face_support": statistics.mean(all_supports) if all_supports else 0.0,
        "median_face_support": _median(all_supports),
        "mean_reprojection_iou": statistics.mean(all_ious) if all_ious else 0.0,
        "median_reprojection_iou": _median(all_ious),
        "runtime_seconds": total_runtime,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "timings_seconds": {
            "mesh_load": mesh_load_seconds,
            "rasterization": rasterization_seconds,
            "evidence_accumulation": evidence_accumulation_seconds,
            "surface_extraction": surface_extraction_seconds,
            "preview": preview_seconds,
        },
        "warnings": [],
    }
    write_json(output_dir / "diagnostics.json", diagnostics)
    raster_manifest_path = output_dir / "raw" / "rasterization_manifest.json"
    write_json(
        raster_manifest_path,
        {
            "backend": "nvdiffrast",
            "buffers_persisted": False,
            "processed_frames": [
                {
                    "frame_id": frame.frame_id,
                    "processed_face_count": frame.raster.processed_face_count,
                    "culled_face_count": frame.raster.culled_face_count,
                    "near_plane_arbitrary_units": frame.raster.near_plane,
                    "far_plane_arbitrary_units": frame.raster.far_plane,
                }
                for frame in frames
            ],
        },
    )
    raw_paths.extend(
        [
            "reconstruction/object_surfaces/evidence_manifest.json",
            "reconstruction/object_surfaces/diagnostics.json",
            "reconstruction/object_surfaces/preview_manifest.json",
            "reconstruction/object_surfaces/raw/rasterization_manifest.json",
            "reconstruction/object_surfaces/previews/global_face_assignment.png",
            "reconstruction/object_surfaces/previews/object_surface_contact_sheet.png",
            "reconstruction/object_surfaces/previews/reprojection_contact_sheet.png",
            "reconstruction/object_surfaces/previews/conflict_heatmap.png",
            *object_preview_paths.values(),
        ]
    )
    import nvdiffrast

    worker_manifest = {
        "schema_version": "0.1.0",
        "worker_version": __version__,
        "backend": "nvdiffrast",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "nvdiffrast_version": getattr(nvdiffrast, "__version__", "unknown"),
        "device": "cuda",
        "device_name": torch.cuda.get_device_name(0),
        "request_sha256": sha256_file(request_path),
        "manifest_sha256": request.manifest_sha256,
        "frame_sequence_digest": request.frame_sequence_digest,
        "camera_reconstruction_sha256": request.camera_reconstruction_sha256,
        "segmentation_tracking_sha256": request.segmentation_tracking_sha256,
        "global_reconstruction_sha256": request.global_reconstruction_sha256,
        "global_mesh_sha256": request.global_mesh_sha256,
        "processed_registered_frame_ids": [frame.frame_id for frame in frames],
        "global_vertex_count": len(vertices),
        "global_face_count": len(faces),
        "processed_face_count_by_frame": {
            frame.frame_id: frame.raster.processed_face_count for frame in frames
        },
        "culled_face_count_by_frame": {
            frame.frame_id: frame.raster.culled_face_count for frame in frames
        },
        "mesh_load_seconds": mesh_load_seconds,
        "rasterization_seconds": rasterization_seconds,
        "evidence_accumulation_seconds": evidence_accumulation_seconds,
        "surface_extraction_seconds": surface_extraction_seconds,
        "preview_seconds": preview_seconds,
        "runtime_seconds": total_runtime,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_host_memory_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "raw_output_paths": sorted(set(raw_paths)),
        "warnings": [],
    }
    write_json(output_dir / "worker_manifest.json", worker_manifest)
    if sha256_file(mesh_path) != request.global_mesh_sha256:
        raise RuntimeError("global mesh changed during object-lifting execution")
    del rasterizer
    del vertices
    del faces
    torch.cuda.empty_cache()
