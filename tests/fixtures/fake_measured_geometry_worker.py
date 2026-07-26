from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_points(path: Path, offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "property float nx",
                "property float ny",
                "property float nz",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "element face 0",
                "property list uchar int vertex_indices",
                "end_header",
                f"{offset - 0.1} -0.1 2 0 0 -1 220 70 50",
                f"{offset + 0.1} -0.1 2 0 0 -1 220 70 50",
                f"{offset + 0.1} 0.1 2 0 0 -1 220 70 50",
                f"{offset - 0.1} 0.1 2 0 0 -1 220 70 50",
                "",
            ]
        ),
        encoding="ascii",
    )


def write_mesh(path: Path, offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "element face 2",
                "property list uchar int vertex_indices",
                "end_header",
                f"{offset - 0.1} -0.1 2",
                f"{offset + 0.1} -0.1 2",
                f"{offset + 0.1} 0.1 2",
                f"{offset - 0.1} 0.1 2",
                "3 0 1 2",
                "3 0 2 3",
                "",
            ]
        ),
        encoding="ascii",
    )


def preview(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (520, 300), (245, 246, 247))
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), title, fill=(20, 30, 40))
    draw.ellipse((145, 75, 375, 265), outline=(30, 140, 210), width=4)
    draw.line((80, 260, 440, 80), fill=(220, 80, 50), width=3)
    image.save(path)


def infer(request_path: Path, input_root: Path, output_dir: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    mode = request["backprojection_configuration"].get("fake_mode", "success")
    if mode == "timeout":
        time.sleep(60)
    if mode == "nonzero":
        print("fake measured geometry failed", file=sys.stderr)
        return 7
    if mode == "oom":
        print("out of memory while fusing surfels", file=sys.stderr)
        return 8
    manifest = json.loads((input_root / request["manifest_path"]).read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    hypotheses = []
    total_raw = total_validated = total_surfels = 0
    for index, track in enumerate(request["object_tracks"]):
        object_id = track["object_id"]
        frames = list(track["mask_paths_by_frame"])
        unresolved = mode == "unresolved" or not frames
        if mode == "one_view_only" and index == 0:
            frames = frames[:1]
            unresolved = True
        if mode == "multiple_objects" and index > 1:
            unresolved = True
        observations = [
            {
                "frame_id": frame_id,
                "registered": True,
                "raw_sample_count": 48,
                "validated_sample_count": 32,
                "supporting_view_count": max(1, len(frames) - 1),
                "contradicting_view_count": 0,
                "depth_residual_median": 0.01,
                "mask_support_fraction": 0.82,
            }
            for frame_id in frames
        ]
        provenance = {
            "adapter_name": "measured_object_geometry",
            "adapter_version": "0.1.0",
            "configuration": {"backend": "fake", "mode": mode},
            "input_artifact_paths": [
                request["camera_reconstruction_path"],
                request["segmentation_tracking_path"],
                request["depth_manifest_path"],
            ],
            "output_artifact_paths": [],
            "timestamp": manifest["provenance"]["timestamp"],
            "confidence": {
                "score": 0.0 if unresolved else 0.78,
                "method": "measured_multiview_depth_support",
                "notes": "visible measured surface only",
            },
            "source": "measured",
        }
        if unresolved:
            hypotheses.append(
                {
                    "object_id": object_id,
                    "semantic_label": track["semantic_label"],
                    "prompt_id": track["prompt_id"],
                    "asset_type_hint": track["asset_type_hint"],
                    "status": "unresolved",
                    "reason": "insufficient_multiview_dense_support",
                    "registered_mask_observations": len(frames),
                    "observations_with_valid_dense_depth": len(frames),
                    "raw_measured_sample_count": 0,
                    "validated_sample_count": 0,
                    "fused_surfel_count": 0,
                    "supporting_view_count": len(frames),
                    "point_cloud": None,
                    "surfel_cloud": None,
                    "observed_surface": None,
                    "observations": observations,
                    "depth_consistency": 0.0,
                    "normal_consistency": 0.0,
                    "reprojection_precision": 0.0,
                    "reprojection_recall": 0.0,
                    "reprojection_iou": 0.0,
                    "visible_mask_coverage": 0.0,
                    "connected_component_count": 0,
                    "measurement_confidence": 0.0,
                    "completeness_confidence": 0.0,
                    "surfel_spacing": None,
                    "geometry_source": "measured",
                    "geometry_status": "partial_measured",
                    "hidden_surface_completion": "not_implemented",
                    "watertight": False,
                    "sim_ready": False,
                    "metric_scale_known": False,
                    "canonical_gravity_alignment_known": False,
                    "coordinate_convention": request["coordinate_convention"],
                    "scale_status": "scale_ambiguous",
                    "provenance": provenance,
                    "warnings": ["no reliable multi-view measured surface"],
                }
            )
            continue
        object_dir = output_dir / "objects" / object_id
        points = object_dir / "measured_points.ply"
        surfels = object_dir / "surfels.ply"
        surface = object_dir / "observed_surface.ply"
        write_points(points, index * 0.4)
        write_points(surfels, index * 0.4)
        write_mesh(surface, index * 0.4)
        with zipfile.ZipFile(object_dir / "surfels.npz", "w") as archive:
            archive.writestr("README.txt", "fake deterministic surfel fixture\n")
        write_json(
            object_dir / "view_support.json",
            {"object_id": object_id, "supporting_frame_ids": frames},
        )
        raw_count = 48 * len(frames)
        validated = 32 * len(frames)
        total_raw += raw_count
        total_validated += validated
        total_surfels += 4
        provenance["output_artifact_paths"] = [
            f"reconstruction/measured_objects/objects/{object_id}/measured_points.ply",
            f"reconstruction/measured_objects/objects/{object_id}/surfels.ply",
        ]
        hypotheses.append(
            {
                "object_id": object_id,
                "semantic_label": track["semantic_label"],
                "prompt_id": track["prompt_id"],
                "asset_type_hint": track["asset_type_hint"],
                "status": "accepted" if len(frames) >= 2 else "partial",
                "reason": None,
                "registered_mask_observations": len(frames),
                "observations_with_valid_dense_depth": len(frames),
                "raw_measured_sample_count": raw_count,
                "validated_sample_count": validated,
                "fused_surfel_count": 4,
                "supporting_view_count": len(frames),
                "point_cloud": {
                    "relative_path": points.relative_to(input_root).as_posix(),
                    "sha256": sha256(points),
                    "point_count": 4,
                    "has_normals": True,
                    "has_colors": True,
                },
                "surfel_cloud": {
                    "relative_path": surfels.relative_to(input_root).as_posix(),
                    "sha256": sha256(surfels),
                    "point_count": 4,
                    "has_normals": True,
                    "has_colors": True,
                },
                "observed_surface": {
                    "relative_path": surface.relative_to(input_root).as_posix(),
                    "sha256": sha256(surface),
                    "vertex_count": 4,
                    "face_count": 2,
                    "surface_type": "observed_depth_triangulation",
                    "watertight": False,
                },
                "observations": observations,
                "depth_consistency": 0.92,
                "normal_consistency": 0.88,
                "reprojection_precision": 0.85,
                "reprojection_recall": 0.73,
                "reprojection_iou": 0.64,
                "visible_mask_coverage": 0.73,
                "connected_component_count": 1,
                "measurement_confidence": 0.78,
                "completeness_confidence": 0.0,
                "surfel_spacing": {
                    "method": "coordinate_hash_kdtree_nearest_neighbor_v1",
                    "source_point_count": validated,
                    "sampled_point_count": 4,
                    "nearest_neighbor_p10": 0.2,
                    "nearest_neighbor_median": 0.2,
                    "nearest_neighbor_p90": 0.2,
                    "voxel_size": 0.3,
                    "coordinate_hash_digest": hashlib.sha256(
                        f"{object_id}:spacing".encode()
                    ).hexdigest(),
                },
                "geometry_source": "measured",
                "geometry_status": "partial_measured",
                "hidden_surface_completion": "not_implemented",
                "watertight": False,
                "sim_ready": False,
                "metric_scale_known": False,
                "canonical_gravity_alignment_known": False,
                "coordinate_convention": request["coordinate_convention"],
                "scale_status": "scale_ambiguous",
                "provenance": provenance,
                "warnings": ["visible surface only; hidden geometry is not reconstructed"],
            }
        )
        preview(output_dir / "previews" / "objects" / f"{object_id}.png", object_id)
    write_json(
        output_dir / "geometry_manifest.json",
        {
            "schema_version": "0.1.0",
            "manifest_sha256": request["manifest_sha256"],
            "frame_sequence_digest": request["frame_sequence_digest"],
            "camera_reconstruction_sha256": request["camera_reconstruction_sha256"],
            "segmentation_tracking_sha256": request["segmentation_tracking_sha256"],
            "dense_workspace_manifest_sha256": request["dense_workspace_manifest_sha256"],
            "undistortion_manifest_sha256": request["undistortion_manifest_sha256"],
            "depth_manifest_sha256": request["depth_manifest_sha256"],
            "hypotheses": hypotheses,
            "coordinate_convention": request["coordinate_convention"],
            "scale_status": "scale_ambiguous",
            "generated_geometry_used_as_source": False,
        },
    )
    statuses = [item["status"] for item in hypotheses]
    write_json(
        output_dir / "diagnostics.json",
        {
            "schema_version": "0.1.0",
            "track_count": len(hypotheses),
            "accepted_object_count": statuses.count("accepted"),
            "partial_object_count": statuses.count("partial"),
            "unresolved_object_count": statuses.count("unresolved"),
            "raw_sample_count": total_raw,
            "validated_sample_count": total_validated,
            "fused_surfel_count": total_surfels,
            "mask_mapping_seconds": 0.01,
            "backprojection_seconds": 0.01,
            "multiview_validation_seconds": 0.01,
            "surfel_fusion_seconds": 0.01,
            "observed_mesh_seconds": 0.01,
            "preview_seconds": 0.01,
            "total_runtime_seconds": 0.06,
            "peak_gpu_memory_bytes": 0,
            "peak_host_memory_bytes": 2048,
            "warnings": [],
        },
    )
    write_json(
        output_dir / "worker_manifest.json",
        {
            "schema_version": "0.1.0",
            "worker_version": "0.1.0",
            "backend": "fake",
            "request_sha256": sha256(request_path),
            "manifest_sha256": request["manifest_sha256"],
            "frame_sequence_digest": request["frame_sequence_digest"],
            "camera_reconstruction_sha256": request["camera_reconstruction_sha256"],
            "segmentation_tracking_sha256": request["segmentation_tracking_sha256"],
            "depth_manifest_sha256": request["depth_manifest_sha256"],
            "runtime_seconds": 0.06,
            "peak_gpu_memory_bytes": 0,
            "peak_host_memory_bytes": 2048,
            "raw_output_paths": [],
            "warnings": [],
        },
    )
    for name in (
        "measured_object_contact_sheet",
        "depth_mask_contact_sheet",
        "reprojection_contact_sheet",
        "object_point_clouds",
    ):
        preview(output_dir / "previews" / f"{name}.png", name.replace("_", " "))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    health = subparsers.add_parser("healthcheck")
    health.add_argument("--config", type=Path, required=True)
    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--request", type=Path, required=True)
    infer_parser.add_argument("--input-root", type=Path, required=True)
    infer_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "healthcheck":
        print(json.dumps({"available": True, "backend": "fake", "worker_version": "0.1.0"}))
        return 0
    return infer(args.request, args.input_root, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
