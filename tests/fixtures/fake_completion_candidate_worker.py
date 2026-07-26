from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_mesh(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 4\nproperty list uchar int vertex_indices\nend_header\n"
        "-0.5 -0.5 0\n0.5 -0.5 0\n0 0.5 0\n0 0 1\n"
        "3 0 1 2\n3 0 1 3\n3 1 2 3\n3 2 0 3\n",
        encoding="ascii",
    )


def write_gaussians(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float opacity\nend_header\n"
        "-0.2 0 0 0.9\n0.2 0 0 0.9\n0 0.2 0 0.9\n0 0 0.2 0.9\n",
        encoding="ascii",
    )


def generate(request_path: Path, output_dir: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    mode = request["generation_configuration"].get("fake_mode", "success")
    if mode == "timeout":
        time.sleep(60)
    if mode == "oom":
        print("out of memory", file=sys.stderr)
        return 9
    if mode in {"nonzero", "gated_access_failure"}:
        print(
            "official checkpoint access denied" if mode == "gated_access_failure" else "failed",
            file=sys.stderr,
        )
        return 7
    candidate_id = request["generation_configuration"]["candidate_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = request["backend"]
    if backend == "sam3d_objects":
        asset = output_dir / "native" / "gaussians.ply"
        write_gaussians(asset)
        visual_asset = output_dir / "native" / "visual_asset.ply"
        write_mesh(visual_asset)
        native_format = "gaussian_splat_ply"
        gaussian_count = 4
        vertex_count = face_count = 4
        renderer = "official_sam3d_gaussian_renderer"
    else:
        asset = output_dir / "native" / "visual_asset.glb"
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(b"glTF\x02\x00\x00\x00\x0c\x00\x00\x00")
        native_format = "pbr_glb"
        gaussian_count = None
        vertex_count, face_count = 4, 4
        renderer = "trellis2_mesh_renderer"
    if mode == "empty_geometry":
        asset.write_bytes(b"")
    if mode == "malformed_candidate":
        (output_dir / "candidate.json").write_text("{bad", encoding="utf-8")
        return 0
    provenance = {
        "adapter_name": f"{backend}_candidate_generation",
        "adapter_version": "0.1.0",
        "configuration": {
            "official_code_commit": request["official_code_commit"],
            "checkpoint_revision": request["checkpoint_revision"],
            "seed": request["generation_seed"],
        },
        "input_artifact_paths": [request["anchor_crop_path"]],
        "output_artifact_paths": [asset.relative_to(output_dir.parents[4]).as_posix()],
        "timestamp": "2025-01-01T00:00:00Z",
        "confidence": {
            "score": 0.5,
            "method": "unvalidated_generated_candidate",
            "notes": "held-out evaluation occurs downstream",
        },
        "source": "generated",
    }
    relative_asset = asset.relative_to(output_dir.parents[4]).as_posix()
    asset_id = "native_gaussian" if backend == "sam3d_objects" else "official_pbr_glb"
    native_assets = [
        {
            "asset_id": asset_id,
            "relative_path": relative_asset,
            "sha256": sha256(asset),
            "format": native_format,
            "size_bytes": asset.stat().st_size,
            "role": "official_native_output",
        }
    ]
    selected_asset_id = asset_id
    selected_asset_path = relative_asset
    if backend == "sam3d_objects":
        relative_visual = visual_asset.relative_to(output_dir.parents[4]).as_posix()
        native_assets.append(
            {
                "asset_id": "official_visual_glb",
                "relative_path": relative_visual,
                "sha256": sha256(visual_asset),
                "format": "mesh_ply",
                "size_bytes": visual_asset.stat().st_size,
                "role": "official_optional_visual_glb",
            }
        )
        selected_asset_id = "official_visual_glb"
        selected_asset_path = relative_visual
    candidate = {
        "candidate_id": candidate_id,
        "object_id": request["object_id"],
        "semantic_label": request["semantic_label"],
        "backend": backend,
        "anchor_frame_id": request["anchor_frame_id"],
        "generation_seed": request["generation_seed"],
        "native_assets": native_assets,
        "registration_asset_id": selected_asset_id,
        "registration_asset_path": selected_asset_path,
        "evaluation_asset_id": selected_asset_id,
        "evaluation_asset_path": selected_asset_path,
        "selection_asset_id": selected_asset_id,
        "selection_asset_path": selected_asset_path,
        "native_coordinate_convention": "backend_canonical_object",
        "native_bounds_min": [-0.5, -0.5, 0],
        "native_bounds_max": [0.5, 0.5, 1],
        "native_center": [0, 0, 0.5],
        "native_scale": 1.0,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "material_count": 1 if backend == "trellis2" else None,
        "texture_count": 1 if backend == "trellis2" else None,
        "gaussian_count": gaussian_count,
        "backend_predicted_layout": {"scale": 1.0, "rotation_xyzw": [0, 0, 0, 1]},
        "backend_anchor_camera": (
            {
                "width": 1024,
                "height": 1024,
                "normalized_intrinsics": [1.0, 0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0],
                "pixel_intrinsics": [1024.0, 1024.0, 512.0, 512.0],
                "camera_axes": "x_right_y_down_z_forward",
                "source": "official_pointmap_intrinsics",
            }
            if backend == "sam3d_objects"
            else None
        ),
        "render_capability": {
            "renderer": renderer,
            "supports_rgba": True,
            "supports_depth": True,
            "supports_normals": backend == "trellis2",
            "camera_axes": "x_right_y_down_z_forward",
        },
        "sampling_method": (
            "opacity_filtered_gaussian_centers"
            if backend == "sam3d_objects"
            else "area_weighted_triangle_sampling"
        ),
        "generation_runtime_seconds": 0.01,
        "peak_gpu_memory_bytes": 0,
        "license_record": request["license_policy"],
        "provenance": provenance,
        "warnings": [],
    }
    write_json(output_dir / "candidate.json", candidate)
    write_json(
        output_dir / "worker_manifest.json",
        {
            "worker_name": f"{backend}_worker",
            "worker_version": "0.1.0",
            "action": "generate",
            "backend": "fake",
            "request_sha256": sha256(request_path),
            "official_repository": request["official_repository"],
            "official_code_commit": request["official_code_commit"],
            "checkpoint_repository": request["checkpoint_repository"],
            "checkpoint_revision": request["checkpoint_revision"],
            "checkpoint_hashes": request["checkpoint_hashes"],
            "runtime_seconds": 0.01,
            "peak_gpu_memory_bytes": 0,
            "peak_host_memory_bytes": 0,
            "warnings": [],
        },
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["healthcheck", "generate", "render"])
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.action == "healthcheck":
        print('{"available": true, "backend": "fake"}')
        return
    if args.output_dir is None:
        parser.error("--output-dir is required")
    raise SystemExit(generate(args.request, args.output_dir))


if __name__ == "__main__":
    main()
