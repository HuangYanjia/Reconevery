from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import time
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


def write_preview(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (480, 280), (242, 245, 248))
    draw = ImageDraw.Draw(image)
    draw.text((20, 18), title, fill=(20, 30, 40))
    for x in range(30, 450, 8):
        value = int(80 + 150 * x / 450)
        draw.line((x, 65, x, 250), fill=(value, 120, 255 - value // 2))
    image.save(path)


def write_array(path: Path, width: int, height: int, channels: int, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        f"{width}&{height}&{channels}&".encode("ascii") + struct.pack(f"<{len(values)}f", *values)
    )


def write_ply(path: Path) -> None:
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
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "-0.5 -0.5 2 255 0 0",
                "0.5 -0.5 2 0 255 0",
                "0.5 0.5 2 0 0 255",
                "-0.5 0.5 2 255 255 255",
                "",
            ]
        ),
        encoding="ascii",
    )


def infer(request_path: Path, input_root: Path, output_dir: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    mode = request["patchmatch_configuration"].get("fake_mode", "success")
    if mode == "timeout":
        time.sleep(60)
    if mode == "nonzero":
        print("fake PatchMatch failed", file=sys.stderr)
        return 7
    if mode == "oom":
        print("CUDA out of memory in PatchMatch", file=sys.stderr)
        return 8
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((input_root / request["manifest_path"]).read_text(encoding="utf-8"))
    frame_by_id = {frame["frame_id"]: frame for frame in manifest["frames"]}
    workspace = output_dir / "workspace"
    images = workspace / "images"
    sparse = workspace / "sparse"
    stereo = workspace / "stereo"
    for path in (
        images,
        sparse,
        stereo / "depth_maps",
        stereo / "normal_maps",
        stereo / "consistency_graphs",
    ):
        path.mkdir(parents=True, exist_ok=True)
    patch_cfg = stereo / "patch-match.cfg"
    patch_cfg.write_text(
        "\n".join(
            f"{frame_by_id[frame_id]['relative_path']}\n__auto__, 20"
            for frame_id in request["registered_frame_ids"]
        )
        + "\n",
        encoding="utf-8",
    )
    records = []
    undistortion = []
    depth_records = []
    failed = []
    for index, frame_id in enumerate(request["registered_frame_ids"], 1):
        frame = frame_by_id[frame_id]
        source = input_root / frame["relative_path"]
        filename = f"{index:08d}.png"
        destination = images / filename
        destination.write_bytes(source.read_bytes())
        width, height = Image.open(source).size
        intrinsics = [float(max(width, height)), float(max(width, height)), width / 2, height / 2]
        records.append(
            {
                "frame_id": frame_id,
                "source_relative_path": frame["relative_path"],
                "source_sha256": frame["sha256"],
                "colmap_image_id": index,
                "workspace_filename": f"reconstruction/dense/workspace/images/{filename}",
                "source_dimensions": [width, height],
                "dense_dimensions": [width, height],
                "dense_camera_id": index,
                "dense_camera_model": "PINHOLE",
                "dense_intrinsics": intrinsics,
            }
        )
        map_hash = hashlib.sha256(f"{frame_id}:{width}:{height}:{intrinsics}".encode()).hexdigest()
        undistortion.append(
            {
                "frame_id": frame_id,
                "source_camera_model": "PINHOLE",
                "source_intrinsics": intrinsics,
                "source_distortion": [],
                "source_dimensions": [width, height],
                "dense_camera_model": "PINHOLE",
                "dense_intrinsics": intrinsics,
                "dense_dimensions": [width, height],
                "roi_xywh": [0, 0, width, height],
                "map_hash": map_hash,
                "source_rgb_hash": sha256(source),
                "dense_rgb_hash": sha256(destination),
                "rgb_remap_mean_absolute_error": 0.0,
                "mask_resampling": "nearest",
            }
        )
        if mode == "one_failed_frame" and index == len(request["registered_frame_ids"]):
            failed.append(frame_id)
            continue
        map_width, map_height = min(width, 8), min(height, 6)
        pixels = map_width * map_height
        depth = stereo / "depth_maps" / f"{filename}.geometric.bin"
        normal = stereo / "normal_maps" / f"{filename}.geometric.bin"
        graph = stereo / "consistency_graphs" / f"{filename}.geometric.bin"
        write_array(depth, map_width, map_height, 1, [2.0 + index * 0.01] * pixels)
        write_array(normal, map_width, map_height, 3, [0.0] * (pixels * 2) + [1.0] * pixels)
        source_index = index % len(request["registered_frame_ids"])
        graph.write_bytes(
            f"{map_width}&{map_height}&1&".encode("ascii")
            + struct.pack("<6i", 0, 0, 3, source_index, source_index, source_index)
        )
        depth_records.append(
            {
                "frame_id": frame_id,
                "depth_path": depth.relative_to(input_root).as_posix(),
                "normal_path": normal.relative_to(input_root).as_posix(),
                "consistency_graph_path": graph.relative_to(input_root).as_posix(),
                "dimensions": [map_width, map_height],
                "depth_channels": 1,
                "normal_channels": 3,
                "positive_finite_depth_count": pixels,
                "valid_depth_ratio": 1.0,
                "depth_percentiles": {"p10": 2.0, "p50": 2.0, "p90": 2.0},
                "finite_normal_ratio": 1.0,
                "consistency_valid_pixel_count": 1,
                "mean_consistency_source_count": 3.0,
                "median_consistency_source_count": 3.0,
                "source_view_ids": [source_index],
                "depth_sha256": sha256(depth),
                "normal_sha256": sha256(normal),
                "consistency_sha256": sha256(graph),
                "warnings": [],
            }
        )
    for item in request["selected_sparse_model_files"]:
        source = input_root / item["relative_path"]
        (sparse / Path(item["relative_path"]).name).write_bytes(source.read_bytes())
    fused = output_dir / "fused.ply"
    write_ply(fused)
    selected_hashes = {
        Path(item["relative_path"]).name: item["sha256"]
        for item in request["selected_sparse_model_files"]
    }
    write_json(
        output_dir / "workspace_manifest.json",
        {
            "schema_version": "0.1.0",
            "manifest_sha256": request["manifest_sha256"],
            "frame_sequence_digest": request["frame_sequence_digest"],
            "camera_reconstruction_sha256": request["camera_reconstruction_sha256"],
            "selected_sparse_model_hashes": selected_hashes,
            "registered_frame_ids": request["registered_frame_ids"],
            "frames": records,
            "patch_match_config_path": patch_cfg.relative_to(input_root).as_posix(),
            "patch_match_config_sha256": sha256(patch_cfg),
            "workspace_path": "reconstruction/dense/workspace",
            "coordinate_convention": json.loads(
                (input_root / request["camera_reconstruction_path"]).read_text()
            )["coordinate_convention"],
        },
    )
    write_json(
        output_dir / "undistortion_manifest.json",
        {
            "schema_version": "0.1.0",
            "policy": "official_colmap_image_undistorter",
            "records": undistortion,
            "rgb_remap_tolerance": request["undistortion_configuration"]["rgb_remap_tolerance"],
        },
    )
    write_json(
        output_dir / "depth_manifest.json",
        {
            "schema_version": "0.1.0",
            "map_type": "geometric",
            "records": depth_records,
            "failed_frame_ids": failed,
        },
    )
    convention = json.loads((input_root / request["camera_reconstruction_path"]).read_text())[
        "coordinate_convention"
    ]
    write_json(
        output_dir / "fusion.json",
        {
            "schema_version": "0.1.0",
            "fused_point_cloud_path": "reconstruction/dense/fused.ply",
            "fused_point_cloud_sha256": sha256(fused),
            "point_count": 4,
            "normal_count": 0,
            "bounds_min": [-0.5, -0.5, 2.0],
            "bounds_max": [0.5, 0.5, 2.0],
            "scene_diagonal_arbitrary_units": math.sqrt(2.0),
            "coordinate_convention": convention,
            "scale_status": "scale_ambiguous",
        },
    )
    write_json(
        output_dir / "diagnostics.json",
        {
            "schema_version": "0.1.0",
            "registered_frame_count": len(request["registered_frame_ids"]),
            "successful_depth_map_count": len(depth_records),
            "failed_depth_map_count": len(failed),
            "fused_point_count": 4,
            "image_undistortion_seconds": 0.01,
            "patchmatch_seconds": 0.02,
            "fusion_seconds": 0.01,
            "total_runtime_seconds": 0.04,
            "peak_gpu_memory_bytes": 0,
            "peak_host_memory_bytes": 1024,
            "warnings": [],
        },
    )
    commands = {
        "image_undistorter": ["colmap", "image_undistorter"],
        "patch_match_stereo": ["colmap", "patch_match_stereo"],
        "stereo_fusion": ["colmap", "stereo_fusion"],
    }
    write_json(
        output_dir / "worker_manifest.json",
        {
            "schema_version": "0.1.0",
            "worker_version": "0.1.0",
            "official_colmap_repository": request["official_colmap_repository"],
            "official_colmap_version": request["official_colmap_version"],
            "official_colmap_commit": request["official_colmap_commit"],
            "colmap_license": "BSD-3-Clause",
            "build_configuration": {"backend": "fake"},
            "cuda_version": None,
            "compiler": None,
            "request_sha256": sha256(request_path),
            "manifest_sha256": request["manifest_sha256"],
            "frame_sequence_digest": request["frame_sequence_digest"],
            "camera_reconstruction_sha256": request["camera_reconstruction_sha256"],
            "registered_frame_ids": request["registered_frame_ids"],
            "command_arguments": commands,
            "return_codes": {name: 0 for name in commands},
            "runtime_seconds": 0.04,
            "peak_gpu_memory_bytes": 0,
            "raw_output_paths": ["reconstruction/dense/workspace/stereo/patch-match.cfg"],
            "warnings": [],
        },
    )
    for name in (
        "depth_contact_sheet",
        "normal_contact_sheet",
        "consistency_contact_sheet",
        "fused_point_cloud",
        "camera_dense_coverage",
    ):
        write_preview(output_dir / "previews" / f"{name}.png", name.replace("_", " "))
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
