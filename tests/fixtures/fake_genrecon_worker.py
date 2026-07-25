from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw

OFFICIAL_REPOSITORY = "https://github.com/kasothaphie/GenRecon"
OFFICIAL_COMMIT = "eaf1468118d20469d17079a4a19737297d2ef87b"
SUBMODULES = {"o-voxel/third_party/eigen": "21e4582d1739107337a03460c81412981130373e"}
CHECKPOINT_URLS = {
    "sparse_structure": "https://kaldir.vc.cit.tum.de/genrecon/sparse_structure.pt",
    "shape_slat": "https://kaldir.vc.cit.tum.de/genrecon/shape_slat.pt",
    "texture_slat": "https://kaldir.vc.cit.tum.de/genrecon/texture_slat.pt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_ply(path: Path, mode: str) -> None:
    if mode == "zero_mesh":
        path.write_text(
            "ply\nformat ascii 1.0\nelement vertex 0\n"
            "property float x\nproperty float y\nproperty float z\n"
            "element face 0\nproperty list uchar int vertex_indices\nend_header\n",
            encoding="ascii",
        )
        return
    nan_value = "nan" if mode == "nonfinite_mesh" else "0"
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 8\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 12\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
        f"{nan_value} 0 0\n"
        "1 0 0\n"
        "1 1 0\n"
        "0 1 0\n"
        "0 0 1\n"
        "1 0 1\n"
        "1 1 1\n"
        "0 1 1\n"
        "3 0 1 2\n3 0 2 3\n"
        "3 4 6 5\n3 4 7 6\n"
        "3 0 4 5\n3 0 5 1\n"
        "3 1 5 6\n3 1 6 2\n"
        "3 2 6 7\n3 2 7 3\n"
        "3 3 7 4\n3 3 4 0\n",
        encoding="ascii",
    )


def write_glb(path: Path, invalid: bool = False) -> None:
    if invalid:
        path.write_bytes(b"not-a-glb")
        return
    document = {
        "asset": {"version": "2.0", "generator": "fake_genrecon_worker"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"material": 0}]}],
        "materials": [{"name": "fake_pbr"}],
        "images": [{"uri": "data:image/png;base64,"}],
        "textures": [{"source": 0}],
    }
    json_payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_payload += b" " * ((4 - len(json_payload) % 4) % 4)
    total_length = 12 + 8 + len(json_payload)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total_length)
        + struct.pack("<II", len(json_payload), 0x4E4F534A)
        + json_payload
    )


def checkpoint_records(request: dict[str, object]) -> list[dict[str, object]]:
    paths = request["checkpoint_paths"]
    hashes = request["checkpoint_hashes"]
    assert isinstance(paths, dict)
    assert isinstance(hashes, dict)
    records = []
    for checkpoint_id in ("sparse_structure", "shape_slat", "texture_slat"):
        path = Path(str(paths[checkpoint_id]))
        records.append(
            {
                "checkpoint_id": checkpoint_id,
                "source_url": CHECKPOINT_URLS[checkpoint_id],
                "local_filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": str(hashes[checkpoint_id]),
                "resolved_at": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
                "access_mode": "fake",
            }
        )
    return records


def healthcheck(config_path: Path) -> int:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    for path in payload.get("checkpoint_paths", {}).values():
        checkpoint = Path(path)
        if not checkpoint.is_file():
            print(f"checkpoint missing: {checkpoint}", file=sys.stderr)
            return 2
    print(
        json.dumps(
            {
                "available": True,
                "backend": "fake_worker",
                "official_repository": payload.get("official_repository"),
                "official_code_commit": payload.get("official_code_commit"),
                "submodule_commits": payload.get("submodule_commits"),
            },
            sort_keys=True,
        )
    )
    return 0


def infer(request_path: Path, output_dir: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    mode = request["reconstruction_parameters"].get("fake_mode", "success")
    if mode == "nonzero_exit":
        print("simulated GenRecon failure", file=sys.stderr)
        return 17
    if mode == "oom":
        print("CUDA out of memory during GenRecon inference", file=sys.stderr)
        return 18
    if mode == "checkpoint_missing":
        print("checkpoint missing", file=sys.stderr)
        return 19
    if mode == "cuda_extension_failure":
        print("CUDA extension import failed: undefined symbol", file=sys.stderr)
        return 20
    if mode == "dinov3_unauthorized":
        print("403 GatedRepoError: DINOv3 access is not authorized", file=sys.stderr)
        return 21
    if mode == "timeout":
        time.sleep(120)
    if mode == "interruption":
        os.kill(os.getpid(), 2)

    output_dir.mkdir(parents=True, exist_ok=True)
    selected = list(request["eligible_frame_ids"])[: int(request["requested_max_views"])]
    registered = list(request["registered_frame_ids"])
    if mode == "frame_order_mismatch":
        selected.reverse()
    if mode == "registered_mismatch":
        registered = registered[:-1]

    transform = {
        "strategy": request["working_transform_strategy"],
        "matrix_colmap_to_working": [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "matrix_working_to_colmap": [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "determinant": 1.0,
        "roundtrip_max_error": 0.0,
        "semantic_status": "internal_unoriented_preprocessing",
    }
    if mode == "bad_transform":
        transform["matrix_working_to_colmap"][0][0] = 1.0

    write_ply(output_dir / "mesh.ply", mode)
    if mode != "missing_scene":
        write_glb(output_dir / "scene.glb", invalid=mode == "invalid_glb")
    if mode != "missing_intermediate":
        (output_dir / "to_glb_inputs.pt").write_bytes(b"fake-to-glb-inputs\n")
    (output_dir / "chunk_inputs.pt").write_bytes(b"fake-chunk-inputs\n")
    write_json(
        output_dir / "args.json",
        {
            "mode": "Iphone",
            "num_imgs_per_scene": len(selected),
            "seed": request["seed"],
        },
    )
    write_json(
        output_dir / "chunk_transforms.json",
        {
            "M_colmap_to_genrecon_working": transform["matrix_colmap_to_working"],
            "M_genrecon_working_to_colmap": transform["matrix_working_to_colmap"],
        },
    )
    layout = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(layout)
    draw.rectangle((40, 40, 180, 180), outline="black", fill="#a8dadc")
    layout.save(output_dir / "chunk_layout.png")
    write_ply(output_dir / "clean_points.ply", "success")
    write_ply(output_dir / "coords_000.ply", "success")

    records = checkpoint_records(request)
    if mode == "wrong_checkpoint":
        records[0]["sha256"] = "0" * 64
    manifest = {
        "schema_version": "0.1.0",
        "official_repository": OFFICIAL_REPOSITORY,
        "official_code_commit": ("0" * 40 if mode == "wrong_commit" else OFFICIAL_COMMIT),
        "submodule_commits": SUBMODULES,
        "official_license": "MIT",
        "checkpoint_records": records,
        "runtime_model_repository": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "runtime_model_revision": "1" * 40,
        "worker_version": "0.1.1",
        "python_version": sys.version.split()[0],
        "torch_version": None,
        "torchvision_version": None,
        "cuda_version": None,
        "device_name": "deterministic fake H100",
        "device": "fake",
        "precision": "float16",
        "seed": request["seed"],
        "request_sha256": sha256(request_path),
        "frame_sequence_digest": request["frame_sequence_digest"],
        "camera_package_sha256": request["camera_package_sha256"],
        "registered_frame_ids": registered,
        "selected_frame_ids": selected,
        "working_transform": transform,
        "reconstruct_return_code": 0,
        "glb_conversion_return_code": 0,
        "runtime_seconds": 0.25,
        "peak_gpu_memory_bytes": 1024,
        "raw_output_paths": [
            "reconstruction/global/raw/args.json",
            "reconstruction/global/raw/chunk_transforms.json",
            "reconstruction/global/raw/chunk_layout.png",
            "reconstruction/global/raw/clean_points.ply",
            "reconstruction/global/raw/mesh.ply",
            "reconstruction/global/raw/coords_000.ply",
            "reconstruction/global/raw/to_glb_inputs.pt",
            "reconstruction/global/raw/chunk_inputs.pt",
            "reconstruction/global/raw/scene.glb",
        ],
        "warnings": [],
    }
    if mode == "malformed_manifest":
        write_json(output_dir / "worker_manifest.json", {"bad": True})
    else:
        write_json(output_dir / "worker_manifest.json", manifest)

    chunks_after = 0 if mode == "empty_chunks" else 1
    write_json(
        output_dir / "worker_diagnostics.json",
        {
            "initial_sparse_points": 64,
            "cleaned_sparse_points": 60,
            "robust_bounds_min": [-1.0, -1.0, -1.0],
            "robust_bounds_max": [1.0, 1.0, 1.0],
            "scene_diagonal_arbitrary_units": math.sqrt(12.0),
            "chunks_before_filtering": 2,
            "chunks_after_filtering": chunks_after,
            "chunks": [
                {
                    "chunk_id": "0",
                    "point_count": 60,
                    "selected_view_count": len(selected),
                    "dropped": False,
                    "reason": None,
                }
            ],
            "chosen_parameters": request["reconstruction_parameters"],
            "warnings": [],
        },
    )
    if mode == "path_escape":
        (output_dir / "escape").symlink_to("/tmp")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    health = subparsers.add_parser("healthcheck")
    health.add_argument("--config", type=Path, required=True)
    inference = subparsers.add_parser("infer")
    inference.add_argument("--request", type=Path, required=True)
    inference.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "healthcheck":
        return healthcheck(args.config)
    return infer(args.request, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
