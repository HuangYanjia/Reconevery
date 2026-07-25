from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import trimesh

from genrecon_worker.checkpoint_loader import sha256_file, verify_checkpoints
from genrecon_worker.commit_verification import verify_checkout
from genrecon_worker.glb_conversion import conversion_command
from genrecon_worker.input_package import prepared_scene
from genrecon_worker.mesh_inspection import transform_outputs_to_colmap
from genrecon_worker.runtime_assets import (
    DINOV3_REPOSITORY,
    prepare_dinov3_runtime_asset,
    resolve_dinov3_revision,
)
from genrecon_worker.schema import InferenceRequest
from genrecon_worker.version import OFFICIAL_LICENSE, OFFICIAL_SUBMODULES, WORKER_VERSION


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gpu_process_memory_bytes(process_id: int) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    if result.returncode != 0:
        return 0
    memory_mib = 0
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
            used = int(fields[1])
        except ValueError:
            continue
        if pid == process_id:
            memory_mib = max(memory_mib, used)
    return memory_mib * 1024 * 1024


def _run_with_gpu_monitor(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, int]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    peak = 0
    stop = threading.Event()
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )

        def monitor() -> None:
            nonlocal peak
            while not stop.wait(0.25):
                try:
                    peak = max(peak, _gpu_process_memory_bytes(process.pid))
                except (OSError, subprocess.SubprocessError):
                    continue

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            return_code = process.wait()
        finally:
            stop.set()
            thread.join(timeout=2)
    return return_code, peak


def _official_command(
    checkout: Path,
    prepared_root: Path,
    output_dir: Path,
    request: InferenceRequest,
) -> list[str]:
    parameters = request.reconstruction_parameters
    command = [
        sys.executable,
        str(checkout / "reconstruct_scene.py"),
        "--mode",
        "Iphone",
        "--path",
        str(prepared_root),
        "--output_path",
        str(output_dir),
        "--ss_ckpt",
        request.checkpoint_paths["sparse_structure"],
        "--shape_ckpt",
        request.checkpoint_paths["shape_slat"],
        "--tex_ckpt",
        request.checkpoint_paths["texture_slat"],
        "--num_imgs_per_scene",
        str(min(request.requested_max_views, len(request.eligible_frame_ids))),
        "--chunk_size_factor",
        str(parameters["chunk_size_factor"]),
        "--stat_std_ratio",
        str(parameters["stat_std_ratio"]),
        "--radius_nb_points",
        str(parameters["radius_nb_points"]),
        "--radius_m",
        str(parameters["radius_m"]),
        "--min_points_per_chunk",
        str(parameters["min_points_per_chunk"]),
        "--pipeline_config",
        str(checkout / request.pipeline_config),
        "--proj_batch_voxels",
        str(parameters["proj_batch_voxels"]),
        "--seed",
        str(request.seed),
        "--colmap_subdir",
        "colmap",
    ]
    if parameters.get("skip_point_cleaning"):
        command.append("--skip_point_cleaning")
    return command


def _selected_frame_ids(
    output_dir: Path,
    request: InferenceRequest,
) -> list[str]:
    camera_log = output_dir / "cameras.json"
    if not camera_log.is_file():
        raise RuntimeError("official GenRecon did not write cameras.json")
    payload = json.loads(camera_log.read_text(encoding="utf-8"))
    names = {
        Path(item["img_path"]).name
        for item in payload.get("scene", [])
        if isinstance(item, dict) and isinstance(item.get("img_path"), str)
    }
    name_by_id = {
        frame_id: Path(request.normalized_frame_paths[frame_id]).name
        for frame_id in request.eligible_frame_ids
    }
    selected = [
        frame_id for frame_id in request.eligible_frame_ids if name_by_id[frame_id] in names
    ]
    if not selected:
        raise RuntimeError("official GenRecon selected no registered input views")
    return selected


def _point_count(path: Path) -> int:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.PointCloud):
        return len(loaded.vertices)
    if isinstance(loaded, trimesh.Trimesh):
        return len(loaded.vertices)
    if isinstance(loaded, trimesh.Scene):
        return sum(len(geometry.vertices) for geometry in loaded.geometry.values())
    return 0


def _chunk_diagnostics(
    output_dir: Path,
    points_working: np.ndarray,
    selected_view_count: int,
) -> tuple[list[dict[str, object]], int]:
    transform_path = output_dir / "chunk_transforms.json"
    if not transform_path.is_file():
        raise RuntimeError("official GenRecon did not write chunk_transforms.json")
    payload = json.loads(transform_path.read_text(encoding="utf-8"))
    raw_chunks = payload.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise RuntimeError("official GenRecon produced an empty chunk transform set")
    homogeneous = np.concatenate(
        [points_working, np.ones((len(points_working), 1), dtype=np.float64)],
        axis=1,
    )
    chunks: list[dict[str, object]] = []
    for item in raw_chunks:
        if not isinstance(item, dict):
            raise RuntimeError("official chunk transform entry is malformed")
        matrix = np.asarray(item["M_original_to_chunk"], dtype=np.float64)
        chunk_points = (matrix @ homogeneous.T).T[:, :3]
        inside = np.all(np.abs(chunk_points) <= 0.5 + 1e-9, axis=1)
        chunks.append(
            {
                "chunk_id": str(item["index"]),
                "point_count": int(inside.sum()),
                "selected_view_count": selected_view_count,
                "dropped": False,
                "reason": None,
            }
        )
    return chunks, len(raw_chunks)


def run_inference(
    request_path: Path,
    output_dir: Path,
) -> None:
    started = time.monotonic()
    request = InferenceRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
    checkout = Path(request.official_checkout_path).resolve()
    submodules = verify_checkout(
        checkout,
        request.official_code_commit,
        OFFICIAL_SUBMODULES,
    )
    checkpoint_records = verify_checkpoints(
        request.checkpoint_paths,
        request.checkpoint_hashes,
    )
    dinov3_revision = prepare_dinov3_runtime_asset(resolve_dinov3_revision())
    root = request_path.resolve().parents[2]
    output_dir = output_dir.resolve()
    output_dir.relative_to(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs = output_dir / "logs"
    logs.mkdir(exist_ok=True)

    environment = dict(os.environ)
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(checkout) if not python_path else f"{checkout}{os.pathsep}{python_path}"
    )
    environment["PYTHONHASHSEED"] = str(request.seed)
    environment["HF_HUB_OFFLINE"] = "1"
    peak_gpu_memory = 0
    with prepared_scene(root, request) as (
        prepared_root,
        working_to_colmap,
        transform_record,
        original_points,
    ):
        colmap_to_working = np.asarray(
            transform_record["matrix_colmap_to_working"],
            dtype=np.float64,
        )
        points_working = (colmap_to_working[:3, :3] @ original_points.T).T + colmap_to_working[
            :3, 3
        ]
        reconstruct_return_code, reconstruct_peak = _run_with_gpu_monitor(
            _official_command(checkout, prepared_root, output_dir, request),
            cwd=checkout,
            environment=environment,
            stdout_path=logs / "reconstruct_scene.stdout.log",
            stderr_path=logs / "reconstruct_scene.stderr.log",
        )
        peak_gpu_memory = max(peak_gpu_memory, reconstruct_peak)
        if reconstruct_return_code != 0:
            tail = (logs / "reconstruct_scene.stderr.log").read_text(
                encoding="utf-8", errors="replace"
            )[-4000:]
            raise RuntimeError(
                f"official reconstruct_scene.py failed with {reconstruct_return_code}: {tail}"
            )
        for required in ("to_glb_inputs.pt", "chunk_inputs.pt", "mesh.ply"):
            if not (output_dir / required).is_file():
                raise RuntimeError(
                    f"official reconstruct_scene.py omitted required output {required}"
                )
        glb_return_code, glb_peak = _run_with_gpu_monitor(
            conversion_command(checkout, output_dir),
            cwd=checkout,
            environment=environment,
            stdout_path=logs / "chunked_to_glb.stdout.log",
            stderr_path=logs / "chunked_to_glb.stderr.log",
        )
        peak_gpu_memory = max(peak_gpu_memory, glb_peak)
        if glb_return_code != 0 or not (output_dir / "scene.glb").is_file():
            tail = (logs / "chunked_to_glb.stderr.log").read_text(
                encoding="utf-8", errors="replace"
            )[-4000:]
            raise RuntimeError(f"official chunked_to_glb.py failed with {glb_return_code}: {tail}")
        selected = _selected_frame_ids(output_dir, request)
        chunks, chunk_count = _chunk_diagnostics(
            output_dir,
            points_working,
            len(selected),
        )
        transform_outputs_to_colmap(output_dir, working_to_colmap)

    _write_json(output_dir / "working_transform.json", transform_record)
    cleaned_count = _point_count(output_dir / "clean_points.ply")
    if cleaned_count <= 0:
        raise RuntimeError("official point cleaning retained zero sparse points")
    low = np.percentile(original_points, 0.5, axis=0)
    high = np.percentile(original_points, 99.5, axis=0)
    diagonal = float(np.linalg.norm(high - low))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise RuntimeError("COLMAP sparse point bounds are degenerate")
    _write_json(
        output_dir / "worker_diagnostics.json",
        {
            "initial_sparse_points": len(original_points),
            "cleaned_sparse_points": cleaned_count,
            "robust_bounds_min": low.tolist(),
            "robust_bounds_max": high.tolist(),
            "scene_diagonal_arbitrary_units": diagonal,
            "chunks_before_filtering": chunk_count,
            "chunks_after_filtering": chunk_count,
            "chunks": chunks,
            "chosen_parameters": request.reconstruction_parameters,
            "warnings": [
                "The official Iphone chunker exposes retained chunks only; "
                "chunks_before_filtering records the retained structured set."
            ],
        },
    )

    import torch
    import torchvision

    raw_paths = [
        path.relative_to(root).as_posix()
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    ]
    manifest = {
        "schema_version": "0.1.0",
        "official_repository": request.official_repository,
        "official_code_commit": request.official_code_commit,
        "submodule_commits": submodules,
        "official_license": OFFICIAL_LICENSE,
        "checkpoint_records": checkpoint_records,
        "runtime_model_repository": DINOV3_REPOSITORY,
        "runtime_model_revision": dinov3_revision,
        "worker_version": WORKER_VERSION,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "device": "cuda",
        "precision": "float16",
        "seed": request.seed,
        "request_sha256": sha256_file(request_path),
        "frame_sequence_digest": request.frame_sequence_digest,
        "camera_package_sha256": request.camera_package_sha256,
        "registered_frame_ids": request.registered_frame_ids,
        "selected_frame_ids": selected,
        "working_transform": transform_record,
        "reconstruct_return_code": reconstruct_return_code,
        "glb_conversion_return_code": glb_return_code,
        "runtime_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": peak_gpu_memory or None,
        "raw_output_paths": raw_paths,
        "image_identifier": None,
        "warnings": [],
    }
    _write_json(output_dir / "worker_manifest.json", manifest)
