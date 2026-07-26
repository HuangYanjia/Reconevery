from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from sam3d_objects_worker.checkpoint_loader import sha256, verify_checkpoint_files
from sam3d_objects_worker.commit_verification import verify_checkout
from sam3d_objects_worker.native_export import (
    export_native,
    ply_vertex_count,
    serializable_layout,
)
from sam3d_objects_worker.official_api import load_inference_class
from sam3d_objects_worker.schema import CandidateRequest
from sam3d_objects_worker.version import WORKER_VERSION


def infer(request_path: Path, input_root: Path, output_dir: Path) -> None:
    request = CandidateRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
    config = request.generation_configuration
    checkout = Path(str(config["official_checkout_path"]))
    checkpoint_root = Path(str(config["checkpoint_root"]))
    pipeline_config = Path(str(config["pipeline_config"]))
    verify_checkout(checkout, request.official_code_commit)
    verify_checkpoint_files(checkpoint_root, request.checkpoint_hashes)
    if not pipeline_config.is_file() or not pipeline_config.resolve().is_relative_to(
        checkpoint_root.resolve()
    ):
        raise RuntimeError("pipeline_config must be a file inside the verified checkpoint root")
    import torch

    crop_path = input_root / request.anchor_crop_path
    crop = Image.open(crop_path).convert("RGBA")
    array = np.asarray(crop)
    image = array[..., :3]
    mask = array[..., 3] > 0
    torch.manual_seed(request.generation_seed)
    started = time.monotonic()
    inference_class = load_inference_class(checkout)
    predictor = inference_class(str(pipeline_config), compile=bool(config.get("compile", False)))
    output = predictor(image, mask, seed=request.generation_seed)
    assets = export_native(output, output_dir / "native")
    gaussian_count = next(
        ply_vertex_count(path)
        for path, format_name, _ in assets
        if format_name == "gaussian_splat_ply"
    )
    runtime = time.monotonic() - started
    relative_assets = [
        {
            "relative_path": path.relative_to(input_root).as_posix(),
            "sha256": sha256(path),
            "format": native_format,
            "size_bytes": path.stat().st_size,
            "role": role,
        }
        for path, native_format, role in assets
    ]
    candidate_id = str(config["candidate_id"])
    provenance = {
        "adapter_name": "sam3d_object_candidates",
        "adapter_version": WORKER_VERSION,
        "configuration": {
            "official_code_commit": request.official_code_commit,
            "checkpoint_revision": request.checkpoint_revision,
            "runtime_model_revisions": request.runtime_model_revisions,
            "runtime_model_hashes": request.runtime_model_hashes,
            "seed": request.generation_seed,
        },
        "input_artifact_paths": [request.anchor_crop_path],
        "output_artifact_paths": [item["relative_path"] for item in relative_assets],
        "timestamp": "1970-01-01T00:00:00Z",
        "confidence": {
            "score": 0.5,
            "method": "unvalidated_official_sam3d_candidate",
            "notes": "held-out evaluation occurs downstream",
        },
        "source": "generated",
    }
    candidate = {
        "candidate_id": candidate_id,
        "object_id": request.object_id,
        "semantic_label": request.semantic_label,
        "backend": "sam3d_objects",
        "anchor_frame_id": request.anchor_frame_id,
        "generation_seed": request.generation_seed,
        "native_assets": relative_assets,
        "native_coordinate_convention": "official_sam3d_object_canonical",
        "native_bounds_min": None,
        "native_bounds_max": None,
        "native_center": None,
        "native_scale": None,
        "vertex_count": None,
        "face_count": None,
        "material_count": None,
        "texture_count": None,
        "gaussian_count": gaussian_count,
        "backend_predicted_layout": serializable_layout(output),
        "render_capability": {
            "renderer": "official_optional_visual_glb_via_nvdiffrast",
            "supports_rgba": True,
            "supports_depth": True,
            "supports_normals": False,
            "camera_axes": "x_right_y_down_z_forward",
        },
        "sampling_method": "area_weighted_official_optional_visual_glb",
        "generation_runtime_seconds": runtime,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "license_record": request.license_policy,
        "provenance": provenance,
        "warnings": [
            "native Gaussian PLY is preserved but is not represented as a watertight surface",
            "registration and held-out rendering use the official optional visual GLB",
        ],
    }
    _write_json(output_dir / "candidate.json", candidate)
    _write_json(
        output_dir / "worker_manifest.json",
        {
            "worker_name": "sam3d_objects_worker",
            "worker_version": WORKER_VERSION,
            "action": "generate",
            "backend": "official_sam3d_objects",
            "request_sha256": _sha256(request_path),
            "official_repository": request.official_repository,
            "official_code_commit": request.official_code_commit,
            "checkpoint_repository": request.checkpoint_repository,
            "checkpoint_revision": request.checkpoint_revision,
            "checkpoint_hashes": request.checkpoint_hashes,
            "runtime_model_revisions": request.runtime_model_revisions,
            "runtime_model_hashes": request.runtime_model_hashes,
            "runtime_seconds": runtime,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "peak_host_memory_bytes": None,
            "warnings": [],
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
