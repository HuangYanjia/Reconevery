from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

from PIL import Image

from trellis2_objects_worker.commit_verification import verify_checkout
from trellis2_objects_worker.glb_export import export_official_glb
from trellis2_objects_worker.official_api import load_rgba_pipeline
from trellis2_objects_worker.runtime_assets import sha256, verify_assets
from trellis2_objects_worker.schema import CandidateRequest
from trellis2_objects_worker.version import WORKER_VERSION


def infer(request_path: Path, input_root: Path, output_dir: Path) -> None:
    request = CandidateRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
    config = request.generation_configuration
    checkout = Path(str(config["official_checkout_path"]))
    snapshot = Path(str(config["checkpoint_snapshot_path"]))
    verify_checkout(checkout, request.official_code_commit)
    verify_assets(snapshot, request.checkpoint_hashes)
    runtime_paths = {
        name: Path(str(path)) for name, path in dict(config.get("runtime_model_paths", {})).items()
    }
    if set(runtime_paths) != set(request.runtime_model_hashes) or set(
        request.runtime_model_revisions
    ) != set(request.runtime_model_hashes):
        raise RuntimeError("TRELLIS.2 runtime model paths, revisions, and hashes must match")
    for name, hashes in request.runtime_model_hashes.items():
        verify_assets(runtime_paths[name], hashes)
    sys.path.insert(0, str(checkout))
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("TRELLIS.2 requires an NVIDIA GPU")
    torch.manual_seed(request.generation_seed)
    image = Image.open(input_root / request.anchor_crop_path).convert("RGBA")
    if image.getchannel("A").getextrema() == (255, 255):
        raise RuntimeError("TRELLIS.2 crop must contain a canonical non-opaque alpha mask")
    started = time.monotonic()
    pipeline = load_rgba_pipeline(snapshot)
    pipeline.cuda()
    output = pipeline.run(
        image,
        seed=request.generation_seed,
        **dict(config.get("pipeline_arguments", {})),
    )[0]
    vertices = output.vertices.detach()
    native_bounds_min = vertices.amin(dim=0).cpu().tolist()
    native_bounds_max = vertices.amax(dim=0).cpu().tolist()
    native_center = ((vertices.amin(dim=0) + vertices.amax(dim=0)) * 0.5).cpu().tolist()
    vertex_count = int(output.vertices.shape[0])
    face_count = int(output.faces.shape[0])
    glb_path = output_dir / "native" / "visual_asset.glb"
    export_official_glb(
        output,
        glb_path,
        texture_size=int(config.get("texture_size", 2048)),
        decimation_target=int(config.get("decimation_target", 1_000_000)),
    )
    runtime = time.monotonic() - started
    relative = glb_path.relative_to(input_root).as_posix()
    candidate_id = str(config["candidate_id"])
    asset = {
        "relative_path": relative,
        "sha256": sha256(glb_path),
        "format": "pbr_glb",
        "size_bytes": glb_path.stat().st_size,
        "role": "official_pbr_glb",
    }
    provenance = {
        "adapter_name": "trellis2_object_candidates",
        "adapter_version": WORKER_VERSION,
        "configuration": {
            "official_code_commit": request.official_code_commit,
            "checkpoint_revision": request.checkpoint_revision,
            "runtime_model_revisions": request.runtime_model_revisions,
            "runtime_model_hashes": request.runtime_model_hashes,
            "seed": request.generation_seed,
        },
        "input_artifact_paths": [request.anchor_crop_path],
        "output_artifact_paths": [relative],
        "timestamp": "1970-01-01T00:00:00Z",
        "confidence": {
            "score": 0.5,
            "method": "unvalidated_official_trellis2_candidate",
            "notes": "held-out evaluation occurs downstream",
        },
        "source": "generated",
    }
    _write_json(
        output_dir / "candidate.json",
        {
            "candidate_id": candidate_id,
            "object_id": request.object_id,
            "semantic_label": request.semantic_label,
            "backend": "trellis2",
            "anchor_frame_id": request.anchor_frame_id,
            "generation_seed": request.generation_seed,
            "native_assets": [asset],
            "native_coordinate_convention": "official_trellis2_object_canonical",
            "native_bounds_min": native_bounds_min,
            "native_bounds_max": native_bounds_max,
            "native_center": native_center,
            "native_scale": None,
            "vertex_count": vertex_count,
            "face_count": face_count,
            "material_count": 1,
            "texture_count": 1,
            "gaussian_count": None,
            "backend_predicted_layout": {},
            "render_capability": {
                "renderer": "completion_evaluation_glb_renderer",
                "supports_rgba": True,
                "supports_depth": True,
                "supports_normals": True,
                "camera_axes": "x_right_y_down_z_forward",
            },
            "sampling_method": "area_weighted_triangle_sampling",
            "generation_runtime_seconds": runtime,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "license_record": request.license_policy,
            "provenance": provenance,
            "warnings": [],
        },
    )
    _write_json(
        output_dir / "worker_manifest.json",
        {
            "worker_name": "trellis2_objects_worker",
            "worker_version": WORKER_VERSION,
            "action": "generate",
            "backend": "official_trellis2",
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
