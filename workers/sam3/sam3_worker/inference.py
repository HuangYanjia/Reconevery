from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from sam3_worker.frame_preparation import (
    prepared_video_frames,
    resolve_worker_output_directory,
)
from sam3_worker.official_compat import official_propagation_directions
from sam3_worker.schema import (
    InferenceRequest,
    RawObservation,
    RawTrack,
    WorkerConfiguration,
    WorkerPrompt,
)
from sam3_worker.version import WORKER_VERSION


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prompt_request(
    prompt: WorkerPrompt,
    frame_index: int,
    dimensions: tuple[int, int],
    object_id: int,
    threshold: float,
) -> dict[str, Any]:
    width, height = dimensions
    output_threshold = (
        prompt.confidence_threshold if prompt.confidence_threshold is not None else threshold
    )
    request: dict[str, Any] = {
        "type": "add_prompt",
        "frame_index": frame_index,
        "output_prob_thresh": output_threshold,
        "rel_coordinates": True,
    }
    if prompt.prompt_type == "text":
        if not prompt.positive:
            raise RuntimeError(
                "the pinned official predictor does not expose negative text prompts"
            )
        request["text"] = prompt.text
    elif prompt.prompt_type == "box":
        assert prompt.box_xyxy is not None
        x0, y0, x1, y1 = prompt.box_xyxy
        request["bounding_boxes"] = [
            [x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height]
        ]
        request["bounding_box_labels"] = [1 if prompt.positive else 0]
    elif prompt.prompt_type == "point":
        assert prompt.points is not None
        request["points"] = [[point.x / width, point.y / height] for point in prompt.points]
        request["point_labels"] = [point.label for point in prompt.points]
        request["obj_id"] = object_id
    else:
        raise RuntimeError(
            "the pinned official build_sam3_predictor handle_request API does not expose "
            "mask seed prompts; use text, box, or point prompts with this pinned backend"
        )
    return request


def _native_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return list(value)


def _scalar(value: Any) -> Any:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _binary_mask_image(value: Any) -> Image.Image:
    if hasattr(value, "detach"):
        value = value.detach().to(device="cpu").tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    while (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list)
        and value[0]
        and isinstance(value[0][0], list)
    ):
        value = value[0]
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(row, list) and row for row in value)
    ):
        raise RuntimeError("official SAM mask output is not a non-empty 2D array")
    width = len(value[0])
    if any(len(row) != width for row in value):
        raise RuntimeError("official SAM mask output has inconsistent row widths")
    pixels = bytes(255 if bool(pixel) else 0 for row in value for pixel in row)
    return Image.frombytes("L", (width, len(value)), pixels)


def _collect_outputs(
    root: Path,
    output_dir: Path,
    request: InferenceRequest,
    prompt: WorkerPrompt,
    frame_index: int,
    outputs: dict[str, Any],
    tracks: dict[str, dict[str, RawObservation]],
) -> None:
    object_ids = _native_list(outputs.get("out_obj_ids", []))
    masks = outputs.get("out_binary_masks", [])
    if "out_probs" not in outputs:
        raise RuntimeError("official SAM output is missing required out_probs confidence scores")
    probabilities = _native_list(outputs["out_probs"])
    boxes_xywh = _native_list(outputs.get("out_boxes_xywh", [None] * len(object_ids)))
    if not (len(object_ids) == len(masks) == len(probabilities)):
        raise RuntimeError("official SAM output arrays have inconsistent lengths")
    frame_id = request.frame_order[frame_index]
    frame_width, frame_height = request.frame_dimensions[frame_id]
    for index, object_id in enumerate(object_ids):
        score = float(_scalar(probabilities[index]))
        if not math.isfinite(score):
            raise RuntimeError("official SAM output contains a non-finite confidence score")
        if score < 0:
            # Multiplex emits a negative sentinel when a propagated object is absent.
            continue
        if score > 1:
            raise RuntimeError("official SAM output contains a confidence score above one")
        raw_id = f"{prompt.prompt_id}:{int(_scalar(object_id))}"
        mask_path = output_dir / "masks" / raw_id / f"{frame_id}.png"
        mask = _binary_mask_image(masks[index])
        if mask.getbbox() is None:
            continue
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask.save(mask_path, format="PNG", compress_level=6, optimize=False)
        relative_mask = mask_path.relative_to(root).as_posix()
        box = boxes_xywh[index] if index < len(boxes_xywh) else None
        model_box = None
        if box is not None:
            x, y, width, height = (float(value) for value in box)
            model_box = (
                x * frame_width,
                y * frame_height,
                (x + width) * frame_width,
                (y + height) * frame_height,
            )
        tracks[raw_id][frame_id] = RawObservation(
            frame_id=frame_id,
            raw_model_object_id=raw_id,
            prompt_id=prompt.prompt_id,
            semantic_label=prompt.label,
            score=score,
            mask_path=relative_mask,
            model_box_xyxy=model_box,
        )


def _run_prompts(
    predictor: Any,
    root: Path,
    output_dir: Path,
    request: InferenceRequest,
    video_dir: Path,
) -> list[RawTrack]:
    frame_index = {frame_id: index for index, frame_id in enumerate(request.frame_order)}
    default_anchor = request.anchor_frames[0].frame_id
    threshold = float(request.postprocessing_configuration["score_threshold"])
    all_tracks: list[RawTrack] = []
    for prompt_index, prompt in enumerate(request.prompt_manifest.prompts):
        if not prompt.enabled:
            continue
        anchor_id = prompt.frame_id or default_anchor
        anchor_index = frame_index[anchor_id]
        tracks: dict[str, dict[str, RawObservation]] = defaultdict(dict)
        for direction in official_propagation_directions(request.tracking_direction):
            start = predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(video_dir),
                    "offload_video_to_cpu": False,
                    "offload_state_to_cpu": False,
                }
            )
            session_id = start["session_id"]
            try:
                add_request = _prompt_request(
                    prompt,
                    anchor_index,
                    request.frame_dimensions[anchor_id],
                    prompt_index + 1,
                    threshold,
                )
                add_request["session_id"] = session_id
                response = predictor.handle_request(add_request)
                _collect_outputs(
                    root,
                    output_dir,
                    request,
                    prompt,
                    response["frame_index"],
                    response["outputs"],
                    tracks,
                )
                output_threshold = (
                    prompt.confidence_threshold
                    if prompt.confidence_threshold is not None
                    else threshold
                )
                for propagated in predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": direction,
                        "start_frame_index": anchor_index,
                        "output_prob_thresh": output_threshold,
                    }
                ):
                    _collect_outputs(
                        root,
                        output_dir,
                        request,
                        prompt,
                        propagated["frame_index"],
                        propagated["outputs"],
                        tracks,
                    )
            finally:
                predictor.handle_request({"type": "close_session", "session_id": session_id})
        for raw_id in sorted(tracks):
            observations = [
                tracks[raw_id][frame_id]
                for frame_id in request.frame_order
                if frame_id in tracks[raw_id]
            ]
            all_tracks.append(
                RawTrack(
                    raw_model_object_id=raw_id,
                    prompt_id=prompt.prompt_id,
                    semantic_label=prompt.label,
                    observations=observations,
                )
            )
    return all_tracks


def run_inference(request_path: Path, output_dir: Path) -> None:
    from sam3_worker.model_loader import load_predictor, sha256_file

    started = time.monotonic()
    request_path = request_path.resolve()
    root = request_path.parent.parent
    output_dir = resolve_worker_output_directory(root, output_dir)
    request = InferenceRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
    config = WorkerConfiguration.model_validate(request.model_configuration)
    if config.device != "cuda" or config.precision != "bfloat16":
        raise RuntimeError("the pinned official SAM video predictor requires cuda/bfloat16")
    import torch
    import torchvision

    random.seed(request.seed)
    torch.manual_seed(request.seed)
    torch.cuda.manual_seed_all(request.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    predictor, checkpoint = load_predictor(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    with prepared_video_frames(
        root,
        request.frame_order,
        request.frame_paths,
        request.frame_dimensions,
    ) as video_dir:
        all_tracks = _run_prompts(predictor, root, output_dir, request, video_dir)
    runtime = time.monotonic() - started
    _write_json(
        output_dir / "worker_result.json",
        {
            "schema_version": "0.1.0",
            "tracks": [track.model_dump(mode="json") for track in all_tracks],
            "warnings": [],
        },
    )
    _write_json(
        output_dir / "worker_manifest.json",
        {
            "schema_version": "0.1.0",
            "official_repository": config.official_repository,
            "official_code_commit": config.official_code_commit,
            "checkpoint_repository": config.checkpoint_repository,
            "checkpoint_revision": config.checkpoint_revision,
            "checkpoint_hash": sha256_file(checkpoint),
            "checkpoint_access_mode": config.checkpoint_access_mode,
            "official_license": "SAM License",
            "worker_version": WORKER_VERSION,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0),
            "device": config.device,
            "precision": config.precision,
            "seed": request.seed,
            "runtime_seconds": runtime,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "prompt_manifest_hash": request.prompt_manifest_sha256,
            "frame_manifest_hash": request.frame_manifest_sha256,
            "frame_sequence_digest": request.frame_sequence_digest,
            "strategy": request.strategy,
            "model_mode": config.model_mode,
            "image_identifier": None,
            "warnings": [],
        },
    )
