#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

WORKER_VERSION = "0.1.0-fake"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    health = subparsers.add_parser("healthcheck")
    health.add_argument("--config", required=True)
    infer = subparsers.add_parser("infer")
    infer.add_argument("--request", required=True)
    infer.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _healthcheck(config_path: Path) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("fake_mode") == "healthcheck_failure":
        print("fake worker dependency check failed", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "available": True,
                "backend": "fake_worker",
                "worker_version": WORKER_VERSION,
            },
            sort_keys=True,
        )
    )
    return 0


def _rectangle(
    width: int,
    height: int,
    index: int,
) -> tuple[int, int, int, int]:
    rectangle_width = max(4, width // 4)
    rectangle_height = max(4, height // 4)
    x0 = 1 + (index * max(2, rectangle_width // 2)) % max(1, width - rectangle_width - 1)
    y0 = 1 + (index * 2) % max(1, height - rectangle_height - 1)
    return x0, y0, x0 + rectangle_width, y0 + rectangle_height


def _mask(
    path: Path,
    width: int,
    height: int,
    rectangle: tuple[int, int, int, int],
    mode: str,
) -> None:
    if mode == "invalid_dimensions":
        width += 1
    image = Image.new("L", (width, height), 0)
    if mode != "empty_mask":
        value = 128 if mode == "non_binary_mask" else 255
        draw = ImageDraw.Draw(image)
        x0, y0, x1, y1 = rectangle
        draw.rectangle(
            (
                min(x0, width - 1),
                min(y0, height - 1),
                min(x1 - 1, width - 1),
                min(y1 - 1, height - 1),
            ),
            fill=value,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", compress_level=6, optimize=False)


def _raw_ids(mode: str, count: int) -> list[str]:
    if mode == "raw_ids_b":
        return [f"raw-b-{count - index:04d}" for index in range(count)]
    return [f"raw-{index + 1:04d}" for index in range(count)]


def _track_specs(request: dict[str, Any], mode: str) -> list[tuple[dict[str, Any], int]]:
    prompts = [
        prompt for prompt in request["prompt_manifest"]["prompts"] if prompt.get("enabled", True)
    ]
    if mode == "no_detections" or not prompts:
        return []
    if mode in {
        "one_object",
        "fragmented_track",
        "duplicate_tracks",
        "invalid_dimensions",
        "empty_mask",
        "non_binary_mask",
        "invalid_box",
        "non_finite_score",
        "out_of_range_score",
        "unknown_frame",
        "duplicate_observation",
        "raw_ids_a",
        "raw_ids_b",
    }:
        count = 2 if mode == "duplicate_tracks" else 1
        return [(prompts[0], 0) for _ in range(count)]
    if mode == "multiple_instances":
        return [(prompts[0], 0), (prompts[0], 1)]
    return [(prompt, index) for index, prompt in enumerate(prompts)]


def _infer(request_path: Path, output_dir: Path) -> int:
    request_path = request_path.resolve()
    root = request_path.parent.parent
    request = json.loads(request_path.read_text(encoding="utf-8"))
    config = request["model_configuration"]
    mode = str(config.get("fake_mode", "success_multi"))
    if mode == "leak_token":
        print(f"worker stdout token={os.environ.get('HF_TOKEN', '')}")
        print(f"worker stderr token={os.environ.get('HF_TOKEN', '')}", file=sys.stderr)
    if mode == "nonzero_exit":
        print("fake worker configured nonzero exit", file=sys.stderr)
        return 17
    if mode == "timeout":
        time.sleep(3600)
        return 0
    if mode == "interruption":
        os.kill(os.getpid(), signal.SIGINT)
        return 130
    if mode == "oom":
        print("CUDA out of memory while allocating fake tensor", file=sys.stderr)
        return 1
    if mode == "unauthorized":
        print("401 Unauthorized: gated repo terms not accepted", file=sys.stderr)
        return 1

    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "malformed_json":
        (output_dir / "worker_result.json").write_text("{not json", encoding="utf-8")
    frame_order = list(request["frame_order"])
    dimensions = request["frame_dimensions"]
    specs = _track_specs(request, mode)
    raw_ids = _raw_ids(mode, len(specs))
    tracks: list[dict[str, Any]] = []
    for track_index, ((prompt, geometry_index), raw_id) in enumerate(
        zip(specs, raw_ids, strict=True)
    ):
        observations: list[dict[str, Any]] = []
        observation_frames = frame_order[:1] if mode == "fragmented_track" else frame_order
        if mode == "unknown_frame":
            observation_frames = ["unknown_frame"]
        for frame_index, frame_id in enumerate(observation_frames):
            dimension_frame = frame_order[0] if frame_id == "unknown_frame" else frame_id
            width, height = dimensions[dimension_frame]
            rectangle = _rectangle(width, height, geometry_index)
            mask_frame_id = frame_id
            relative_mask = f"{request['output_directory']}/masks/{raw_id}/{mask_frame_id}.png"
            _mask(
                root / relative_mask,
                width,
                height,
                rectangle,
                mode,
            )
            score = 0.92 - track_index * 0.02 - frame_index * 0.001
            if mode == "non_finite_score":
                score = float("nan")
            elif mode == "out_of_range_score":
                score = 1.5
            model_box = (
                [-1.0, -1.0, float(width + 1), float(height + 1)]
                if mode == "invalid_box"
                else [float(value) for value in rectangle]
            )
            observations.append(
                {
                    "frame_id": frame_id,
                    "raw_model_object_id": raw_id,
                    "prompt_id": prompt["prompt_id"],
                    "semantic_label": prompt["label"],
                    "score": score,
                    "mask_path": relative_mask,
                    "mask_encoding": "binary_png",
                    "model_box_xyxy": model_box,
                    "occluded": None,
                }
            )
        if mode == "duplicate_observation":
            observations.append(dict(observations[0]))
        tracks.append(
            {
                "raw_model_object_id": raw_id,
                "prompt_id": prompt["prompt_id"],
                "semantic_label": prompt["label"],
                "observations": observations,
            }
        )
    if mode != "malformed_json" and mode != "missing_output":
        _write_json(
            output_dir / "worker_result.json",
            {
                "schema_version": "0.1.0",
                "tracks": tracks,
                "warnings": [],
            },
        )
    _write_json(
        output_dir / "worker_manifest.json",
        {
            "schema_version": "0.1.0",
            "official_repository": config["official_repository"],
            "official_code_commit": config["official_code_commit"],
            "checkpoint_repository": config["checkpoint_repository"],
            "checkpoint_revision": config["checkpoint_revision"],
            "checkpoint_hash": None,
            "checkpoint_access_mode": config["checkpoint_access_mode"],
            "official_license": "SAM License (fake worker; no checkpoint loaded)",
            "worker_version": WORKER_VERSION,
            "torch_version": None,
            "torchvision_version": None,
            "cuda_version": None,
            "device_name": "deterministic fake CPU",
            "device": config["device"],
            "precision": config["precision"],
            "seed": request["seed"],
            "runtime_seconds": 0.125,
            "peak_gpu_memory_bytes": None,
            "prompt_manifest_hash": request["prompt_manifest_sha256"],
            "frame_manifest_hash": request["frame_manifest_sha256"],
            "strategy": request["strategy"],
            "model_mode": config["model_mode"],
            "image_identifier": None,
            "warnings": ["fake worker: no official SAM checkpoint executed"],
        },
    )
    return 0


def main() -> int:
    args = _arguments()
    if args.action == "healthcheck":
        return _healthcheck(Path(args.config))
    return _infer(Path(args.request), Path(args.output_dir))


if __name__ == "__main__":
    raise SystemExit(main())
