from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from particulate_worker import __version__
from particulate_worker.normalize import (
    axis_point_from_prediction,
    load_mesh,
    source_to_working,
)
from particulate_worker.verification import sha256, verify_official_runtime


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _point_from_plucker(plucker: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = plucker[:3]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    point = np.cross(plucker[3:], axis)
    return axis, point


def _official_infer(
    repository: Path,
    checkpoint: Path,
    source: Path,
    native_dir: Path,
    up: str,
    configuration: dict[str, object],
) -> int | None:
    command = [
        sys.executable,
        str(repository / "infer.py"),
        "--input_mesh",
        str(source),
        "--output_dir",
        str(native_dir),
        "--model_config",
        str(repository / "configs/particulate-B.yaml"),
        "--ckpt_path",
        str(checkpoint),
        "--up_dir",
        up,
        "--num_points",
        str(int(configuration.get("num_points", 102400))),
        "--animation_frames",
        str(int(configuration.get("animation_frames", 50))),
        "--eval",
    ]
    if not bool(configuration.get("strict_connected_components", True)):
        command.append("--no_strict")
    environment = dict(os.environ)
    environment.setdefault("HF_HUB_OFFLINE", "1")
    peak_gpu_memory_bytes: int | None = None
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout_file:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr_file:
            process = subprocess.Popen(
                command,
                cwd=repository,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                env=environment,
            )
            while process.poll() is None:
                try:
                    memory = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=pid,used_gpu_memory",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                except OSError:
                    memory = None
                if memory is not None and memory.returncode == 0:
                    for line in memory.stdout.splitlines():
                        fields = [value.strip() for value in line.split(",")]
                        if len(fields) == 2 and fields[0] == str(process.pid):
                            value = int(fields[1]) * 1024 * 1024
                            peak_gpu_memory_bytes = max(peak_gpu_memory_bytes or 0, value)
                time.sleep(0.1)
            returncode = process.wait()
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read()
            stderr = stderr_file.read()
    logs = native_dir / "official_inference.log"
    logs.parent.mkdir(parents=True, exist_ok=True)
    logs.write_text(stdout + "\n--- stderr ---\n" + stderr, encoding="utf-8")
    if returncode != 0:
        raise RuntimeError(
            f"official Particulate inference failed ({returncode}): {stderr[-2000:]}"
        )
    return peak_gpu_memory_bytes


def _normalize_candidate(
    item: dict[str, object],
    input_root: Path,
    output_dir: Path,
    request_hash: str,
    repository: Path,
    checkpoint: Path,
    partfield_checkpoint: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    started = time.monotonic()
    candidate_id = str(item["candidate_id"])
    source = input_root / str(item["source_mesh_path"])
    if sha256(source) != item["source_mesh_sha256"]:
        raise ValueError("Particulate source mesh hash mismatch")
    source_mesh = load_mesh(source)
    working_hypothesis = str(item["working_frame_hypothesis"])
    hypotheses_evaluated = item.get("hypotheses_evaluated")
    if not isinstance(hypotheses_evaluated, list) or hypotheses_evaluated != [working_hypothesis]:
        raise ValueError(
            "Particulate request must explicitly record the single configured "
            "working-frame hypothesis"
        )
    configured = item.get("generation_configuration")
    configuration = configured if isinstance(configured, dict) else {}
    configured_up = str(configuration.get("up_axis", working_hypothesis))
    if configured_up != working_hypothesis:
        raise ValueError("generation up_axis conflicts with the audited working-frame hint")
    up = working_hypothesis.removeprefix("+")
    native_dir = output_dir / "candidates" / candidate_id / "native"
    native_dir.mkdir(parents=True, exist_ok=True)
    peak_gpu_memory_bytes = _official_infer(
        repository,
        checkpoint,
        source,
        native_dir,
        up,
        configuration,
    )
    prediction_path = native_dir / "eval/pred.npz"
    if not prediction_path.is_file():
        raise RuntimeError("official Particulate did not produce eval/pred.npz")
    prediction = np.load(prediction_path, allow_pickle=False)
    face_part_ids = np.asarray(prediction["face_part_ids"], dtype=np.int64)
    if len(face_part_ids) != len(source_mesh.faces):
        raise RuntimeError("Particulate face-part prediction does not match source mesh")
    source_to_particulate, particulate_to_source = source_to_working(source_mesh, up)
    unique_parts = sorted(int(value) for value in np.unique(face_part_ids))
    link_ids = {part_id: f"part_{part_id:03d}" for part_id in unique_parts}
    links = []
    link_root = output_dir / "candidates" / candidate_id / "links"
    for part_id in unique_parts:
        link_mesh = source_mesh.submesh([face_part_ids == part_id], append=True, repair=False)
        path = link_root / f"{link_ids[part_id]}.ply"
        path.parent.mkdir(parents=True, exist_ok=True)
        link_mesh.export(path)
        relative = _relative(path, input_root)
        links.append(
            {
                "link_id": link_ids[part_id],
                "name": link_ids[part_id],
                "visual_asset_paths": [relative],
                "visual_asset_hashes": {relative: sha256(path)},
                "native_bounds_min": [float(value) for value in link_mesh.bounds[0]],
                "native_bounds_max": [float(value) for value in link_mesh.bounds[1]],
            }
        )
    hierarchy = np.asarray(prediction["motion_hierarchy"], dtype=np.int64).reshape(-1, 2)
    revolute = np.asarray(prediction["is_part_revolute"], dtype=bool)
    prismatic = np.asarray(prediction["is_part_prismatic"], dtype=bool)
    revolute_plucker = np.asarray(prediction["revolute_plucker"], dtype=np.float64)
    revolute_range = np.asarray(prediction["revolute_range"], dtype=np.float64)
    prismatic_axis = np.asarray(prediction["prismatic_axis"], dtype=np.float64)
    prismatic_range = np.asarray(prediction["prismatic_range"], dtype=np.float64)
    joints = []
    for joint_index, (parent, child) in enumerate(hierarchy):
        child_index = int(child)
        if child_index >= len(unique_parts) or int(parent) >= len(unique_parts):
            continue
        pivot = None
        if revolute[child_index]:
            joint_type = "revolute"
            axis_working, pivot_working = _point_from_plucker(revolute_plucker[child_index])
            limits = revolute_range[child_index]
        elif prismatic[child_index]:
            joint_type = "prismatic"
            axis_working = prismatic_axis[child_index]
            pivot_working = None
            limits = prismatic_range[child_index]
        else:
            joint_type = "fixed"
            axis_working = np.array([1.0, 0.0, 0.0])
            pivot_working = None
            limits = np.array([0.0, 0.0])
        axis, pivot_value = axis_point_from_prediction(
            axis_working,
            pivot_working,
            particulate_to_source,
        )
        if pivot_value is not None:
            pivot = [float(value) for value in pivot_value]
        joints.append(
            {
                "joint_id": f"joint_{joint_index:03d}",
                "parent_link_id": link_ids[unique_parts[int(parent)]],
                "child_link_id": link_ids[unique_parts[child_index]],
                "joint_type": joint_type,
                "axis": [float(value) for value in axis],
                "pivot": pivot,
                "candidate_limit_lower": float(limits[0]),
                "candidate_limit_upper": float(limits[1]),
                "limit_source": "candidate_prior",
            }
        )
    native_paths = sorted(
        path
        for path in native_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".glb", ".obj", ".npz", ".log"}
    )
    native_relative = [_relative(path, input_root) for path in native_paths]
    license_record = dict(item["source_license"])
    license_record.update(
        {
            "source_family": "particulate",
            "commercial_review_status": "research_only",
            "production_selectable": False,
        }
    )
    candidate = {
        "candidate_id": candidate_id,
        "articulated_object_id": item["articulated_object_id"],
        "source_family": "particulate",
        "source_asset_id": item["source_backend"],
        "links": links,
        "joints": joints,
        "states": [],
        "native_coordinate_convention": (
            f"source candidate frame; temporary {up} to Particulate +Z transform recorded"
        ),
        "native_units": "source_arbitrary_units",
        "native_output_paths": native_relative,
        "native_output_hashes": {
            relative: sha256(path)
            for relative, path in zip(native_relative, native_paths, strict=True)
        },
        "working_transform_source_to_particulate": [
            float(value) for value in source_to_particulate.reshape(-1)
        ],
        "working_transform_particulate_to_source": [
            float(value) for value in particulate_to_source.reshape(-1)
        ],
        "working_frame_hypothesis": working_hypothesis,
        "working_frame_hypotheses_evaluated": hypotheses_evaluated,
        "working_frame_selection_evidence": item["hypothesis_selection_evidence"],
        "license_record": license_record,
        "production_selectable": False,
        "provenance": {
            "adapter_name": "particulate_candidates",
            "adapter_version": "0.1.0",
            "configuration": configuration,
            "input_artifact_paths": [item["source_mesh_path"]],
            "output_artifact_paths": [
                *(path for link in links for path in link["visual_asset_paths"]),
                *native_relative,
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            "confidence": {
                "score": 0.5,
                "method": "official_particulate_prior_pending_multistate_validation",
                "notes": None,
            },
            "source": "generated",
        },
        "warnings": [
            "PartField runtime checkpoint is non-commercial research-only",
            "candidate kinematics require frozen-structure held-out validation",
        ],
    }
    worker = {
        "schema_version": "0.1.0",
        "worker_version": __version__,
        "request_sha256": request_hash,
        "official_repository": item["official_repository"],
        "official_code_commit": item["official_code_commit"],
        "checkpoint_repository": item["checkpoint_repository"],
        "checkpoint_revision": item["checkpoint_revision"],
        "checkpoint_hashes": item["checkpoint_hashes"],
        "runtime_model_hashes": item["runtime_model_hashes"],
        "runtime_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "peak_host_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "warnings": [],
    }
    return candidate, worker


def generate(request_path: Path, input_root: Path, output_dir: Path) -> None:
    request = read_json(request_path)
    repository_value = request.get("official_repository_path")
    checkpoint_value = request.get("checkpoint_path")
    partfield_value = request.get("partfield_checkpoint_path")
    if not all(
        isinstance(value, str) for value in (repository_value, checkpoint_value, partfield_value)
    ):
        raise ValueError("Particulate runtime paths are required")
    repository = Path(str(repository_value)).expanduser().resolve()
    checkpoint = Path(str(checkpoint_value)).expanduser().resolve()
    partfield_checkpoint = Path(str(partfield_value)).expanduser().resolve()
    verify_official_runtime(repository, checkpoint, partfield_checkpoint)
    request_hash = sha256(request_path)
    candidates = []
    manifests = []
    failures = []
    started = time.monotonic()
    for item in request["requests"]:
        try:
            candidate, manifest = _normalize_candidate(
                item,
                input_root,
                output_dir,
                request_hash,
                repository,
                checkpoint,
                partfield_checkpoint,
            )
            candidates.append(candidate)
            manifests.append(manifest)
        except Exception:
            failures.append(str(item["candidate_id"]))
            raise
    write_json(
        output_dir / "candidate_manifest.json",
        {
            "schema_version": "0.1.0",
            "measured_motion_sha256": request["measured_motion_sha256"],
            "retrieval_manifest_sha256": request["retrieval_manifest_sha256"],
            "candidates": candidates,
            "worker_manifests": manifests,
            "failed_candidate_ids": failures,
            "runtime_seconds": time.monotonic() - started,
            "warnings": [],
        },
    )


def healthcheck(request_path: Path) -> None:
    request = read_json(request_path)
    repository = request.get("official_repository_path")
    checkpoint = request.get("checkpoint_path")
    partfield = request.get("partfield_checkpoint_path")
    if all(isinstance(value, str) for value in (repository, checkpoint, partfield)):
        verify_official_runtime(
            Path(str(repository)).expanduser().resolve(),
            Path(str(checkpoint)).expanduser().resolve(),
            Path(str(partfield)).expanduser().resolve(),
        )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Particulate healthcheck requires CUDA")
    print(
        f"particulate_worker {__version__}: official commit/checkpoints verified; "
        f"CUDA={torch.version.cuda}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("healthcheck", "generate"))
    parser.add_argument("--request", required=True)
    parser.add_argument("--input-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    request_path = Path(args.request).resolve()
    if args.action == "healthcheck":
        healthcheck(request_path)
        return 0
    if not args.input_root or not args.output_dir:
        parser.error("generate requires --input-root and --output-dir")
    generate(
        request_path,
        Path(args.input_root).resolve(),
        Path(args.output_dir).resolve(),
    )
    return 0
