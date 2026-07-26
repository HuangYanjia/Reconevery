from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_points(path: Path, object_id: str) -> None:
    offset = (int(hashlib.sha256(object_id.encode()).hexdigest()[:4], 16) % 50) / 100
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 0\nproperty list uchar int vertex_indices\nend_header\n"
        f"{offset} 0 1\n{offset + 0.1} 0 1\n{offset} 0.1 1\n{offset + 0.1} 0.1 1\n",
        encoding="ascii",
    )


def manifest(
    output_dir: Path,
    request_path: Path,
    action: str,
    name: str,
) -> None:
    write_json(
        output_dir / f"{action}_worker_manifest.json",
        {
            "worker_name": name,
            "worker_version": "0.1.0",
            "action": action,
            "backend": "fake",
            "request_sha256": sha256(request_path),
            "official_repository": None,
            "official_code_commit": None,
            "checkpoint_repository": None,
            "checkpoint_revision": None,
            "checkpoint_hashes": {},
            "runtime_seconds": 0.01,
            "peak_gpu_memory_bytes": 0,
            "peak_host_memory_bytes": 0,
            "warnings": [],
        },
    )


def prepare_evidence(request_path: Path, input_root: Path, output_dir: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    objects = []
    for object_id, values in sorted(request["object_inputs"].items()):
        point_path = output_dir / object_id / "training_points.ply"
        write_points(point_path, object_id)
        heldout_path = output_dir / object_id / "heldout_measurements.json"
        write_json(
            heldout_path,
            {
                "object_id": object_id,
                "heldout_frame_ids": values["heldout_frame_ids"],
                "pixel_evidence_materialized": False,
            },
        )
        objects.append(
            {
                "object_id": object_id,
                "training_frame_ids": values["training_frame_ids"],
                "heldout_frame_ids": values["heldout_frame_ids"],
                "training_points_path": point_path.relative_to(input_root).as_posix(),
                "training_points_sha256": sha256(point_path),
                "training_point_count": 4,
                "heldout_measurement_manifest_path": heldout_path.relative_to(
                    input_root
                ).as_posix(),
                "heldout_measurement_manifest_sha256": sha256(heldout_path),
            }
        )
    write_json(
        output_dir / "evidence_package.json",
        {
            "manifest_sha256": request["manifest_sha256"],
            "frame_sequence_digest": request["frame_sequence_digest"],
            "camera_reconstruction_sha256": request["camera_reconstruction_sha256"],
            "segmentation_tracking_sha256": request["segmentation_tracking_sha256"],
            "dense_depth_manifest_sha256": request["dense_depth_manifest_sha256"],
            "measured_geometry_sha256": request["measured_geometry_sha256"],
            "evidence_split_sha256": request["evidence_split_sha256"],
            "crop_manifest_sha256": request["crop_manifest_sha256"],
            "objects": objects,
            "coordinate_convention": request["coordinate_convention"],
            "scale_status": "scale_ambiguous",
        },
    )
    manifest(output_dir, request_path, "evidence", "completion_evidence_worker")
    shutil.move(output_dir / "evidence_worker_manifest.json", output_dir / "worker_manifest.json")
    return 0


def load_generations(input_root: Path) -> tuple[list[dict[str, object]], dict[str, dict]]:
    candidates: list[dict[str, object]] = []
    by_id: dict[str, dict] = {}
    for name in (
        "sam3d_generation_manifest.json",
        "trellis2_generation_manifest.json",
        "measured_generation_manifest.json",
    ):
        generation = json.loads(
            (input_root / "reconstruction" / "completion" / name).read_text(encoding="utf-8")
        )
        for candidate in generation["candidates"]:
            candidates.append(candidate)
            by_id[candidate["candidate_id"]] = candidate
    return candidates, by_id


def register(request_path: Path, input_root: Path, output_dir: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    mode = request["registration_configuration"].get("fake_mode", "success")
    if mode == "timeout":
        time.sleep(60)
    if mode == "oom":
        print("out of memory", file=sys.stderr)
        return 9
    package = json.loads(
        (input_root / "reconstruction/completion/evidence/evidence_package.json").read_text(
            encoding="utf-8"
        )
    )
    split = json.loads(
        (input_root / "reconstruction/completion/evidence_split.json").read_text(encoding="utf-8")
    )
    candidates, by_id = load_generations(input_root)
    split_by_id = {item["object_id"]: item for item in split["objects"]}
    evidence_by_id = {item["object_id"]: item for item in package["objects"]}
    registrations = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        object_id = candidate["object_id"]
        if mode == "registration_failure" and candidate["backend"] != "measured_partial_baseline":
            registrations.append(
                {
                    "candidate_id": candidate_id,
                    "object_id": object_id,
                    "status": "registration_failed",
                    "frozen_transform": None,
                    "fitting_frame_ids": split_by_id[object_id]["registration_fitting_frames"],
                    "heldout_frame_ids": split_by_id[object_id]["heldout_validation_frames"],
                    "training_points_sha256": evidence_by_id[object_id]["training_points_sha256"],
                    "fitting_objective": None,
                    "failure_reason": "fake registration failure",
                    "warnings": [],
                }
            )
            continue
        matrix = [1.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        inverse = [1.0, 0.0, 0.0, -0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        registrations.append(
            {
                "candidate_id": candidate_id,
                "object_id": object_id,
                "status": "registered",
                "frozen_transform": {
                    "matrix_world_from_candidate": matrix,
                    "inverse_matrix": inverse,
                    "scale": 1.0,
                    "rotation_determinant": 1.0,
                    "rotation_degrees": 0.0,
                    "translation": [0.1, 0.0, 0.0],
                    "measured_surface_median_residual": 0.01,
                    "measured_surface_p90_residual": 0.025,
                    "normal_agreement": 0.9,
                    "symmetry_ambiguous": mode == "symmetry_ambiguous",
                },
                "fitting_frame_ids": split_by_id[object_id]["registration_fitting_frames"],
                "heldout_frame_ids": split_by_id[object_id]["heldout_validation_frames"],
                "training_points_sha256": evidence_by_id[object_id]["training_points_sha256"],
                "fitting_objective": 0.02,
                "failure_reason": None,
                "warnings": [],
            }
        )
    write_json(
        output_dir / "registration_manifest.json",
        {
            "request_sha256": sha256(request_path),
            "registrations": registrations,
            "runtime_seconds": 0.02 * len(registrations),
            "peak_gpu_memory_bytes": 0,
            "peak_host_memory_bytes": 0,
        },
    )
    manifest(output_dir, request_path, "registration", "completion_registration_worker")
    return 0


def metric_payload(backend: str, heldout: list[str], mode: str) -> dict[str, object]:
    if backend == "measured_partial_baseline":
        precision, recall, iou, depth, inliers, negative, front = (
            0.90,
            0.35,
            0.34,
            0.025,
            0.82,
            0.01,
            0.005,
        )
    elif backend == "sam3d_objects":
        precision, recall, iou, depth, inliers, negative, front = (
            0.78,
            0.62,
            0.52,
            0.045,
            0.72,
            0.04,
            0.02,
        )
    else:
        precision, recall, iou, depth, inliers, negative, front = (
            0.82,
            0.70,
            0.60,
            0.04,
            0.76,
            0.03,
            0.015,
        )
    if (
        mode in {"poor_heldout", "negative_space_violation"}
        and backend != "measured_partial_baseline"
    ):
        precision, recall, iou, depth, inliers, negative, front = (
            0.30,
            0.80,
            0.28,
            0.2,
            0.1,
            0.4,
            0.3,
        )
    return {
        "mask_precision": precision,
        "mask_recall": recall,
        "mask_iou": iou,
        "per_frame_iou": {frame_id: iou for frame_id in heldout},
        "dense_depth_relative_residual": depth,
        "depth_inlier_fraction": inliers,
        "negative_space_violation_ratio": negative,
        "front_of_scene_violation_ratio": front,
        "measured_point_to_candidate_median": 0.01,
        "measured_point_to_candidate_p90": 0.03,
        "normal_agreement": 0.88,
        "candidate_visible_coverage": recall,
        "validation_view_count": len(heldout),
        "visible_candidate_area": 100,
        "occluded_candidate_area": 20,
        "negative_space_violation_area": round(100 * negative),
        "front_of_scene_violation_area": round(100 * front),
    }


def write_fake_render(
    path: Path,
    candidate_id: str,
    frame_id: str,
    *,
    passed: bool,
) -> None:
    image = Image.new("RGB", (480, 270), (238, 241, 245))
    draw = ImageDraw.Draw(image)
    color = (35, 155, 85) if passed else (190, 55, 55)
    draw.rectangle((80, 55, 400, 220), outline=color, width=8)
    draw.text((18, 16), candidate_id, fill=(25, 35, 48))
    draw.text((18, 238), f"{frame_id} | {'PASS' if passed else 'REJECT'}", fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", compress_level=9)


def evaluate(request_path: Path, input_root: Path, output_dir: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    mode = request["evaluation_configuration"].get("fake_mode", "success")
    registration = json.loads(
        (input_root / request["registration_manifest_path"]).read_text(encoding="utf-8")
    )
    _, candidates = load_generations(input_root)
    baseline_by_object = {
        candidate["object_id"]: metric_payload(
            "measured_partial_baseline",
            request["heldout_inputs"][candidate["object_id"]]["frame_ids"],
            mode,
        )
        for candidate in candidates.values()
        if candidate["backend"] == "measured_partial_baseline"
    }
    evaluations = []
    config = request["evaluation_configuration"]
    for item in registration["registrations"]:
        if item["status"] == "registration_failed":
            continue
        candidate = candidates[item["candidate_id"]]
        backend = candidate["backend"]
        heldout = item["heldout_frame_ids"]
        metrics = metric_payload(backend, heldout, mode)
        baseline = baseline_by_object[item["object_id"]]
        gain = {
            "recall_gain_vs_measured_baseline": metrics["mask_recall"] - baseline["mask_recall"],
            "iou_gain_vs_measured_baseline": metrics["mask_iou"] - baseline["mask_iou"],
            "precision_change_vs_measured_baseline": (
                metrics["mask_precision"] - baseline["mask_precision"]
            ),
            "depth_residual_change": (
                metrics["dense_depth_relative_residual"] - baseline["dense_depth_relative_residual"]
            ),
            "visible_coverage_gain": (
                metrics["candidate_visible_coverage"] - baseline["candidate_visible_coverage"]
            ),
            "negative_space_change": (
                metrics["negative_space_violation_ratio"]
                - baseline["negative_space_violation_ratio"]
            ),
        }
        failed = []
        generated = backend != "measured_partial_baseline"
        if len(heldout) < config["minimum_validation_views"]:
            failed.append("minimum_validation_views")
        if generated and metrics["mask_iou"] < config["minimum_mask_iou"]:
            failed.append("minimum_mask_iou")
        if generated and metrics["mask_precision"] < config["minimum_mask_precision"]:
            failed.append("minimum_mask_precision")
        if (
            generated
            and metrics["dense_depth_relative_residual"]
            > config["maximum_median_relative_depth_residual"]
        ):
            failed.append("maximum_median_relative_depth_residual")
        if generated and metrics["depth_inlier_fraction"] < config["minimum_depth_inlier_fraction"]:
            failed.append("minimum_depth_inlier_fraction")
        if (
            generated
            and metrics["negative_space_violation_ratio"]
            > config["maximum_negative_space_violation_ratio"]
        ):
            failed.append("maximum_negative_space_violation_ratio")
        if (
            generated
            and metrics["front_of_scene_violation_ratio"]
            > config["maximum_front_of_scene_violation_ratio"]
        ):
            failed.append("maximum_front_of_scene_violation_ratio")
        if (
            generated
            and gain["recall_gain_vs_measured_baseline"]
            < config["minimum_recall_gain_over_measured_baseline"]
        ):
            failed.append("minimum_recall_gain_over_measured_baseline")
        if (
            generated
            and gain["precision_change_vs_measured_baseline"]
            < -config["maximum_precision_drop_from_measured_baseline"]
        ):
            failed.append("maximum_precision_drop_from_measured_baseline")
        render_paths = {}
        for frame_id in heldout:
            render_path = output_dir / "renders" / item["candidate_id"] / f"{frame_id}.png"
            write_fake_render(
                render_path,
                item["candidate_id"],
                frame_id,
                passed=not failed,
            )
            render_paths[frame_id] = render_path.relative_to(input_root).as_posix()
        evaluations.append(
            {
                "candidate_id": item["candidate_id"],
                "object_id": item["object_id"],
                "backend": backend,
                "transform_sha256": digest(item["frozen_transform"]),
                "heldout_frame_ids": heldout,
                "metrics": metrics,
                "measured_baseline_metrics": baseline,
                "completion_gain": gain,
                "passed_hard_gates": not failed,
                "failed_gates": failed,
                "evaluation_runtime_seconds": 0.01,
                "license_record": candidate["license_record"],
                "render_paths": render_paths,
            }
        )
    write_json(
        output_dir / "evaluation_manifest.json",
        {
            "registration_manifest_sha256": request["registration_manifest_sha256"],
            "evaluation_configuration": request["evaluation_configuration"],
            "evaluations": evaluations,
            "transforms_frozen_before_heldout_evaluation": True,
            "runtime_seconds": 0.01 * len(evaluations),
            "peak_gpu_memory_bytes": 0,
            "peak_host_memory_bytes": 0,
        },
    )
    manifest(output_dir, request_path, "evaluation", "completion_evaluation_worker")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["healthcheck", "prepare-evidence", "register", "evaluate"],
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.action == "healthcheck":
        print('{"available": true, "worker": "fake_completion_evaluation"}')
        return
    if args.input_root is None or args.output_dir is None:
        parser.error("--input-root and --output-dir are required")
    if args.action == "prepare-evidence":
        code = prepare_evidence(args.request, args.input_root, args.output_dir)
    elif args.action == "register":
        code = register(args.request, args.input_root, args.output_dir)
    else:
        code = evaluate(args.request, args.input_root, args.output_dir)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
