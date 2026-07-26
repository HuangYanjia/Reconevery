from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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


def preview(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (640, 360), (244, 246, 248))
    draw = ImageDraw.Draw(image)
    draw.text((28, 24), title, fill=(20, 30, 40))
    draw.line((60, 300, 570, 70), fill=(60, 120, 190), width=3)
    image.save(path, format="PNG")


def metrics(
    median: float,
    p90: float,
    inlier: float,
    *,
    observations: int = 120,
    coverage: float = 0.8,
    point_distance: float = 0.05,
) -> dict[str, object]:
    return {
        "observation_count": observations,
        "sparse_depth_residual_median": median,
        "sparse_depth_residual_p75": (median + p90) / 2,
        "sparse_depth_residual_p90": p90,
        "sparse_depth_residual_p95": p90 * 1.1,
        "log_depth_residual_median": median,
        "inlier_fractions": {"0.05": inlier / 2, "0.10": inlier, "0.20": min(1.0, inlier * 1.4)},
        "mesh_pixel_coverage": coverage,
        "point_to_surface_median_scene_diagonal": point_distance,
        "point_to_surface_p90_scene_diagonal": point_distance * 2,
        "point_to_plane_median_scene_diagonal": None,
        "bad_frame_fraction": 0.0 if inlier >= 0.5 else 0.5,
    }


def transform(scale: float, translation: tuple[float, float, float]) -> dict[str, object]:
    matrix = [
        [scale, 0.0, 0.0, translation[0]],
        [0.0, scale, 0.0, translation[1]],
        [0.0, 0.0, scale, translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    inverse_scale = 1.0 / scale
    inverse = [
        [inverse_scale, 0.0, 0.0, -translation[0] * inverse_scale],
        [0.0, inverse_scale, 0.0, -translation[1] * inverse_scale],
        [0.0, 0.0, inverse_scale, -translation[2] * inverse_scale],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return {
        "matrix_original_mesh_to_aligned_colmap": matrix,
        "inverse_matrix": inverse,
        "scale": scale,
        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "rotation_axis_angle": [0.0, 0.0, 0.0],
        "rotation_degrees": 0.0,
        "translation": list(translation),
        "translation_scene_diagonal_ratio": math.sqrt(sum(v * v for v in translation)) / 10,
        "determinant": scale**3,
        "roundtrip_error": 0.0,
    }


def healthcheck(config_path: Path) -> int:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "available": True,
                "worker_version": payload["worker_version"],
                "backend": "fake",
            },
            sort_keys=True,
        )
    )
    return 0


def infer(request_path: Path, input_root: Path, output_dir: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    mode = request["optimization_configuration"].get("fake_mode", "success_full_sim3")
    if mode == "timeout":
        time.sleep(60)
    if mode == "nonzero":
        print("fake alignment worker failed", file=sys.stderr)
        return 9
    if mode == "oom":
        print("CUDA out of memory during fake alignment", file=sys.stderr)
        return 10
    if mode == "interruption":
        os.kill(os.getpid(), 2)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((input_root / request["manifest_path"]).read_text(encoding="utf-8"))
    registered = request["registered_frame_ids"]
    training_frames = registered[::2]
    validation_frames = registered[1::2] or registered[-1:]
    training_points = [2, 4, 6, 8]
    validation_points = [1, 3, 5, 7]
    chain_consistent = mode != "transform_chain_bug"
    identity = mode == "identity"
    accepted = True
    scale = 1.0 if identity or mode == "success_rigid" else 1.2
    translation = (0.0, 0.0, 0.0) if identity or mode == "success_scale" else (0.2, -0.1, 0.05)
    if mode == "implausible_scale":
        scale = 8.0
    candidate_transform = transform(scale, translation)
    baseline = metrics(0.50, 0.75, 0.10, point_distance=0.20)
    aligned = metrics(0.10, 0.20, 0.65, coverage=0.79, point_distance=0.04)
    status = "accepted_global_sim3"
    failure_reason = None
    if identity:
        status = "identity_already_consistent"
        baseline = metrics(0.04, 0.08, 0.75, point_distance=0.02)
        aligned = baseline
        candidate_transform = transform(1.0, (0.0, 0.0, 0.0))
    elif mode == "transform_chain_bug":
        status = "global_sim3_insufficient"
        accepted = False
        failure_reason = "transform chain audit failed"
    elif mode in {"no_improvement", "validation_regression"}:
        status = "rejected_no_validation_improvement"
        accepted = False
        aligned = metrics(0.55, 0.80, 0.08, point_distance=0.21)
        failure_reason = "held-out validation did not improve"
    elif mode in {"implausible_scale", "excessive_rotation", "excessive_translation"}:
        status = "rejected_implausible_transform"
        accepted = False
        failure_reason = "candidate violates conservative bounds"
    elif mode in {"insufficient_observations", "correspondence_collapse"}:
        status = "global_sim3_insufficient"
        accepted = False
        aligned = metrics(0.30, 0.50, 0.20, observations=20)
        failure_reason = mode.replace("_", " ")
    elif mode == "local_deformation":
        status = "global_sim3_insufficient"
        accepted = False
        aligned = metrics(0.20, 0.55, 0.35, point_distance=0.08)
        failure_reason = "residual remains locally structured"
    elif mode == "symmetric_ambiguity":
        status = "global_sim3_insufficient"
        accepted = False
        failure_reason = "multiple equivalent global similarity candidates"
    audit = {
        "schema_version": "0.1.0",
        "status": "consistent" if chain_consistent else "transform_chain_bug",
        "stages": [
            {
                "stage_id": "A_colmap_arbitrary",
                "transform_source": "fake identity",
                "matrix_from_previous": transform(1.0, (0.0, 0.0, 0.0))[
                    "matrix_original_mesh_to_aligned_colmap"
                ],
                "matrix_to_previous": transform(1.0, (0.0, 0.0, 0.0))["inverse_matrix"],
                "determinant": 1.0,
                "rotation_orthogonality_error": 0.0,
                "scale": 1.0,
                "translation": [0.0, 0.0, 0.0],
                "roundtrip_error": 0.0,
                "mesh_bounds_min": [-1.0, -1.0, -1.0],
                "mesh_bounds_max": [1.0, 1.0, 1.0],
                "camera_center_bounds_min": [-1.0, -1.0, -1.0],
                "camera_center_bounds_max": [1.0, 1.0, 1.0],
                "sparse_point_bounds_min": [-1.0, -1.0, -1.0],
                "sparse_point_bounds_max": [1.0, 1.0, 1.0],
            }
        ],
        "colmap_working_roundtrip_error": 0.0 if chain_consistent else 1.0,
        "camera_basis_roundtrip_error": 0.0,
        "sampled_mesh_roundtrip_error": 0.0,
        "pre_post_render_depth_error": 0.0,
        "pre_post_render_silhouette_iou": 1.0,
        "pre_post_render_equivalent": chain_consistent,
        "raw_working_mesh_available": False,
        "raw_working_scene_available": False,
        "checks": {"roundtrip": chain_consistent},
        "warnings": [],
    }
    sparse = {
        "schema_version": "0.1.0",
        "observations": [],
        "total_colmap_points": 8,
        "total_raw_observations": 0,
        "retained_observations": 0,
        "rejected_observations": 0,
        "filtering_configuration": request["sparse_observation_configuration"],
        "undistortion_records": [],
        "warnings": ["Fake worker omits verbose sparse observations."],
    }
    split = {
        "schema_version": "0.1.0",
        "strategy": "alternating_registered_frames_and_point_ids",
        "training_frame_ids": training_frames,
        "validation_frame_ids": validation_frames,
        "training_point_ids": training_points,
        "validation_point_ids": validation_points,
        "training_observation_count": 120,
        "validation_observation_count": 120,
        "split_seed": request["seed"],
    }
    checks = {
        "transform_chain_consistent": chain_consistent,
        "minimum_validation_observations": aligned["observation_count"] >= 100,
        "median_residual_improvement": aligned["sparse_depth_residual_median"]
        < baseline["sparse_depth_residual_median"],
        "p90_residual_improvement": aligned["sparse_depth_residual_p90"]
        < baseline["sparse_depth_residual_p90"],
        "inlier_fraction_improvement": aligned["inlier_fractions"]["0.10"]
        > baseline["inlier_fractions"]["0.10"],
        "mesh_coverage_preserved": True,
        "bad_frame_fraction": aligned["bad_frame_fraction"] <= 0.3,
        "finite_transform": mode != "non_finite_transform",
        "positive_scale": mode != "singular_transform",
        "proper_rotation": True,
        "scale_plausible": 0.25 <= scale <= 4.0,
        "rotation_plausible": mode != "excessive_rotation",
        "translation_plausible": mode != "excessive_translation",
        "point_surface_not_degraded": aligned["point_to_surface_median_scene_diagonal"]
        <= baseline["point_to_surface_median_scene_diagonal"],
    }
    provenance = {
        "adapter_name": "camera_mesh_alignment",
        "adapter_version": "0.1.0",
        "configuration": {"fake_mode": mode},
        "input_artifact_paths": [
            request["manifest_path"],
            request["camera_reconstruction_path"],
            request["global_mesh_path"],
        ],
        "output_artifact_paths": ["reconstruction/alignment/alignment.json"],
        "timestamp": manifest["provenance"]["timestamp"],
        "confidence": {
            "score": aligned["inlier_fractions"]["0.10"],
            "method": "fake_heldout_depth",
            "notes": None,
        },
        "source": "fused",
    }
    alignment = {
        "schema_version": "0.1.0",
        "status": status,
        "accepted": accepted,
        "transform": candidate_transform,
        "baseline_training_metrics": baseline,
        "aligned_training_metrics": aligned,
        "baseline_validation_metrics": baseline,
        "aligned_validation_metrics": aligned,
        "acceptance_checks": checks,
        "failure_reason": failure_reason,
        "coordinate_convention": request["coordinate_convention"],
        "scale_status": "scale_ambiguous",
        "transform_chain_audit_path": "reconstruction/alignment/transform_chain_audit.json",
        "dataset_split_path": "reconstruction/alignment/dataset_split.json",
        "candidate_id": "candidate_00",
        "provenance": provenance,
        "warnings": [],
    }
    initialization = {
        "initialization_id": "identity",
        "strategy": "identity",
        "matrix": transform(1.0, (0.0, 0.0, 0.0))["matrix_original_mesh_to_aligned_colmap"],
        "initial_scale": 1.0,
        "initial_rotation_degrees": 0.0,
        "initial_translation_scene_diagonal_ratio": 0.0,
        "selected_for_optimization": True,
        "rationale": "deterministic fake identity baseline",
    }
    candidate = {
        "candidate_id": "candidate_00",
        "initialization_id": "identity",
        "matrix_original_mesh_to_aligned_colmap": candidate_transform[
            "matrix_original_mesh_to_aligned_colmap"
        ],
        "scale": candidate_transform["scale"],
        "rotation_degrees": candidate_transform["rotation_degrees"],
        "translation_scene_diagonal_ratio": candidate_transform["translation_scene_diagonal_ratio"],
        "finite": mode != "non_finite_transform",
        "hit_parameter_bound": mode
        in {
            "implausible_scale",
            "excessive_rotation",
            "excessive_translation",
        },
        "correspondence_collapsed": mode == "correspondence_collapse",
        "training_metrics": aligned,
        "validation_metrics": aligned,
        "objective": aligned["point_to_surface_median_scene_diagonal"],
        "selected": True,
        "rejection_reason": failure_reason,
    }
    camera_metrics = [
        {
            "frame_id": frame_id,
            "camera_id": "camera_0001",
            "valid_sparse_observations": 20,
            "mesh_pixel_coverage": 0.79,
            "baseline_median_residual": 0.5,
            "aligned_median_residual": 0.1,
            "baseline_p90_residual": 0.75,
            "aligned_p90_residual": 0.2,
            "baseline_inlier_fraction": 0.1,
            "aligned_inlier_fraction": 0.65,
            "visible_mesh_face_count": 12,
            "outlier": False,
            "outlier_reason": None,
            "split": "training" if frame_id in training_frames else "validation",
        }
        for frame_id in registered
    ]
    diagnostics = {
        "schema_version": "0.1.0",
        "initializations": [initialization],
        "camera_metrics": camera_metrics,
        "chunk_metrics": [
            {
                "chunk_id": "0",
                "observation_count": 120,
                "baseline_median_residual": 0.5,
                "aligned_median_residual": 0.1,
                "aligned_p90_residual": 0.2,
                "aligned_inlier_fraction": 0.65,
            }
        ],
        "residual_is_locally_structured": mode == "local_deformation",
        "candidate_solution_ambiguous": mode == "symmetric_ambiguity",
        "competing_candidate_ids": (
            ["candidate_01_symmetric"] if mode == "symmetric_ambiguity" else []
        ),
        "global_similarity_sufficient": accepted,
        "transform_chain_consistent": chain_consistent,
        "camera_outlier_frame_ids": [],
        "best_candidate_id": "candidate_00",
        "diagnosis": "deterministic fake alignment result",
        "performance_seconds": {
            "mesh_load": 0.01,
            "sparse_observation_preparation": 0.01,
            "baseline_rendering": 0.01,
            "correspondence": 0.01,
            "optimization": 0.01,
            "validation_rendering": 0.01,
            "preview": 0.01,
        },
        "peak_gpu_memory_bytes": None,
        "peak_host_memory_bytes": 1024,
        "warnings": [],
    }
    write_json(output_dir / "transform_chain_audit.json", audit)
    write_json(output_dir / "sparse_observations.json", sparse)
    write_json(output_dir / "dataset_split.json", split)
    write_json(output_dir / "alignment.json", alignment)
    write_json(
        output_dir / "candidates.json",
        {"schema_version": "0.1.0", "candidates": [candidate]},
    )
    write_json(
        output_dir / "iterations.json",
        {
            "schema_version": "0.1.0",
            "iterations": [
                {
                    "candidate_id": "candidate_00",
                    "iteration": 0,
                    "correspondence_count": 120,
                    "inlier_count": 100,
                    "loss": 0.1,
                    "scale": scale,
                    "rotation_degrees": 0.0,
                    "translation_scene_diagonal_ratio": candidate_transform[
                        "translation_scene_diagonal_ratio"
                    ],
                    "validation_point_to_surface_median": aligned[
                        "point_to_surface_median_scene_diagonal"
                    ],
                    "converged": True,
                }
            ],
        },
    )
    write_json(output_dir / "diagnostics.json", diagnostics)
    preview_names = (
        "transform_chain_comparison",
        "baseline_depth_residual",
        "aligned_depth_residual",
        "baseline_vs_aligned_scatter",
        "per_camera_residuals",
        "per_chunk_residuals",
        "sparse_points_and_mesh_before",
        "sparse_points_and_mesh_after",
        "heldout_validation_summary",
    )
    for name in preview_names:
        preview(output_dir / "previews" / f"{name}.png", name.replace("_", " "))
    preview_manifest = {
        f"{name}_path": f"reconstruction/alignment/previews/{name}.png" for name in preview_names
    }
    write_json(output_dir / "preview_manifest.json", preview_manifest)
    if mode == "path_escape":
        (output_dir / "raw" / "escape").symlink_to("/tmp")
    worker = {
        "schema_version": "0.1.0",
        "worker_version": "0.1.0",
        "backend": "fake",
        "python_version": sys.version.split()[0],
        "numpy_version": None,
        "scipy_version": None,
        "torch_version": None,
        "cuda_version": None,
        "nvdiffrast_version": None,
        "device": "cpu",
        "device_name": "deterministic fake device",
        "request_sha256": sha256(request_path),
        "manifest_sha256": request["manifest_sha256"],
        "frame_sequence_digest": request["frame_sequence_digest"],
        "camera_reconstruction_sha256": (
            "0" * 64 if mode == "wrong_camera_hash" else request["camera_reconstruction_sha256"]
        ),
        "camera_package_sha256": (
            "0" * 64 if mode == "wrong_package_hash" else request["camera_package_sha256"]
        ),
        "global_reconstruction_sha256": request["global_reconstruction_sha256"],
        "global_mesh_sha256": (
            "0" * 64 if mode == "wrong_mesh_hash" else request["global_mesh_sha256"]
        ),
        "mesh_load_seconds": 0.01,
        "sparse_observation_seconds": 0.01,
        "baseline_render_seconds": 0.01,
        "correspondence_seconds": 0.01,
        "optimization_seconds": 0.01,
        "validation_render_seconds": 0.01,
        "preview_seconds": 0.01,
        "runtime_seconds": 0.08,
        "peak_gpu_memory_bytes": None,
        "peak_host_memory_bytes": 1024,
        "raw_output_paths": [
            f"reconstruction/alignment/previews/{name}.png" for name in preview_names
        ],
        "warnings": [],
    }
    if mode == "malformed_manifest":
        write_json(output_dir / "worker_manifest.json", {"bad": True})
    else:
        write_json(output_dir / "worker_manifest.json", worker)
    if mode == "missing_output":
        (output_dir / "alignment.json").unlink()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    health = commands.add_parser("healthcheck")
    health.add_argument("--config", type=Path, required=True)
    infer_parser = commands.add_parser("infer")
    infer_parser.add_argument("--request", type=Path, required=True)
    infer_parser.add_argument("--input-root", type=Path, required=True)
    infer_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "healthcheck":
        return healthcheck(args.config)
    return infer(args.request, args.input_root, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
