from __future__ import annotations

import hashlib
import json
import platform
import resource
import time
from pathlib import Path
from typing import Any

from alignment_worker.colmap_observations import (
    deterministic_split,
    prepare_sparse_observations,
)
from alignment_worker.depth_rendering import (
    merge_camera_metrics,
    render_alignment_metrics,
)
from alignment_worker.diagnostics import (
    ambiguous_candidate_ids,
    chunk_residual_metrics,
    residual_is_structured,
)
from alignment_worker.mesh_sampling import sample_mesh_surface
from alignment_worker.optimizer import (
    optimize_candidate,
    point_surface_metrics,
    select_candidate_by_training_objective,
)
from alignment_worker.previews import write_previews
from alignment_worker.schema import load_request
from alignment_worker.sim3 import (
    decompose_similarity,
    initialization_matrices,
)
from alignment_worker.transform_chain import audit_transform_chain
from alignment_worker.validation import evaluate_acceptance
from alignment_worker.version import __version__


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_hash(root: Path, relative_path: str, expected: str) -> Path:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"required alignment input is missing: {relative_path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"alignment input hash mismatch for {relative_path}: expected {expected}, got {actual}"
        )
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_mesh(path: Path) -> tuple[Any, Any]:
    import numpy as np
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("global alignment mesh is not a single triangle mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not len(vertices) or not len(faces):
        raise ValueError("global alignment mesh is empty")
    if not np.isfinite(vertices).all():
        raise ValueError("global alignment mesh contains non-finite vertices")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("global alignment mesh contains invalid face indices")
    return vertices, faces


def _unique_points(
    observations: list[dict[str, object]],
    point_ids: list[int],
) -> Any:
    import numpy as np

    point_set = set(point_ids)
    values: dict[int, tuple[float, float, float]] = {}
    for observation in observations:
        point_id = int(observation["point3d_id"])
        if point_id in point_set:
            values[point_id] = tuple(observation["point_world"])
    return np.asarray([values[key] for key in sorted(values)], dtype=np.float64)


def _point_only_metrics(point_metrics: dict[str, float | None]) -> dict[str, object]:
    return {
        "observation_count": 0,
        "sparse_depth_residual_median": None,
        "sparse_depth_residual_p75": None,
        "sparse_depth_residual_p90": None,
        "sparse_depth_residual_p95": None,
        "log_depth_residual_median": None,
        "inlier_fractions": {"0.05": 0.0, "0.10": 0.0, "0.20": 0.0},
        "mesh_pixel_coverage": 0.0,
        "point_to_surface_median_scene_diagonal": point_metrics["median"],
        "point_to_surface_p90_scene_diagonal": point_metrics["p90"],
        "point_to_plane_median_scene_diagonal": None,
        "bad_frame_fraction": 1.0,
    }


def _initialization_record(
    initialization: dict[str, object],
    scene_diagonal: float,
) -> dict[str, object]:
    decomposition = decompose_similarity(initialization["matrix"], scene_diagonal)
    return {
        "initialization_id": initialization["id"],
        "strategy": initialization["strategy"],
        "matrix": decomposition["matrix_original_mesh_to_aligned_colmap"],
        "initial_scale": decomposition["scale"],
        "initial_rotation_degrees": decomposition["rotation_degrees"],
        "initial_translation_scene_diagonal_ratio": decomposition[
            "translation_scene_diagonal_ratio"
        ],
        "selected_for_optimization": True,
        "rationale": {
            "identity": "required no-correction baseline",
            "centroid": "robust median-centroid translation",
            "extent": "robust 5-95 percentile extent ratio",
            "pca_0": "right-handed principal-axis hypothesis scored on held-out evidence",
        }[str(initialization["id"])],
    }


def run_inference(
    request_path: Path,
    input_root: Path,
    output_dir: Path,
) -> None:
    import numpy as np
    import scipy
    import torch

    started = time.monotonic()
    input_root = input_root.resolve()
    request_path = request_path.resolve()
    request_path.relative_to(input_root)
    output_dir = output_dir.resolve()
    output_dir.relative_to(input_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw" / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "previews").mkdir(parents=True, exist_ok=True)
    request = load_request(request_path)
    manifest_path = require_hash(input_root, request.manifest_path, request.manifest_sha256)
    camera_path = require_hash(
        input_root,
        request.camera_reconstruction_path,
        request.camera_reconstruction_sha256,
    )
    package_path = require_hash(
        input_root,
        request.camera_package_manifest_path,
        request.camera_package_sha256,
    )
    require_hash(input_root, request.cameras_txt_path, request.cameras_txt_sha256)
    images_path = require_hash(input_root, request.images_txt_path, request.images_txt_sha256)
    points_path = require_hash(input_root, request.points3d_txt_path, request.points3d_txt_sha256)
    global_path = require_hash(
        input_root,
        request.global_reconstruction_path,
        request.global_reconstruction_sha256,
    )
    mesh_path = require_hash(input_root, request.global_mesh_path, request.global_mesh_sha256)
    require_hash(
        input_root,
        request.global_worker_manifest_path,
        request.global_worker_manifest_sha256,
    )
    working_transform_path = require_hash(
        input_root,
        request.working_transform_path,
        request.working_transform_sha256,
    )
    chunk_transforms_path = require_hash(
        input_root,
        request.chunk_transforms_path,
        request.chunk_transforms_sha256,
    )
    require_hash(
        input_root,
        request.genrecon_camera_debug_path,
        request.genrecon_camera_debug_sha256,
    )
    working_mesh_path = None
    if request.working_mesh_path is not None and request.working_mesh_sha256 is not None:
        working_mesh_path = require_hash(
            input_root,
            request.working_mesh_path,
            request.working_mesh_sha256,
        )
    manifest = _load_json(manifest_path)
    camera = _load_json(camera_path)
    package = _load_json(package_path)
    _load_json(global_path)
    working_transform = _load_json(working_transform_path)
    chunk_transforms = _load_json(chunk_transforms_path)

    mesh_start = time.monotonic()
    vertices, faces = _load_mesh(mesh_path)
    working_vertices = _load_mesh(working_mesh_path)[0] if working_mesh_path is not None else None
    mesh_samples, _mesh_normals, scene_diagonal = sample_mesh_surface(
        vertices,
        faces,
        maximum_vertices=int(request.mesh_sampling_configuration["maximum_sample_vertices"]),
        maximum_face_centroids=int(
            request.mesh_sampling_configuration["maximum_sample_face_centroids"]
        ),
    )
    mesh_load_seconds = time.monotonic() - mesh_start

    sparse_start = time.monotonic()
    sparse_manifest, undistortion_records, all_sparse_points = prepare_sparse_observations(
        camera=camera,
        package_manifest=package,
        images_path=images_path,
        points3d_path=points_path,
        configuration=request.sparse_observation_configuration,
    )
    observations = sparse_manifest["observations"]
    split = deterministic_split(
        observations,
        request.registered_frame_ids,
        seed=request.seed,
    )
    training_points = _unique_points(observations, split["training_point_ids"])
    validation_points = _unique_points(observations, split["validation_point_ids"])
    minimum_correspondences = int(request.optimization_configuration["minimum_correspondences"])
    if len(training_points) < minimum_correspondences or len(validation_points) < 3:
        raise RuntimeError(
            "insufficient disjoint sparse points for bounded Sim(3) fitting and validation"
        )
    sparse_observation_seconds = time.monotonic() - sparse_start

    camera_centers = np.asarray(
        [
            pose["transform_world_from_camera"]["translation"]
            for pose in camera["poses"]
            if pose["frame_id"] in request.registered_frame_ids
        ],
        dtype=np.float64,
    )
    audit = audit_transform_chain(
        working_transform=working_transform,
        chunk_transforms=chunk_transforms,
        final_mesh_vertices=vertices,
        camera_centers=camera_centers,
        sparse_points=all_sparse_points,
        working_mesh_vertices=working_vertices,
        final_mesh_faces=faces,
        camera_pose=next(
            pose for pose in camera["poses"] if pose["frame_id"] == request.registered_frame_ids[0]
        ),
        undistorted_intrinsics=undistortion_records[request.registered_frame_ids[0]][
            "undistorted_intrinsics"
        ],
        face_chunk_size=int(request.mesh_sampling_configuration["face_chunk_size"]),
        tolerance=float(request.audit_configuration["roundtrip_tolerance"]),
    )

    correspondence_start = time.monotonic()
    initializations = initialization_matrices(mesh_samples, training_points)
    initialization_records = [
        _initialization_record(initialization, scene_diagonal) for initialization in initializations
    ]
    candidates = []
    iterations = []
    for index, initialization in enumerate(initializations):
        candidate, candidate_iterations = optimize_candidate(
            candidate_id=f"candidate_{index:02d}_{initialization['id']}",
            initialization_id=str(initialization["id"]),
            initial_matrix=initialization["matrix"],
            mesh_samples=mesh_samples,
            training_points=training_points,
            scene_diagonal=scene_diagonal,
            configuration=request.optimization_configuration,
        )
        candidates.append(candidate)
        iterations.extend(candidate_iterations)
    correspondence_seconds = time.monotonic() - correspondence_start
    best = select_candidate_by_training_objective(candidates)
    best["selected"] = True
    competing_candidates = ambiguous_candidate_ids(candidates, scene_diagonal=scene_diagonal)
    candidate_matrix = np.asarray(
        best["matrix_original_mesh_to_aligned_colmap"],
        dtype=np.float64,
    )
    identity = np.eye(4, dtype=np.float64)
    baseline_train_point = point_surface_metrics(
        mesh_samples,
        training_points,
        identity,
        scene_diagonal,
    )
    baseline_validation_point = point_surface_metrics(
        mesh_samples,
        validation_points,
        identity,
        scene_diagonal,
    )
    aligned_train_point = point_surface_metrics(
        mesh_samples,
        training_points,
        candidate_matrix,
        scene_diagonal,
    )
    aligned_validation_point = point_surface_metrics(
        mesh_samples,
        validation_points,
        candidate_matrix,
        scene_diagonal,
    )
    train_frames = set(split["training_frame_ids"])
    validation_frames = set(split["validation_frame_ids"])
    train_points = set(split["training_point_ids"])
    validation_point_ids = set(split["validation_point_ids"])
    training_observations = [
        item
        for item in observations
        if item["frame_id"] in train_frames and item["point3d_id"] in train_points
    ]
    validation_observations = [
        item
        for item in observations
        if item["frame_id"] in validation_frames and item["point3d_id"] in validation_point_ids
    ]

    baseline_render_start = time.monotonic()
    baseline_training, baseline_train_frames, _, _ = render_alignment_metrics(
        vertices=vertices,
        faces=faces,
        camera=camera,
        observations=training_observations,
        frame_ids=split["training_frame_ids"],
        undistortion_records=undistortion_records,
        matrix=identity,
        face_chunk_size=int(request.mesh_sampling_configuration["face_chunk_size"]),
        point_metrics=baseline_train_point,
        bad_frame_threshold=float(request.acceptance_configuration["bad_frame_threshold"]),
        split="training",
    )
    baseline_validation, baseline_validation_frames, baseline_pairs, baseline_records = (
        render_alignment_metrics(
            vertices=vertices,
            faces=faces,
            camera=camera,
            observations=validation_observations,
            frame_ids=split["validation_frame_ids"],
            undistortion_records=undistortion_records,
            matrix=identity,
            face_chunk_size=int(request.mesh_sampling_configuration["face_chunk_size"]),
            point_metrics=baseline_validation_point,
            bad_frame_threshold=float(request.acceptance_configuration["bad_frame_threshold"]),
            split="validation",
        )
    )
    baseline_render_seconds = time.monotonic() - baseline_render_start

    validation_render_start = time.monotonic()
    aligned_training, aligned_train_frames, _, _ = render_alignment_metrics(
        vertices=vertices,
        faces=faces,
        camera=camera,
        observations=training_observations,
        frame_ids=split["training_frame_ids"],
        undistortion_records=undistortion_records,
        matrix=candidate_matrix,
        face_chunk_size=int(request.mesh_sampling_configuration["face_chunk_size"]),
        point_metrics=aligned_train_point,
        bad_frame_threshold=float(request.acceptance_configuration["bad_frame_threshold"]),
        split="training",
    )
    aligned_validation, aligned_validation_frames, aligned_pairs, aligned_records = (
        render_alignment_metrics(
            vertices=vertices,
            faces=faces,
            camera=camera,
            observations=validation_observations,
            frame_ids=split["validation_frame_ids"],
            undistortion_records=undistortion_records,
            matrix=candidate_matrix,
            face_chunk_size=int(request.mesh_sampling_configuration["face_chunk_size"]),
            point_metrics=aligned_validation_point,
            bad_frame_threshold=float(request.acceptance_configuration["bad_frame_threshold"]),
            split="validation",
        )
    )
    validation_render_seconds = time.monotonic() - validation_render_start
    camera_metrics = merge_camera_metrics(
        [*baseline_train_frames, *baseline_validation_frames],
        [*aligned_train_frames, *aligned_validation_frames],
    )
    transform = best["transform"]
    status, accepted, acceptance_checks, failure_reason, global_sufficient = evaluate_acceptance(
        audit=audit,
        transform=transform,
        baseline=baseline_validation,
        aligned=aligned_validation,
        configuration={
            **request.acceptance_configuration,
            **{
                key: request.optimization_configuration[key]
                for key in (
                    "min_scale",
                    "max_scale",
                    "max_rotation_degrees_from_identity",
                    "max_translation_scene_diagonals",
                )
            },
        },
    )
    if status == "identity_already_consistent":
        transform = decompose_similarity(identity, scene_diagonal)
        aligned_training = baseline_training
        aligned_validation = baseline_validation
        aligned_pairs = baseline_pairs
        accepted = True
        global_sufficient = True
    elif (
        not accepted
        and float(baseline_validation["inlier_fractions"]["0.10"]) < 0.05
        and float(aligned_validation["inlier_fractions"]["0.10"]) < 0.05
        and (
            aligned_validation["point_to_surface_median_scene_diagonal"] is None
            or baseline_validation["point_to_surface_median_scene_diagonal"] is None
            or float(aligned_validation["point_to_surface_median_scene_diagonal"])
            >= float(baseline_validation["point_to_surface_median_scene_diagonal"]) * 0.95
        )
    ):
        status = "generecon_geometry_inconsistent_with_colmap"
        failure_reason = (
            "neither global depth agreement nor sampled point-to-surface agreement "
            "improves enough to support one global correction"
        )

    selected_id = str(best["candidate_id"])
    for candidate in candidates:
        train_point = candidate.pop("training_point_metrics")
        validation_point = point_surface_metrics(
            mesh_samples,
            validation_points,
            candidate["matrix_original_mesh_to_aligned_colmap"],
            scene_diagonal,
        )
        candidate.pop("transform")
        if candidate["candidate_id"] == selected_id:
            candidate["training_metrics"] = aligned_training
            candidate["validation_metrics"] = aligned_validation
        else:
            candidate["training_metrics"] = _point_only_metrics(train_point)
            candidate["validation_metrics"] = _point_only_metrics(validation_point)
        if not candidate["selected"] and candidate["rejection_reason"] is None:
            candidate["rejection_reason"] = "higher_training_point_surface_objective"

    chunks = chunk_residual_metrics(
        baseline_records=baseline_records,
        aligned_records=aligned_records,
        working_transform=working_transform,
        chunk_transforms=chunk_transforms,
    )
    structured = residual_is_structured(chunks)
    if structured:
        global_sufficient = False
        if accepted:
            status = "global_sim3_insufficient"
            accepted = False
            failure_reason = "held-out residual remains strongly structured across GenRecon chunks"
    if competing_candidates:
        global_sufficient = False
        accepted = False
        status = "global_sim3_insufficient"
        failure_reason = (
            "multiple materially different Sim(3) candidates have equivalent held-out objectives"
        )

    provenance = {
        "adapter_name": "camera_mesh_alignment",
        "adapter_version": "0.1.0",
        "configuration": {
            "audit": request.audit_configuration,
            "sparse_observations": request.sparse_observation_configuration,
            "mesh_sampling": request.mesh_sampling_configuration,
            "optimization": request.optimization_configuration,
            "acceptance": request.acceptance_configuration,
        },
        "input_artifact_paths": [
            request.manifest_path,
            request.camera_reconstruction_path,
            request.camera_package_manifest_path,
            request.global_reconstruction_path,
            request.global_mesh_path,
            request.working_transform_path,
            request.chunk_transforms_path,
            request.genrecon_camera_debug_path,
        ],
        "output_artifact_paths": [
            "reconstruction/alignment/alignment.json",
            "reconstruction/alignment/diagnostics.json",
        ],
        "timestamp": manifest["provenance"]["timestamp"],
        "confidence": {
            "score": float(aligned_validation["inlier_fractions"]["0.10"]),
            "method": "heldout_sparse_depth_inlier_fraction",
            "notes": "Association of an existing mesh to arbitrary COLMAP coordinates only.",
        },
        "source": "fused",
    }
    alignment = {
        "schema_version": "0.1.0",
        "status": status,
        "accepted": accepted,
        "transform": transform,
        "baseline_training_metrics": baseline_training,
        "aligned_training_metrics": aligned_training,
        "baseline_validation_metrics": baseline_validation,
        "aligned_validation_metrics": aligned_validation,
        "acceptance_checks": acceptance_checks,
        "failure_reason": failure_reason,
        "coordinate_convention": request.coordinate_convention,
        "scale_status": "scale_ambiguous",
        "transform_chain_audit_path": "reconstruction/alignment/transform_chain_audit.json",
        "dataset_split_path": "reconstruction/alignment/dataset_split.json",
        "candidate_id": selected_id,
        "provenance": provenance,
        "warnings": [
            "A fitted similarity scale remains arbitrary and is not metric.",
            *audit["warnings"],
        ],
    }
    outlier_ids = [str(item["frame_id"]) for item in camera_metrics if bool(item["outlier"])]
    timings = {
        "mesh_load": mesh_load_seconds,
        "sparse_observation_preparation": sparse_observation_seconds,
        "baseline_rendering": baseline_render_seconds,
        "correspondence": correspondence_seconds,
        "optimization": correspondence_seconds,
        "validation_rendering": validation_render_seconds,
    }
    peak_gpu = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    peak_host = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    diagnostics = {
        "schema_version": "0.1.0",
        "initializations": initialization_records,
        "camera_metrics": camera_metrics,
        "chunk_metrics": chunks,
        "residual_is_locally_structured": structured,
        "candidate_solution_ambiguous": bool(competing_candidates),
        "competing_candidate_ids": competing_candidates,
        "global_similarity_sufficient": global_sufficient,
        "transform_chain_consistent": audit["status"] == "consistent",
        "camera_outlier_frame_ids": outlier_ids,
        "best_candidate_id": selected_id,
        "diagnosis": (
            "One bounded global Sim(3) passed held-out gates."
            if accepted
            else "Residual evidence does not support applying one global Sim(3)."
        ),
        "performance_seconds": timings,
        "peak_gpu_memory_bytes": peak_gpu,
        "peak_host_memory_bytes": peak_host,
        "warnings": [],
    }

    write_json(output_dir / "transform_chain_audit.json", audit)
    write_json(output_dir / "sparse_observations.json", sparse_manifest)
    write_json(output_dir / "dataset_split.json", split)
    write_json(output_dir / "alignment.json", alignment)
    write_json(
        output_dir / "candidates.json",
        {"schema_version": "0.1.0", "candidates": candidates},
    )
    write_json(
        output_dir / "iterations.json",
        {"schema_version": "0.1.0", "iterations": iterations},
    )
    write_json(output_dir / "diagnostics.json", diagnostics)
    preview_start = time.monotonic()
    homogeneous_samples = np.concatenate(
        (mesh_samples, np.ones((len(mesh_samples), 1), dtype=np.float64)),
        axis=1,
    )
    aligned_mesh_samples = (candidate_matrix @ homogeneous_samples.T).T[:, :3]
    preview_manifest = write_previews(
        output_dir=output_dir,
        audit=audit,
        baseline_metrics=baseline_validation,
        aligned_metrics=aligned_validation,
        camera_metrics=camera_metrics,
        chunk_metrics=chunks,
        baseline_pairs=baseline_pairs,
        aligned_pairs=aligned_pairs,
        status=status,
        transform=transform,
        mesh_samples=mesh_samples,
        sparse_points=all_sparse_points,
        aligned_mesh_samples=aligned_mesh_samples,
    )
    preview_seconds = time.monotonic() - preview_start
    timings["preview"] = preview_seconds
    write_json(output_dir / "preview_manifest.json", preview_manifest)

    raw_paths = [
        path.relative_to(input_root).as_posix()
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    ]
    worker_manifest = {
        "schema_version": "0.1.0",
        "worker_version": __version__,
        "backend": "nvdiffrast_scipy",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "nvdiffrast_version": "installed",
        "device": "cuda",
        "device_name": torch.cuda.get_device_name(0),
        "request_sha256": sha256_file(request_path),
        "manifest_sha256": request.manifest_sha256,
        "frame_sequence_digest": request.frame_sequence_digest,
        "camera_reconstruction_sha256": request.camera_reconstruction_sha256,
        "camera_package_sha256": request.camera_package_sha256,
        "global_reconstruction_sha256": request.global_reconstruction_sha256,
        "global_mesh_sha256": request.global_mesh_sha256,
        "mesh_load_seconds": mesh_load_seconds,
        "sparse_observation_seconds": sparse_observation_seconds,
        "baseline_render_seconds": baseline_render_seconds,
        "correspondence_seconds": correspondence_seconds,
        "optimization_seconds": correspondence_seconds,
        "validation_render_seconds": validation_render_seconds,
        "preview_seconds": preview_seconds,
        "runtime_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": peak_gpu,
        "peak_host_memory_bytes": peak_host,
        "raw_output_paths": raw_paths,
        "warnings": [],
    }
    write_json(output_dir / "worker_manifest.json", worker_manifest)
