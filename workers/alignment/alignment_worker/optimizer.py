from __future__ import annotations

from typing import Any

from alignment_worker.robust_losses import cauchy_loss, robust_inlier_mask
from alignment_worker.sim3 import apply_transform, decompose_similarity, umeyama_similarity


def point_surface_metrics(
    mesh_samples: Any,
    sparse_points: Any,
    matrix: Any,
    scene_diagonal: float,
) -> dict[str, float | None]:
    import numpy as np
    from scipy.spatial import cKDTree

    transformed = apply_transform(mesh_samples, matrix)
    tree = cKDTree(transformed)
    distances, _ = tree.query(np.asarray(sparse_points, dtype=np.float64), workers=-1)
    normalized = distances / max(scene_diagonal, 1e-12)
    return {
        "median": float(np.median(normalized)) if len(normalized) else None,
        "p90": float(np.percentile(normalized, 90)) if len(normalized) else None,
    }


def optimize_candidate(
    *,
    candidate_id: str,
    initialization_id: str,
    initial_matrix: Any,
    mesh_samples: Any,
    training_points: Any,
    scene_diagonal: float,
    configuration: dict[str, Any],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    import numpy as np
    from scipy.spatial import cKDTree

    matrix = np.asarray(initial_matrix, dtype=np.float64)
    iterations: list[dict[str, object]] = []
    correspondence_collapsed = False
    converged = False
    maximum_iterations = int(configuration["maximum_iterations"])
    for iteration in range(maximum_iterations):
        transformed = apply_transform(mesh_samples, matrix)
        tree = cKDTree(transformed)
        distances, indices = tree.query(training_points, workers=-1)
        mask = robust_inlier_mask(distances, float(configuration["correspondence_mad_multiplier"]))
        inlier_count = int(mask.sum())
        if inlier_count < int(configuration["minimum_correspondences"]):
            correspondence_collapsed = True
            break
        source = np.asarray(mesh_samples, dtype=np.float64)[indices[mask]]
        target = np.asarray(training_points, dtype=np.float64)[mask]
        updated = umeyama_similarity(source, target)
        delta = float(np.linalg.norm(updated - matrix, ord="fro"))
        matrix = updated
        decomposition = decompose_similarity(matrix, scene_diagonal)
        loss = cauchy_loss(
            distances[mask] / max(scene_diagonal, 1e-12),
            float(configuration["cauchy_scale"]),
        )
        converged = delta <= float(configuration["convergence_tolerance"])
        iterations.append(
            {
                "candidate_id": candidate_id,
                "iteration": iteration,
                "correspondence_count": len(distances),
                "inlier_count": inlier_count,
                "loss": loss,
                "scale": decomposition["scale"],
                "rotation_degrees": decomposition["rotation_degrees"],
                "translation_scene_diagonal_ratio": decomposition[
                    "translation_scene_diagonal_ratio"
                ],
                "validation_point_to_surface_median": None,
                "converged": converged,
            }
        )
        if converged:
            break
    decomposition = decompose_similarity(matrix, scene_diagonal)
    bounds = {
        "scale": (
            float(configuration["min_scale"])
            <= decomposition["scale"]
            <= float(configuration["max_scale"])
        ),
        "rotation": decomposition["rotation_degrees"]
        <= float(configuration["max_rotation_degrees_from_identity"]),
        "translation": decomposition["translation_scene_diagonal_ratio"]
        <= float(configuration["max_translation_scene_diagonals"]),
    }
    train_point = point_surface_metrics(
        mesh_samples,
        training_points,
        matrix,
        scene_diagonal,
    )
    objective = train_point["median"] if train_point["median"] is not None else float("inf")
    candidate = {
        "candidate_id": candidate_id,
        "initialization_id": initialization_id,
        "matrix_original_mesh_to_aligned_colmap": decomposition[
            "matrix_original_mesh_to_aligned_colmap"
        ],
        "scale": decomposition["scale"],
        "rotation_degrees": decomposition["rotation_degrees"],
        "translation_scene_diagonal_ratio": decomposition["translation_scene_diagonal_ratio"],
        "finite": bool(np.isfinite(matrix).all()),
        "hit_parameter_bound": not all(bounds.values()),
        "correspondence_collapsed": correspondence_collapsed,
        "training_point_metrics": train_point,
        "objective": float(objective),
        "selected": False,
        "rejection_reason": (
            "correspondence_collapse"
            if correspondence_collapsed
            else "implausible_transform"
            if not all(bounds.values())
            else None
        ),
        "transform": decomposition,
    }
    return candidate, iterations


def select_candidate_by_training_objective(
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    plausible = [
        candidate
        for candidate in candidates
        if candidate["finite"]
        and not candidate["hit_parameter_bound"]
        and not candidate["correspondence_collapsed"]
    ]
    return min(
        plausible or candidates,
        key=lambda item: (float(item["objective"]), str(item["candidate_id"])),
    )
