from __future__ import annotations

from typing import Any

from alignment_worker.sim3 import apply_transform


def ambiguous_candidate_ids(
    candidates: list[dict[str, object]],
    *,
    scene_diagonal: float,
    relative_objective_tolerance: float = 0.01,
    minimum_rotation_degrees: float = 10.0,
    minimum_log_scale_difference: float = 0.05,
    minimum_translation_scene_diagonals: float = 0.05,
) -> list[str]:
    import math

    import numpy as np

    viable = [
        candidate
        for candidate in candidates
        if bool(candidate["finite"])
        and not bool(candidate["hit_parameter_bound"])
        and not bool(candidate["correspondence_collapsed"])
    ]
    if len(viable) < 2:
        return []
    viable.sort(key=lambda item: float(item["objective"]))
    best = viable[0]
    best_objective = max(float(best["objective"]), 1e-12)
    best_matrix = np.asarray(
        best["matrix_original_mesh_to_aligned_colmap"],
        dtype=np.float64,
    )
    best_scale = float(abs(np.linalg.det(best_matrix[:3, :3])) ** (1.0 / 3.0))
    competitors: list[str] = []
    for candidate in viable[1:]:
        objective = float(candidate["objective"])
        if objective > best_objective * (1.0 + relative_objective_tolerance) + 1e-12:
            break
        matrix = np.asarray(
            candidate["matrix_original_mesh_to_aligned_colmap"],
            dtype=np.float64,
        )
        scale = float(abs(np.linalg.det(matrix[:3, :3])) ** (1.0 / 3.0))
        relative_rotation = (matrix[:3, :3] / max(scale, 1e-12)) @ (
            best_matrix[:3, :3] / max(best_scale, 1e-12)
        ).T
        cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
        rotation_degrees = math.degrees(math.acos(cosine))
        log_scale_difference = abs(math.log(scale / max(best_scale, 1e-12)))
        translation_difference = float(
            np.linalg.norm(matrix[:3, 3] - best_matrix[:3, 3]) / max(scene_diagonal, 1e-12)
        )
        if (
            rotation_degrees >= minimum_rotation_degrees
            or log_scale_difference >= minimum_log_scale_difference
            or translation_difference >= minimum_translation_scene_diagonals
        ):
            competitors.append(str(candidate["candidate_id"]))
    return competitors


def chunk_residual_metrics(
    *,
    baseline_records: list[dict[str, object]],
    aligned_records: list[dict[str, object]],
    working_transform: dict[str, Any],
    chunk_transforms: dict[str, Any],
) -> list[dict[str, object]]:
    import numpy as np

    aligned_by_key = {(item["frame_id"], item["point3d_id"]): item for item in aligned_records}
    colmap_to_working = np.asarray(
        working_transform["matrix_colmap_to_working"],
        dtype=np.float64,
    )
    chunks = chunk_transforms.get("chunks", [])
    grouped: dict[str, tuple[list[float], list[float]]] = {}
    for baseline in baseline_records:
        key = (baseline["frame_id"], baseline["point3d_id"])
        aligned = aligned_by_key.get(key)
        if aligned is None:
            continue
        point_working = apply_transform([baseline["point_world"]], colmap_to_working)[0]
        assigned = "unassigned"
        for index, chunk in enumerate(chunks):
            matrix = chunk.get("M_original_to_chunk")
            if matrix is None:
                continue
            local = apply_transform([point_working], matrix)[0]
            if np.all(np.abs(local) <= 0.5 + 1e-9):
                assigned = str(chunk.get("index", index))
                break
        baseline_values, aligned_values = grouped.setdefault(assigned, ([], []))
        baseline_values.append(float(baseline["residual"]))
        aligned_values.append(float(aligned["residual"]))
    output = []
    for chunk_id in sorted(grouped):
        baseline_values, aligned_values = grouped[chunk_id]
        output.append(
            {
                "chunk_id": chunk_id,
                "observation_count": len(aligned_values),
                "baseline_median_residual": float(np.median(baseline_values)),
                "aligned_median_residual": float(np.median(aligned_values)),
                "aligned_p90_residual": float(np.percentile(aligned_values, 90)),
                "aligned_inlier_fraction": sum(value <= 0.10 for value in aligned_values)
                / len(aligned_values),
            }
        )
    return output


def residual_is_structured(chunks: list[dict[str, object]]) -> bool:
    medians = [
        float(item["aligned_median_residual"])
        for item in chunks
        if item["aligned_median_residual"] is not None and item["observation_count"] >= 5
    ]
    if len(medians) < 2:
        return False
    minimum = min(medians)
    maximum = max(medians)
    return maximum > max(0.20, minimum * 2.0)
