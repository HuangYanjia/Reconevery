from __future__ import annotations

from typing import Any


def relative_improvement(baseline: float | None, aligned: float | None) -> float:
    if baseline is None or aligned is None or baseline <= 0:
        return 0.0
    return (baseline - aligned) / baseline


def evaluate_acceptance(
    *,
    audit: dict[str, Any],
    transform: dict[str, Any],
    baseline: dict[str, Any],
    aligned: dict[str, Any],
    configuration: dict[str, Any],
) -> tuple[str, bool, dict[str, bool], str | None, bool]:
    baseline_inlier = float(baseline["inlier_fractions"]["0.10"])
    aligned_inlier = float(aligned["inlier_fractions"]["0.10"])
    coverage_ratio = float(aligned["mesh_pixel_coverage"]) / max(
        float(baseline["mesh_pixel_coverage"]),
        1e-12,
    )
    checks = {
        "transform_chain_consistent": audit["status"] == "consistent",
        "minimum_validation_observations": int(aligned["observation_count"])
        >= int(configuration["minimum_validation_observations"]),
        "median_residual_improvement": relative_improvement(
            baseline["sparse_depth_residual_median"],
            aligned["sparse_depth_residual_median"],
        )
        >= float(configuration["minimum_median_residual_relative_improvement"]),
        "p90_residual_improvement": relative_improvement(
            baseline["sparse_depth_residual_p90"],
            aligned["sparse_depth_residual_p90"],
        )
        >= float(configuration["minimum_p90_residual_relative_improvement"]),
        "inlier_fraction_improvement": aligned_inlier - baseline_inlier
        >= float(configuration["minimum_inlier_fraction_absolute_improvement"]),
        "mesh_coverage_preserved": coverage_ratio
        >= float(configuration["minimum_mesh_coverage_ratio_vs_baseline"]),
        "bad_frame_fraction": float(aligned["bad_frame_fraction"])
        <= float(configuration["maximum_bad_frame_fraction"]),
        "finite_transform": bool(transform["roundtrip_error"] < 1e-6),
        "positive_scale": float(transform["scale"]) > 0,
        "proper_rotation": float(transform["determinant"]) > 0,
        "scale_plausible": float(configuration["min_scale"])
        <= float(transform["scale"])
        <= float(configuration["max_scale"]),
        "rotation_plausible": float(transform["rotation_degrees"])
        <= float(configuration["max_rotation_degrees_from_identity"]),
        "translation_plausible": float(transform["translation_scene_diagonal_ratio"])
        <= float(configuration["max_translation_scene_diagonals"]),
        "point_surface_not_degraded": (
            aligned["point_to_surface_median_scene_diagonal"] is not None
            and baseline["point_to_surface_median_scene_diagonal"] is not None
            and float(aligned["point_to_surface_median_scene_diagonal"])
            <= float(baseline["point_to_surface_median_scene_diagonal"]) * 1.05
        ),
    }
    identity_consistent = (
        baseline["sparse_depth_residual_median"] is not None
        and float(baseline["sparse_depth_residual_median"])
        <= float(configuration["identity_median_residual_threshold"])
        and baseline_inlier >= float(configuration["identity_inlier_fraction_threshold"])
    )
    global_sufficient = aligned_inlier >= float(
        configuration["sufficient_inlier_fraction_threshold"]
    )
    plausibility = all(
        checks[key]
        for key in (
            "finite_transform",
            "positive_scale",
            "proper_rotation",
            "scale_plausible",
            "rotation_plausible",
            "translation_plausible",
        )
    )
    if not checks["transform_chain_consistent"]:
        return (
            "global_sim3_insufficient",
            False,
            checks,
            "transform chain audit failed; a fitted correction cannot hide it",
            False,
        )
    if identity_consistent:
        return "identity_already_consistent", True, checks, None, True
    if not plausibility:
        return (
            "rejected_implausible_transform",
            False,
            checks,
            "best candidate violates conservative Sim(3) plausibility bounds",
            False,
        )
    gate_keys = (
        "minimum_validation_observations",
        "median_residual_improvement",
        "p90_residual_improvement",
        "inlier_fraction_improvement",
        "mesh_coverage_preserved",
        "bad_frame_fraction",
        "point_surface_not_degraded",
    )
    if not all(checks[key] for key in gate_keys):
        improvement = any(
            checks[key]
            for key in (
                "median_residual_improvement",
                "p90_residual_improvement",
                "inlier_fraction_improvement",
            )
        )
        return (
            "global_sim3_insufficient" if improvement else "rejected_no_validation_improvement",
            False,
            checks,
            "held-out validation gates did not all pass",
            False,
        )
    if not global_sufficient:
        return (
            "global_sim3_insufficient",
            False,
            checks,
            "global transform improves validation but residual alignment remains insufficient",
            False,
        )
    return "accepted_global_sim3", True, checks, None, True
