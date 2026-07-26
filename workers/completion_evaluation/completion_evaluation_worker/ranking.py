def hard_gate_failures(metrics: dict[str, float], config: dict[str, float]) -> list[str]:
    checks = (
        ("minimum_mask_iou", metrics["mask_iou"] >= config["minimum_mask_iou"]),
        (
            "minimum_mask_precision",
            metrics["mask_precision"] >= config["minimum_mask_precision"],
        ),
        (
            "maximum_median_relative_depth_residual",
            metrics["dense_depth_relative_residual"]
            <= config["maximum_median_relative_depth_residual"],
        ),
        (
            "minimum_depth_inlier_fraction",
            metrics["depth_inlier_fraction"] >= config["minimum_depth_inlier_fraction"],
        ),
        (
            "maximum_negative_space_violation_ratio",
            metrics["negative_space_violation_ratio"]
            <= config["maximum_negative_space_violation_ratio"],
        ),
        (
            "maximum_front_of_scene_violation_ratio",
            metrics["front_of_scene_violation_ratio"]
            <= config["maximum_front_of_scene_violation_ratio"],
        ),
    )
    return [name for name, passed in checks if not passed]
