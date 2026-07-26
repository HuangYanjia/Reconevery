from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from articulation_alignment_worker.sim3 import rotation_axis_angle


@dataclass(frozen=True)
class JointEstimate:
    joint_type: str
    axis: np.ndarray | None
    pivot: np.ndarray | None
    positions: list[float]
    orthogonal_residual: float | None
    rotation_leakage_degrees: float | None
    axis_consistency_degrees: float | None
    pivot_residual: float | None


def estimate_joint(
    transforms: list[np.ndarray],
    *,
    max_fixed_translation: float,
    max_fixed_rotation_degrees: float,
    max_prismatic_rotation_degrees: float,
    max_prismatic_orthogonal_residual: float,
    max_revolute_axis_error_degrees: float,
) -> JointEstimate:
    translations = np.asarray([matrix[:3, 3] for matrix in transforms])
    rotations = [rotation_axis_angle(matrix) for matrix in transforms]
    rotation_degrees = np.degrees([angle for _, angle in rotations])
    centered = translations - translations[0]
    translation_magnitudes = np.linalg.norm(centered, axis=1)
    if (
        float(np.max(translation_magnitudes)) <= max_fixed_translation
        and float(np.max(rotation_degrees)) <= max_fixed_rotation_degrees
    ):
        return JointEstimate("fixed", None, None, [0.0] * len(transforms), None, None, None, None)

    _, _, right = np.linalg.svd(centered, full_matrices=False)
    translation_axis = right[0]
    if np.dot(translation_axis, centered[-1]) < 0:
        translation_axis = -translation_axis
    positions = centered @ translation_axis
    orthogonal = centered - positions[:, None] * translation_axis
    extent = max(float(np.ptp(positions)), np.finfo(np.float64).eps)
    orthogonal_residual = float(np.median(np.linalg.norm(orthogonal, axis=1)) / extent)
    if (
        float(np.max(rotation_degrees)) <= max_prismatic_rotation_degrees
        and orthogonal_residual <= max_prismatic_orthogonal_residual
    ):
        return JointEstimate(
            "prismatic",
            translation_axis,
            None,
            positions.tolist(),
            orthogonal_residual,
            float(np.max(rotation_degrees)),
            None,
            None,
        )

    moving = [
        (axis, angle, matrix)
        for (axis, angle), matrix in zip(rotations, transforms, strict=True)
        if angle > 1e-5
    ]
    if moving:
        reference_axis = moving[0][0]
        aligned_axes = [
            axis if np.dot(axis, reference_axis) >= 0 else -axis for axis, _, _ in moving
        ]
        axis = np.mean(aligned_axes, axis=0)
        axis /= np.linalg.norm(axis)
        axis_errors = [
            np.degrees(np.arccos(np.clip(abs(np.dot(item, axis)), -1.0, 1.0)))
            for item in aligned_axes
        ]
        axis_error = float(max(axis_errors))
        if axis_error <= max_revolute_axis_error_degrees:
            equations: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            for _, _, matrix in moving:
                rotation = matrix[:3, :3]
                equations.append(np.eye(3) - rotation)
                targets.append(matrix[:3, 3])
            coefficient = np.concatenate(equations, axis=0)
            target = np.concatenate(targets, axis=0)
            coefficient = np.concatenate([coefficient, axis[None, :]], axis=0)
            target = np.concatenate([target, np.zeros(1)], axis=0)
            pivot, *_ = np.linalg.lstsq(coefficient, target, rcond=None)
            residuals = [
                np.linalg.norm((np.eye(3) - matrix[:3, :3]) @ pivot - matrix[:3, 3])
                for _, _, matrix in moving
            ]
            angles = []
            for item_axis, angle in rotations:
                angles.append(float(angle if np.dot(item_axis, axis) >= 0 else -angle))
            return JointEstimate(
                "revolute",
                axis,
                pivot,
                angles,
                None,
                None,
                axis_error,
                float(np.median(residuals)),
            )
    return JointEstimate("unknown", None, None, [0.0] * len(transforms), None, None, None, None)
