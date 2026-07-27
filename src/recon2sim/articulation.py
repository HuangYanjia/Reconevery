from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from recon2sim.artifacts import (
    ArticulatedCandidateEvaluation,
    ArticulatedLicenseMode,
    ArticulationEvidenceLevel,
    ArticulationEvidenceSplit,
)


@dataclass(frozen=True)
class AnalyticJointEstimate:
    joint_type: str
    axis: tuple[float, float, float] | None
    pivot: tuple[float, float, float] | None
    positions: tuple[float, ...]
    residual: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_evidence_tier(state_count: int) -> ArticulationEvidenceLevel:
    if state_count < 1:
        raise ValueError("articulation requires at least one static state")
    if state_count == 1:
        return ArticulationEvidenceLevel.SINGLE_STATE_PRIOR_ONLY
    if state_count == 2:
        return ArticulationEvidenceLevel.TWO_STATE_MOTION_SUPPORTED
    return ArticulationEvidenceLevel.MULTI_STATE_HELDOUT_AVAILABLE


def effective_evidence_level(
    accepted_alignment_state_count: int,
    *,
    valid_measured_motion: bool,
    heldout_evaluation_ran: bool = False,
    heldout_candidate_passed: bool = False,
) -> ArticulationEvidenceLevel:
    if accepted_alignment_state_count < 2 or not valid_measured_motion:
        return ArticulationEvidenceLevel.SINGLE_STATE_PRIOR_ONLY
    if accepted_alignment_state_count < 3 or not heldout_evaluation_ran:
        return ArticulationEvidenceLevel.TWO_STATE_MOTION_SUPPORTED
    if heldout_candidate_passed:
        return ArticulationEvidenceLevel.MULTI_STATE_HELDOUT_VALIDATED
    return ArticulationEvidenceLevel.MULTI_STATE_HELDOUT_AVAILABLE


def ordered_motion_state_ids(
    reference_state_id: str,
    state_ids: Sequence[str],
    accepted_state_ids: set[str],
) -> list[str]:
    if reference_state_id not in state_ids:
        raise ValueError("declared articulation reference state is absent")
    if reference_state_id not in accepted_state_ids:
        raise ValueError("declared articulation reference state was not accepted")
    return [
        reference_state_id,
        *[
            state_id
            for state_id in state_ids
            if state_id != reference_state_id and state_id in accepted_state_ids
        ],
    ]


# Kept as a source-compatible migration alias for Phase 5C fake fixtures.
def evidence_level(state_count: int) -> ArticulationEvidenceLevel:
    return capture_evidence_tier(state_count)


def split_articulation_evidence(
    articulated_object_id: str,
    state_ids: Sequence[str],
    registered_frames_by_state: Mapping[str, Sequence[str]],
    *,
    seed: int,
) -> ArticulationEvidenceSplit:
    ordered = list(state_ids)
    if not ordered or len(ordered) != len(set(ordered)):
        raise ValueError("articulation state order must be non-empty and unique")
    if len(ordered) == 1:
        generation: list[str] = ordered
        fitting: list[str] = []
        heldout: list[str] = []
    elif len(ordered) == 2:
        generation, fitting, heldout = [ordered[0]], [ordered[1]], []
    else:
        generation = [ordered[0]]
        fitting = ordered[1:-1]
        heldout = [ordered[-1]]
    heldout_views: dict[str, list[str]] = {}
    for state_id in heldout:
        frames = list(registered_frames_by_state.get(state_id, ()))
        heldout_views[state_id] = frames[::2] or frames[:1]
    return ArticulationEvidenceSplit(
        articulated_object_id=articulated_object_id,
        candidate_generation_states=generation,
        kinematic_fitting_states=fitting,
        heldout_validation_states=heldout,
        heldout_views_by_state=heldout_views,
        seed=seed,
    )


def matrix4_multiply(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, ...]:
    if len(left) != 16 or len(right) != 16:
        raise ValueError("4x4 matrices require exactly sixteen values")
    return tuple(
        sum(left[row * 4 + inner] * right[inner * 4 + column] for inner in range(4))
        for row in range(4)
        for column in range(4)
    )


def sim3_roundtrip_error(
    matrix: Sequence[float],
    inverse: Sequence[float],
) -> float:
    product = matrix4_multiply(matrix, inverse)
    identity = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    return max(abs(actual - expected) for actual, expected in zip(product, identity, strict=True))


def proper_positive_sim3(
    matrix: Sequence[float],
    inverse: Sequence[float],
    *,
    tolerance: float = 1e-5,
) -> bool:
    if len(matrix) != 16 or len(inverse) != 16:
        return False
    if not all(math.isfinite(value) for value in (*matrix, *inverse)):
        return False
    if any(
        abs(matrix[index] - expected) > tolerance
        for index, expected in zip(
            (12, 13, 14, 15),
            (0.0, 0.0, 0.0, 1.0),
            strict=True,
        )
    ):
        return False
    row0 = matrix[0:3]
    row1 = matrix[4:7]
    row2 = matrix[8:11]
    determinant = (
        row0[0] * (row1[1] * row2[2] - row1[2] * row2[1])
        - row0[1] * (row1[0] * row2[2] - row1[2] * row2[0])
        + row0[2] * (row1[0] * row2[1] - row1[1] * row2[0])
    )
    return determinant > 0 and sim3_roundtrip_error(matrix, inverse) <= tolerance


def invert_sim3(matrix: Sequence[float]) -> tuple[float, ...] | None:
    """Invert a row-major affine Sim(3) without adding a numeric core dependency."""
    if len(matrix) != 16 or not all(math.isfinite(value) for value in matrix):
        return None
    a, b, c = matrix[0:3]
    d, e, f = matrix[4:7]
    g, h, i = matrix[8:11]
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if determinant <= 1e-15:
        return None
    inverse3 = (
        (e * i - f * h) / determinant,
        (c * h - b * i) / determinant,
        (b * f - c * e) / determinant,
        (f * g - d * i) / determinant,
        (a * i - c * g) / determinant,
        (c * d - a * f) / determinant,
        (d * h - e * g) / determinant,
        (b * g - a * h) / determinant,
        (a * e - b * d) / determinant,
    )
    tx, ty, tz = matrix[3], matrix[7], matrix[11]
    inverse_translation = tuple(
        -sum(inverse3[row * 3 + column] * value for column, value in enumerate((tx, ty, tz)))
        for row in range(3)
    )
    return (
        inverse3[0],
        inverse3[1],
        inverse3[2],
        inverse_translation[0],
        inverse3[3],
        inverse3[4],
        inverse3[5],
        inverse_translation[1],
        inverse3[6],
        inverse3[7],
        inverse3[8],
        inverse_translation[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _normalize(vector: Sequence[float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return tuple(value / norm for value in vector)  # type: ignore[return-value]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _rotation_axis_angle(matrix: Sequence[float]) -> tuple[tuple[float, float, float], float]:
    cosine = max(
        -1.0,
        min(1.0, (matrix[0] + matrix[5] + matrix[10] - 1.0) / 2.0),
    )
    angle = math.acos(cosine)
    if angle < 1e-8:
        return (1.0, 0.0, 0.0), 0.0
    axis = _normalize(
        (
            matrix[9] - matrix[6],
            matrix[2] - matrix[8],
            matrix[4] - matrix[1],
        )
    )
    return axis, angle


def _solve_three_by_three(
    matrix: Sequence[Sequence[float]],
    target: Sequence[float],
) -> tuple[float, float, float]:
    augmented = [list(row) + [value] for row, value in zip(matrix, target, strict=True)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-12:
            raise ValueError("joint pivot system is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * reference
                for value, reference in zip(
                    augmented[row],
                    augmented[column],
                    strict=True,
                )
            ]
    return tuple(augmented[row][3] for row in range(3))  # type: ignore[return-value]


def estimate_analytic_joint(
    transforms: Sequence[Sequence[float]],
    *,
    maximum_prismatic_rotation_degrees: float = 5.0,
    maximum_prismatic_orthogonal_residual: float = 0.05,
    maximum_revolute_axis_error_degrees: float = 15.0,
) -> AnalyticJointEstimate:
    if len(transforms) < 2 or any(len(matrix) != 16 for matrix in transforms):
        raise ValueError("analytic joint estimation requires at least two 4x4 transforms")
    translations = [(matrix[3], matrix[7], matrix[11]) for matrix in transforms]
    relative = [
        tuple(value - translations[0][index] for index, value in enumerate(item))
        for item in translations
    ]
    axes_angles = [_rotation_axis_angle(matrix) for matrix in transforms]
    maximum_rotation = max(math.degrees(angle) for _, angle in axes_angles)
    farthest = max(relative, key=lambda item: math.sqrt(_dot(item, item)))
    translation_extent = math.sqrt(_dot(farthest, farthest))
    if maximum_rotation <= maximum_prismatic_rotation_degrees and translation_extent > 1e-9:
        axis = _normalize(farthest)
        positions = tuple(_dot(item, axis) for item in relative)
        orthogonal = [
            math.sqrt(sum((item[index] - position * axis[index]) ** 2 for index in range(3)))
            for item, position in zip(relative, positions, strict=True)
        ]
        residual = max(orthogonal) / translation_extent
        if residual <= maximum_prismatic_orthogonal_residual:
            return AnalyticJointEstimate(
                "prismatic",
                axis,
                None,
                positions,
                residual,
            )
    moving = [(axis, angle) for axis, angle in axes_angles if angle > 1e-8]
    if moving:
        reference = moving[0][0]
        aligned = [
            axis if _dot(axis, reference) >= 0 else tuple(-value for value in axis)
            for axis, _ in moving
        ]
        axis = _normalize(tuple(sum(item[index] for item in aligned) for index in range(3)))
        axis_error = max(
            math.degrees(math.acos(max(-1.0, min(1.0, abs(_dot(item, axis)))))) for item in aligned
        )
        if axis_error <= maximum_revolute_axis_error_degrees:
            rows: list[tuple[float, float, float]] = []
            targets: list[float] = []
            for matrix in transforms[1:]:
                rows.extend(
                    [
                        (1.0 - matrix[0], -matrix[1], -matrix[2]),
                        (-matrix[4], 1.0 - matrix[5], -matrix[6]),
                        (-matrix[8], -matrix[9], 1.0 - matrix[10]),
                    ]
                )
                targets.extend((matrix[3], matrix[7], matrix[11]))
            rows.append(axis)
            targets.append(0.0)
            normal = [[sum(row[i] * row[j] for row in rows) for j in range(3)] for i in range(3)]
            normal_target = [
                sum(row[i] * value for row, value in zip(rows, targets, strict=True))
                for i in range(3)
            ]
            pivot = _solve_three_by_three(normal, normal_target)
            residuals = [
                abs(sum(row[index] * pivot[index] for index in range(3)) - target)
                for row, target in zip(rows[:-1], targets[:-1], strict=True)
            ]
            positions = tuple(
                angle if _dot(item_axis, axis) >= 0 else -angle for item_axis, angle in axes_angles
            )
            return AnalyticJointEstimate(
                "revolute",
                axis,
                pivot,
                positions,
                max(residuals, default=0.0),
            )
    return AnalyticJointEstimate(
        "unknown",
        None,
        None,
        tuple(0.0 for _ in transforms),
        float("inf"),
    )


def select_articulated_candidate(
    evaluations: Sequence[ArticulatedCandidateEvaluation],
    *,
    production_selectable: Mapping[str, bool],
    mode: ArticulatedLicenseMode,
) -> tuple[str | None, str | None, str | None]:
    passing = sorted(
        (item for item in evaluations if item.passed_hard_gates),
        key=lambda item: (
            -min(
                (
                    state.movable_part_mask_iou
                    for state in item.state_evaluations
                    if state.heldout and state.movable_part_mask_iou is not None
                ),
                default=0.0,
            ),
            -min(
                (
                    state.depth_inlier_fraction
                    for state in item.state_evaluations
                    if state.heldout and state.depth_inlier_fraction is not None
                ),
                default=0.0,
            ),
            item.candidate_id,
        ),
    )
    research = passing[0].candidate_id if passing else None
    production = next(
        (
            item.candidate_id
            for item in passing
            if production_selectable.get(item.candidate_id, False)
        ),
        None,
    )
    selected = research if mode is ArticulatedLicenseMode.RESEARCH_EVALUATION else production
    return research, production, selected


__all__ = [
    "capture_evidence_tier",
    "effective_evidence_level",
    "evidence_level",
    "estimate_analytic_joint",
    "matrix4_multiply",
    "ordered_motion_state_ids",
    "proper_positive_sim3",
    "select_articulated_candidate",
    "sha256_file",
    "sim3_roundtrip_error",
    "split_articulation_evidence",
    "stable_digest",
]
