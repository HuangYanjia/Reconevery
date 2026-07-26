from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from pathlib import Path

from recon2sim.artifacts import (
    CandidateHeldoutEvaluation,
    CompletionEligibilityStatus,
    CompletionObjectEvidenceSplit,
)
from recon2sim.ir import AssetType

ARTICULATED_LABELS = {
    "cabinet",
    "cabinet door",
    "door",
    "drawer",
    "refrigerator",
    "scissors",
}
HUMAN_LABELS = {"human", "person", "body", "man", "woman", "child"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_label(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", " ", value.strip().lower())
    return " ".join(result.split())


def completion_eligibility(
    label: str,
    asset_type_hint: AssetType | None,
    *,
    allow_unclassified: bool,
    override: CompletionEligibilityStatus | None = None,
) -> tuple[CompletionEligibilityStatus, str, bool]:
    if override is not None:
        return override, "explicit per-object completion eligibility override", True
    semantic = normalized_label(label)
    if semantic in HUMAN_LABELS:
        return (
            CompletionEligibilityStatus.DEFERRED_HUMAN,
            "human/body completion is excluded",
            False,
        )
    if semantic in ARTICULATED_LABELS:
        return (
            CompletionEligibilityStatus.DEFERRED_ARTICULATED,
            "label is conservatively routed to future articulated reconstruction",
            False,
        )
    if asset_type_hint is AssetType.ARTICULATED:
        return (
            CompletionEligibilityStatus.DEFERRED_ARTICULATED,
            "configured asset-type hint is articulated",
            False,
        )
    if asset_type_hint is AssetType.DEFORMABLE:
        return (
            CompletionEligibilityStatus.DEFERRED_DEFORMABLE,
            "configured asset-type hint is deformable",
            False,
        )
    if asset_type_hint is AssetType.FLUID_OR_PARTICLE:
        return (
            CompletionEligibilityStatus.DEFERRED_FLUID,
            "configured asset-type hint is fluid or particle",
            False,
        )
    if asset_type_hint is AssetType.IGNORE:
        return CompletionEligibilityStatus.DEFERRED_UNKNOWN, "object is configured to ignore", False
    if asset_type_hint is AssetType.STATIC_STRUCTURE:
        return (
            CompletionEligibilityStatus.ELIGIBLE_STATIC,
            "static visual completion is allowed",
            False,
        )
    if asset_type_hint is AssetType.RIGID:
        return (
            CompletionEligibilityStatus.ELIGIBLE_RIGID,
            "rigid visual completion is allowed",
            False,
        )
    if asset_type_hint in {None, AssetType.UNCLASSIFIED} and allow_unclassified:
        return (
            CompletionEligibilityStatus.ELIGIBLE_RIGID,
            "unclassified object explicitly allowed by completion configuration",
            False,
        )
    return (
        CompletionEligibilityStatus.DEFERRED_UNKNOWN,
        "unclassified objects require an explicit configuration allowance",
        False,
    )


def select_diverse_anchors(
    scored_frames: list[tuple[str, float, tuple[float, float, float]]],
    *,
    maximum_count: int,
    minimum_angle_degrees: float,
) -> list[str]:
    if maximum_count <= 0:
        return []
    ordered = sorted(scored_frames, key=lambda item: (-item[1], item[0]))
    selected: list[tuple[str, float, tuple[float, float, float]]] = []
    minimum_cosine = math.cos(math.radians(minimum_angle_degrees))
    for candidate in ordered:
        direction = candidate[2]
        norm = math.sqrt(sum(value * value for value in direction))
        if norm == 0:
            continue
        normalized = tuple(value / norm for value in direction)
        if selected:
            sufficiently_diverse = True
            for previous in selected:
                previous_norm = math.sqrt(sum(value * value for value in previous[2]))
                if previous_norm == 0:
                    continue
                cosine = sum(
                    normalized[index] * previous[2][index] / previous_norm for index in range(3)
                )
                if cosine > minimum_cosine:
                    sufficiently_diverse = False
                    break
            if not sufficiently_diverse:
                continue
        selected.append(candidate)
        if len(selected) == maximum_count:
            break
    if len(selected) < maximum_count:
        selected_ids = {item[0] for item in selected}
        selected.extend(item for item in ordered if item[0] not in selected_ids)
    return [item[0] for item in selected[:maximum_count]]


def split_object_evidence(
    object_id: str,
    ordered_frames: list[str],
    anchors: list[str],
    *,
    minimum_heldout_frames: int,
    fitting_fraction: float,
) -> CompletionObjectEvidenceSplit:
    unique = list(dict.fromkeys(ordered_frames))
    anchor_set = set(anchors)
    remaining = [frame for frame in unique if frame not in anchor_set]
    heldout_count = min(minimum_heldout_frames, max(0, len(remaining) - 1))
    heldout = remaining[1::2][:heldout_count]
    if len(heldout) < heldout_count:
        heldout.extend(frame for frame in reversed(remaining) if frame not in heldout)
        heldout = heldout[:heldout_count]
    fitting_candidates = [frame for frame in remaining if frame not in set(heldout)]
    fitting_count = max(1, math.floor(len(unique) * fitting_fraction)) if fitting_candidates else 0
    fitting = fitting_candidates[:fitting_count]
    degraded = len(anchors) < 1 or len(heldout) < minimum_heldout_frames or not fitting
    return CompletionObjectEvidenceSplit(
        object_id=object_id,
        generation_anchor_frames=anchors,
        registration_fitting_frames=fitting,
        heldout_validation_frames=heldout,
        degraded_split=degraded,
        limitation=(
            "insufficient registered observations for the preferred disjoint split"
            if degraded
            else None
        ),
    )


def candidate_id(object_id: str, backend: str, anchor_frame_id: str, seed: int) -> str:
    return f"{object_id}__{backend}__{anchor_frame_id}__seed_{seed}"


def positive_scale_sim3(matrix: Iterable[float], *, tolerance: float = 1e-5) -> bool:
    values = list(matrix)
    if len(values) != 16 or any(not math.isfinite(value) for value in values):
        return False
    if any(
        abs(values[index] - expected) > tolerance
        for index, expected in zip((12, 13, 14, 15), (0.0, 0.0, 0.0, 1.0), strict=True)
    ):
        return False
    columns = [
        (values[0], values[4], values[8]),
        (values[1], values[5], values[9]),
        (values[2], values[6], values[10]),
    ]
    scales = [math.sqrt(sum(value * value for value in column)) for column in columns]
    if min(scales) <= 0 or max(scales) - min(scales) > tolerance * max(scales):
        return False
    scale = sum(scales) / 3
    rotation = [[columns[column][row] / scale for column in range(3)] for row in range(3)]
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    return abs(determinant - 1.0) <= 1e-4


def evaluation_rank_key(evaluation: CandidateHeldoutEvaluation) -> tuple[object, ...]:
    metrics = evaluation.metrics
    gain = evaluation.completion_gain
    representation_rank = {
        "trellis2": 0,
        "sam3d_objects": 1,
        "measured_partial_baseline": 2,
    }.get(evaluation.backend.value, 3)
    return (
        not evaluation.passed_hard_gates,
        metrics.negative_space_violation_ratio,
        metrics.front_of_scene_violation_ratio,
        -metrics.depth_inlier_fraction,
        -metrics.mask_iou,
        -gain.recall_gain_vs_measured_baseline,
        representation_rank,
        evaluation.candidate_id,
    )


def pareto_front(
    evaluations: list[CandidateHeldoutEvaluation],
) -> list[CandidateHeldoutEvaluation]:
    passing = [item for item in evaluations if item.passed_hard_gates]
    front: list[CandidateHeldoutEvaluation] = []
    for candidate in passing:
        dominated = False
        for other in passing:
            if other is candidate:
                continue
            better_or_equal = (
                other.metrics.mask_iou >= candidate.metrics.mask_iou
                and other.metrics.depth_inlier_fraction >= candidate.metrics.depth_inlier_fraction
                and other.metrics.negative_space_violation_ratio
                <= candidate.metrics.negative_space_violation_ratio
            )
            strictly_better = (
                other.metrics.mask_iou > candidate.metrics.mask_iou
                or other.metrics.depth_inlier_fraction > candidate.metrics.depth_inlier_fraction
                or other.metrics.negative_space_violation_ratio
                < candidate.metrics.negative_space_violation_ratio
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(front, key=evaluation_rank_key)
