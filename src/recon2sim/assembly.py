from __future__ import annotations

import hashlib
import json
import math
from collections import deque

from recon2sim.artifacts import (
    SceneAssemblyCalibrationPolicy,
    SceneAssemblyInputManifest,
    SceneAssemblyWorldMode,
    SceneAssemblyWorldRecord,
    WorldCalibrationStatus,
)

IDENTITY_MATRIX4: tuple[float, ...] = (
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


def stable_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def multiply_matrix4(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(
        sum(left[row * 4 + inner] * right[inner * 4 + column] for inner in range(4))
        for row in range(4)
        for column in range(4)
    )


def transform_point(
    matrix: tuple[float, ...],
    point: tuple[float, float, float],
) -> tuple[float, float, float]:
    x, y, z = point
    return (
        matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3],
        matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7],
        matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11],
    )


def validate_proper_sim3(matrix: tuple[float, ...], *, tolerance: float = 1e-7) -> float:
    if len(matrix) != 16 or any(not math.isfinite(value) for value in matrix):
        raise ValueError("assembly transform must contain 16 finite values")
    if (
        max(
            abs(matrix[index] - expected)
            for index, expected in zip((12, 13, 14, 15), (0.0, 0.0, 0.0, 1.0), strict=True)
        )
        > tolerance
    ):
        raise ValueError("assembly transform must be affine")
    columns = tuple((matrix[column], matrix[4 + column], matrix[8 + column]) for column in range(3))
    norms = tuple(math.sqrt(sum(value * value for value in column)) for column in columns)
    scale = sum(norms) / 3.0
    if scale <= 0 or max(abs(value - scale) for value in norms) > tolerance * max(1.0, scale):
        raise ValueError("assembly transform must use one positive uniform scale")
    unit = tuple(tuple(value / scale for value in column) for column in columns)
    for left in range(3):
        for right in range(3):
            dot = sum(unit[left][index] * unit[right][index] for index in range(3))
            expected = 1.0 if left == right else 0.0
            if abs(dot - expected) > tolerance:
                raise ValueError("assembly transform rotation must be orthonormal")
    determinant = (
        unit[0][0] * (unit[1][1] * unit[2][2] - unit[1][2] * unit[2][1])
        - unit[1][0] * (unit[0][1] * unit[2][2] - unit[0][2] * unit[2][1])
        + unit[2][0] * (unit[0][1] * unit[1][2] - unit[0][2] * unit[1][1])
    )
    if abs(determinant - 1.0) > tolerance:
        raise ValueError("assembly transform rotation must be proper and right handed")
    return scale


def transformed_bounds(
    bounds: tuple[float, float, float, float, float, float] | None,
    matrix: tuple[float, ...],
) -> tuple[float, float, float, float, float, float] | None:
    if bounds is None:
        return None
    minimum = bounds[:3]
    maximum = bounds[3:]
    points = [
        transform_point(matrix, (x, y, z))
        for x in (minimum[0], maximum[0])
        for y in (minimum[1], maximum[1])
        for z in (minimum[2], maximum[2])
    ]
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        min(point[2] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
        max(point[2] for point in points),
    )


def bounds_overlap_ratio(
    first: tuple[float, float, float, float, float, float] | None,
    second: tuple[float, float, float, float, float, float] | None,
) -> float | None:
    if first is None or second is None:
        return None
    lengths = tuple(
        max(0.0, min(first[index + 3], second[index + 3]) - max(first[index], second[index]))
        for index in range(3)
    )
    intersection = lengths[0] * lengths[1] * lengths[2]
    first_volume = math.prod(max(0.0, first[index + 3] - first[index]) for index in range(3))
    second_volume = math.prod(max(0.0, second[index + 3] - second[index]) for index in range(3))
    denominator = min(first_volume, second_volume)
    return intersection / denominator if denominator > 0 else 0.0


def bounds_center_distance(
    first: tuple[float, float, float, float, float, float] | None,
    second: tuple[float, float, float, float, float, float] | None,
) -> float | None:
    if first is None or second is None:
        return None
    first_center = tuple((first[index] + first[index + 3]) / 2 for index in range(3))
    second_center = tuple((second[index] + second[index + 3]) / 2 for index in range(3))
    return math.sqrt(sum((first_center[index] - second_center[index]) ** 2 for index in range(3)))


def connected_lineages(manifest: SceneAssemblyInputManifest) -> set[str]:
    neighbors: dict[str, set[str]] = {item.lineage_id: set() for item in manifest.lineages}
    for item in manifest.lineages:
        if item.connected_to_lineage_id is None:
            continue
        if item.connected_to_lineage_id not in neighbors:
            raise ValueError(
                f"lineage {item.lineage_id!r} connects to undeclared lineage "
                f"{item.connected_to_lineage_id!r}"
            )
        assert item.transform_connected_from_lineage is not None
        validate_proper_sim3(item.transform_connected_from_lineage)
        neighbors[item.lineage_id].add(item.connected_to_lineage_id)
        neighbors[item.connected_to_lineage_id].add(item.lineage_id)
    visited = {manifest.primary_lineage_id}
    queue = deque([manifest.primary_lineage_id])
    while queue:
        current = queue.popleft()
        for neighbor in sorted(neighbors[current]):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited


def resolve_world(manifest: SceneAssemblyInputManifest) -> SceneAssemblyWorldRecord:
    policy = manifest.calibration_policy
    status = manifest.calibration_status
    source_world_warnings = (
        ["gravity_evidence_available_but_no_typed_orientation_transform"]
        if status is WorldCalibrationStatus.ACCEPTED_GRAVITY_ONLY
        else []
    )
    if policy is SceneAssemblyCalibrationPolicy.PRESERVE_SOURCE_WORLD:
        return SceneAssemblyWorldRecord(
            world_mode=SceneAssemblyWorldMode.SOURCE_ARBITRARY,
            calibration_policy=policy,
            calibration_status=status,
            source_world_to_assembly_world=IDENTITY_MATRIX4,
            linear_units="arbitrary_units",
            alignment_status="unoriented",
            full_canonical_world_used=False,
            metric_scale_known=False,
            gravity_alignment_known=False,
            world_wrapper_required=False,
            warnings=source_world_warnings,
        )
    if policy is SceneAssemblyCalibrationPolicy.REQUIRE_FULL_CANONICAL and (
        status is not WorldCalibrationStatus.ACCEPTED_FULL_CANONICAL
    ):
        raise ValueError("assembly policy requires an accepted full-canonical calibration")
    if status is WorldCalibrationStatus.ACCEPTED_FULL_CANONICAL:
        if manifest.source_world_to_assembly_world is None:
            raise ValueError("full-canonical assembly requires an accepted world transform")
        validate_proper_sim3(manifest.source_world_to_assembly_world)
        return SceneAssemblyWorldRecord(
            world_mode=SceneAssemblyWorldMode.CANONICAL_METRIC,
            calibration_policy=policy,
            calibration_status=status,
            source_world_to_assembly_world=manifest.source_world_to_assembly_world,
            linear_units="meters",
            alignment_status="canonical",
            full_canonical_world_used=True,
            metric_scale_known=True,
            gravity_alignment_known=True,
            world_wrapper_required=True,
        )
    if status is WorldCalibrationStatus.ACCEPTED_METRIC_ONLY:
        if manifest.source_world_to_assembly_world is None:
            raise ValueError("metric-only assembly requires its accepted scale transform")
        validate_proper_sim3(manifest.source_world_to_assembly_world)
        return SceneAssemblyWorldRecord(
            world_mode=SceneAssemblyWorldMode.METRIC_UNORIENTED,
            calibration_policy=policy,
            calibration_status=status,
            source_world_to_assembly_world=manifest.source_world_to_assembly_world,
            linear_units="meters",
            alignment_status="unoriented",
            full_canonical_world_used=False,
            metric_scale_known=True,
            gravity_alignment_known=False,
            world_wrapper_required=True,
        )
    if status is WorldCalibrationStatus.ACCEPTED_GRAVITY_ONLY:
        return SceneAssemblyWorldRecord(
            world_mode=SceneAssemblyWorldMode.SOURCE_ARBITRARY,
            calibration_policy=policy,
            calibration_status=status,
            source_world_to_assembly_world=IDENTITY_MATRIX4,
            linear_units="arbitrary_units",
            alignment_status="unoriented",
            full_canonical_world_used=False,
            metric_scale_known=False,
            gravity_alignment_known=False,
            world_wrapper_required=False,
            warnings=source_world_warnings,
        )
    return SceneAssemblyWorldRecord(
        world_mode=SceneAssemblyWorldMode.SOURCE_ARBITRARY,
        calibration_policy=policy,
        calibration_status=status,
        source_world_to_assembly_world=IDENTITY_MATRIX4,
        linear_units="arbitrary_units",
        alignment_status="unoriented",
        full_canonical_world_used=False,
        metric_scale_known=False,
        gravity_alignment_known=False,
        world_wrapper_required=False,
    )


__all__ = [
    "IDENTITY_MATRIX4",
    "bounds_center_distance",
    "bounds_overlap_ratio",
    "connected_lineages",
    "multiply_matrix4",
    "resolve_world",
    "stable_digest",
    "transform_point",
    "transformed_bounds",
    "validate_proper_sim3",
]
