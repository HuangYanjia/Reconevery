from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import median

Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class _Node:
    point: Point3
    axis: int
    left: _Node | None
    right: _Node | None


def _build(points: list[Point3], depth: int = 0) -> _Node | None:
    if not points:
        return None
    axis = depth % 3
    ordered = sorted(points, key=lambda point: (point[axis], point))
    middle = len(ordered) // 2
    return _Node(
        point=ordered[middle],
        axis=axis,
        left=_build(ordered[:middle], depth + 1),
        right=_build(ordered[middle + 1 :], depth + 1),
    )


def _distance_squared(left: Point3, right: Point3) -> float:
    return sum((a - b) * (a - b) for a, b in zip(left, right, strict=True))


def _nearest_nonself(node: _Node | None, target: Point3, best: float) -> float:
    if node is None:
        return best
    distance = _distance_squared(node.point, target)
    if 1e-24 < distance < best:
        best = distance
    delta = target[node.axis] - node.point[node.axis]
    near, far = (node.left, node.right) if delta <= 0 else (node.right, node.left)
    best = _nearest_nonself(near, target, best)
    if delta * delta < best:
        best = _nearest_nonself(far, target, best)
    return best


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("nearest-neighbor spacing requires positive distances")
    position = quantile * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def estimate_spacing(
    points: Iterable[Point3],
    *,
    multiplier: float,
    maximum_sample_count: int = 4096,
) -> dict[str, float | int | str]:
    source = [tuple(float(value) for value in point) for point in points]
    if not source:
        raise ValueError("nearest-neighbor spacing requires at least one point")
    if multiplier <= 0:
        raise ValueError("voxel-size multiplier must be positive")
    keyed = []
    for point in source:
        encoded = struct.pack("<3d", *point)
        keyed.append((hashlib.sha256(encoded).digest(), point, encoded))
    keyed.sort(key=lambda item: (item[0], item[1]))
    sampled = [item[1] for item in keyed[:maximum_sample_count]]
    unique = sorted(set(sampled))
    if len(unique) < 2:
        fallback = 1e-3
        digest = hashlib.sha256(b"".join(item[2] for item in keyed)).hexdigest()
        return {
            "method": "coordinate_hash_kdtree_nearest_neighbor_v1",
            "source_point_count": len(source),
            "sampled_point_count": len(unique),
            "nearest_neighbor_p10": fallback,
            "nearest_neighbor_median": fallback,
            "nearest_neighbor_p90": fallback,
            "voxel_size": fallback * multiplier,
            "coordinate_hash_digest": digest,
        }
    tree = _build(unique)
    distances = sorted(math.sqrt(_nearest_nonself(tree, point, math.inf)) for point in unique)
    distances = [value for value in distances if math.isfinite(value) and value > 0]
    spacing = median(distances)
    digest = hashlib.sha256(b"".join(item[2] for item in keyed)).hexdigest()
    return {
        "method": "coordinate_hash_kdtree_nearest_neighbor_v1",
        "source_point_count": len(source),
        "sampled_point_count": len(unique),
        "nearest_neighbor_p10": _percentile(distances, 0.1),
        "nearest_neighbor_median": spacing,
        "nearest_neighbor_p90": _percentile(distances, 0.9),
        "voxel_size": max(spacing * multiplier, 1e-8),
        "coordinate_hash_digest": digest,
    }
