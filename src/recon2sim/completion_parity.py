from __future__ import annotations

import math
from collections.abc import Iterable

Matrix4 = list[list[float]]


def world_from_camera_matrix(
    rotation_xyzw: tuple[float, float, float, float],
    translation: tuple[float, float, float],
) -> Matrix4:
    x, y, z, w = rotation_xyzw
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rotation = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    return [
        [*rotation[0], translation[0]],
        [*rotation[1], translation[1]],
        [*rotation[2], translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def invert_rigid_matrix(matrix: Matrix4) -> Matrix4:
    rotation = [[matrix[row][column] for row in range(3)] for column in range(3)]
    translation = [matrix[row][3] for row in range(3)]
    inverse_translation = [
        -sum(rotation[row][column] * translation[column] for column in range(3)) for row in range(3)
    ]
    return [
        [*rotation[0], inverse_translation[0]],
        [*rotation[1], inverse_translation[1]],
        [*rotation[2], inverse_translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def flatten_matrix(matrix: Matrix4) -> list[float]:
    return [value for row in matrix for value in row]


def matrix_product(left: Matrix4, right: Matrix4) -> Matrix4:
    return [
        [sum(left[row][index] * right[index][column] for index in range(4)) for column in range(4)]
        for row in range(4)
    ]


def backend_layout_world_matrix(layout: dict[str, object], world_from_camera: Matrix4) -> Matrix4:
    scale_values = list(_flatten_numeric(layout["scale"]))
    rotation_values = list(_flatten_numeric(layout["rotation"]))
    translation = list(_flatten_numeric(layout["translation"]))
    if (
        len(scale_values) != 3
        or len(rotation_values) != 4
        or len(translation) != 3
        or max(scale_values) - min(scale_values) > 1e-6
    ):
        raise ValueError("backend layout is not a uniform local-to-camera Sim(3)")
    w, x, y, z = rotation_values
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    scale = scale_values[0]
    rotation = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    camera_from_candidate = [
        [*(scale * rotation[column][0] for column in range(3)), translation[0]],
        [*(scale * rotation[column][1] for column in range(3)), translation[1]],
        [*(scale * rotation[column][2] for column in range(3)), translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    pytorch3d_to_opencv = [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return matrix_product(
        world_from_camera,
        matrix_product(pytorch3d_to_opencv, camera_from_candidate),
    )


def _flatten_numeric(value: object) -> Iterable[float]:
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_numeric(item)
    elif isinstance(value, (int, float)):
        yield float(value)
    else:
        raise ValueError("backend layout contains a non-numeric value")


def binary_mask_metrics(
    first: Iterable[int],
    second: Iterable[int],
    width: int,
    height: int,
) -> tuple[float, float, float | None, int, int]:
    first_values = [value > 0 for value in first]
    second_values = [value > 0 for value in second]
    if len(first_values) != width * height or len(second_values) != width * height:
        raise ValueError("representation masks do not match declared dimensions")
    intersection = sum(a and b for a, b in zip(first_values, second_values, strict=True))
    union = sum(a or b for a, b in zip(first_values, second_values, strict=True))
    first_count = sum(first_values)
    second_count = sum(second_values)
    silhouette_iou = intersection / union if union else 1.0

    def bounds(values: list[bool]) -> tuple[int, int, int, int] | None:
        indices = [index for index, present in enumerate(values) if present]
        if not indices:
            return None
        columns = [index % width for index in indices]
        rows = [index // width for index in indices]
        return min(columns), min(rows), max(columns) + 1, max(rows) + 1

    first_bounds, second_bounds = bounds(first_values), bounds(second_values)
    if first_bounds is None or second_bounds is None:
        bbox_iou = 1.0 if first_bounds == second_bounds else 0.0
        centroid_distance = None
    else:
        left = max(first_bounds[0], second_bounds[0])
        top = max(first_bounds[1], second_bounds[1])
        right = min(first_bounds[2], second_bounds[2])
        bottom = min(first_bounds[3], second_bounds[3])
        intersect_area = max(0, right - left) * max(0, bottom - top)
        first_area = (first_bounds[2] - first_bounds[0]) * (first_bounds[3] - first_bounds[1])
        second_area = (second_bounds[2] - second_bounds[0]) * (second_bounds[3] - second_bounds[1])
        bbox_iou = intersect_area / max(first_area + second_area - intersect_area, 1)
        first_centroid = (
            (first_bounds[0] + first_bounds[2]) / 2,
            (first_bounds[1] + first_bounds[3]) / 2,
        )
        second_centroid = (
            (second_bounds[0] + second_bounds[2]) / 2,
            (second_bounds[1] + second_bounds[3]) / 2,
        )
        centroid_distance = math.hypot(
            first_centroid[0] - second_centroid[0],
            first_centroid[1] - second_centroid[1],
        ) / math.hypot(width, height)
    return silhouette_iou, bbox_iou, centroid_distance, first_count, second_count


def target_mask_metrics(
    rendered: Iterable[int],
    target: Iterable[int],
) -> tuple[float, float, float]:
    rendered_values = [value > 0 for value in rendered]
    target_values = [value > 0 for value in target]
    if len(rendered_values) != len(target_values):
        raise ValueError("rendered and target masks do not have the same dimensions")
    intersection = sum(
        candidate and expected
        for candidate, expected in zip(rendered_values, target_values, strict=True)
    )
    rendered_count = sum(rendered_values)
    target_count = sum(target_values)
    union = rendered_count + target_count - intersection
    return (
        intersection / rendered_count if rendered_count else 0.0,
        intersection / target_count if target_count else 0.0,
        intersection / union if union else 1.0,
    )
