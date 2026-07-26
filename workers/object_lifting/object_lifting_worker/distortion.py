from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

SUPPORTED_CAMERA_MODELS = {
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "OPENCV",
}


def distortion_coefficients(
    camera_model: str,
    distortion: Sequence[float],
) -> tuple[float, float, float, float, float]:
    if camera_model not in SUPPORTED_CAMERA_MODELS:
        raise ValueError(
            f"unsupported camera model {camera_model}; supported models: "
            + ", ".join(sorted(SUPPORTED_CAMERA_MODELS))
        )
    values = list(distortion)
    if camera_model in {"SIMPLE_PINHOLE", "PINHOLE"}:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    if camera_model == "SIMPLE_RADIAL":
        return (values[0] if values else 0.0), 0.0, 0.0, 0.0, 0.0
    if camera_model == "RADIAL":
        return (
            values[0] if len(values) > 0 else 0.0,
            values[1] if len(values) > 1 else 0.0,
            0.0,
            0.0,
            0.0,
        )
    return (
        values[0] if len(values) > 0 else 0.0,
        values[1] if len(values) > 1 else 0.0,
        values[2] if len(values) > 2 else 0.0,
        values[3] if len(values) > 3 else 0.0,
        values[4] if len(values) > 4 else 0.0,
    )


def distort_normalized(
    x: float,
    y: float,
    camera_model: str,
    distortion: Sequence[float],
) -> tuple[float, float]:
    k1, k2, p1, p2, k3 = distortion_coefficients(camera_model, distortion)
    radius2 = x * x + y * y
    radial = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
    tangential_x = 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x)
    tangential_y = p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return x * radial + tangential_x, y * radial + tangential_y


@dataclass(frozen=True)
class UndistortedMask:
    mask: Any
    width: int
    height: int
    intrinsics: dict[str, Any]
    map_hash: str


def undistort_binary_mask(
    mask: Any,
    *,
    camera_model: str,
    intrinsics: dict[str, Any],
    raster_scale: float,
) -> UndistortedMask:
    import cv2
    import numpy as np

    if camera_model not in SUPPORTED_CAMERA_MODELS:
        raise ValueError(
            f"unsupported camera model {camera_model}; supported models: "
            + ", ".join(sorted(SUPPORTED_CAMERA_MODELS))
        )
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    if mask.shape != (height, width):
        raise ValueError(
            f"mask dimensions {mask.shape[::-1]} do not match camera {(width, height)}"
        )
    camera_matrix = np.asarray(
        [
            [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
            [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    coefficients = np.asarray(
        distortion_coefficients(camera_model, intrinsics.get("distortion", [])),
        dtype=np.float64,
    )
    new_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        coefficients,
        (width, height),
        0.0,
        (width, height),
        centerPrincipalPoint=False,
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        coefficients,
        None,
        new_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    undistorted = cv2.remap(
        mask.astype(np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    output_width = max(1, int(round(width * raster_scale)))
    output_height = max(1, int(round(height * raster_scale)))
    if (output_width, output_height) != (width, height):
        undistorted = cv2.resize(
            undistorted,
            (output_width, output_height),
            interpolation=cv2.INTER_NEAREST,
        )
    undistorted = np.where(undistorted > 0, 255, 0).astype(np.uint8)
    scale_x = output_width / width
    scale_y = output_height / height
    output_intrinsics = {
        "width": output_width,
        "height": output_height,
        "fx": float(new_matrix[0, 0]) * scale_x,
        "fy": float(new_matrix[1, 1]) * scale_y,
        "cx": float(new_matrix[0, 2]) * scale_x,
        "cy": float(new_matrix[1, 2]) * scale_y,
        "distortion": [],
    }
    digest = hashlib.sha256()
    digest.update(map_x.tobytes(order="C"))
    digest.update(map_y.tobytes(order="C"))
    digest.update(str((output_width, output_height)).encode())
    return UndistortedMask(
        mask=undistorted,
        width=output_width,
        height=output_height,
        intrinsics=output_intrinsics,
        map_hash=digest.hexdigest(),
    )
