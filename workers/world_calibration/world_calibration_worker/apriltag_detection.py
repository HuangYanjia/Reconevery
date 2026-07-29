from __future__ import annotations

from typing import Any

import numpy as np


def detect_official(
    image: np.ndarray,
    *,
    family: str,
    tag_id: int,
    tag_size_m: float,
    camera_params: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    # The module is installed by the pinned official AprilTag CMake project.
    from apriltag import apriltag  # type: ignore[import-not-found]

    detector = apriltag(family)
    records: list[dict[str, Any]] = []
    for detection in detector.detect(image):
        if int(detection["id"]) != tag_id:
            continue
        pose = detector.estimate_tag_pose(
            detection,
            tag_size_m,
            *camera_params,
        )
        rotation_camera_from_tag = np.asarray(pose["R"], dtype=float).reshape(3, 3)
        translation_camera_from_tag = np.asarray(pose["t"], dtype=float).reshape(3)
        rotation_tag_from_camera = rotation_camera_from_tag.T
        camera_center_tag = -rotation_tag_from_camera @ translation_camera_from_tag
        records.append(
            {
                "tag_id": tag_id,
                "corners_xy": np.asarray(detection["lb-rb-rt-lt"], dtype=float).tolist(),
                "decision_margin": float(detection.get("margin", 0.0)),
                "hamming": int(detection.get("hamming", 0)),
                "camera_center_tag_m": camera_center_tag.tolist(),
                "rotation_tag_from_camera": rotation_tag_from_camera.reshape(-1).tolist(),
                "pose_error": float(pose["error"]),
            }
        )
    return records


__all__ = ["detect_official"]
