from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from alignment_worker.camera_projection import (
    distortion_coefficients,
    transform_world_point_to_camera,
    undistort_points,
)


def _data_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _undistortion_record(
    frame_id: str,
    camera: dict[str, Any],
    *,
    raster_scale: float,
) -> dict[str, object]:
    import cv2
    import numpy as np

    intrinsics = camera["intrinsics"]
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    source_matrix = np.asarray(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    coefficients = np.asarray(
        distortion_coefficients(camera["model"], intrinsics.get("distortion", [])),
        dtype=np.float64,
    )
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(
        source_matrix,
        coefficients,
        (width, height),
        0.0,
        (width, height),
        centerPrincipalPoint=False,
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        source_matrix,
        coefficients,
        None,
        new_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    output_width = max(1, int(round(width * raster_scale)))
    output_height = max(1, int(round(height * raster_scale)))
    digest = hashlib.sha256()
    digest.update(map_x.tobytes(order="C"))
    digest.update(map_y.tobytes(order="C"))
    digest.update(str((output_width, output_height)).encode())
    scale_x = output_width / width
    scale_y = output_height / height
    undistorted = {
        "width": output_width,
        "height": output_height,
        "fx": float(new_matrix[0, 0]) * scale_x,
        "fy": float(new_matrix[1, 1]) * scale_y,
        "cx": float(new_matrix[0, 2]) * scale_x,
        "cy": float(new_matrix[1, 2]) * scale_y,
        "distortion": [],
    }
    return {
        "frame_id": frame_id,
        "source_camera_model": camera["model"],
        "source_intrinsics": intrinsics,
        "source_distortion": intrinsics.get("distortion", []),
        "undistorted_width": output_width,
        "undistorted_height": output_height,
        "undistorted_intrinsics": undistorted,
        "roi_xywh": [int(value) for value in roi],
        "crop_policy": "full_image_alpha_0",
        "map_hash": digest.hexdigest(),
    }


def prepare_sparse_observations(
    *,
    camera: dict[str, Any],
    package_manifest: dict[str, Any],
    images_path: Path,
    points3d_path: Path,
    configuration: dict[str, Any],
) -> tuple[dict[str, object], dict[str, dict[str, object]], Any]:
    import numpy as np

    point_records: dict[int, dict[str, object]] = {}
    for line in _data_lines(points3d_path):
        fields = line.split()
        point_id = int(fields[0])
        point_records[point_id] = {
            "point_world": (float(fields[1]), float(fields[2]), float(fields[3])),
            "error": float(fields[7]),
            "track_length": max(1, (len(fields) - 8) // 2),
        }
    frame_by_name = {
        item["package_image_name"]: item["frame_id"]
        for item in package_manifest["registered_frames"]
    }
    pose_by_frame = {pose["frame_id"]: pose for pose in camera["poses"]}
    raster_scale = float(configuration["raster_scale"])
    undistortion_records = {
        frame_id: _undistortion_record(frame_id, camera, raster_scale=raster_scale)
        for frame_id in camera["registered_frame_ids"]
    }
    observations: list[dict[str, object]] = []
    raw_count = 0
    rejected = 0
    lines = _data_lines(images_path)
    for index in range(0, len(lines), 2):
        image_fields = lines[index].split()
        if len(image_fields) < 10:
            raise ValueError("COLMAP images.txt registered-image row is malformed")
        frame_id = frame_by_name.get(image_fields[9])
        if frame_id is None or frame_id not in pose_by_frame:
            continue
        values = lines[index + 1].split()
        distorted = []
        candidates = []
        for offset in range(0, len(values), 3):
            point_id = int(values[offset + 2])
            if point_id < 0 or point_id not in point_records:
                continue
            raw_count += 1
            distorted.append((float(values[offset]), float(values[offset + 1])))
            candidates.append((offset // 3, point_id, point_records[point_id]))
        if not candidates:
            continue
        mapped = undistort_points(
            distorted,
            camera_model=camera["model"],
            intrinsics=camera["intrinsics"],
            raster_scale=raster_scale,
        )
        width = undistortion_records[frame_id]["undistorted_width"]
        height = undistortion_records[frame_id]["undistorted_height"]
        transform = pose_by_frame[frame_id]["transform_world_from_camera"]
        for distorted_pixel, undistorted_pixel, candidate in zip(
            distorted, mapped, candidates, strict=True
        ):
            point2d_index, point_id, point = candidate
            world = point["point_world"]
            depth = transform_world_point_to_camera(
                world,
                transform["translation"],
                transform["rotation_xyzw"],
            )[2]
            keep = (
                float(point["error"]) <= float(configuration["max_colmap_reprojection_error"])
                and int(point["track_length"]) >= int(configuration["min_track_length"])
                and depth > float(configuration["minimum_camera_depth"])
            )
            if bool(configuration["require_inside_undistorted_image"]):
                keep = keep and (
                    0 <= undistorted_pixel[0] < width and 0 <= undistorted_pixel[1] < height
                )
            if not keep:
                rejected += 1
                continue
            observations.append(
                {
                    "point3d_id": point_id,
                    "frame_id": frame_id,
                    "point2d_index": point2d_index,
                    "distorted_pixel": distorted_pixel,
                    "undistorted_pixel": (
                        float(undistorted_pixel[0]),
                        float(undistorted_pixel[1]),
                    ),
                    "point_world": world,
                    "camera_depth": float(depth),
                    "colmap_reprojection_error": float(point["error"]),
                    "track_length": int(point["track_length"]),
                    "camera_model": camera["model"],
                }
            )
    observations.sort(key=lambda item: (item["frame_id"], item["point2d_index"]))
    unique_points = np.asarray(
        [point_records[point_id]["point_world"] for point_id in sorted(point_records)],
        dtype=np.float64,
    )
    manifest = {
        "schema_version": "0.1.0",
        "observations": observations,
        "total_colmap_points": len(point_records),
        "total_raw_observations": raw_count,
        "retained_observations": len(observations),
        "rejected_observations": rejected,
        "filtering_configuration": configuration,
        "undistortion_records": [
            undistortion_records[frame_id] for frame_id in camera["registered_frame_ids"]
        ],
        "warnings": [],
    }
    return manifest, undistortion_records, unique_points


def deterministic_split(
    observations: list[dict[str, object]],
    registered_frame_ids: list[str],
    *,
    seed: int,
) -> dict[str, object]:
    training_frames = registered_frame_ids[::2]
    validation_frames = registered_frame_ids[1::2]
    if not validation_frames and training_frames:
        validation_frames = training_frames[-1:]
        training_frames = training_frames[:-1]
    all_points = sorted({int(item["point3d_id"]) for item in observations})
    training_points = [point_id for point_id in all_points if (point_id + seed) % 2 == 0]
    validation_points = [point_id for point_id in all_points if (point_id + seed) % 2 == 1]
    training_set = set(training_frames)
    validation_set = set(validation_frames)
    training_point_set = set(training_points)
    validation_point_set = set(validation_points)
    training_count = sum(
        item["frame_id"] in training_set and item["point3d_id"] in training_point_set
        for item in observations
    )
    validation_count = sum(
        item["frame_id"] in validation_set and item["point3d_id"] in validation_point_set
        for item in observations
    )
    return {
        "schema_version": "0.1.0",
        "strategy": "alternating_registered_frames_and_point_ids",
        "training_frame_ids": training_frames,
        "validation_frame_ids": validation_frames,
        "training_point_ids": training_points,
        "validation_point_ids": validation_points,
        "training_observation_count": training_count,
        "validation_observation_count": validation_count,
        "split_seed": seed,
    }
