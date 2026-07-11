"""Small, dependency-free reader for COLMAP sparse binary models.

The binary layout and camera-model table follow COLMAP's BSD-licensed model I/O
implementation and official output-format documentation at
https://colmap.github.io/format.html. This module is an independent typed reader;
it does not copy/import COLMAP or PyCOLMAP code.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from recon2sim.ir import CameraIntrinsics


@dataclass(frozen=True)
class CameraModelSpec:
    model_id: int
    name: str
    parameter_count: int


CAMERA_MODELS = {
    spec.model_id: spec
    for spec in (
        CameraModelSpec(0, "SIMPLE_PINHOLE", 3),
        CameraModelSpec(1, "PINHOLE", 4),
        CameraModelSpec(2, "SIMPLE_RADIAL", 4),
        CameraModelSpec(3, "RADIAL", 5),
        CameraModelSpec(4, "OPENCV", 8),
        CameraModelSpec(5, "OPENCV_FISHEYE", 8),
        CameraModelSpec(6, "FULL_OPENCV", 12),
        CameraModelSpec(7, "FOV", 5),
        CameraModelSpec(8, "SIMPLE_RADIAL_FISHEYE", 4),
        CameraModelSpec(9, "RADIAL_FISHEYE", 5),
        CameraModelSpec(10, "THIN_PRISM_FISHEYE", 12),
        CameraModelSpec(11, "RAD_TAN_THIN_PRISM_FISHEYE", 16),
    )
}
SUPPORTED_CAMERA_MODELS = {
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "OPENCV",
}
MAX_BINARY_RECORDS = 10_000_000
MAX_IMAGE_NAME_BYTES = 1_048_576


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model_name: str
    width: int
    height: int
    parameters: tuple[float, ...]


@dataclass(frozen=True)
class ColmapPoint2D:
    x: float
    y: float
    point3d_id: int


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qvec_wxyz: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str
    points2d: tuple[ColmapPoint2D, ...]


@dataclass(frozen=True)
class ColmapTrackElement:
    image_id: int
    point2d_index: int


@dataclass(frozen=True)
class ColmapPoint3D:
    point3d_id: int
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int]
    error: float
    track: tuple[ColmapTrackElement, ...]


@dataclass(frozen=True)
class ColmapModel:
    cameras: dict[int, ColmapCamera]
    images: dict[int, ColmapImage]
    points3d: dict[int, ColmapPoint3D]

    @property
    def mean_track_length(self) -> float:
        if not self.points3d:
            return 0.0
        return sum(len(point.track) for point in self.points3d.values()) / len(self.points3d)

    @property
    def mean_reprojection_error(self) -> float | None:
        if not self.points3d:
            return None
        return sum(point.error for point in self.points3d.values()) / len(self.points3d)


class ColmapFormatError(ValueError):
    """Raised when a COLMAP sparse binary file is missing or malformed."""


def _read_exact(file: BinaryIO, size: int, label: str) -> bytes:
    data = file.read(size)
    if len(data) != size:
        raise ColmapFormatError(f"truncated COLMAP binary while reading {label}")
    return data


def _read_struct(file: BinaryIO, format_string: str, label: str) -> tuple[int | float, ...]:
    binary_format = "<" + format_string
    return struct.unpack(
        binary_format,
        _read_exact(file, struct.calcsize(binary_format), label),
    )


def _read_count(file: BinaryIO, label: str) -> int:
    count = int(_read_struct(file, "Q", label)[0])
    if count > MAX_BINARY_RECORDS:
        raise ColmapFormatError(f"unreasonable {label} in COLMAP binary: {count}")
    return count


def _require_eof(file: BinaryIO, path: Path) -> None:
    if file.read(1):
        raise ColmapFormatError(f"unexpected trailing bytes in COLMAP binary: {path}")


def read_cameras_binary(path: Path) -> dict[int, ColmapCamera]:
    if not path.is_file():
        raise FileNotFoundError(f"missing COLMAP cameras binary: {path}")
    cameras: dict[int, ColmapCamera] = {}
    with path.open("rb") as file:
        for _ in range(_read_count(file, "camera count")):
            camera_id, model_id, width, height = _read_struct(file, "iiQQ", "camera record")
            model_spec = CAMERA_MODELS.get(int(model_id))
            if model_spec is None:
                raise ColmapFormatError(f"unknown COLMAP camera model ID: {model_id}")
            parameters = tuple(
                float(value)
                for value in _read_struct(
                    file,
                    "d" * model_spec.parameter_count,
                    f"camera {camera_id} parameters",
                )
            )
            camera = ColmapCamera(
                int(camera_id), model_spec.name, int(width), int(height), parameters
            )
            if camera.camera_id <= 0 or camera.width <= 0 or camera.height <= 0:
                raise ColmapFormatError(
                    f"COLMAP camera record has invalid ID or dimensions: {camera}"
                )
            if not all(math.isfinite(value) for value in camera.parameters):
                raise ColmapFormatError(
                    f"COLMAP camera {camera.camera_id} contains non-finite parameters"
                )
            if camera.camera_id in cameras:
                raise ColmapFormatError(f"duplicate COLMAP camera ID: {camera.camera_id}")
            cameras[camera.camera_id] = camera
        _require_eof(file, path)
    return cameras


def _read_null_terminated_name(file: BinaryIO) -> str:
    data = bytearray()
    for _ in range(MAX_IMAGE_NAME_BYTES):
        value = _read_exact(file, 1, "image name")
        if value == b"\0":
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ColmapFormatError("COLMAP image name is not valid UTF-8") from exc
        data.extend(value)
    raise ColmapFormatError("COLMAP image name exceeds the safety limit")


def read_images_binary(path: Path) -> dict[int, ColmapImage]:
    if not path.is_file():
        raise FileNotFoundError(f"missing COLMAP images binary: {path}")
    images: dict[int, ColmapImage] = {}
    with path.open("rb") as file:
        for _ in range(_read_count(file, "registered image count")):
            values = _read_struct(file, "idddddddi", "registered image record")
            image_id = int(values[0])
            name = _read_null_terminated_name(file)
            points: list[ColmapPoint2D] = []
            for _ in range(_read_count(file, f"image {image_id} observation count")):
                x, y, point3d_id = _read_struct(file, "ddq", "2D observation")
                point = ColmapPoint2D(float(x), float(y), int(point3d_id))
                if not math.isfinite(point.x) or not math.isfinite(point.y):
                    raise ColmapFormatError(
                        f"COLMAP image {image_id} contains a non-finite 2D observation"
                    )
                if point.point3d_id < -1:
                    raise ColmapFormatError(
                        f"COLMAP image {image_id} contains invalid point3D ID {point.point3d_id}"
                    )
                points.append(point)
            image = ColmapImage(
                image_id=image_id,
                qvec_wxyz=(
                    float(values[1]),
                    float(values[2]),
                    float(values[3]),
                    float(values[4]),
                ),
                tvec=(float(values[5]), float(values[6]), float(values[7])),
                camera_id=int(values[8]),
                name=name,
                points2d=tuple(points),
            )
            if image.image_id <= 0 or image.camera_id <= 0 or not image.name:
                raise ColmapFormatError(
                    f"COLMAP registered image has invalid ID, camera, or name: {image}"
                )
            if not all(math.isfinite(value) for value in (*image.qvec_wxyz, *image.tvec)):
                raise ColmapFormatError(
                    f"COLMAP registered image {image_id} contains a non-finite pose"
                )
            if image_id in images:
                raise ColmapFormatError(f"duplicate COLMAP image ID: {image_id}")
            images[image_id] = image
        _require_eof(file, path)
    return images


def read_points3d_binary(path: Path) -> dict[int, ColmapPoint3D]:
    if not path.is_file():
        raise FileNotFoundError(f"missing COLMAP points3D binary: {path}")
    points: dict[int, ColmapPoint3D] = {}
    with path.open("rb") as file:
        for _ in range(_read_count(file, "sparse point count")):
            values = _read_struct(file, "QdddBBBd", "sparse point record")
            point_id = int(values[0])
            track: list[ColmapTrackElement] = []
            for _ in range(_read_count(file, f"point {point_id} track length")):
                image_id, point2d_index = _read_struct(file, "ii", "track element")
                element = ColmapTrackElement(int(image_id), int(point2d_index))
                if element.image_id <= 0 or element.point2d_index < 0:
                    raise ColmapFormatError(
                        f"COLMAP point {point_id} contains an invalid track element: {element}"
                    )
                track.append(element)
            point = ColmapPoint3D(
                point3d_id=point_id,
                xyz=(float(values[1]), float(values[2]), float(values[3])),
                rgb=(int(values[4]), int(values[5]), int(values[6])),
                error=float(values[7]),
                track=tuple(track),
            )
            if (
                point.point3d_id <= 0
                or point.error < 0
                or not all(math.isfinite(value) for value in (*point.xyz, point.error))
            ):
                raise ColmapFormatError(f"COLMAP point {point_id} contains invalid numeric values")
            if point_id in points:
                raise ColmapFormatError(f"duplicate COLMAP point ID: {point_id}")
            points[point_id] = point
        _require_eof(file, path)
    return points


def read_colmap_model(path: Path) -> ColmapModel:
    cameras = read_cameras_binary(path / "cameras.bin")
    images = read_images_binary(path / "images.bin")
    points = read_points3d_binary(path / "points3D.bin")
    missing_camera_ids = sorted({image.camera_id for image in images.values()} - set(cameras))
    if missing_camera_ids:
        raise ColmapFormatError(
            f"registered images reference missing camera IDs: {missing_camera_ids}"
        )
    image_ids = set(images)
    point_ids = set(points)
    for image in images.values():
        missing_point_ids = sorted(
            {
                observation.point3d_id
                for observation in image.points2d
                if observation.point3d_id != -1
            }
            - point_ids
        )
        if missing_point_ids:
            raise ColmapFormatError(
                f"image {image.image_id} observations reference missing point IDs: "
                f"{missing_point_ids}"
            )
    for point in points.values():
        missing_image_ids = sorted({element.image_id for element in point.track} - image_ids)
        if missing_image_ids:
            raise ColmapFormatError(
                f"point {point.point3d_id} track references missing image IDs: {missing_image_ids}"
            )
        for element in point.track:
            image = images[element.image_id]
            if element.point2d_index >= len(image.points2d):
                raise ColmapFormatError(
                    f"point {point.point3d_id} track index {element.point2d_index} is out of "
                    f"range for image {element.image_id}"
                )
            observation = image.points2d[element.point2d_index]
            if observation.point3d_id != point.point3d_id:
                raise ColmapFormatError(
                    f"point {point.point3d_id} track disagrees with image "
                    f"{element.image_id} observation {element.point2d_index}"
                )
    return ColmapModel(cameras, images, points)


def camera_intrinsics(camera: ColmapCamera) -> CameraIntrinsics:
    parameters = camera.parameters
    if camera.model_name == "SIMPLE_PINHOLE":
        focal, cx, cy = parameters
        fx = fy = focal
        distortion: list[float] = []
    elif camera.model_name == "PINHOLE":
        fx, fy, cx, cy = parameters
        distortion = []
    elif camera.model_name == "SIMPLE_RADIAL":
        focal, cx, cy, radial = parameters
        fx = fy = focal
        distortion = [radial]
    elif camera.model_name == "RADIAL":
        focal, cx, cy, radial_1, radial_2 = parameters
        fx = fy = focal
        distortion = [radial_1, radial_2]
    elif camera.model_name == "OPENCV":
        fx, fy, cx, cy, radial_1, radial_2, tangent_1, tangent_2 = parameters
        distortion = [radial_1, radial_2, tangent_1, tangent_2]
    else:
        raise ValueError(
            f"unsupported COLMAP camera model {camera.model_name!r}; supported models are "
            f"{sorted(SUPPORTED_CAMERA_MODELS)}. The raw COLMAP workspace has been preserved."
        )
    return CameraIntrinsics(
        width=camera.width,
        height=camera.height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        distortion=distortion,
    )
