from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from recon2sim.ir import CameraIntrinsics

# Binary layouts follow COLMAP's BSD-licensed scripts/python/read_write_model.py helper.
# https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py

CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
    11: ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
}
SUPPORTED_CAMERA_MODELS = {
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "OPENCV",
}


class ColmapModelError(ValueError):
    """Raised when a COLMAP sparse model is missing, malformed, or unsupported."""


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model_id: int
    model_name: str
    width: int
    height: int
    params: tuple[float, ...]


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
    def average_reprojection_error(self) -> float | None:
        if not self.points3d:
            return None
        return sum(point.error for point in self.points3d.values()) / len(self.points3d)


class _Reader:
    def __init__(self, file: BinaryIO, path: Path) -> None:
        self.file = file
        self.path = path

    def unpack(self, format_string: str) -> tuple[int | float, ...]:
        size = struct.calcsize(format_string)
        payload = self.file.read(size)
        if len(payload) != size:
            raise ColmapModelError(
                f"malformed COLMAP binary {self.path}: expected {size} bytes, got {len(payload)}"
            )
        values = struct.unpack(format_string, payload)
        if any(not isinstance(value, (int, float)) for value in values):
            raise ColmapModelError(
                f"internal parser error for numeric COLMAP format {format_string!r}"
            )
        return values

    def c_string(self) -> str:
        payload = bytearray()
        while len(payload) <= 1024 * 1024:
            value = self.file.read(1)
            if not value:
                raise ColmapModelError(
                    f"malformed COLMAP binary {self.path}: unterminated image name"
                )
            if value == b"\0":
                try:
                    return payload.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ColmapModelError(
                        f"malformed COLMAP binary {self.path}: image name is not UTF-8"
                    ) from exc
            payload.extend(value)
        raise ColmapModelError(f"malformed COLMAP binary {self.path}: image name is too long")

    def require_eof(self) -> None:
        if self.file.read(1):
            raise ColmapModelError(
                f"malformed COLMAP binary {self.path}: unexpected trailing bytes"
            )


def _required_file(model_dir: Path, name: str) -> Path:
    path = model_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"COLMAP sparse model is missing required binary file: {path}")
    return path


def read_cameras(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with path.open("rb") as file:
        reader = _Reader(file, path)
        (count_raw,) = reader.unpack("<Q")
        count = int(count_raw)
        for _ in range(count):
            camera_id_raw, model_id_raw, width_raw, height_raw = reader.unpack("<iiQQ")
            camera_id = int(camera_id_raw)
            model_id = int(model_id_raw)
            definition = CAMERA_MODELS.get(model_id)
            if definition is None:
                raise ColmapModelError(
                    f"unsupported unknown COLMAP camera model ID {model_id} in {path}"
                )
            model_name, parameter_count = definition
            params = tuple(float(value) for value in reader.unpack(f"<{parameter_count}d"))
            if camera_id in cameras:
                raise ColmapModelError(f"duplicate COLMAP camera ID {camera_id} in {path}")
            cameras[camera_id] = ColmapCamera(
                camera_id=camera_id,
                model_id=model_id,
                model_name=model_name,
                width=int(width_raw),
                height=int(height_raw),
                params=params,
            )
        reader.require_eof()
    return cameras


def read_images(path: Path) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    with path.open("rb") as file:
        reader = _Reader(file, path)
        (count_raw,) = reader.unpack("<Q")
        for _ in range(int(count_raw)):
            properties = reader.unpack("<idddddddi")
            image_id = int(properties[0])
            qvec = tuple(float(value) for value in properties[1:5])
            tvec = tuple(float(value) for value in properties[5:8])
            camera_id = int(properties[8])
            name = reader.c_string()
            (point_count_raw,) = reader.unpack("<Q")
            points = []
            for _ in range(int(point_count_raw)):
                x_raw, y_raw, point3d_id_raw = reader.unpack("<ddq")
                points.append(
                    ColmapPoint2D(
                        x=float(x_raw),
                        y=float(y_raw),
                        point3d_id=int(point3d_id_raw),
                    )
                )
            if image_id in images:
                raise ColmapModelError(f"duplicate COLMAP image ID {image_id} in {path}")
            images[image_id] = ColmapImage(
                image_id=image_id,
                qvec_wxyz=(qvec[0], qvec[1], qvec[2], qvec[3]),
                tvec=(tvec[0], tvec[1], tvec[2]),
                camera_id=camera_id,
                name=name,
                points2d=tuple(points),
            )
        reader.require_eof()
    return images


def read_points3d(path: Path) -> dict[int, ColmapPoint3D]:
    points: dict[int, ColmapPoint3D] = {}
    with path.open("rb") as file:
        reader = _Reader(file, path)
        (count_raw,) = reader.unpack("<Q")
        for _ in range(int(count_raw)):
            properties = reader.unpack("<QdddBBBd")
            point3d_id = int(properties[0])
            (track_count_raw,) = reader.unpack("<Q")
            track = []
            for _ in range(int(track_count_raw)):
                image_id_raw, point2d_index_raw = reader.unpack("<ii")
                track.append(
                    ColmapTrackElement(
                        image_id=int(image_id_raw),
                        point2d_index=int(point2d_index_raw),
                    )
                )
            if point3d_id in points:
                raise ColmapModelError(f"duplicate COLMAP point3D ID {point3d_id} in {path}")
            points[point3d_id] = ColmapPoint3D(
                point3d_id=point3d_id,
                xyz=(float(properties[1]), float(properties[2]), float(properties[3])),
                rgb=(int(properties[4]), int(properties[5]), int(properties[6])),
                error=float(properties[7]),
                track=tuple(track),
            )
        reader.require_eof()
    return points


def read_model(model_dir: Path) -> ColmapModel:
    cameras = read_cameras(_required_file(model_dir, "cameras.bin"))
    images = read_images(_required_file(model_dir, "images.bin"))
    points3d = read_points3d(_required_file(model_dir, "points3D.bin"))
    missing_cameras = {image.camera_id for image in images.values()} - set(cameras)
    if missing_cameras:
        raise ColmapModelError(
            f"COLMAP images reference unknown camera IDs: {sorted(missing_cameras)}"
        )
    for image in images.values():
        missing_points = {
            point.point3d_id
            for point in image.points2d
            if point.point3d_id >= 0 and point.point3d_id not in points3d
        }
        if missing_points:
            raise ColmapModelError(
                f"COLMAP image {image.name!r} references unknown point3D IDs: "
                f"{sorted(missing_points)}"
            )
    for point in points3d.values():
        for track in point.track:
            track_image = images.get(track.image_id)
            if track_image is None:
                raise ColmapModelError(
                    f"COLMAP point3D {point.point3d_id} track references unknown image "
                    f"{track.image_id}"
                )
            if track.point2d_index < 0 or track.point2d_index >= len(track_image.points2d):
                raise ColmapModelError(
                    f"COLMAP point3D {point.point3d_id} track has invalid point2D index "
                    f"{track.point2d_index} for image {track_image.name!r}"
                )
            if track_image.points2d[track.point2d_index].point3d_id != point.point3d_id:
                raise ColmapModelError(
                    f"COLMAP point3D {point.point3d_id} track is inconsistent with image "
                    f"{track_image.name!r} point2D {track.point2d_index}"
                )
    return ColmapModel(cameras=cameras, images=images, points3d=points3d)


def camera_intrinsics(camera: ColmapCamera) -> CameraIntrinsics:
    name = camera.model_name
    params = camera.params
    if name == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1:3]
        distortion: list[float] = []
    elif name == "PINHOLE":
        fx, fy, cx, cy = params
        distortion = []
    elif name == "SIMPLE_RADIAL":
        fx = fy = params[0]
        cx, cy = params[1:3]
        distortion = [params[3]]
    elif name == "RADIAL":
        fx = fy = params[0]
        cx, cy = params[1:3]
        distortion = [params[3], params[4]]
    elif name == "OPENCV":
        fx, fy, cx, cy = params[:4]
        distortion = list(params[4:8])
    else:
        raise ColmapModelError(
            f"unsupported COLMAP camera model {name}; supported models are "
            f"{sorted(SUPPORTED_CAMERA_MODELS)}"
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


__all__ = [
    "CAMERA_MODELS",
    "SUPPORTED_CAMERA_MODELS",
    "ColmapCamera",
    "ColmapImage",
    "ColmapModel",
    "ColmapModelError",
    "ColmapPoint2D",
    "ColmapPoint3D",
    "ColmapTrackElement",
    "camera_intrinsics",
    "read_cameras",
    "read_images",
    "read_model",
    "read_points3d",
]
