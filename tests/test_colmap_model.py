from __future__ import annotations

import math
import struct
from pathlib import Path

import pytest

from recon2sim.colmap import (
    ColmapCamera,
    camera_intrinsics,
    colmap_pose_to_world_from_camera,
    normalize_quaternion_xyzw,
    quaternion_xyzw_to_rotation_matrix,
    read_model,
    rotation_matrix_to_quaternion_xyzw,
)
from recon2sim.colmap.model import ColmapModelError


def _write_model(root: Path, *, camera_model_id: int = 4) -> None:
    root.mkdir(parents=True)
    params_by_model = {
        4: (500.0, 510.0, 320.0, 240.0, 0.1, -0.02, 0.003, -0.004),
        5: (500.0, 510.0, 320.0, 240.0, 0.1, -0.02, 0.003, -0.004),
    }
    params = params_by_model[camera_model_id]
    (root / "cameras.bin").write_bytes(
        struct.pack("<QiiQQ", 1, 1, camera_model_id, 640, 480)
        + struct.pack(f"<{len(params)}d", *params)
    )
    image = (
        struct.pack("<Q", 1)
        + struct.pack("<idddddddi", 3, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 1)
        + b"frame_000000.png\0"
        + struct.pack("<Qddq", 1, 10.0, 20.0, 7)
    )
    (root / "images.bin").write_bytes(image)
    point = (
        struct.pack("<Q", 1)
        + struct.pack("<QdddBBBd", 7, 1.0, 2.0, 3.0, 10, 20, 30, 0.5)
        + struct.pack("<Qii", 1, 3, 0)
    )
    (root / "points3D.bin").write_bytes(point)


def test_colmap_binary_parser_reads_typed_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "0"
    _write_model(model_dir)
    model = read_model(model_dir)

    assert model.cameras[1].model_name == "OPENCV"
    assert model.images[3].name == "frame_000000.png"
    assert model.images[3].points2d[0].point3d_id == 7
    assert model.points3d[7].track[0].image_id == 3
    assert model.average_reprojection_error == 0.5
    intrinsics = camera_intrinsics(model.cameras[1])
    assert (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy) == (
        500.0,
        510.0,
        320.0,
        240.0,
    )
    assert intrinsics.distortion == [0.1, -0.02, 0.003, -0.004]


@pytest.mark.parametrize(
    ("name", "params", "expected"),
    [
        ("SIMPLE_PINHOLE", (5.0, 3.0, 2.0), (5.0, 5.0, [])),
        ("PINHOLE", (5.0, 6.0, 3.0, 2.0), (5.0, 6.0, [])),
        ("SIMPLE_RADIAL", (5.0, 3.0, 2.0, 0.1), (5.0, 5.0, [0.1])),
        ("RADIAL", (5.0, 3.0, 2.0, 0.1, 0.2), (5.0, 5.0, [0.1, 0.2])),
    ],
)
def test_supported_camera_model_mapping(
    name: str,
    params: tuple[float, ...],
    expected: tuple[float, float, list[float]],
) -> None:
    camera = ColmapCamera(1, 0, name, 10, 8, params)
    intrinsics = camera_intrinsics(camera)
    assert (intrinsics.fx, intrinsics.fy, intrinsics.distortion) == expected


def test_unsupported_camera_model_names_supported_set(tmp_path: Path) -> None:
    model_dir = tmp_path / "0"
    _write_model(model_dir, camera_model_id=5)
    camera = read_model(model_dir).cameras[1]
    with pytest.raises(
        ColmapModelError,
        match="unsupported COLMAP camera model OPENCV_FISHEYE.*SIMPLE_PINHOLE",
    ):
        camera_intrinsics(camera)


def test_missing_and_malformed_colmap_binary_files_are_actionable(tmp_path: Path) -> None:
    model_dir = tmp_path / "0"
    model_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="missing required binary file.*cameras.bin"):
        read_model(model_dir)
    (model_dir / "cameras.bin").write_bytes(struct.pack("<Q", 1))
    with pytest.raises(ColmapModelError, match="malformed COLMAP binary"):
        read_model(model_dir)


def test_identity_and_translation_pose_inversion() -> None:
    identity = colmap_pose_to_world_from_camera((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert identity.translation_m == (0.0, 0.0, 0.0)
    assert identity.rotation_xyzw == (0.0, 0.0, 0.0, 1.0)

    translated = colmap_pose_to_world_from_camera(
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 2.0, 3.0),
    )
    assert translated.translation_m == (-1.0, -2.0, -3.0)


def test_known_ninety_degree_rotation_is_inverted() -> None:
    half = math.sqrt(0.5)
    transform = colmap_pose_to_world_from_camera(
        (half, 0.0, 0.0, half),
        (0.0, 0.0, 0.0),
    )
    assert transform.rotation_xyzw == pytest.approx((0.0, 0.0, -half, half))


def test_quaternion_normalization_and_matrix_round_trip() -> None:
    assert normalize_quaternion_xyzw((0.0, 0.0, 0.0, 2.0)) == (0.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="norm is zero"):
        normalize_quaternion_xyzw((0.0, 0.0, 0.0, 0.0))

    source = normalize_quaternion_xyzw((0.2, -0.3, 0.1, 0.9))
    matrix = quaternion_xyzw_to_rotation_matrix(source)
    round_trip = rotation_matrix_to_quaternion_xyzw(matrix)
    assert round_trip == pytest.approx(source)
