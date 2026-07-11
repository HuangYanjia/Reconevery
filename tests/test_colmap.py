from __future__ import annotations

import math
import struct
from pathlib import Path

import pytest

from recon2sim.adapters.colmap import (
    rank_sparse_models,
    reconstruction_confidence,
    select_sparse_model,
)
from recon2sim.colmap import (
    ColmapCamera,
    ColmapFormatError,
    ColmapImage,
    ColmapModel,
    ColmapPoint3D,
    ColmapTrackElement,
    camera_intrinsics,
    colmap_world_to_camera_to_world_from_camera,
    normalize_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
    read_colmap_model,
    transpose,
)


def _camera_parameters(model_id: int) -> tuple[float, ...]:
    return {
        0: (500.0, 320.0, 240.0),
        1: (500.0, 510.0, 320.0, 240.0),
        2: (500.0, 320.0, 240.0, 0.01),
        3: (500.0, 320.0, 240.0, 0.01, -0.001),
        4: (500.0, 510.0, 320.0, 240.0, 0.01, -0.001, 0.002, -0.002),
        5: (500.0, 510.0, 320.0, 240.0, 0.01, -0.001, 0.002, -0.002),
    }[model_id]


def write_synthetic_colmap_model(
    path: Path,
    image_names: list[str],
    *,
    camera_model_id: int = 4,
    point_count: int = 2,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    parameters = _camera_parameters(camera_model_id)
    with (path / "cameras.bin").open("wb") as file:
        file.write(struct.pack("<Q", 1))
        file.write(struct.pack("<iiQQ", 1, camera_model_id, 640, 480))
        file.write(struct.pack("<" + "d" * len(parameters), *parameters))
    with (path / "images.bin").open("wb") as file:
        file.write(struct.pack("<Q", len(image_names)))
        for index, name in enumerate(image_names, start=1):
            file.write(
                struct.pack(
                    "<idddddddi",
                    index,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    float(index - 1),
                    0.0,
                    0.0,
                    1,
                )
            )
            file.write(name.encode("utf-8") + b"\0")
            observation_count = 1 if point_count else 0
            file.write(struct.pack("<Q", observation_count))
            if observation_count:
                file.write(struct.pack("<ddq", 10.0, 20.0, 1))
    with (path / "points3D.bin").open("wb") as file:
        file.write(struct.pack("<Q", point_count))
        for point_id in range(1, point_count + 1):
            file.write(
                struct.pack(
                    "<QdddBBBd",
                    point_id,
                    float(point_id),
                    0.0,
                    1.0,
                    100,
                    120,
                    140,
                    0.5,
                )
            )
            track_images = list(range(1, min(len(image_names), 2) + 1)) if point_id == 1 else []
            file.write(struct.pack("<Q", len(track_images)))
            for image_id in track_images:
                file.write(struct.pack("<ii", image_id, 0))


def _memory_model(image_count: int, point_count: int) -> ColmapModel:
    camera = ColmapCamera(1, "PINHOLE", 640, 480, (500.0, 500.0, 320.0, 240.0))
    images = {
        index: ColmapImage(
            index,
            (1.0, 0.0, 0.0, 0.0),
            (float(index), 0.0, 0.0),
            1,
            f"frame_{index:06d}.png",
            (),
        )
        for index in range(1, image_count + 1)
    }
    points = {
        index: ColmapPoint3D(
            index,
            (float(index), 0.0, 0.0),
            (1, 2, 3),
            0.5,
            (ColmapTrackElement(1, 0),) if image_count else (),
        )
        for index in range(1, point_count + 1)
    }
    return ColmapModel({1: camera}, images, points)


def test_binary_parser_reads_tiny_synthetic_model(tmp_path: Path) -> None:
    model_path = tmp_path / "0"
    write_synthetic_colmap_model(
        model_path,
        ["frame_000000.png", "frame_000001.png"],
        point_count=2,
    )
    model = read_colmap_model(model_path)
    assert model.cameras[1].model_name == "OPENCV"
    assert model.images[2].name == "frame_000001.png"
    assert model.images[2].tvec == (1.0, 0.0, 0.0)
    assert len(model.points3d) == 2
    assert model.mean_track_length == 1.0
    assert model.mean_reprojection_error == 0.5


def test_binary_parser_rejects_truncated_model(tmp_path: Path) -> None:
    model_path = tmp_path / "0"
    write_synthetic_colmap_model(model_path, ["frame_000000.png"])
    (model_path / "images.bin").write_bytes(b"\x01\x00")
    with pytest.raises(ColmapFormatError, match="truncated COLMAP binary"):
        read_colmap_model(model_path)


def test_binary_parser_reports_missing_required_files(tmp_path: Path) -> None:
    model_path = tmp_path / "empty_model"
    model_path.mkdir()
    with pytest.raises(FileNotFoundError, match="missing COLMAP cameras binary"):
        read_colmap_model(model_path)


@pytest.mark.parametrize(
    ("model_name", "parameters", "fx", "fy", "distortion"),
    [
        ("SIMPLE_PINHOLE", (500.0, 320.0, 240.0), 500.0, 500.0, []),
        ("PINHOLE", (500.0, 510.0, 320.0, 240.0), 500.0, 510.0, []),
        ("SIMPLE_RADIAL", (500.0, 320.0, 240.0, 0.1), 500.0, 500.0, [0.1]),
        ("RADIAL", (500.0, 320.0, 240.0, 0.1, -0.01), 500.0, 500.0, [0.1, -0.01]),
        (
            "OPENCV",
            (500.0, 510.0, 320.0, 240.0, 0.1, -0.01, 0.02, -0.02),
            500.0,
            510.0,
            [0.1, -0.01, 0.02, -0.02],
        ),
    ],
)
def test_supported_camera_model_mapping(
    model_name: str,
    parameters: tuple[float, ...],
    fx: float,
    fy: float,
    distortion: list[float],
) -> None:
    intrinsics = camera_intrinsics(ColmapCamera(1, model_name, 640, 480, parameters))
    assert intrinsics.fx == fx
    assert intrinsics.fy == fy
    assert intrinsics.distortion == distortion


def test_unsupported_camera_model_names_supported_set() -> None:
    camera = ColmapCamera(
        1,
        "OPENCV_FISHEYE",
        640,
        480,
        (500.0, 500.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="unsupported COLMAP camera model 'OPENCV_FISHEYE'"):
        camera_intrinsics(camera)


def test_colmap_pose_identity_translation_and_normalization() -> None:
    identity = colmap_world_to_camera_to_world_from_camera((2.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert identity.translation_m == (0.0, 0.0, 0.0)
    assert identity.rotation_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))

    translated = colmap_world_to_camera_to_world_from_camera((1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    assert translated.translation_m == pytest.approx((-1.0, -2.0, -3.0))
    assert normalize_quaternion_wxyz((2.0, 0.0, 0.0, 0.0)) == (1.0, 0.0, 0.0, 0.0)


def test_colmap_pose_known_90_degree_rotation_and_round_trip() -> None:
    angle = math.pi / 2
    qvec = (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))
    transform = colmap_world_to_camera_to_world_from_camera(qvec, (0.0, 0.0, 0.0))
    assert transform.rotation_xyzw == pytest.approx(
        (0.0, 0.0, -math.sin(angle / 2), math.cos(angle / 2))
    )
    matrix = quaternion_wxyz_to_matrix(qvec)
    round_trip = transpose(transpose(matrix))
    for actual_row, expected_row in zip(round_trip, matrix, strict=True):
        assert actual_row == pytest.approx(expected_row)


def test_colmap_pose_rejects_zero_quaternion() -> None:
    with pytest.raises(ValueError, match="finite and non-zero"):
        normalize_quaternion_wxyz((0.0, 0.0, 0.0, 0.0))


def test_model_selection_is_deterministic_and_enforces_thresholds() -> None:
    models = {
        "2": _memory_model(8, 100),
        "0": _memory_model(9, 50),
        "1": _memory_model(9, 75),
    }
    ranked = rank_sparse_models(models, input_frame_count=10)
    assert [candidate.model_id for candidate in ranked] == ["1", "0", "2"]
    selected, diagnostics = select_sparse_model(
        models,
        input_frame_count=10,
        min_registered_frames=8,
        min_registration_ratio=0.8,
    )
    assert selected is not None and selected.model_id == "1"
    assert diagnostics[0].selected is True
    assert 0 <= reconstruction_confidence(diagnostics[0]) <= 1

    rejected, rejected_diagnostics = select_sparse_model(
        models,
        input_frame_count=10,
        min_registered_frames=10,
        min_registration_ratio=1.0,
    )
    assert rejected is None
    assert "below minimum" in (rejected_diagnostics[0].rejection_reason or "")
