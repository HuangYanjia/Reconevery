from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

WORKER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKER_ROOT))

from object_lifting_worker.rasterization import (  # noqa: E402
    NvdiffrastRasterizer,
    cpu_rasterize_face_ids,
)


@pytest.mark.integration
@pytest.mark.requires_object_lifting_gpu
def test_nvdiffrast_matches_cpu_in_triangle_interiors() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    vertices = np.asarray(
        [
            (-1.4, -0.7, 2.0),
            (-0.2, -0.7, 2.0),
            (-0.8, 0.7, 2.0),
            (0.2, -0.7, 2.0),
            (1.4, -0.7, 2.0),
            (0.8, 0.7, 2.0),
        ],
        dtype=np.float32,
    )
    faces = np.asarray([(0, 1, 2), (3, 4, 5)], dtype=np.int64)
    intrinsics = {
        "width": 96,
        "height": 64,
        "fx": 55.0,
        "fy": 58.0,
        "cx": 43.25,
        "cy": 29.75,
    }
    pose = {
        "transform_world_from_camera": {
            "translation": [0.0, 0.0, 0.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    }
    rasterizer = NvdiffrastRasterizer(vertices, faces, face_chunk_size=2)
    gpu = rasterizer.rasterize(pose, intrinsics)
    cpu = np.asarray(
        cpu_rasterize_face_ids(
            vertices.tolist(),
            faces.tolist(),
            translation_world_from_camera=(0.0, 0.0, 0.0),
            rotation_xyzw_world_from_camera=(0.0, 0.0, 0.0, 1.0),
            intrinsics=intrinsics,
            near_plane=gpu.near_plane,
            far_plane=gpu.far_plane,
        ),
        dtype=np.int64,
    )

    def interior(buffer: object) -> object:
        value = np.asarray(buffer)
        result = value >= 0
        for row_offset, column_offset in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ):
            shifted = np.roll(value, (row_offset, column_offset), axis=(0, 1))
            result &= shifted == value
        result[[0, -1], :] = False
        result[:, [0, -1]] = False
        return result

    comparison = interior(cpu) & interior(gpu.face_ids)
    assert int(comparison.sum()) > 100
    assert np.array_equal(cpu[comparison], gpu.face_ids[comparison])


@pytest.mark.integration
@pytest.mark.requires_object_lifting_gpu
def test_nvdiffrast_keeps_triangle_crossing_near_and_camera_planes() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    vertices = np.asarray(
        [(-0.2, -0.2, -0.1), (0.3, -0.2, 1.5), (0.0, 0.4, 1.5)],
        dtype=np.float32,
    )
    faces = np.asarray([(0, 1, 2)], dtype=np.int64)
    intrinsics = {
        "width": 64,
        "height": 48,
        "fx": 36.0,
        "fy": 36.0,
        "cx": 30.25,
        "cy": 20.75,
    }
    pose = {
        "transform_world_from_camera": {
            "translation": [0.0, 0.0, 0.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    }
    rasterizer = NvdiffrastRasterizer(vertices, faces, face_chunk_size=1)
    raster = rasterizer.rasterize(pose, intrinsics)
    assert raster.processed_face_count == 1
    assert int((raster.face_ids == 0).sum()) > 0
