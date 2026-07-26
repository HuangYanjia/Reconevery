from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from articulation_evaluation_worker.cli import _heldout_joint_position
from articulation_evaluation_worker.dense_io import read_dense_array
from articulation_evaluation_worker.rendering import (
    classify_depth,
    mask_metrics,
    render_mesh_depth,
)


def _write_quad(path: Path) -> None:
    path.write_text(
        """ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 2
property list uchar int vertex_indices
end_header
-0.5 -0.75 2
0.5 -0.75 2
0.5 -0.25 2
-0.5 -0.25 2
3 0 1 2
3 0 2 3
""",
        encoding="ascii",
    )


def _write_points(path: Path, points: np.ndarray) -> None:
    rows = "\n".join(" ".join(f"{value:.12g}" for value in row) for row in points)
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
        f"{rows}\n",
        encoding="ascii",
    )


@pytest.mark.parametrize("channels", [1, 3])
def test_colmap_dense_array_contract(tmp_path: Path, channels: int) -> None:
    width, height = 3, 2
    values = np.arange(width * height * channels, dtype="<f4").reshape(
        (width, height, channels),
        order="F",
    )
    path = tmp_path / "dense.bin"
    with path.open("wb") as stream:
        stream.write(f"{width}&{height}&{channels}&".encode())
        values.reshape(-1, order="F").tofile(stream)
    decoded = read_dense_array(path, channels)
    expected = values.transpose(1, 0, 2)
    if channels == 1:
        expected = expected[..., 0]
    assert decoded.tobytes() == expected.tobytes()


def test_nvdiffrast_articulated_link_render_and_occlusion(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("nvdiffrast.torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the articulated renderer test")
    mesh = tmp_path / "quad.ply"
    _write_quad(mesh)
    depth = render_mesh_depth(
        mesh,
        np.eye(4),
        np.eye(4),
        (32.0, 32.0, 31.5, 31.5),
        (64, 64),
    )
    rows, columns = np.nonzero(np.isfinite(depth))
    assert (int(columns.min()), int(columns.max()) + 1) == (24, 40)
    assert (int(rows.min()), int(rows.max()) + 1) == (20, 28)
    assert np.median(depth[np.isfinite(depth)]) == pytest.approx(2.0, abs=1e-6)
    target = np.isfinite(depth)
    scene = np.full(depth.shape, 1.0, dtype=np.float32)
    classified = classify_depth(depth, scene, target)
    assert np.count_nonzero(classified["occluded"]) == np.count_nonzero(target)
    precision, recall, iou = mask_metrics(classified["visible"], target)
    assert (precision, recall, iou) == (0.0, 0.0, 0.0)


def test_heldout_geometry_recovers_only_prismatic_q(tmp_path: Path) -> None:
    generator = np.random.default_rng(7)
    candidate_points = generator.normal(size=(500, 3))
    measured_points = candidate_points + np.asarray([0.7, 0.0, 0.0])
    _write_points(tmp_path / "candidate.ply", candidate_points)
    _write_points(tmp_path / "measured.ply", measured_points)
    position = _heldout_joint_position(
        input_root=tmp_path,
        link={"visual_asset_paths": ["candidate.ply"]},
        joint={
            "joint_id": "drawer_joint",
            "joint_type": "prismatic",
            "axis": [1.0, 0.0, 0.0],
            "pivot": None,
            "candidate_limit_lower": 0.0,
            "candidate_limit_upper": 1.0,
            "limit_source": "candidate_prior",
        },
        measured_joint=None,
        measured_part={"measured_point_cloud_path": "measured.ply"},
        base_matrix=np.eye(4),
        reference_from_state=np.eye(4),
    )
    assert position == pytest.approx(0.7, abs=2e-3)
