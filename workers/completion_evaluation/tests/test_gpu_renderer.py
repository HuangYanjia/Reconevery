from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from completion_evaluation_worker.native_render_dispatch import render_mesh_candidate


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


def test_nvdiffrast_output_is_normalized_to_top_left_pixels(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("nvdiffrast.torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the manual renderer parity test")
    mesh = tmp_path / "quad.ply"
    _write_quad(mesh)
    result = render_mesh_candidate(
        mesh,
        np.eye(4),
        {
            "camera_from_world": np.eye(4),
            "width": 64,
            "height": 64,
            "fx": 32.0,
            "fy": 32.0,
            "cx": 31.5,
            "cy": 31.5,
            "near": 0.1,
            "far": 10.0,
        },
    )
    rows, columns = np.nonzero(result.valid)
    assert (int(columns.min()), int(columns.max()) + 1) == (24, 40)
    assert (int(rows.min()), int(rows.max()) + 1) == (20, 28)
    assert np.isfinite(result.depth[result.valid]).all()
    assert np.median(result.depth[result.valid]) == pytest.approx(2.0, abs=1e-6)
