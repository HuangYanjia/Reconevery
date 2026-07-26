from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

WORKER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKER_ROOT))

from object_lifting_worker.surface_extraction import (  # noqa: E402
    seam_aware_component_diagnostics,
)
from object_lifting_worker.surface_samples import SurfaceSampleFusion  # noqa: E402


def test_surface_samples_fuse_views_and_preserve_original_face_ids() -> None:
    fusion = SurfaceSampleFusion(
        origin=(0.0, 0.0, 0.0),
        voxel_edge=0.5,
        core_weight=1.0,
        boundary_weight=0.25,
    )
    for frame_index, face_id, point in (
        (0, 5, (0.10, 0.10, 1.0)),
        (1, 8, (0.12, 0.11, 1.0)),
    ):
        fusion.accumulate(
            frame_index=frame_index,
            face_ids=np.asarray([[face_id]], dtype=np.int64),
            world_points=np.asarray([[point]], dtype=np.float64),
            barycentric=np.asarray([[[0.2, 0.3, 0.5]]], dtype=np.float64),
            depth=np.asarray([[1.0]], dtype=np.float64),
            core=np.asarray([[True]]),
            boundary=np.asarray([[False]]),
            frame_score=1.0,
        )
    result = fusion.finalize(
        min_supporting_views=2,
        min_positive_weight=2.0,
        accepted_score=0.65,
        ambiguous_score=0.40,
    )
    assert result.accepted_faces == [5, 8]
    assert result.cell_count == 1
    assert result.face_support[5].supporting_views == 2
    assert result.face_support[8].supporting_views == 2
    assert result.face_support[5].direct_sample_support == 1.0
    assert result.face_support[5].propagated_support == 0.0


def test_surface_sample_negative_evidence_prevents_false_acceptance() -> None:
    fusion = SurfaceSampleFusion(
        origin=(0.0, 0.0, 0.0),
        voxel_edge=1.0,
        core_weight=1.0,
        boundary_weight=0.25,
    )
    for frame_index in (0, 1):
        fusion.accumulate(
            frame_index=frame_index,
            face_ids=np.asarray([[3]], dtype=np.int64),
            world_points=np.asarray([[[0.1, 0.1, 0.1]]], dtype=np.float64),
            barycentric=np.asarray([[[0.3, 0.3, 0.4]]], dtype=np.float64),
            depth=np.asarray([[1.0]], dtype=np.float64),
            core=np.asarray([[True]]),
            boundary=np.asarray([[False]]),
            frame_score=1.0,
        )
    fusion.accumulate_negative(
        face_ids=np.full((4, 4), 3, dtype=np.int64),
        world_points=np.full((4, 4, 3), 0.1, dtype=np.float64),
        exterior=np.ones((4, 4), dtype=bool),
        frame_score=1.0,
        negative_weight=1.0,
    )
    result = fusion.finalize(
        min_supporting_views=2,
        min_positive_weight=2.0,
        accepted_score=0.65,
        ambiguous_score=0.40,
    )
    assert result.accepted_faces == []
    assert result.ambiguous_faces == []


def test_seam_diagnostic_groups_duplicated_boundary_vertices_without_new_faces() -> None:
    vertices = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ],
        dtype=np.float64,
    )
    faces = np.asarray([(0, 1, 2), (3, 4, 5)], dtype=np.int64)
    result = seam_aware_component_diagnostics(
        vertices,
        faces,
        [0, 1],
        median_edge=1.0,
        centroid_distance_multiplier=1.0,
        endpoint_distance_multiplier=0.01,
        normal_cosine=0.99,
    )
    assert result == {
        "exact_component_count": 2,
        "seam_aware_component_count": 1,
        "potential_chunk_seam_merges": 1,
    }
