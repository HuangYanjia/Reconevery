from __future__ import annotations

import numpy as np
from alignment_worker.diagnostics import ambiguous_candidate_ids, residual_is_structured
from alignment_worker.optimizer import select_candidate_by_training_objective
from alignment_worker.sim3 import apply_transform, decompose_similarity, umeyama_similarity
from alignment_worker.transform_chain import _render_equivalence


def _points() -> np.ndarray:
    return np.asarray(
        [
            [-1.0, -0.5, 0.2],
            [0.1, 1.2, -0.3],
            [1.4, -0.2, 0.7],
            [0.6, 0.8, 1.3],
            [-0.7, 0.4, 1.1],
            [1.1, 1.5, -0.8],
        ],
        dtype=np.float64,
    )


def test_identity_similarity() -> None:
    source = _points()
    matrix = umeyama_similarity(source, source)
    np.testing.assert_allclose(matrix, np.eye(4), atol=1e-10)


def test_known_translation_rotation_scale_and_full_sim3() -> None:
    source = _points()
    angle = np.deg2rad(25.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    expected = np.eye(4)
    expected[:3, :3] = 1.7 * rotation
    expected[:3, 3] = [0.4, -1.2, 0.8]
    target = apply_transform(source, expected)
    recovered = umeyama_similarity(source, target)
    np.testing.assert_allclose(recovered, expected, atol=1e-10)
    decomposition = decompose_similarity(recovered, scene_diagonal=4.0)
    assert abs(float(decomposition["scale"]) - 1.7) < 1e-10
    assert abs(float(decomposition["rotation_degrees"]) - 25.0) < 1e-9


def test_heldout_points_follow_recovered_transform() -> None:
    training = _points()
    expected = np.eye(4)
    expected[:3, :3] *= 0.6
    expected[:3, 3] = [-0.5, 0.25, 1.0]
    recovered = umeyama_similarity(training, apply_transform(training, expected))
    heldout = np.asarray([[2.0, -1.0, 0.5], [-1.4, 0.3, 2.2]])
    np.testing.assert_allclose(
        apply_transform(heldout, recovered),
        apply_transform(heldout, expected),
        atol=1e-10,
    )


def test_symmetric_candidates_are_reported_as_ambiguous() -> None:
    identity = np.eye(4)
    rotated = np.eye(4)
    rotated[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    candidates = [
        {
            "candidate_id": "identity",
            "matrix_original_mesh_to_aligned_colmap": identity,
            "objective": 0.1,
            "finite": True,
            "hit_parameter_bound": False,
            "correspondence_collapsed": False,
        },
        {
            "candidate_id": "symmetric_rotation",
            "matrix_original_mesh_to_aligned_colmap": rotated,
            "objective": 0.1005,
            "finite": True,
            "hit_parameter_bound": False,
            "correspondence_collapsed": False,
        },
    ]
    assert ambiguous_candidate_ids(candidates, scene_diagonal=4.0) == ["symmetric_rotation"]


def test_local_deformation_residuals_are_structured() -> None:
    chunks = [
        {"aligned_median_residual": 0.08, "observation_count": 50},
        {"aligned_median_residual": 0.42, "observation_count": 50},
    ]
    assert residual_is_structured(chunks)


def test_working_and_colmap_frame_renders_are_equivalent() -> None:
    vertices = np.asarray(
        [
            [-1.0, -1.0, 3.0],
            [1.0, -1.0, 3.0],
            [1.0, 1.0, 3.0],
            [-1.0, 1.0, 3.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    angle = np.deg2rad(18.0)
    colmap_to_working = np.eye(4)
    colmap_to_working[:3, :3] = 1.4 * np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    colmap_to_working[:3, 3] = [0.4, -0.2, 0.7]
    working_vertices = apply_transform(vertices, colmap_to_working)
    pose = {
        "transform_world_from_camera": {
            "translation": [0.0, 0.0, 0.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    }
    intrinsics = {
        "width": 96,
        "height": 72,
        "fx": 70.0,
        "fy": 68.0,
        "cx": 45.0,
        "cy": 32.0,
    }

    depth_error, silhouette_iou = _render_equivalence(
        final_vertices=vertices,
        working_vertices=working_vertices,
        faces=faces,
        camera_pose=pose,
        intrinsics=intrinsics,
        colmap_to_working=colmap_to_working,
        face_chunk_size=100,
    )

    assert depth_error is not None
    assert depth_error < 1e-5
    assert silhouette_iou >= 0.999


def test_validation_evidence_cannot_change_candidate_selection() -> None:
    candidates = [
        {
            "candidate_id": "training_winner",
            "objective": 0.1,
            "finite": True,
            "hit_parameter_bound": False,
            "correspondence_collapsed": False,
            "validation_metrics": {"median": 10.0},
        },
        {
            "candidate_id": "validation_winner",
            "objective": 0.2,
            "finite": True,
            "hit_parameter_bound": False,
            "correspondence_collapsed": False,
            "validation_metrics": {"median": 0.0},
        },
    ]
    selected = select_candidate_by_training_objective(candidates)
    candidates[0]["validation_metrics"] = {"median": 1000.0}
    candidates[1]["validation_metrics"] = {"median": 0.0}

    assert selected["candidate_id"] == "training_winner"
    assert select_candidate_by_training_objective(candidates)["candidate_id"] == "training_winner"
