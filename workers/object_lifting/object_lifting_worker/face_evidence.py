from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class FaceStatistics:
    visible_pixels: float = 0.0
    core_positive_pixels: float = 0.0
    boundary_positive_pixels: float = 0.0
    exterior_negative_pixels: float = 0.0
    positive_views: int = 0
    negative_views: int = 0
    first_frame_index: int = 2**31 - 1
    last_frame_index: int = -1
    depth_sum: float = 0.0
    depth_pixels: int = 0
    support_score: float = 0.0


def _counts(values: Any) -> tuple[Any, Any]:
    import numpy as np

    if values.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    return np.unique(values.astype(np.int64), return_counts=True)


def accumulate_positive(
    stats: dict[int, FaceStatistics],
    *,
    face_ids: Any,
    depth: Any,
    core: Any,
    boundary: Any,
    frame_index: int,
    frame_score: float,
) -> None:
    valid = face_ids >= 0
    per_frame_faces: set[int] = set()
    for region, attribute in (
        (core, "core_positive_pixels"),
        (boundary, "boundary_positive_pixels"),
    ):
        identifiers, counts = _counts(face_ids[valid & region])
        for face_id, count in zip(identifiers.tolist(), counts.tolist(), strict=True):
            item = stats.setdefault(int(face_id), FaceStatistics())
            setattr(item, attribute, getattr(item, attribute) + count * frame_score)
            item.first_frame_index = min(item.first_frame_index, frame_index)
            item.last_frame_index = max(item.last_frame_index, frame_index)
            per_frame_faces.add(int(face_id))
    for face_id in per_frame_faces:
        stats[face_id].positive_views += 1
        pixels = valid & (face_ids == face_id) & (core | boundary)
        if pixels.any():
            stats[face_id].depth_sum += float(depth[pixels].sum())
            stats[face_id].depth_pixels += int(pixels.sum())


def accumulate_visibility_and_negative(
    stats: dict[int, FaceStatistics],
    *,
    face_ids: Any,
    exterior: Any,
    frame_score: float,
) -> None:
    import numpy as np

    if not stats:
        return
    candidates = np.fromiter(stats, dtype=np.int64)
    candidate_pixels = np.isin(face_ids, candidates)
    visible_ids, visible_counts = _counts(face_ids[candidate_pixels])
    for face_id, count in zip(visible_ids.tolist(), visible_counts.tolist(), strict=True):
        stats[int(face_id)].visible_pixels += float(count)
    negative_ids, negative_counts = _counts(face_ids[candidate_pixels & exterior])
    for face_id, count in zip(negative_ids.tolist(), negative_counts.tolist(), strict=True):
        item = stats[int(face_id)]
        item.exterior_negative_pixels += count * frame_score
        item.negative_views += 1


def score_faces(
    stats: dict[int, FaceStatistics],
    *,
    configuration: dict[str, Any],
) -> tuple[list[int], list[int]]:
    core_weight = float(configuration["core_positive_weight"])
    boundary_weight = float(configuration["boundary_positive_weight"])
    negative_weight = float(configuration["exterior_negative_weight"])
    min_visible = int(configuration["min_visible_pixels_per_face"])
    min_positive = int(configuration["min_positive_pixels_per_face"])
    min_views = int(configuration["min_supporting_views"])
    accepted_score = float(configuration["accepted_face_score"])
    ambiguous_score = float(configuration["ambiguous_face_score"])
    accepted: list[int] = []
    ambiguous: list[int] = []
    for face_id, item in stats.items():
        positive = (
            core_weight * item.core_positive_pixels
            + boundary_weight * item.boundary_positive_pixels
        )
        negative = negative_weight * item.exterior_negative_pixels
        item.support_score = positive / (positive + negative + 1e-12)
        eligible = (
            item.visible_pixels >= min_visible
            and positive >= min_positive
            and item.positive_views >= min_views
        )
        if eligible and item.support_score >= accepted_score:
            accepted.append(face_id)
        elif positive >= min_positive and item.support_score >= ambiguous_score:
            ambiguous.append(face_id)
    return sorted(accepted), sorted(ambiguous)


def write_evidence_npz(
    path: Path,
    stats: dict[int, FaceStatistics],
    *,
    sample_face_support: dict[int, Any] | None = None,
) -> list[dict[str, object]]:
    import numpy as np

    all_face_ids = set(stats)
    if sample_face_support:
        all_face_ids.update(sample_face_support)
    face_ids = np.asarray(sorted(all_face_ids), dtype=np.uint64)
    ordered = [stats.get(int(face_id), FaceStatistics()) for face_id in face_ids]
    arrays = {
        "global_face_ids": face_ids,
        "visible_pixel_count": np.asarray(
            [item.visible_pixels for item in ordered], dtype=np.float64
        ),
        "core_positive_pixel_count": np.asarray(
            [item.core_positive_pixels for item in ordered], dtype=np.float64
        ),
        "boundary_positive_pixel_count": np.asarray(
            [item.boundary_positive_pixels for item in ordered], dtype=np.float64
        ),
        "exterior_negative_pixel_count": np.asarray(
            [item.exterior_negative_pixels for item in ordered], dtype=np.float64
        ),
        "positive_view_count": np.asarray(
            [item.positive_views for item in ordered], dtype=np.uint32
        ),
        "negative_view_count": np.asarray(
            [item.negative_views for item in ordered], dtype=np.uint32
        ),
        "first_supporting_frame_index": np.asarray(
            [item.first_frame_index for item in ordered], dtype=np.int32
        ),
        "last_supporting_frame_index": np.asarray(
            [item.last_frame_index for item in ordered], dtype=np.int32
        ),
        "mean_depth": np.asarray(
            [item.depth_sum / item.depth_pixels if item.depth_pixels else 0.0 for item in ordered],
            dtype=np.float64,
        ),
        "support_score": np.asarray([item.support_score for item in ordered], dtype=np.float64),
    }
    if sample_face_support is not None:
        arrays.update(
            {
                "direct_sample_support": np.asarray(
                    [
                        getattr(sample_face_support.get(int(face_id)), "direct_sample_support", 0.0)
                        for face_id in face_ids
                    ],
                    dtype=np.float64,
                ),
                "patch_support": np.asarray(
                    [
                        getattr(sample_face_support.get(int(face_id)), "patch_support", 0.0)
                        for face_id in face_ids
                    ],
                    dtype=np.float64,
                ),
                "propagated_support": np.asarray(
                    [
                        getattr(sample_face_support.get(int(face_id)), "propagated_support", 0.0)
                        for face_id in face_ids
                    ],
                    dtype=np.float64,
                ),
                "sample_supporting_view_count": np.asarray(
                    [
                        getattr(sample_face_support.get(int(face_id)), "supporting_views", 0)
                        for face_id in face_ids
                    ],
                    dtype=np.uint32,
                ),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    records = []
    for name, array in arrays.items():
        records.append(
            {
                "name": name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "content_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        )
    return records
