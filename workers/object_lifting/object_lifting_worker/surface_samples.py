from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SurfaceSampleCell:
    supporting_frames: set[int] = field(default_factory=set)
    positive_weight: float = 0.0
    negative_weight: float = 0.0
    weighted_point_sum: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    minimum: list[float] = field(default_factory=lambda: [float("inf")] * 3)
    maximum: list[float] = field(default_factory=lambda: [float("-inf")] * 3)
    depth_sum: float = 0.0
    barycentric_sum: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    member_face_weights: dict[int, float] = field(default_factory=dict)
    support_score: float = 0.0

    @property
    def centroid(self) -> tuple[float, float, float]:
        denominator = max(self.positive_weight, 1e-12)
        return tuple(value / denominator for value in self.weighted_point_sum)


@dataclass(frozen=True)
class SampleFaceSupport:
    direct_sample_support: float
    patch_support: float
    propagated_support: float
    supporting_views: int


@dataclass(frozen=True)
class SurfaceSampleFusionResult:
    accepted_faces: list[int]
    ambiguous_faces: list[int]
    cell_count: int
    accepted_cell_count: int
    ambiguous_cell_count: int
    face_support: dict[int, SampleFaceSupport]
    cell_centroids: list[tuple[float, float, float]]


class SurfaceSampleFusion:
    """Streaming deterministic voxel fusion of visible positive surface samples."""

    def __init__(
        self,
        *,
        origin: Any,
        voxel_edge: float,
        core_weight: float,
        boundary_weight: float,
    ) -> None:
        import numpy as np

        if voxel_edge <= 0:
            raise ValueError("surface-sample voxel edge must be positive")
        self.np = np
        self.origin = np.asarray(origin, dtype=np.float64)
        self.voxel_edge = float(voxel_edge)
        self.core_weight = float(core_weight)
        self.boundary_weight = float(boundary_weight)
        self.cells: dict[tuple[int, int, int], SurfaceSampleCell] = {}

    def accumulate(
        self,
        *,
        frame_index: int,
        face_ids: Any,
        world_points: Any,
        barycentric: Any,
        depth: Any,
        core: Any,
        boundary: Any,
        frame_score: float,
    ) -> None:
        np = self.np
        positive = (face_ids >= 0) & (core | boundary)
        rows, columns = np.nonzero(positive)
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            region_weight = self.core_weight if bool(core[row, column]) else self.boundary_weight
            weight = region_weight * frame_score
            if weight <= 0:
                continue
            point = np.asarray(world_points[row, column], dtype=np.float64)
            if not np.isfinite(point).all():
                continue
            key = tuple(np.floor((point - self.origin) / self.voxel_edge).astype(np.int64))
            cell = self.cells.setdefault(key, SurfaceSampleCell())
            cell.supporting_frames.add(frame_index)
            cell.positive_weight += weight
            for axis in range(3):
                value = float(point[axis])
                cell.weighted_point_sum[axis] += weight * value
                cell.minimum[axis] = min(cell.minimum[axis], value)
                cell.maximum[axis] = max(cell.maximum[axis], value)
                cell.barycentric_sum[axis] += weight * float(barycentric[row, column, axis])
            cell.depth_sum += weight * float(depth[row, column])
            face_id = int(face_ids[row, column])
            cell.member_face_weights[face_id] = cell.member_face_weights.get(face_id, 0.0) + weight

    def accumulate_negative(
        self,
        *,
        face_ids: Any,
        world_points: Any,
        exterior: Any,
        frame_score: float,
        negative_weight: float,
    ) -> None:
        """Accumulate exterior evidence only into cells established by positive samples."""
        np = self.np
        negative = (face_ids >= 0) & exterior
        points = np.asarray(world_points[negative], dtype=np.float64)
        if not len(points):
            return
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            return
        keys = np.floor((points - self.origin) / self.voxel_edge).astype(np.int64)
        unique_keys, counts = np.unique(keys, axis=0, return_counts=True)
        weight = frame_score * negative_weight
        for key, count in zip(unique_keys.tolist(), counts.tolist(), strict=True):
            cell = self.cells.get(tuple(key))
            if cell is not None:
                cell.negative_weight += float(count) * weight

    def finalize(
        self,
        *,
        min_supporting_views: int,
        min_positive_weight: float,
        accepted_score: float,
        ambiguous_score: float,
    ) -> SurfaceSampleFusionResult:
        face_to_cells: dict[int, list[SurfaceSampleCell]] = {}
        for cell in self.cells.values():
            for face_id in cell.member_face_weights:
                face_to_cells.setdefault(face_id, []).append(cell)
        accepted_cells: list[SurfaceSampleCell] = []
        ambiguous_cells: list[SurfaceSampleCell] = []
        for key in sorted(self.cells):
            cell = self.cells[key]
            cell.support_score = cell.positive_weight / (
                cell.positive_weight + cell.negative_weight + 1e-12
            )
            eligible = (
                len(cell.supporting_frames) >= min_supporting_views
                and cell.positive_weight >= min_positive_weight
            )
            if eligible and cell.support_score >= accepted_score:
                accepted_cells.append(cell)
            elif (
                cell.positive_weight >= min_positive_weight
                and cell.support_score >= ambiguous_score
            ):
                ambiguous_cells.append(cell)
        accepted_faces = sorted(
            {face_id for cell in accepted_cells for face_id in cell.member_face_weights}
        )
        ambiguous_faces = sorted(
            {face_id for cell in ambiguous_cells for face_id in cell.member_face_weights}
            - set(accepted_faces)
        )
        face_support: dict[int, SampleFaceSupport] = {}
        for face_id, cells in face_to_cells.items():
            direct = sum(cell.member_face_weights[face_id] for cell in cells)
            face_support[face_id] = SampleFaceSupport(
                direct_sample_support=direct,
                patch_support=max(cell.support_score for cell in cells),
                propagated_support=0.0,
                supporting_views=len(set().union(*(cell.supporting_frames for cell in cells))),
            )
        return SurfaceSampleFusionResult(
            accepted_faces=accepted_faces,
            ambiguous_faces=ambiguous_faces,
            cell_count=len(self.cells),
            accepted_cell_count=len(accepted_cells),
            ambiguous_cell_count=len(ambiguous_cells),
            face_support=face_support,
            cell_centroids=[cell.centroid for cell in accepted_cells],
        )
