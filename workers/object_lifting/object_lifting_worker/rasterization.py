from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from object_lifting_worker.camera_projection import (
    camera_from_world,
    homogeneous_clip_coordinates,
    ndc_to_pixel,
    transform_world_point_to_camera,
)


@dataclass(frozen=True)
class RasterResult:
    face_ids: Any
    depth: Any
    valid: Any
    barycentric: Any
    world_points: Any
    candidate_face_ids: Any
    processed_face_count: int
    culled_face_count: int
    near_plane: float
    far_plane: float


def _clip_plane_distances(vertex: tuple[float, float, float, float]) -> tuple[float, ...]:
    x, y, z, w = vertex
    return (x + w, w - x, y + w, w - y, z + w, w - z, w)


def triangle_outside_clip(
    vertices: list[tuple[float, float, float, float]],
) -> bool:
    """Conservatively reject only triangles wholly outside one clip half-space."""
    if len(vertices) != 3:
        raise ValueError("clip-space triangles require exactly three vertices")
    distances = [_clip_plane_distances(vertex) for vertex in vertices]
    return any(all(item[plane] <= 0 for item in distances) for plane in range(7))


def _clip_polygon(
    vertices: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    polygon = vertices
    for plane in range(7):
        if not polygon:
            break
        output: list[tuple[float, float, float, float]] = []
        previous = polygon[-1]
        previous_distance = _clip_plane_distances(previous)[plane]
        for current in polygon:
            current_distance = _clip_plane_distances(current)[plane]
            previous_inside = previous_distance > 0
            current_inside = current_distance > 0
            if current_inside != previous_inside:
                denominator = previous_distance - current_distance
                if abs(denominator) > 1e-20:
                    amount = previous_distance / denominator
                    output.append(
                        tuple(
                            previous[index] + amount * (current[index] - previous[index])
                            for index in range(4)
                        )
                    )  # type: ignore[arg-type]
            if current_inside:
                output.append(current)
            previous = current
            previous_distance = current_distance
        polygon = output
    return polygon


def _clip_planes(
    camera_vertices: list[tuple[float, float, float]],
    scene_diagonal: float,
) -> tuple[float, float]:
    positive = sorted(vertex[2] for vertex in camera_vertices if vertex[2] > 0)
    if not positive:
        raise ValueError("global mesh lies entirely behind a registered camera")
    near = max(scene_diagonal * 1e-5, positive[0] * 0.25, 1e-8)
    far = max(near * 10.0, positive[-1] * 1.25)
    return near, far


def cpu_rasterize_face_ids(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    *,
    translation_world_from_camera: tuple[float, float, float],
    rotation_xyzw_world_from_camera: tuple[float, float, float, float],
    intrinsics: dict[str, float | int],
    near_plane: float | None = None,
    far_plane: float | None = None,
) -> list[list[int]]:
    """Deterministic clipped reference rasterizer for synthetic parity tests."""
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    face_buffer = [[-1 for _ in range(width)] for _ in range(height)]
    depth_buffer = [[float("inf") for _ in range(width)] for _ in range(height)]
    camera_vertices = [
        transform_world_point_to_camera(
            vertex,
            translation_world_from_camera,
            rotation_xyzw_world_from_camera,
        )
        for vertex in vertices
    ]
    minimum = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
    maximum = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
    diagonal = math.sqrt(sum((maximum[axis] - minimum[axis]) ** 2 for axis in range(3)))
    if near_plane is not None and far_plane is not None:
        near, far = near_plane, far_plane
    else:
        automatic_near, automatic_far = _clip_planes(camera_vertices, max(diagonal, 1e-8))
        near = near_plane if near_plane is not None else automatic_near
        far = far_plane if far_plane is not None else automatic_far
    clip_vertices = [
        homogeneous_clip_coordinates(
            vertex,
            fx=float(intrinsics["fx"]),
            fy=float(intrinsics["fy"]),
            cx=float(intrinsics["cx"]),
            cy=float(intrinsics["cy"]),
            width=width,
            height=height,
            near=near,
            far=far,
        )
        for vertex in camera_vertices
    ]
    for face_id, face in enumerate(faces):
        triangle = [clip_vertices[index] for index in face]
        if triangle_outside_clip(triangle):
            continue
        polygon = _clip_polygon(triangle)
        if len(polygon) < 3:
            continue
        for triangle_index in range(1, len(polygon) - 1):
            clipped = [polygon[0], polygon[triangle_index], polygon[triangle_index + 1]]
            projected = [
                ndc_to_pixel(vertex[0] / vertex[3], vertex[1] / vertex[3], width, height)
                for vertex in clipped
            ]
            a, b, c = projected
            minimum_x = max(0, math.floor(min(a[0], b[0], c[0])))
            maximum_x = min(width - 1, math.ceil(max(a[0], b[0], c[0])))
            minimum_y = max(0, math.floor(min(a[1], b[1], c[1])))
            maximum_y = min(height - 1, math.ceil(max(a[1], b[1], c[1])))
            denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(denominator) < 1e-12:
                continue
            for y in range(minimum_y, maximum_y + 1):
                for x in range(minimum_x, maximum_x + 1):
                    sample_x = float(x)
                    sample_y = float(y)
                    weight_a = (
                        (b[1] - c[1]) * (sample_x - c[0]) + (c[0] - b[0]) * (sample_y - c[1])
                    ) / denominator
                    weight_b = (
                        (c[1] - a[1]) * (sample_x - c[0]) + (a[0] - c[0]) * (sample_y - c[1])
                    ) / denominator
                    weight_c = 1.0 - weight_a - weight_b
                    if min(weight_a, weight_b, weight_c) < -1e-9:
                        continue
                    inverse_depth = sum(
                        weight / clipped[index][3]
                        for index, weight in enumerate((weight_a, weight_b, weight_c))
                    )
                    if inverse_depth <= 0:
                        continue
                    depth = 1.0 / inverse_depth
                    if depth < depth_buffer[y][x]:
                        depth_buffer[y][x] = depth
                        face_buffer[y][x] = face_id
    return face_buffer


class NvdiffrastRasterizer:
    def __init__(self, vertices: Any, faces: Any, *, face_chunk_size: int) -> None:
        import numpy as np
        import nvdiffrast.torch as dr
        import torch

        self.dr = dr
        self.np = np
        self.torch = torch
        self.vertices_cpu = np.asarray(vertices, dtype=np.float32)
        self.faces_cpu = np.asarray(faces, dtype=np.int64)
        if self.vertices_cpu.ndim != 2 or self.vertices_cpu.shape[1] != 3:
            raise ValueError("global mesh vertices must be an Nx3 array")
        if self.faces_cpu.ndim != 2 or self.faces_cpu.shape[1] != 3:
            raise ValueError("global mesh faces must be an Mx3 array")
        if not np.isfinite(self.vertices_cpu).all():
            raise ValueError("global mesh contains non-finite vertices")
        if self.faces_cpu.size and (
            self.faces_cpu.min() < 0 or self.faces_cpu.max() >= self.vertices_cpu.shape[0]
        ):
            raise ValueError("global mesh contains invalid face indices")
        self.face_chunk_size = face_chunk_size
        self.device = torch.device("cuda")
        self.vertices = torch.as_tensor(self.vertices_cpu, device=self.device, dtype=torch.float32)
        self.context = dr.RasterizeCudaContext(device=self.device)
        bounds = self.vertices_cpu.max(axis=0) - self.vertices_cpu.min(axis=0)
        self.scene_diagonal = float(np.linalg.norm(bounds))
        if not self.scene_diagonal > 0:
            raise ValueError("global mesh bounds are degenerate")

    def _camera_space_and_clip(
        self,
        pose: dict[str, Any],
        intrinsics: dict[str, Any],
    ) -> tuple[Any, Any, float, float]:
        torch = self.torch
        transform = pose["transform_world_from_camera"]
        rotation, translation = camera_from_world(
            transform["translation"],
            transform["rotation_xyzw"],
        )
        rotation_tensor = torch.tensor(rotation, dtype=torch.float32, device=self.device)
        translation_tensor = torch.tensor(translation, dtype=torch.float32, device=self.device)
        camera_vertices = self.vertices @ rotation_tensor.T + translation_tensor
        z = camera_vertices[:, 2]
        positive = z[z > 0]
        if positive.numel() == 0:
            raise ValueError("global mesh lies entirely behind a registered camera")
        near = max(
            self.scene_diagonal * 1e-5,
            float(torch.quantile(positive, 0.001).item()) * 0.25,
            1e-8,
        )
        far = max(
            near * 10.0,
            float(torch.quantile(positive, 0.999).item()) * 1.25,
        )
        width = int(intrinsics["width"])
        height = int(intrinsics["height"])
        x = camera_vertices[:, 0]
        y = camera_vertices[:, 1]
        x_clip = (2.0 * float(intrinsics["fx"]) / width) * x + (
            2.0 * (float(intrinsics["cx"]) + 0.5) / width - 1.0
        ) * z
        y_clip = (-2.0 * float(intrinsics["fy"]) / height) * y + (
            1.0 - 2.0 * (float(intrinsics["cy"]) + 0.5) / height
        ) * z
        z_clip = ((far + near) / (far - near)) * z - (2.0 * far * near / (far - near))
        clip = torch.stack((x_clip, y_clip, z_clip, z), dim=1)
        return camera_vertices, clip, near, far

    def _candidate_faces(self, clip: Any) -> Any:
        torch = self.torch
        candidate_chunks = []
        face_count = self.faces_cpu.shape[0]
        for start in range(0, face_count, self.face_chunk_size):
            stop = min(start + self.face_chunk_size, face_count)
            face_chunk = torch.as_tensor(
                self.faces_cpu[start:stop],
                dtype=torch.int64,
                device=self.device,
            )
            triangle = clip[face_chunk]
            x = triangle[:, :, 0]
            y = triangle[:, :, 1]
            z = triangle[:, :, 2]
            w = triangle[:, :, 3]
            outside = (
                (x < -w).all(dim=1)
                | (x > w).all(dim=1)
                | (y < -w).all(dim=1)
                | (y > w).all(dim=1)
                | (z < -w).all(dim=1)
                | (z > w).all(dim=1)
                | (w <= 0).all(dim=1)
            )
            local = torch.nonzero(~outside, as_tuple=False).flatten()
            if local.numel():
                candidate_chunks.append(local.cpu() + start)
        if not candidate_chunks:
            return self.np.empty((0,), dtype=self.np.int64)
        return torch.cat(candidate_chunks).numpy().astype(self.np.int64)

    def rasterize(
        self,
        pose: dict[str, Any],
        intrinsics: dict[str, Any],
    ) -> RasterResult:
        torch = self.torch
        np = self.np
        camera_vertices, clip, near, far = self._camera_space_and_clip(pose, intrinsics)
        candidates_cpu = self._candidate_faces(clip)
        width = int(intrinsics["width"])
        height = int(intrinsics["height"])
        face_count = self.faces_cpu.shape[0]
        if not len(candidates_cpu):
            return RasterResult(
                face_ids=np.full((height, width), -1, dtype=np.int64),
                depth=np.full((height, width), np.inf, dtype=np.float32),
                valid=np.zeros((height, width), dtype=bool),
                barycentric=np.zeros((height, width, 3), dtype=np.float32),
                world_points=np.full((height, width, 3), np.nan, dtype=np.float32),
                candidate_face_ids=candidates_cpu,
                processed_face_count=0,
                culled_face_count=face_count,
                near_plane=near,
                far_plane=far,
            )
        triangles = torch.as_tensor(
            self.faces_cpu[candidates_cpu],
            dtype=torch.int32,
            device=self.device,
        )
        raster, _ = self.dr.rasterize(
            self.context,
            clip.unsqueeze(0),
            triangles,
            resolution=[height, width],
        )
        local_ids = raster[0, :, :, 3].to(torch.int64) - 1
        valid = local_ids >= 0
        candidate_tensor = torch.as_tensor(candidates_cpu, dtype=torch.int64, device=self.device)
        global_ids = torch.full_like(local_ids, -1)
        global_ids[valid] = candidate_tensor[local_ids[valid]]
        interpolated_depth, _ = self.dr.interpolate(
            camera_vertices[:, 2].reshape(1, -1, 1).contiguous(),
            raster.contiguous(),
            triangles.contiguous(),
        )
        depth = interpolated_depth[0, :, :, 0]
        depth[~valid] = torch.inf
        world_points, _ = self.dr.interpolate(
            self.vertices.reshape(1, -1, 3).contiguous(),
            raster.contiguous(),
            triangles.contiguous(),
        )
        world_points = world_points[0]
        world_points[~valid] = torch.nan
        barycentric = torch.stack(
            (
                raster[0, :, :, 0],
                raster[0, :, :, 1],
                1.0 - raster[0, :, :, 0] - raster[0, :, :, 1],
            ),
            dim=2,
        )
        barycentric[~valid] = 0
        return RasterResult(
            face_ids=global_ids.cpu().numpy(),
            depth=depth.cpu().numpy(),
            valid=valid.cpu().numpy(),
            barycentric=barycentric.cpu().numpy(),
            world_points=world_points.cpu().numpy(),
            candidate_face_ids=candidates_cpu,
            processed_face_count=int(len(candidates_cpu)),
            culled_face_count=int(face_count - len(candidates_cpu)),
            near_plane=near,
            far_plane=far,
        )

    def rasterize_face_subset(
        self,
        pose: dict[str, Any],
        intrinsics: dict[str, Any],
        face_ids: Any,
    ) -> Any:
        np = self.np
        face_ids = np.asarray(face_ids, dtype=np.int64)
        if face_ids.size == 0:
            return np.zeros(
                (int(intrinsics["height"]), int(intrinsics["width"])),
                dtype=bool,
            )
        _camera_vertices, clip, _near, _far = self._camera_space_and_clip(pose, intrinsics)
        triangles = self.torch.as_tensor(
            self.faces_cpu[face_ids],
            dtype=self.torch.int32,
            device=self.device,
        )
        raster, _ = self.dr.rasterize(
            self.context,
            clip.unsqueeze(0),
            triangles,
            resolution=[int(intrinsics["height"]), int(intrinsics["width"])],
        )
        return (raster[0, :, :, 3] > 0).cpu().numpy()
