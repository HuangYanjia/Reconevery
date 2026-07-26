from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from object_lifting_worker.camera_projection import (
    camera_from_world,
    project_pinhole,
    transform_world_point_to_camera,
)


@dataclass(frozen=True)
class RasterResult:
    face_ids: Any
    depth: Any
    valid: Any
    candidate_face_ids: Any
    processed_face_count: int
    culled_face_count: int
    near_plane: float
    far_plane: float


def cpu_rasterize_face_ids(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    *,
    translation_world_from_camera: tuple[float, float, float],
    rotation_xyzw_world_from_camera: tuple[float, float, float, float],
    intrinsics: dict[str, float | int],
) -> list[list[int]]:
    """Small deterministic reference rasterizer used only by synthetic tests."""
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
    projected = [
        project_pinhole(
            vertex,
            fx=float(intrinsics["fx"]),
            fy=float(intrinsics["fy"]),
            cx=float(intrinsics["cx"]),
            cy=float(intrinsics["cy"]),
        )
        for vertex in camera_vertices
    ]
    for face_id, face in enumerate(faces):
        if any(camera_vertices[index][2] <= 0 for index in face):
            continue
        points = [projected[index] for index in face]
        if any(point is None for point in points):
            continue
        a, b, c = points  # type: ignore[misc]
        minimum_x = max(0, int(min(a[0], b[0], c[0])))
        maximum_x = min(width - 1, int(max(a[0], b[0], c[0])) + 1)
        minimum_y = max(0, int(min(a[1], b[1], c[1])))
        maximum_y = min(height - 1, int(max(a[1], b[1], c[1])) + 1)
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-12:
            continue
        for y in range(minimum_y, maximum_y + 1):
            for x in range(minimum_x, maximum_x + 1):
                sample_x = x + 0.5
                sample_y = y + 0.5
                weight_a = (
                    (b[1] - c[1]) * (sample_x - c[0]) + (c[0] - b[0]) * (sample_y - c[1])
                ) / denominator
                weight_b = (
                    (c[1] - a[1]) * (sample_x - c[0]) + (a[0] - c[0]) * (sample_y - c[1])
                ) / denominator
                weight_c = 1.0 - weight_a - weight_b
                if min(weight_a, weight_b, weight_c) < -1e-9:
                    continue
                inverse_depth = (
                    weight_a / camera_vertices[face[0]][2]
                    + weight_b / camera_vertices[face[1]][2]
                    + weight_c / camera_vertices[face[2]][2]
                )
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

    def rasterize(
        self,
        pose: dict[str, Any],
        intrinsics: dict[str, Any],
    ) -> RasterResult:
        torch = self.torch
        np = self.np
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
        )
        far = max(
            near * 10.0,
            float(torch.quantile(positive, 0.999).item()) * 1.25,
        )
        safe_z = torch.clamp(z, min=near * 0.01)
        u = float(intrinsics["fx"]) * camera_vertices[:, 0] / safe_z + float(intrinsics["cx"])
        v = float(intrinsics["fy"]) * camera_vertices[:, 1] / safe_z + float(intrinsics["cy"])
        width = int(intrinsics["width"])
        height = int(intrinsics["height"])
        x_ndc = 2.0 * (u + 0.5) / width - 1.0
        y_ndc = 1.0 - 2.0 * (v + 0.5) / height
        a = (far + near) / (far - near)
        b = -2.0 * far * near / (far - near)
        clip = torch.stack(
            (x_ndc * safe_z, y_ndc * safe_z, a * safe_z + b, safe_z),
            dim=1,
        )
        candidate_chunks = []
        face_count = self.faces_cpu.shape[0]
        for start in range(0, face_count, self.face_chunk_size):
            stop = min(start + self.face_chunk_size, face_count)
            face_chunk = torch.as_tensor(
                self.faces_cpu[start:stop],
                dtype=torch.int64,
                device=self.device,
            )
            triangle_z = z[face_chunk]
            triangle_x = x_ndc[face_chunk]
            triangle_y = y_ndc[face_chunk]
            visible = (
                (triangle_z.max(dim=1).values >= near)
                & (triangle_z.min(dim=1).values <= far)
                & (triangle_x.max(dim=1).values >= -1.0)
                & (triangle_x.min(dim=1).values <= 1.0)
                & (triangle_y.max(dim=1).values >= -1.0)
                & (triangle_y.min(dim=1).values <= 1.0)
            )
            local = torch.nonzero(visible, as_tuple=False).flatten()
            if local.numel():
                candidate_chunks.append(local.cpu() + start)
        if not candidate_chunks:
            empty = np.full((height, width), -1, dtype=np.int64)
            return RasterResult(
                face_ids=empty,
                depth=np.full((height, width), np.inf, dtype=np.float32),
                valid=np.zeros((height, width), dtype=bool),
                candidate_face_ids=np.empty((0,), dtype=np.int64),
                processed_face_count=0,
                culled_face_count=face_count,
                near_plane=near,
                far_plane=far,
            )
        candidates_cpu = torch.cat(candidate_chunks).numpy().astype(np.int64)
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
            z.reshape(1, -1, 1).contiguous(),
            raster.contiguous(),
            triangles.contiguous(),
        )
        depth = interpolated_depth[0, :, :, 0]
        depth[~valid] = torch.inf
        return RasterResult(
            face_ids=global_ids.cpu().numpy(),
            depth=depth.cpu().numpy(),
            valid=valid.cpu().numpy(),
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
        torch = self.torch
        face_ids = np.asarray(face_ids, dtype=np.int64)
        if face_ids.size == 0:
            return np.zeros(
                (int(intrinsics["height"]), int(intrinsics["width"])),
                dtype=bool,
            )
        transform = pose["transform_world_from_camera"]
        rotation, translation = camera_from_world(
            transform["translation"],
            transform["rotation_xyzw"],
        )
        rotation_tensor = torch.tensor(rotation, dtype=torch.float32, device=self.device)
        translation_tensor = torch.tensor(translation, dtype=torch.float32, device=self.device)
        camera_vertices = self.vertices @ rotation_tensor.T + translation_tensor
        z = camera_vertices[:, 2]
        near = max(self.scene_diagonal * 1e-5, 1e-6)
        far = max(near * 10, float(torch.max(z).item()) * 1.25)
        safe_z = torch.clamp(z, min=near * 0.01)
        width = int(intrinsics["width"])
        height = int(intrinsics["height"])
        u = float(intrinsics["fx"]) * camera_vertices[:, 0] / safe_z + float(intrinsics["cx"])
        v = float(intrinsics["fy"]) * camera_vertices[:, 1] / safe_z + float(intrinsics["cy"])
        x_ndc = 2.0 * (u + 0.5) / width - 1.0
        y_ndc = 1.0 - 2.0 * (v + 0.5) / height
        a = (far + near) / (far - near)
        b = -2.0 * far * near / (far - near)
        clip = torch.stack(
            (x_ndc * safe_z, y_ndc * safe_z, a * safe_z + b, safe_z),
            dim=1,
        )
        triangles = torch.as_tensor(
            self.faces_cpu[face_ids],
            dtype=torch.int32,
            device=self.device,
        )
        raster, _ = self.dr.rasterize(
            self.context,
            clip.unsqueeze(0),
            triangles,
            resolution=[height, width],
        )
        return (raster[0, :, :, 3] > 0).cpu().numpy()
