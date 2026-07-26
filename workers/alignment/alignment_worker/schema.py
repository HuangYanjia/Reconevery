from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ValueError("alignment worker paths must be safe run-relative paths")
    return value


class WorkerConfig(StrictModel):
    worker_version: str
    device: Literal["cuda"]
    backend: Literal["nvdiffrast_scipy"]


class AlignmentRequest(StrictModel):
    schema_version: Literal["0.1.0"]
    run_id: str
    manifest_path: str
    manifest_sha256: str
    frame_sequence_digest: str
    camera_reconstruction_path: str
    camera_reconstruction_sha256: str
    registered_frame_ids: list[str]
    unregistered_frame_ids: list[str]
    coordinate_convention: dict[str, Any]
    camera_package_manifest_path: str
    camera_package_sha256: str
    cameras_txt_path: str
    cameras_txt_sha256: str
    images_txt_path: str
    images_txt_sha256: str
    points3d_txt_path: str
    points3d_txt_sha256: str
    global_reconstruction_path: str
    global_reconstruction_sha256: str
    global_mesh_path: str
    global_mesh_sha256: str
    global_worker_manifest_path: str
    global_worker_manifest_sha256: str
    working_transform_path: str
    working_transform_sha256: str
    chunk_transforms_path: str
    chunk_transforms_sha256: str
    genrecon_camera_debug_path: str
    genrecon_camera_debug_sha256: str
    working_mesh_path: str | None = None
    working_mesh_sha256: str | None = None
    working_scene_path: str | None = None
    working_scene_sha256: str | None = None
    audit_configuration: dict[str, Any]
    sparse_observation_configuration: dict[str, Any]
    mesh_sampling_configuration: dict[str, Any]
    optimization_configuration: dict[str, Any]
    acceptance_configuration: dict[str, Any]
    output_directory: str
    seed: int

    @field_validator(
        "manifest_path",
        "camera_reconstruction_path",
        "camera_package_manifest_path",
        "cameras_txt_path",
        "images_txt_path",
        "points3d_txt_path",
        "global_reconstruction_path",
        "global_mesh_path",
        "global_worker_manifest_path",
        "working_transform_path",
        "chunk_transforms_path",
        "genrecon_camera_debug_path",
        "working_mesh_path",
        "working_scene_path",
        "output_directory",
    )
    @classmethod
    def safe_paths(cls, value: str | None) -> str | None:
        return relative_path(value) if value is not None else None

    @model_validator(mode="after")
    def raw_colmap_semantics(self) -> AlignmentRequest:
        expected = {
            "world_frame": "colmap_arbitrary",
            "alignment_status": "unoriented",
            "camera_axes": "x_right_y_down_z_forward",
            "linear_units": "arbitrary_units",
            "scale_status": "scale_ambiguous",
            "transform_direction": "world_from_camera",
        }
        mismatches = {
            key: self.coordinate_convention.get(key)
            for key, expected_value in expected.items()
            if self.coordinate_convention.get(key) != expected_value
        }
        if mismatches:
            raise ValueError(f"alignment requires raw COLMAP semantics: {mismatches}")
        if set(self.registered_frame_ids) & set(self.unregistered_frame_ids):
            raise ValueError("registered and unregistered frames must not overlap")
        return self


def load_request(path: Path) -> AlignmentRequest:
    return AlignmentRequest.model_validate_json(path.read_text(encoding="utf-8"))
