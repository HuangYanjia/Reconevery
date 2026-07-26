from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ValueError("worker artifact paths must be relative and safe")
    return value


class WorkerConfig(StrictModel):
    worker_version: str
    device: Literal["cuda"]
    backend: Literal["nvdiffrast"]


class WorkerTrack(StrictModel):
    object_id: str
    semantic_label: str
    prompt_id: str
    asset_type_hint: str | None = None
    track_coverage: float = Field(ge=0, le=1)
    mask_paths_by_frame: dict[str, str]
    frame_scores: dict[str, float]

    @field_validator("mask_paths_by_frame")
    @classmethod
    def safe_masks(cls, values: dict[str, str]) -> dict[str, str]:
        return {frame_id: relative_path(path) for frame_id, path in values.items()}


class WorkerRequest(StrictModel):
    schema_version: Literal["0.1.0"]
    run_id: str
    manifest_path: str
    manifest_sha256: str
    frame_sequence_digest: str
    master_frame_order: list[str]
    normalized_frame_paths: dict[str, str]
    normalized_frame_hashes: dict[str, str]
    camera_reconstruction_path: str
    camera_reconstruction_sha256: str
    camera_package_manifest_path: str
    camera_package_manifest_sha256: str
    camera_package_images_path: str
    camera_package_images_sha256: str
    camera_package_points3d_path: str
    camera_package_points3d_sha256: str
    camera_package_registered_frames_path: str
    camera_package_registered_frames_sha256: str
    registered_frame_ids: list[str]
    unregistered_frame_ids: list[str]
    coordinate_convention: dict[str, Any]
    segmentation_tracking_path: str
    segmentation_tracking_sha256: str
    object_tracks: list[WorkerTrack]
    global_reconstruction_path: str
    global_reconstruction_sha256: str
    global_mesh_path: str
    global_mesh_sha256: str
    alignment_policy: Literal["none", "use_if_accepted", "require_accepted"] = "none"
    alignment_path: str | None = None
    alignment_sha256: str | None = None
    alignment_status: str | None = None
    alignment_accepted: bool = False
    matrix_original_mesh_to_aligned_colmap: list[list[float]] | None = None
    lifting_method: Literal["exact_face_vote_v1", "surface_sample_fusion_v2"]
    rasterization_configuration: dict[str, Any]
    mask_processing_configuration: dict[str, Any]
    face_evidence_configuration: dict[str, Any]
    surface_sample_configuration: dict[str, Any]
    surface_extraction_configuration: dict[str, Any]
    output_directory: str
    seed: int

    @field_validator(
        "manifest_path",
        "camera_reconstruction_path",
        "camera_package_manifest_path",
        "camera_package_images_path",
        "camera_package_points3d_path",
        "camera_package_registered_frames_path",
        "segmentation_tracking_path",
        "global_reconstruction_path",
        "global_mesh_path",
        "alignment_path",
        "output_directory",
    )
    @classmethod
    def safe_request_paths(cls, value: str | None) -> str | None:
        return relative_path(value) if value is not None else None

    @model_validator(mode="after")
    def raw_colmap_only(self) -> WorkerRequest:
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
            for key, value in expected.items()
            if self.coordinate_convention.get(key) != value
        }
        if mismatches:
            raise ValueError(f"worker requires raw COLMAP coordinate semantics: {mismatches}")
        if self.alignment_policy == "none":
            if (
                self.alignment_path is not None
                or self.alignment_sha256 is not None
                or self.alignment_accepted
                or self.matrix_original_mesh_to_aligned_colmap is not None
            ):
                raise ValueError("alignment_policy=none cannot apply alignment")
        else:
            if self.alignment_path is None or self.alignment_sha256 is None:
                raise ValueError("alignment-aware lifting requires alignment artifact")
            if self.alignment_accepted != (self.matrix_original_mesh_to_aligned_colmap is not None):
                raise ValueError("accepted alignment must carry exactly one matrix")
            if self.alignment_policy == "require_accepted" and not self.alignment_accepted:
                raise ValueError("require_accepted requires accepted alignment")
        return self


def load_request(path: Path) -> WorkerRequest:
    return WorkerRequest.model_validate_json(path.read_text(encoding="utf-8"))
