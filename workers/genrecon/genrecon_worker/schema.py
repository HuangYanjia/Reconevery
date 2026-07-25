from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class WorkerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkerConfiguration(WorkerModel):
    official_repository: str
    official_code_commit: str
    official_checkout_path: str
    submodule_commits: dict[str, str]
    checkpoint_paths: dict[str, str]
    checkpoint_hashes: dict[str, str]
    device: Literal["cuda"]
    precision: Literal["float16"]


class InferenceRequest(WorkerModel):
    schema_version: str
    run_id: str
    official_repository: str
    official_code_commit: str
    official_checkout_path: str
    checkpoint_paths: dict[str, str]
    checkpoint_hashes: dict[str, str]
    checkpoint_manifest_path: str
    manifest_path: str
    manifest_sha256: str
    frame_sequence_digest: str
    camera_reconstruction_path: str
    camera_reconstruction_sha256: str
    camera_package_manifest_path: str
    camera_package_sha256: str
    master_frame_order: list[str]
    normalized_frame_paths: dict[str, str]
    normalized_frame_hashes: dict[str, str]
    registered_frame_ids: list[str]
    unregistered_frame_ids: list[str]
    eligible_frame_ids: list[str]
    requested_max_views: int = Field(gt=0)
    coordinate_convention: dict[str, Any]
    working_transform_strategy: Literal["identity", "pca_scene_axes"]
    pipeline_config: str
    reconstruction_parameters: dict[str, Any]
    output_directory: str
    seed: int
