from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class WorkerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkerConfiguration(WorkerModel):
    official_repository: str
    official_code_commit: str
    checkpoint_repository: str
    checkpoint_revision: str
    checkpoint_access_mode: Literal["authenticated_remote", "local_path", "offline_cache", "fake"]
    local_checkpoint_path: str | None
    model_cache_path: str | None
    offline: bool
    model_mode: Literal["sam3", "sam3.1"]
    device: Literal["cpu", "cuda"]
    precision: Literal["float32", "float16", "bfloat16"]
    seed: int
    fake_mode: str


class PromptPoint(WorkerModel):
    x: float
    y: float
    label: Literal[0, 1]


class WorkerPrompt(WorkerModel):
    prompt_id: str
    label: str
    prompt_type: Literal["text", "box", "point", "mask"]
    text: str | None
    frame_id: str | None
    box_xyxy: tuple[float, float, float, float] | None
    points: list[PromptPoint] | None
    mask_path: str | None
    asset_type_hint: str | None
    confidence_threshold: float | None
    positive: bool
    synonym_group: str | None
    instance_limit: int | None
    notes: str | None
    enabled: bool


class PromptManifest(WorkerModel):
    schema_version: Literal["0.1.0"]
    prompts: list[WorkerPrompt]
    source_path: str | None
    source_sha256: str | None
    input_hashes: dict[str, str]


class Anchor(WorkerModel):
    frame_id: str
    score: float
    camera_pose_available: bool
    selection_reason: str


class InferenceRequest(WorkerModel):
    schema_version: Literal["0.1.0"]
    run_id: str
    frame_manifest_path: str
    frame_manifest_sha256: str
    frame_order: list[str]
    frame_paths: list[str]
    frame_dimensions: dict[str, tuple[int, int]]
    camera_reconstruction_path: str
    camera_reconstruction_sha256: str
    registered_frame_ids: list[str]
    unregistered_frame_ids: list[str]
    prompt_manifest: PromptManifest
    prompt_manifest_sha256: str
    anchor_frames: list[Anchor]
    strategy: Literal["detect_then_track"]
    tracking_direction: Literal["forward", "backward", "forward_backward"]
    model_configuration: dict[str, Any]
    postprocessing_configuration: dict[str, Any]
    output_directory: str
    seed: int


class RawObservation(WorkerModel):
    frame_id: str
    raw_model_object_id: str
    prompt_id: str
    semantic_label: str
    score: float
    mask_path: str
    mask_encoding: Literal["binary_png"] = "binary_png"
    model_box_xyxy: tuple[float, float, float, float] | None
    occluded: bool | None = None


class RawTrack(WorkerModel):
    raw_model_object_id: str
    prompt_id: str
    semantic_label: str
    observations: list[RawObservation] = Field(default_factory=list)
