from __future__ import annotations

import math
import re
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from recon2sim.ir import (
    AssetType,
    CameraIntrinsics,
    CameraPose,
    ConfidenceRecord,
    CoordinateConvention,
    PhysicsProperties,
    ProvenanceRecord,
    ScaleStatus,
    StrictModel,
)


def _relative_artifact_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ValueError("artifact paths must be relative to the run directory")
    return value


class InputSourceType(StrEnum):
    GENERATED_TEST_IMAGE = "generated_test_image"
    IMAGE_DIRECTORY = "image_directory"
    VIDEO = "video"
    MOCK = "mock"


class FrameManifestEntry(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    relative_path: Annotated[str, Field(min_length=1)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    timestamp_s: float = Field(ge=0)
    source_type: InputSourceType
    source_file: str | None = None
    original_frame_index: int | None = Field(default=None, ge=0)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _relative_artifact_path(value)


class IngestManifest(StrictModel):
    source_type: InputSourceType
    frames: Annotated[list[FrameManifestEntry], Field(min_length=1)]
    source_input_path: str | None = None
    source_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    ffmpeg_version: str | None = None
    ffprobe_version: str | None = None
    extraction_config: dict[str, object] = Field(default_factory=dict)
    total_decoded_frames: int | None = Field(default=None, ge=0)
    selected_frames: int | None = Field(default=None, ge=0)
    dropped_frames: int = Field(default=0, ge=0)
    output_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]] = Field(
        default_factory=dict
    )
    frame_qa_path: str | None = None
    frame_sequence_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    provenance: ProvenanceRecord

    @model_validator(mode="after")
    def unique_frames(self) -> Self:
        ids = [frame.frame_id for frame in self.frames]
        paths = [frame.relative_path for frame in self.frames]
        if len(ids) != len(set(ids)):
            raise ValueError("ingest frame IDs must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("ingest frame paths must be unique")
        if any(frame.source_type is not self.source_type for frame in self.frames):
            raise ValueError("ingest frame source types must match the manifest source type")
        if self.selected_frames is not None and self.selected_frames != len(self.frames):
            raise ValueError("selected_frames must match the number of manifest frames")
        if self.output_hashes and self.output_hashes != {
            frame.relative_path: frame.sha256 for frame in self.frames
        }:
            raise ValueError("output_hashes must exactly match manifest frame paths and hashes")
        if self.frame_qa_path is not None:
            _relative_artifact_path(self.frame_qa_path)
        return self


class FrameSelectionStatus(StrEnum):
    SELECTED = "selected"
    REJECTED = "rejected"


class FrameQualityEntry(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    relative_path: Annotated[str, Field(min_length=1)]
    blur_score: float = Field(ge=0)
    mean_brightness: float = Field(ge=0, le=255)
    grayscale_variance: float = Field(ge=0)
    near_duplicate: bool
    duplicate_of_frame_id: str | None = None
    status: FrameSelectionStatus
    rejection_reason: str | None = None

    @field_validator("relative_path")
    @classmethod
    def validate_quality_path(cls, value: str) -> str:
        return _relative_artifact_path(value)

    @model_validator(mode="after")
    def valid_selection(self) -> Self:
        if self.status is FrameSelectionStatus.SELECTED and self.rejection_reason is not None:
            raise ValueError("selected frames cannot have a rejection reason")
        if self.status is FrameSelectionStatus.REJECTED and not self.rejection_reason:
            raise ValueError("rejected frames require a rejection reason")
        return self


class FrameQualityReport(StrictModel):
    entries: Annotated[list[FrameQualityEntry], Field(min_length=1)]
    configuration: dict[str, object] = Field(default_factory=dict)
    provenance: ProvenanceRecord

    @model_validator(mode="after")
    def unique_entries(self) -> Self:
        frame_ids = [entry.frame_id for entry in self.entries]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("frame quality report frame IDs must be unique")
        return self


class CameraReconstruction(StrictModel):
    camera_id: Annotated[str, Field(min_length=1)]
    model: Annotated[str, Field(min_length=1)] = "pinhole"
    intrinsics: CameraIntrinsics
    poses: Annotated[list[CameraPose], Field(min_length=1)]
    registered_frame_ids: list[str] = Field(default_factory=list)
    unregistered_frame_ids: list[str] = Field(default_factory=list)
    sparse_point_count: int = Field(default=0, ge=0)
    average_reprojection_error: float | None = Field(default=None, ge=0)
    confidence: ConfidenceRecord
    coordinate_convention: CoordinateConvention
    scale_status: ScaleStatus = ScaleStatus.METRIC_SCALE_KNOWN
    frame_sequence_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    provenance: ProvenanceRecord

    @model_validator(mode="after")
    def consistent_registration(self) -> Self:
        pose_ids = [pose.frame_id for pose in self.poses]
        if len(pose_ids) != len(set(pose_ids)):
            raise ValueError("camera reconstruction pose frame IDs must be unique")
        if len(self.registered_frame_ids) != len(set(self.registered_frame_ids)):
            raise ValueError("registered frame IDs must be unique")
        if len(self.unregistered_frame_ids) != len(set(self.unregistered_frame_ids)):
            raise ValueError("unregistered frame IDs must be unique")
        if not self.registered_frame_ids:
            self.registered_frame_ids = pose_ids
        if set(self.registered_frame_ids) != set(pose_ids):
            raise ValueError("registered frame IDs must exactly match camera poses")
        if set(self.registered_frame_ids) & set(self.unregistered_frame_ids):
            raise ValueError("registered and unregistered frame IDs must not overlap")
        return self


class SparseModelDiagnostics(StrictModel):
    model_id: Annotated[str, Field(min_length=1)]
    registered_frames: int = Field(ge=0)
    registration_ratio: float = Field(ge=0, le=1)
    sparse_points: int = Field(ge=0)
    average_reprojection_error: float | None = Field(default=None, ge=0)
    selected: bool = False
    rejection_reason: str | None = None


class CameraDiagnostics(StrictModel):
    input_frame_count: int = Field(ge=0)
    selected_frame_count: int = Field(ge=0)
    models: list[SparseModelDiagnostics] = Field(default_factory=list)
    selected_model: str | None = None
    warnings: list[str] = Field(default_factory=list)
    failed_subcommand: str | None = None


class ColmapCommandRecord(StrictModel):
    name: Annotated[str, Field(min_length=1)]
    command: Annotated[list[str], Field(min_length=1)]
    return_code: int | None
    duration_s: float = Field(ge=0)
    timed_out: bool
    stdout_path: Annotated[str, Field(min_length=1)]
    stderr_path: Annotated[str, Field(min_length=1)]

    @field_validator("stdout_path", "stderr_path")
    @classmethod
    def validate_log_paths(cls, value: str) -> str:
        return _relative_artifact_path(value)


class ColmapWorkspaceManifest(StrictModel):
    execution_mode: Literal["local", "docker"]
    executable_or_image: Annotated[str, Field(min_length=1)]
    tool_version: str | None = None
    image_identifier: str | None = None
    database_path: Annotated[str, Field(min_length=1)]
    image_path: Annotated[str, Field(min_length=1)]
    sparse_path: Annotated[str, Field(min_length=1)]
    selected_model: str | None = None
    commands: list[ColmapCommandRecord] = Field(default_factory=list)
    failed_subcommand: str | None = None

    @field_validator("database_path", "image_path", "sparse_path")
    @classmethod
    def validate_workspace_paths(cls, value: str) -> str:
        return _relative_artifact_path(value)


class SegmentationPromptType(StrEnum):
    TEXT = "text"
    BOX = "box"
    POINT = "point"
    MASK = "mask"


class SegmentationPoint(StrictModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    label: Literal[0, 1]


class SegmentationPrompt(StrictModel):
    prompt_id: Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
    label: Annotated[str, Field(min_length=1)]
    prompt_type: SegmentationPromptType | None = None
    text: str | None = None
    frame_id: str | None = None
    box_xyxy: tuple[float, float, float, float] | None = None
    points: list[SegmentationPoint] | None = None
    mask_path: str | None = None
    asset_type_hint: AssetType | None = None
    confidence_threshold: float | None = Field(default=None, ge=0, le=1)
    positive: bool = True
    synonym_group: str | None = None
    instance_limit: int | None = Field(default=None, gt=0)
    notes: str | None = None
    enabled: bool = True

    @field_validator("label")
    @classmethod
    def nonempty_label(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("segmentation prompt labels must not be blank")
        return stripped

    @field_validator("text")
    @classmethod
    def nonempty_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("text prompts must not be blank")
        return value.strip() if value is not None else None

    @field_validator("mask_path")
    @classmethod
    def relative_mask_seed_path(cls, value: str | None) -> str | None:
        return _relative_artifact_path(value) if value is not None else None

    @model_validator(mode="after")
    def infer_and_validate_prompt_type(self) -> Self:
        populated = {
            SegmentationPromptType.TEXT: self.text is not None,
            SegmentationPromptType.BOX: self.box_xyxy is not None,
            SegmentationPromptType.POINT: self.points is not None,
            SegmentationPromptType.MASK: self.mask_path is not None,
        }
        active = [prompt_type for prompt_type, present in populated.items() if present]
        if len(active) != 1:
            raise ValueError("each prompt must specify exactly one of text, box, points, or mask")
        inferred = active[0]
        if self.prompt_type is not None and self.prompt_type is not inferred:
            raise ValueError(
                f"prompt_type={self.prompt_type.value!r} does not match {inferred.value!r} fields"
            )
        object.__setattr__(self, "prompt_type", inferred)
        if inferred is not SegmentationPromptType.TEXT and not self.frame_id:
            raise ValueError(f"{inferred.value} prompts require frame_id")
        if self.box_xyxy is not None:
            x0, y0, x1, y1 = self.box_xyxy
            if x1 <= x0 or y1 <= y0:
                raise ValueError("box prompts require x1>x0 and y1>y0")
        if self.points is not None and not self.points:
            raise ValueError("point prompts require at least one point")
        return self


class SegmentationPromptManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    prompts: list[SegmentationPrompt]
    source_path: str | None = None
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    input_hashes: dict[str, str] = Field(default_factory=dict)

    @field_validator("source_path")
    @classmethod
    def relative_source_path(cls, value: str | None) -> str | None:
        return _relative_artifact_path(value) if value is not None else None

    @field_validator("input_hashes")
    @classmethod
    def validate_input_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        for path, sha256 in value.items():
            _relative_artifact_path(path)
            if not re.fullmatch(r"[0-9a-f]{64}", sha256):
                raise ValueError(f"prompt input {path!r} has an invalid SHA-256")
        return value

    @model_validator(mode="after")
    def unique_prompt_ids(self) -> Self:
        prompt_ids = [prompt.prompt_id for prompt in self.prompts]
        if len(prompt_ids) != len(set(prompt_ids)):
            raise ValueError("segmentation prompt IDs must be unique")
        return self


class Sam3AnchorFrame(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    score: float
    camera_pose_available: bool
    selection_reason: Annotated[str, Field(min_length=1)]


class Sam3InferenceRequest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    run_id: Annotated[str, Field(min_length=1)]
    frame_manifest_path: Annotated[str, Field(min_length=1)]
    frame_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    frame_order: Annotated[list[str], Field(min_length=1)]
    frame_paths: Annotated[list[str], Field(min_length=1)]
    frame_dimensions: dict[str, tuple[int, int]]
    camera_reconstruction_path: Annotated[str, Field(min_length=1)]
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    registered_frame_ids: list[str]
    unregistered_frame_ids: list[str]
    prompt_manifest: SegmentationPromptManifest
    prompt_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    anchor_frames: Annotated[list[Sam3AnchorFrame], Field(min_length=1)]
    strategy: Literal["detect_then_track"]
    tracking_direction: Literal["forward", "backward", "forward_backward"]
    model_configuration: dict[str, object]
    postprocessing_configuration: dict[str, object]
    output_directory: Annotated[str, Field(min_length=1)]
    seed: int

    @field_validator(
        "frame_manifest_path",
        "camera_reconstruction_path",
        "frame_paths",
        "output_directory",
    )
    @classmethod
    def relative_request_paths(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, str):
            return _relative_artifact_path(value)
        return [_relative_artifact_path(item) for item in value]

    @model_validator(mode="after")
    def consistent_frames(self) -> Self:
        if len(self.frame_order) != len(set(self.frame_order)):
            raise ValueError("inference request frame_order must be unique")
        if len(self.frame_paths) != len(self.frame_order):
            raise ValueError("inference request must contain one frame path per ordered frame")
        if set(self.frame_dimensions) != set(self.frame_order):
            raise ValueError("frame_dimensions must exactly match frame_order")
        known = set(self.frame_order)
        if set(self.registered_frame_ids) | set(self.unregistered_frame_ids) != known:
            raise ValueError("registered and unregistered frame IDs must cover frame_order")
        if set(self.registered_frame_ids) & set(self.unregistered_frame_ids):
            raise ValueError("registered and unregistered frame IDs must not overlap")
        unknown_anchors = {anchor.frame_id for anchor in self.anchor_frames} - known
        if unknown_anchors:
            raise ValueError(f"anchor frames are not in frame_order: {sorted(unknown_anchors)}")
        return self


class Sam3RawObservation(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    raw_model_object_id: Annotated[str, Field(min_length=1)]
    prompt_id: Annotated[str, Field(min_length=1)]
    semantic_label: Annotated[str, Field(min_length=1)]
    score: float
    mask_path: Annotated[str, Field(min_length=1)]
    mask_encoding: Literal["binary_png", "grayscale_probability_png"] = "binary_png"
    model_box_xyxy: tuple[float, float, float, float] | None = None
    occluded: bool | None = None

    @field_validator("mask_path")
    @classmethod
    def relative_raw_mask_path(cls, value: str) -> str:
        return _relative_artifact_path(value)


class Sam3RawTrack(StrictModel):
    raw_model_object_id: Annotated[str, Field(min_length=1)]
    prompt_id: Annotated[str, Field(min_length=1)]
    semantic_label: Annotated[str, Field(min_length=1)]
    observations: list[Sam3RawObservation] = Field(default_factory=list)


class Sam3RawResult(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    tracks: list[Sam3RawTrack]
    warnings: list[str] = Field(default_factory=list)


class Sam3WorkerManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    official_repository: Annotated[str, Field(min_length=1)]
    official_code_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    checkpoint_repository: Annotated[str, Field(min_length=1)]
    checkpoint_revision: Annotated[str, Field(min_length=1)]
    checkpoint_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    checkpoint_access_mode: Literal["authenticated_remote", "local_path", "offline_cache", "fake"]
    official_license: Annotated[str, Field(min_length=1)]
    worker_version: Annotated[str, Field(min_length=1)]
    torch_version: str | None = None
    torchvision_version: str | None = None
    cuda_version: str | None = None
    device_name: str | None = None
    device: Annotated[str, Field(min_length=1)]
    precision: Annotated[str, Field(min_length=1)]
    seed: int
    runtime_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    prompt_manifest_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_manifest_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    strategy: Annotated[str, Field(min_length=1)]
    model_mode: Annotated[str, Field(min_length=1)]
    image_identifier: str | None = None
    warnings: list[str] = Field(default_factory=list)


class AnchorFrameDiagnostic(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    strategy: Annotated[str, Field(min_length=1)]
    selection_score: float
    selection_reason: Annotated[str, Field(min_length=1)]
    camera_pose_available: bool


class DroppedTrackDiagnostic(StrictModel):
    raw_model_object_id: Annotated[str, Field(min_length=1)]
    semantic_label: Annotated[str, Field(min_length=1)]
    prompt_id: Annotated[str, Field(min_length=1)]
    reason_code: Annotated[str, Field(min_length=1)]
    explanation: Annotated[str, Field(min_length=1)]


class SegmentationObservation(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    mask_path: Annotated[str, Field(min_length=1)]
    bbox_xywh: tuple[int, int, int, int]
    model_box_xyxy: tuple[float, float, float, float] | None = None
    frame_score: float = Field(ge=0, le=1)
    mask_area_pixels: int = Field(gt=0)
    mask_area_ratio: float = Field(gt=0, le=1)
    camera_pose_available: bool
    occluded: bool | None = None
    visibility_status: Literal["visible"] = "visible"
    raw_model_object_id: Annotated[str, Field(min_length=1)]
    prompt_id: Annotated[str, Field(min_length=1)]

    @field_validator("mask_path")
    @classmethod
    def relative_canonical_mask_path(cls, value: str) -> str:
        return _relative_artifact_path(value)

    @field_validator("bbox_xywh")
    @classmethod
    def positive_canonical_box(cls, value: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, width, height = value
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("canonical boxes require non-negative origins and positive dimensions")
        return value


class SegmentationTrack(StrictModel):
    object_id: Annotated[str, Field(min_length=1)]
    semantic_label: Annotated[str, Field(min_length=1)]
    normalized_semantic_label: Annotated[str, Field(min_length=1)]
    prompt_id: Annotated[str, Field(min_length=1)]
    raw_model_object_id: Annotated[str, Field(min_length=1)]
    asset_type_hint: AssetType | None = None
    asset_type_hint_source: Literal["configured_semantic_hint"] | None = None
    first_frame_id: Annotated[str, Field(min_length=1)]
    last_frame_id: Annotated[str, Field(min_length=1)]
    observation_count: int = Field(gt=0)
    coverage_ratio: float = Field(gt=0, le=1)
    mean_score: float = Field(ge=0, le=1)
    minimum_score: float = Field(ge=0, le=1)
    observations: Annotated[list[SegmentationObservation], Field(min_length=1)]
    provenance: ProvenanceRecord
    confidence: ConfidenceRecord

    @model_validator(mode="after")
    def consistent_track_summary(self) -> Self:
        if self.observation_count != len(self.observations):
            raise ValueError("observation_count must match observations")
        if self.first_frame_id != self.observations[0].frame_id:
            raise ValueError("first_frame_id must match the first observation")
        if self.last_frame_id != self.observations[-1].frame_id:
            raise ValueError("last_frame_id must match the last observation")
        if self.asset_type_hint is None and self.asset_type_hint_source is not None:
            raise ValueError("asset_type_hint_source requires asset_type_hint")
        if self.asset_type_hint is not None and self.asset_type_hint_source is None:
            raise ValueError("asset_type_hint requires asset_type_hint_source")
        return self


class SegmentationDiagnostics(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    backend_mode: Literal["local_worker", "docker", "fake_worker"]
    input_frame_count: int = Field(ge=0)
    registered_frame_count: int = Field(ge=0)
    unregistered_frame_count: int = Field(ge=0)
    prompt_count: int = Field(ge=0)
    anchor_frames: list[AnchorFrameDiagnostic]
    raw_track_count: int = Field(ge=0)
    kept_track_count: int = Field(ge=0)
    dropped_tracks: list[DroppedTrackDiagnostic]
    mask_count: int = Field(ge=0)
    mean_coverage: float = Field(ge=0, le=1)
    mean_confidence: float = Field(ge=0, le=1)
    runtime_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    thresholds: dict[str, float | int]
    no_matching_prompt_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SegmentationTrackingArtifact(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    frame_count: int = Field(gt=0)
    tracks: list[SegmentationTrack]
    prompt_manifest_path: Annotated[str, Field(min_length=1)]
    worker_manifest_path: Annotated[str, Field(min_length=1)]
    diagnostics_path: Annotated[str, Field(min_length=1)]
    canonicalization_version: Literal["0.1.0"] = "0.1.0"
    frame_sequence_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    provenance: ProvenanceRecord

    @field_validator(
        "prompt_manifest_path",
        "worker_manifest_path",
        "diagnostics_path",
    )
    @classmethod
    def relative_tracking_paths(cls, value: str) -> str:
        return _relative_artifact_path(value)

    @model_validator(mode="after")
    def unique_track_ids(self) -> Self:
        object_ids = [track.object_id for track in self.tracks]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("canonical segmentation object IDs must be unique")
        return self


class ObservationLineage(StrictModel):
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_ids: list[str]
    frame_paths: list[str]
    frame_sha256_by_id: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    registered_frame_ids: list[str]
    unregistered_frame_ids: list[str]
    segmentation_input_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    genrecon_input_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("frame_paths")
    @classmethod
    def validate_lineage_paths(cls, values: list[str]) -> list[str]:
        return [_relative_artifact_path(value) for value in values]

    @model_validator(mode="after")
    def consistent_lineage(self) -> Self:
        if len(self.frame_ids) != len(set(self.frame_ids)):
            raise ValueError("lineage frame IDs must be unique")
        if len(self.frame_paths) != len(self.frame_ids):
            raise ValueError("lineage must contain one path per frame ID")
        if set(self.frame_sha256_by_id) != set(self.frame_ids):
            raise ValueError("lineage hashes must exactly cover frame IDs")
        if set(self.registered_frame_ids) | set(self.unregistered_frame_ids) != set(self.frame_ids):
            raise ValueError("lineage registration sets must cover all frame IDs")
        if set(self.registered_frame_ids) & set(self.unregistered_frame_ids):
            raise ValueError("lineage registration sets must not overlap")
        return self


class GenReconRegisteredFrame(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    source_relative_path: Annotated[str, Field(min_length=1)]
    package_image_name: Annotated[str, Field(min_length=1)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    original_colmap_image_id: int = Field(gt=0)
    package_image_id: int = Field(gt=0)
    original_colmap_camera_id: int = Field(gt=0)
    package_camera_id: int = Field(gt=0)

    @field_validator("source_relative_path", "package_image_name")
    @classmethod
    def validate_registered_paths(cls, value: str) -> str:
        return _relative_artifact_path(value)


class GenReconCameraPackageManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    source_manifest_path: Literal["inputs/manifest.json"] = "inputs/manifest.json"
    source_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_reconstruction_path: Literal["camera/reconstruction.json"] = "camera/reconstruction.json"
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    selected_model_id: Annotated[str, Field(min_length=1)]
    source_model_paths: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    master_frame_ids: list[str]
    registered_frame_ids: list[str]
    unregistered_frame_ids: list[str]
    eligible_frame_ids: list[str]
    registered_frames: list[GenReconRegisteredFrame]
    cameras_path: Literal["camera/genrecon_package/cameras.txt"] = (
        "camera/genrecon_package/cameras.txt"
    )
    images_path: Literal["camera/genrecon_package/images.txt"] = (
        "camera/genrecon_package/images.txt"
    )
    points3d_path: Literal["camera/genrecon_package/points3D.txt"] = (
        "camera/genrecon_package/points3D.txt"
    )
    package_content_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    coordinate_convention: CoordinateConvention

    @model_validator(mode="after")
    def consistent_package_frames(self) -> Self:
        master = set(self.master_frame_ids)
        if len(self.master_frame_ids) != len(master):
            raise ValueError("camera package master frame IDs must be unique")
        if set(self.registered_frame_ids) | set(self.unregistered_frame_ids) != master:
            raise ValueError("camera package registration sets must cover the master frames")
        if set(self.registered_frame_ids) & set(self.unregistered_frame_ids):
            raise ValueError("camera package registration sets must not overlap")
        expected_eligible = [
            frame_id for frame_id in self.master_frame_ids if frame_id in self.registered_frame_ids
        ]
        if self.eligible_frame_ids != expected_eligible:
            raise ValueError("eligible GenRecon frames must be registered frames in master order")
        if [frame.frame_id for frame in self.registered_frames] != self.eligible_frame_ids:
            raise ValueError("registered frame records must follow eligible frame order")
        if set(self.source_model_paths) != {"cameras.bin", "images.bin", "points3D.bin"}:
            raise ValueError("camera package must identify exactly one complete COLMAP model")
        return self


class GenReconCheckpointRecord(StrictModel):
    checkpoint_id: Literal["sparse_structure", "shape_slat", "texture_slat"]
    source_url: Annotated[str, Field(min_length=1)]
    local_filename: Annotated[str, Field(min_length=1)]
    size_bytes: int = Field(gt=0)
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    resolved_at: Annotated[str, Field(min_length=1)]
    access_mode: Literal["downloaded", "local_cache", "fake"]


class GenReconCheckpointManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    official_host: Literal["https://kaldir.vc.cit.tum.de/genrecon/"]
    checkpoints: Annotated[list[GenReconCheckpointRecord], Field(min_length=3, max_length=3)]

    @model_validator(mode="after")
    def complete_checkpoint_set(self) -> Self:
        identifiers = [checkpoint.checkpoint_id for checkpoint in self.checkpoints]
        if set(identifiers) != {"sparse_structure", "shape_slat", "texture_slat"}:
            raise ValueError("checkpoint manifest must contain the three official checkpoints")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("checkpoint manifest checkpoint IDs must be unique")
        return self


class GenReconWorkingTransform(StrictModel):
    strategy: Literal["identity", "pca_scene_axes"]
    matrix_colmap_to_working: list[list[float]]
    matrix_working_to_colmap: list[list[float]]
    determinant: float
    roundtrip_max_error: float = Field(ge=0)
    semantic_status: Literal["internal_unoriented_preprocessing"] = (
        "internal_unoriented_preprocessing"
    )

    @field_validator("matrix_colmap_to_working", "matrix_working_to_colmap")
    @classmethod
    def matrix_is_finite_4x4(cls, value: list[list[float]]) -> list[list[float]]:
        import math

        if len(value) != 4 or any(len(row) != 4 for row in value):
            raise ValueError("working transforms must be 4x4 matrices")
        if any(not math.isfinite(component) for row in value for component in row):
            raise ValueError("working transforms must contain only finite values")
        return value


class GenReconInferenceRequest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    run_id: Annotated[str, Field(min_length=1)]
    official_repository: Literal["https://github.com/kasothaphie/GenRecon"]
    official_code_commit: Literal["eaf1468118d20469d17079a4a19737297d2ef87b"]
    official_checkout_path: Annotated[str, Field(min_length=1)]
    checkpoint_paths: dict[str, str]
    checkpoint_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    checkpoint_manifest_path: Annotated[str, Field(min_length=1)]
    manifest_path: Literal["inputs/manifest.json"] = "inputs/manifest.json"
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_reconstruction_path: Literal["camera/reconstruction.json"] = "camera/reconstruction.json"
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_package_manifest_path: Literal["camera/genrecon_package/package_manifest.json"] = (
        "camera/genrecon_package/package_manifest.json"
    )
    camera_package_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    master_frame_order: list[str]
    normalized_frame_paths: dict[str, str]
    normalized_frame_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    registered_frame_ids: list[str]
    unregistered_frame_ids: list[str]
    eligible_frame_ids: list[str]
    requested_max_views: int = Field(gt=0)
    coordinate_convention: CoordinateConvention
    working_transform_strategy: Literal["identity", "pca_scene_axes"]
    pipeline_config: str
    reconstruction_parameters: dict[str, object]
    output_directory: Literal["reconstruction/global/raw"] = "reconstruction/global/raw"
    seed: int

    @field_validator(
        "checkpoint_manifest_path",
        "normalized_frame_paths",
        "pipeline_config",
    )
    @classmethod
    def validate_genrecon_request_paths(cls, value: str | dict[str, str]) -> str | dict[str, str]:
        if isinstance(value, str):
            return _relative_artifact_path(value)
        return {frame_id: _relative_artifact_path(path) for frame_id, path in value.items()}

    @model_validator(mode="after")
    def consistent_request_frames(self) -> Self:
        master = set(self.master_frame_order)
        if len(master) != len(self.master_frame_order):
            raise ValueError("GenRecon master frame order must be unique")
        if (
            set(self.normalized_frame_paths) != master
            or set(self.normalized_frame_hashes) != master
        ):
            raise ValueError("GenRecon frame paths and hashes must exactly cover master order")
        if set(self.registered_frame_ids) | set(self.unregistered_frame_ids) != master:
            raise ValueError("GenRecon registration sets must cover master frames")
        if self.eligible_frame_ids != [
            frame_id
            for frame_id in self.master_frame_order
            if frame_id in self.registered_frame_ids
        ]:
            raise ValueError("GenRecon eligible frames must be registered frames in master order")
        expected_checkpoints = {"sparse_structure", "shape_slat", "texture_slat"}
        if set(self.checkpoint_paths) != expected_checkpoints:
            raise ValueError("GenRecon requires all three checkpoint paths")
        if set(self.checkpoint_hashes) != expected_checkpoints:
            raise ValueError("GenRecon requires all three checkpoint hashes")
        return self


class GenReconWorkerManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    official_repository: Literal["https://github.com/kasothaphie/GenRecon"]
    official_code_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    submodule_commits: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]]
    official_license: Annotated[str, Field(min_length=1)]
    checkpoint_records: list[GenReconCheckpointRecord]
    runtime_model_repository: Literal["facebook/dinov3-vitl16-pretrain-lvd1689m"]
    runtime_model_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    runtime_repository_revisions: dict[
        str,
        Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")],
    ]
    worker_version: Annotated[str, Field(min_length=1)]
    python_version: Annotated[str, Field(min_length=1)]
    torch_version: str | None = None
    torchvision_version: str | None = None
    cuda_version: str | None = None
    device_name: str | None = None
    device: Annotated[str, Field(min_length=1)]
    precision: Annotated[str, Field(min_length=1)]
    seed: int
    request_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_package_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    registered_frame_ids: list[str]
    selected_frame_ids: list[str]
    working_transform: GenReconWorkingTransform
    reconstruct_return_code: int
    glb_conversion_return_code: int
    runtime_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    raw_output_paths: list[str]
    image_identifier: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("raw_output_paths")
    @classmethod
    def validate_raw_output_paths(cls, values: list[str]) -> list[str]:
        return [_relative_artifact_path(value) for value in values]


class GlobalSceneMeshStatistics(StrictModel):
    vertex_count: int = Field(gt=0)
    face_count: int = Field(gt=0)
    disconnected_components: int = Field(ge=1)
    degenerate_faces: int = Field(ge=0)
    non_manifold_edge_count: int = Field(ge=0)
    finite_coordinates: Literal[True] = True
    bounding_box_min: tuple[float, float, float]
    bounding_box_max: tuple[float, float, float]
    bounding_box_extent: tuple[float, float, float]
    material_count: int = Field(ge=0)
    texture_count: int = Field(ge=0)
    glb_parse_status: Literal["valid"]


class GlobalSceneChunkDiagnostic(StrictModel):
    chunk_id: Annotated[str, Field(min_length=1)]
    point_count: int = Field(ge=0)
    selected_view_count: int = Field(ge=0)
    dropped: bool = False
    reason: str | None = None


class GlobalSceneDiagnostics(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    eligible_frame_count: int = Field(ge=0)
    selected_view_count: int = Field(ge=0)
    registered_coverage: float = Field(ge=0, le=1)
    initial_sparse_points: int = Field(ge=0)
    cleaned_sparse_points: int = Field(ge=0)
    point_retention_ratio: float = Field(ge=0, le=1)
    robust_bounds_min: tuple[float, float, float]
    robust_bounds_max: tuple[float, float, float]
    scene_diagonal_arbitrary_units: float = Field(gt=0)
    chunks_before_filtering: int = Field(ge=0)
    chunks_after_filtering: int = Field(gt=0)
    chunks: list[GlobalSceneChunkDiagnostic]
    mesh: GlobalSceneMeshStatistics
    chosen_parameters: dict[str, object]
    runtime_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    warnings: list[str] = Field(default_factory=list)


class GlobalScenePreviewManifest(StrictModel):
    global_scene_preview_path: Literal["reconstruction/global/previews/global_scene_preview.png"]
    camera_trajectory_path: Literal[
        "reconstruction/global/previews/camera_trajectory_and_sparse_points.png"
    ]
    input_vs_geometry_path: Literal[
        "reconstruction/global/previews/input_vs_geometry_contact_sheet.png"
    ]
    optional_turntable_path: str | None = None

    @field_validator("optional_turntable_path")
    @classmethod
    def validate_optional_preview_path(cls, value: str | None) -> str | None:
        return _relative_artifact_path(value) if value is not None else None


class GlobalSceneReconstructionArtifact(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    scene_asset_path: Literal["reconstruction/global/scene.glb"]
    mesh_asset_path: Literal["reconstruction/global/mesh.ply"]
    scene_ir_path: Literal["scene_ir/scene.json"]
    format: Literal["glb"] = "glb"
    coordinate_convention: CoordinateConvention
    scale_status: ScaleStatus
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_package_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    input_frame_count: int = Field(gt=0)
    registered_frame_count: int = Field(gt=0)
    unregistered_frame_count: int = Field(ge=0)
    eligible_frame_ids: list[str]
    actual_selected_frame_ids: list[str]
    mesh: GlobalSceneMeshStatistics
    chunk_count: int = Field(gt=0)
    checkpoints: list[GenReconCheckpointRecord]
    official_repository: Annotated[str, Field(min_length=1)]
    official_code_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    runtime_model_repository: Literal["facebook/dinov3-vitl16-pretrain-lvd1689m"]
    runtime_model_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    runtime_repository_revisions: dict[
        str,
        Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")],
    ]
    runtime_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    seed: int
    provenance: ProvenanceRecord


class EndToEndConsistencyCheck(StrictModel):
    check_id: Annotated[str, Field(min_length=1)]
    passed: bool
    message: Annotated[str, Field(min_length=1)]


class EndToEndConsistencyReport(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    passed: bool
    checks: list[EndToEndConsistencyCheck]
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    real_modules_share_consistent_inputs: bool
    object_level_2d_3d_fusion_implemented: Literal[False] = False
    sim_ready_scene_implemented: Literal[False] = False
    metric_scale_known: Literal[False] = False
    canonical_gravity_alignment_known: Literal[False] = False
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def summary_matches_checks(self) -> Self:
        expected = all(check.passed for check in self.checks)
        if self.passed != expected:
            raise ValueError("end-to-end report passed must equal all individual checks")
        if self.real_modules_share_consistent_inputs != expected:
            raise ValueError("input consistency summary must equal all individual checks")
        return self


class CompactFaceIndexManifest(StrictModel):
    relative_path: Annotated[str, Field(min_length=1)]
    dtype: Literal["uint32", "uint64"]
    byte_order: Literal["little"] = "little"
    count: int = Field(ge=0)
    global_mesh_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    minimum_face_id: int | None = Field(default=None, ge=0)
    maximum_face_id: int | None = Field(default=None, ge=0)
    content_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @field_validator("relative_path")
    @classmethod
    def relative_face_index_path(cls, value: str) -> str:
        return _relative_artifact_path(value)

    @model_validator(mode="after")
    def consistent_range(self) -> Self:
        if self.count == 0 and (
            self.minimum_face_id is not None or self.maximum_face_id is not None
        ):
            raise ValueError("empty face-index arrays cannot declare a face-ID range")
        if self.count > 0 and (self.minimum_face_id is None or self.maximum_face_id is None):
            raise ValueError("non-empty face-index arrays require a face-ID range")
        if (
            self.minimum_face_id is not None
            and self.maximum_face_id is not None
            and self.minimum_face_id > self.maximum_face_id
        ):
            raise ValueError("face-index minimum must not exceed maximum")
        return self


class ObjectSurfaceTrackRequest(StrictModel):
    object_id: Annotated[str, Field(min_length=1)]
    semantic_label: Annotated[str, Field(min_length=1)]
    prompt_id: Annotated[str, Field(min_length=1)]
    asset_type_hint: AssetType | None = None
    track_coverage: float = Field(ge=0, le=1)
    mask_paths_by_frame: dict[str, str]
    frame_scores: dict[str, float]

    @field_validator("mask_paths_by_frame")
    @classmethod
    def relative_track_masks(cls, values: dict[str, str]) -> dict[str, str]:
        return {frame_id: _relative_artifact_path(path) for frame_id, path in values.items()}

    @model_validator(mode="after")
    def matching_observation_keys(self) -> Self:
        if set(self.mask_paths_by_frame) != set(self.frame_scores):
            raise ValueError("mask paths and frame scores must cover the same observations")
        if any(score < 0 or score > 1 for score in self.frame_scores.values()):
            raise ValueError("object lifting frame scores must be in [0, 1]")
        return self


class ObjectSurfaceLiftingRequest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    run_id: Annotated[str, Field(min_length=1)]
    manifest_path: Literal["inputs/manifest.json"] = "inputs/manifest.json"
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    master_frame_order: Annotated[list[str], Field(min_length=1)]
    normalized_frame_paths: dict[str, str]
    normalized_frame_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    camera_reconstruction_path: Literal["camera/reconstruction.json"] = "camera/reconstruction.json"
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_package_manifest_path: Literal["camera/genrecon_package/package_manifest.json"] = (
        "camera/genrecon_package/package_manifest.json"
    )
    camera_package_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_package_images_path: Literal["camera/genrecon_package/images.txt"] = (
        "camera/genrecon_package/images.txt"
    )
    camera_package_images_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_package_points3d_path: Literal["camera/genrecon_package/points3D.txt"] = (
        "camera/genrecon_package/points3D.txt"
    )
    camera_package_points3d_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_package_registered_frames_path: Literal[
        "camera/genrecon_package/registered_frames.json"
    ] = "camera/genrecon_package/registered_frames.json"
    camera_package_registered_frames_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    registered_frame_ids: Annotated[list[str], Field(min_length=1)]
    unregistered_frame_ids: list[str]
    coordinate_convention: CoordinateConvention
    segmentation_tracking_path: Literal["observations/object_tracks.json"] = (
        "observations/object_tracks.json"
    )
    segmentation_tracking_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    object_tracks: list[ObjectSurfaceTrackRequest]
    global_reconstruction_path: Literal["reconstruction/global/metadata.json"] = (
        "reconstruction/global/metadata.json"
    )
    global_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    global_mesh_path: Literal["reconstruction/global/mesh.ply"] = "reconstruction/global/mesh.ply"
    global_mesh_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    alignment_policy: Literal["none", "use_if_accepted", "require_accepted"] = "none"
    alignment_path: Literal["reconstruction/alignment/alignment.json"] | None = None
    alignment_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    alignment_status: str | None = None
    alignment_accepted: bool = False
    matrix_original_mesh_to_aligned_colmap: list[list[float]] | None = None
    lifting_method: Literal["exact_face_vote_v1", "surface_sample_fusion_v2"]
    rasterization_configuration: dict[str, object]
    mask_processing_configuration: dict[str, object]
    face_evidence_configuration: dict[str, object]
    surface_sample_configuration: dict[str, object]
    surface_extraction_configuration: dict[str, object]
    output_directory: Literal["reconstruction/object_surfaces/raw"] = (
        "reconstruction/object_surfaces/raw"
    )
    seed: int

    @field_validator("normalized_frame_paths")
    @classmethod
    def relative_normalized_frames(cls, values: dict[str, str]) -> dict[str, str]:
        return {frame_id: _relative_artifact_path(path) for frame_id, path in values.items()}

    @field_validator("matrix_original_mesh_to_aligned_colmap")
    @classmethod
    def finite_optional_alignment_matrix(
        cls, value: list[list[float]] | None
    ) -> list[list[float]] | None:
        if value is None:
            return None
        import math

        if len(value) != 4 or any(len(row) != 4 for row in value):
            raise ValueError("object-lifting alignment transform must be a 4x4 matrix")
        if any(not math.isfinite(component) for row in value for component in row):
            raise ValueError("object-lifting alignment transform must be finite")
        return value

    @model_validator(mode="after")
    def consistent_lineage(self) -> Self:
        master = set(self.master_frame_order)
        if len(master) != len(self.master_frame_order):
            raise ValueError("object lifting master frame order must be unique")
        if (
            set(self.normalized_frame_paths) != master
            or set(self.normalized_frame_hashes) != master
        ):
            raise ValueError("normalized paths and hashes must cover the master frame order")
        if set(self.registered_frame_ids) | set(self.unregistered_frame_ids) != master:
            raise ValueError("registered and unregistered sets must cover master frames")
        if set(self.registered_frame_ids) & set(self.unregistered_frame_ids):
            raise ValueError("registered and unregistered frames must not overlap")
        known = master
        for track in self.object_tracks:
            unknown = set(track.mask_paths_by_frame) - known
            if unknown:
                raise ValueError(
                    f"object {track.object_id!r} references unknown frames: {sorted(unknown)}"
                )
        if self.alignment_policy == "none":
            if (
                self.alignment_path is not None
                or self.alignment_sha256 is not None
                or self.alignment_accepted
                or self.matrix_original_mesh_to_aligned_colmap is not None
            ):
                raise ValueError("alignment_policy=none cannot carry an alignment transform")
        else:
            if self.alignment_path is None or self.alignment_sha256 is None:
                raise ValueError("alignment-aware lifting requires a typed alignment artifact")
            if self.alignment_accepted != (self.matrix_original_mesh_to_aligned_colmap is not None):
                raise ValueError("accepted alignment and applied matrix must agree")
            if self.alignment_policy == "require_accepted" and not self.alignment_accepted:
                raise ValueError("require_accepted lifting requires an accepted alignment")
        return self


class ObjectSurfaceObservationSupport(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    registered: Literal[True] = True
    source_camera_model: Annotated[str, Field(min_length=1)]
    source_distortion: list[float]
    undistorted_width: int = Field(gt=0)
    undistorted_height: int = Field(gt=0)
    undistorted_intrinsics: CameraIntrinsics
    undistortion_map_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    visible_face_count: int = Field(ge=0)
    supporting_face_count: int = Field(ge=0)
    iou: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    rendered_area_pixels: int = Field(ge=0)
    mask_area_pixels: int = Field(ge=0)
    false_positive_area_pixels: int = Field(ge=0)
    false_negative_area_pixels: int = Field(ge=0)


class ObjectSurfaceComponent(StrictModel):
    component_id: Annotated[str, Field(min_length=1)]
    face_count: int = Field(gt=0)
    surface_area_arbitrary_units_squared: float = Field(ge=0)
    relative_face_ratio: float = Field(gt=0, le=1)
    relative_surface_area: float = Field(ge=0, le=1)
    retained: bool
    removal_reason: str | None = None


class ObjectSurfaceConflict(StrictModel):
    conflict_type: Literal["same_class_instance", "different_semantic_label"]
    object_ids: Annotated[list[str], Field(min_length=2)]
    face_count: int = Field(gt=0)
    resolution: Literal[
        "winner_by_support",
        "ambiguous_below_margin",
        "multi_label_retained",
    ]


class FaceEvidenceArrayRecord(StrictModel):
    name: Annotated[str, Field(min_length=1)]
    shape: list[int]
    dtype: Annotated[str, Field(min_length=1)]
    content_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ObjectSurfaceHypothesis(StrictModel):
    object_id: Annotated[str, Field(min_length=1)]
    semantic_label: Annotated[str, Field(min_length=1)]
    prompt_id: Annotated[str, Field(min_length=1)]
    asset_type_hint: AssetType | None = None
    status: Literal["accepted", "partial", "ambiguous", "unresolved"]
    unresolved_reason: str | None = None
    source_track_path: Literal["observations/object_tracks.json"]
    source_mask_paths: list[str]
    supporting_frame_ids: list[str]
    supporting_registered_frame_ids: list[str]
    global_mesh_path: Literal["reconstruction/global/mesh.ply"]
    global_mesh_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    global_face_count: int = Field(gt=0)
    accepted_global_face_ids: CompactFaceIndexManifest
    ambiguous_global_face_ids: CompactFaceIndexManifest
    face_evidence_path: str
    face_evidence_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    face_evidence_arrays: list[FaceEvidenceArrayRecord]
    surface_mesh_path: str | None = None
    surface_point_cloud_path: str | None = None
    surface_visual_glb_path: str | None = None
    vertex_count: int = Field(ge=0)
    face_count: int = Field(ge=0)
    component_count: int = Field(ge=0)
    exact_component_count: int = Field(ge=0)
    seam_aware_component_count: int = Field(ge=0)
    potential_chunk_seam_merges: int = Field(ge=0)
    components: list[ObjectSurfaceComponent]
    bbox_min: tuple[float, float, float] | None = None
    bbox_max: tuple[float, float, float] | None = None
    bbox_extent: tuple[float, float, float] | None = None
    centroid: tuple[float, float, float] | None = None
    mean_face_support_score: float = Field(ge=0, le=1)
    median_face_support_score: float = Field(ge=0, le=1)
    supporting_view_count: int = Field(ge=0)
    median_reprojection_iou: float = Field(ge=0, le=1)
    mean_reprojection_iou: float = Field(ge=0, le=1)
    track_coverage: float = Field(ge=0, le=1)
    association_precision: float = Field(ge=0, le=1)
    mask_recall: float = Field(ge=0, le=1)
    reprojection_iou: float = Field(ge=0, le=1)
    multiview_support: float = Field(ge=0, le=1)
    surface_connectedness: float = Field(ge=0, le=1)
    observed_surface_coverage: float = Field(ge=0, le=1)
    association_confidence: float = Field(ge=0, le=1)
    completeness_confidence: float = Field(default=0.0, ge=0.0, le=0.0)
    observation_support: list[ObjectSurfaceObservationSupport]
    geometry_status: Literal["partial_observation_supported"] = "partial_observation_supported"
    completion_status: Literal["not_completed"] = "not_completed"
    hidden_surface_completion: Literal["not_implemented"] = "not_implemented"
    sim_ready: Literal[False] = False
    metric_scale_known: Literal[False] = False
    canonical_gravity_alignment_known: Literal[False] = False
    coordinate_convention: CoordinateConvention
    scale_status: Literal[ScaleStatus.SCALE_AMBIGUOUS] = ScaleStatus.SCALE_AMBIGUOUS
    confidence: ConfidenceRecord
    provenance: ProvenanceRecord
    warnings: list[str] = Field(default_factory=list)

    @field_validator(
        "source_mask_paths",
        "face_evidence_path",
        "surface_mesh_path",
        "surface_point_cloud_path",
        "surface_visual_glb_path",
    )
    @classmethod
    def relative_object_surface_paths(cls, value: list[str] | str | None) -> list[str] | str | None:
        if value is None:
            return None
        if isinstance(value, list):
            return [_relative_artifact_path(path) for path in value]
        return _relative_artifact_path(value)

    @model_validator(mode="after")
    def consistent_surface_status(self) -> Self:
        if self.status == "unresolved":
            if self.accepted_global_face_ids.count != 0 or self.face_count != 0:
                raise ValueError("unresolved objects cannot contain accepted surface faces")
            if not self.unresolved_reason:
                raise ValueError("unresolved objects require an actionable reason")
        elif self.accepted_global_face_ids.count == 0 or self.face_count == 0:
            raise ValueError("accepted or ambiguous objects require a non-empty surface")
        if self.face_count != self.accepted_global_face_ids.count:
            raise ValueError("surface face count must match accepted global face IDs")
        return self


class ObjectSurfaceWorkerManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    worker_version: Annotated[str, Field(min_length=1)]
    backend: Literal["nvdiffrast", "fake"]
    python_version: Annotated[str, Field(min_length=1)]
    torch_version: str | None = None
    cuda_version: str | None = None
    nvdiffrast_version: str | None = None
    device: Annotated[str, Field(min_length=1)]
    device_name: str | None = None
    request_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    segmentation_tracking_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    global_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    global_mesh_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    alignment_policy: Literal["none", "use_if_accepted", "require_accepted"] = "none"
    alignment_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    alignment_status: str | None = None
    alignment_accepted: bool = False
    processed_registered_frame_ids: list[str]
    global_vertex_count: int = Field(gt=0)
    global_face_count: int = Field(gt=0)
    lifting_method: Literal["exact_face_vote_v1", "surface_sample_fusion_v2"]
    median_global_edge_length: float = Field(gt=0)
    sample_voxel_edge_length: float | None = Field(default=None, gt=0)
    fused_sample_cell_count: int = Field(ge=0)
    processed_face_count_by_frame: dict[str, int]
    culled_face_count_by_frame: dict[str, int]
    mesh_load_seconds: float = Field(ge=0)
    rasterization_seconds: float = Field(ge=0)
    evidence_accumulation_seconds: float = Field(ge=0)
    surface_extraction_seconds: float = Field(ge=0)
    preview_seconds: float = Field(ge=0)
    runtime_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    peak_host_memory_bytes: int | None = Field(default=None, ge=0)
    raw_output_paths: list[str]
    warnings: list[str] = Field(default_factory=list)

    @field_validator("raw_output_paths")
    @classmethod
    def relative_worker_outputs(cls, values: list[str]) -> list[str]:
        return [_relative_artifact_path(value) for value in values]


class GlobalFacePartitionSummary(StrictModel):
    global_face_count: int = Field(gt=0)
    unassigned_face_count: int = Field(ge=0)
    exactly_one_object_face_count: int = Field(ge=0)
    multi_label_face_count: int = Field(ge=0)
    same_class_conflict_face_count: int = Field(ge=0)
    assigned_face_count_by_object: dict[str, int]
    ambiguous_face_count_by_object: dict[str, int]
    unassigned_face_ratio: float = Field(ge=0, le=1)


class ObjectSurfaceEvidenceArtifact(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    request_path: Literal["reconstruction/object_surfaces/request.json"] = (
        "reconstruction/object_surfaces/request.json"
    )
    worker_manifest_path: Literal["reconstruction/object_surfaces/worker_manifest.json"] = (
        "reconstruction/object_surfaces/worker_manifest.json"
    )
    diagnostics_path: Literal["reconstruction/object_surfaces/diagnostics.json"] = (
        "reconstruction/object_surfaces/diagnostics.json"
    )
    preview_manifest_path: Literal["reconstruction/object_surfaces/preview_manifest.json"] = (
        "reconstruction/object_surfaces/preview_manifest.json"
    )
    scene_ir_path: Literal["scene_ir/phase4_scene.json"] = "scene_ir/phase4_scene.json"
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    segmentation_tracking_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    global_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    global_mesh_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    alignment_policy: Literal["none", "use_if_accepted", "require_accepted"] = "none"
    alignment_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    alignment_status: str | None = None
    alignment_accepted: bool = False
    coordinate_convention: CoordinateConvention
    scale_status: Literal[ScaleStatus.SCALE_AMBIGUOUS] = ScaleStatus.SCALE_AMBIGUOUS
    geometry_status: Literal["partial_observation_supported"] = "partial_observation_supported"
    hidden_surface_completion: Literal["not_implemented"] = "not_implemented"
    sim_ready: Literal[False] = False
    metric_scale_known: Literal[False] = False
    canonical_gravity_alignment_known: Literal[False] = False
    hypotheses: list[ObjectSurfaceHypothesis]
    partition: GlobalFacePartitionSummary
    conflicts: list[ObjectSurfaceConflict]
    provenance: ProvenanceRecord
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_hypotheses(self) -> Self:
        object_ids = [hypothesis.object_id for hypothesis in self.hypotheses]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("object surface hypothesis IDs must be unique")
        return self


class ObjectSurfaceDiagnostics(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    track_count: int = Field(ge=0)
    accepted_object_count: int = Field(ge=0)
    partial_object_count: int = Field(ge=0)
    ambiguous_object_count: int = Field(ge=0)
    unresolved_object_count: int = Field(ge=0)
    global_vertex_count: int = Field(gt=0)
    global_face_count: int = Field(gt=0)
    processed_camera_count: int = Field(ge=0)
    canonical_mask_count: int = Field(ge=0)
    accepted_face_count: int = Field(ge=0)
    ambiguous_face_count: int = Field(ge=0)
    same_class_conflict_count: int = Field(ge=0)
    different_label_overlap_count: int = Field(ge=0)
    unassigned_face_ratio: float = Field(ge=0, le=1)
    mean_face_support: float = Field(ge=0, le=1)
    median_face_support: float = Field(ge=0, le=1)
    mean_reprojection_iou: float = Field(ge=0, le=1)
    median_reprojection_iou: float = Field(ge=0, le=1)
    alignment_sufficient_for_lifting: bool
    diagnosed_bottleneck: Literal[
        "camera_mesh_alignment",
        "exact_face_granularity",
        "missing_or_hallucinated_geometry",
        "mixed_or_inconclusive",
    ]
    runtime_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    timings_seconds: dict[str, float]
    warnings: list[str] = Field(default_factory=list)


class ObjectSurfacePreviewManifest(StrictModel):
    global_face_assignment_path: Literal[
        "reconstruction/object_surfaces/previews/global_face_assignment.png"
    ]
    object_surface_contact_sheet_path: Literal[
        "reconstruction/object_surfaces/previews/object_surface_contact_sheet.png"
    ]
    reprojection_contact_sheet_path: Literal[
        "reconstruction/object_surfaces/previews/reprojection_contact_sheet.png"
    ]
    conflict_heatmap_path: Literal["reconstruction/object_surfaces/previews/conflict_heatmap.png"]
    global_mesh_depth_contact_sheet_path: Literal[
        "reconstruction/object_surfaces/previews/global_mesh_depth_contact_sheet.png"
    ]
    global_mesh_edge_overlay_path: Literal[
        "reconstruction/object_surfaces/previews/global_mesh_edge_overlay.png"
    ]
    sparse_point_vs_mesh_depth_path: Literal[
        "reconstruction/object_surfaces/previews/sparse_point_vs_mesh_depth.png"
    ]
    surface_sample_fusion_path: Literal[
        "reconstruction/object_surfaces/previews/surface_sample_fusion.png"
    ]
    object_preview_paths: dict[str, str]

    @field_validator("object_preview_paths")
    @classmethod
    def relative_object_previews(cls, values: dict[str, str]) -> dict[str, str]:
        return {object_id: _relative_artifact_path(path) for object_id, path in values.items()}


class ObjectSurfaceMethodMetrics(StrictModel):
    object_id: Annotated[str, Field(min_length=1)]
    method: Literal["exact_face_vote_v1", "surface_sample_fusion_v2"]
    accepted_faces: int = Field(ge=0)
    ambiguous_faces: int = Field(ge=0)
    component_count: int = Field(ge=0)
    surface_area_arbitrary_units_squared: float = Field(ge=0)
    reprojection_iou: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    supporting_views: int = Field(ge=0)
    runtime_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)


class ObjectSurfaceMethodComparison(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    selected_method: Literal["exact_face_vote_v1", "surface_sample_fusion_v2"]
    metrics: list[ObjectSurfaceMethodMetrics]
    conclusion: Annotated[str, Field(min_length=1)]
    warnings: list[str] = Field(default_factory=list)


class CameraMeshFrameAlignment(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    mesh_pixel_coverage: float = Field(ge=0, le=1)
    depth_finite_ratio: float = Field(ge=0, le=1)
    visible_global_face_count: int = Field(ge=0)
    depth_percentiles: dict[str, float]
    sparse_observation_count: int = Field(ge=0)
    normalized_depth_residual_median: float | None = Field(default=None, ge=0)
    normalized_depth_residual_p90: float | None = Field(default=None, ge=0)
    depth_inlier_fraction: float | None = Field(default=None, ge=0, le=1)


class CameraMeshAlignmentArtifact(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    global_mesh_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frames: list[CameraMeshFrameAlignment]
    mesh_pixel_coverage_mean: float = Field(ge=0, le=1)
    sparse_depth_residual_median: float | None = Field(default=None, ge=0)
    sparse_depth_residual_p90: float | None = Field(default=None, ge=0)
    sparse_depth_inlier_fraction: float | None = Field(default=None, ge=0, le=1)
    alignment_sufficient_for_lifting: bool
    diagnosis: Annotated[str, Field(min_length=1)]
    warnings: list[str] = Field(default_factory=list)


class Phase4ConsistencyCheck(StrictModel):
    check_id: Annotated[str, Field(min_length=1)]
    passed: bool
    message: Annotated[str, Field(min_length=1)]


class Phase4ConsistencyReport(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    passed: bool
    checks: list[Phase4ConsistencyCheck]
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    real_2d_tracks_lifted_to_global_3d: bool
    hidden_surface_completion_implemented: Literal[False] = False
    object_replacement_implemented: Literal[False] = False
    sim_ready_scene_implemented: Literal[False] = False
    metric_scale_known: Literal[False] = False
    canonical_gravity_alignment_known: Literal[False] = False
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def summary_matches_checks(self) -> Self:
        expected = all(check.passed for check in self.checks)
        if self.passed != expected:
            raise ValueError("Phase 4 report passed must equal all individual checks")
        return self


class TransformChainStage(StrictModel):
    stage_id: Annotated[str, Field(min_length=1)]
    transform_source: Annotated[str, Field(min_length=1)]
    matrix_from_previous: list[list[float]]
    matrix_to_previous: list[list[float]]
    determinant: float
    rotation_orthogonality_error: float = Field(ge=0)
    scale: float = Field(gt=0)
    translation: tuple[float, float, float]
    roundtrip_error: float = Field(ge=0)
    mesh_bounds_min: tuple[float, float, float] | None = None
    mesh_bounds_max: tuple[float, float, float] | None = None
    camera_center_bounds_min: tuple[float, float, float] | None = None
    camera_center_bounds_max: tuple[float, float, float] | None = None
    sparse_point_bounds_min: tuple[float, float, float] | None = None
    sparse_point_bounds_max: tuple[float, float, float] | None = None

    @field_validator("matrix_from_previous", "matrix_to_previous")
    @classmethod
    def finite_transform_stage_matrix(cls, value: list[list[float]]) -> list[list[float]]:
        import math

        if len(value) != 4 or any(len(row) != 4 for row in value):
            raise ValueError("transform-chain matrices must be 4x4")
        if any(not math.isfinite(component) for row in value for component in row):
            raise ValueError("transform-chain matrices must contain finite values")
        return value


class TransformChainAudit(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    status: Literal["consistent", "transform_chain_bug"]
    stages: Annotated[list[TransformChainStage], Field(min_length=1)]
    colmap_working_roundtrip_error: float = Field(ge=0)
    camera_basis_roundtrip_error: float = Field(ge=0)
    sampled_mesh_roundtrip_error: float = Field(ge=0)
    pre_post_render_depth_error: float | None = Field(default=None, ge=0)
    pre_post_render_silhouette_iou: float | None = Field(default=None, ge=0, le=1)
    pre_post_render_equivalent: bool
    raw_working_mesh_available: bool
    raw_working_scene_available: bool
    checks: dict[str, bool]
    warnings: list[str] = Field(default_factory=list)


class SparseDepthObservation(StrictModel):
    point3d_id: int = Field(gt=0)
    frame_id: Annotated[str, Field(min_length=1)]
    point2d_index: int = Field(ge=0)
    distorted_pixel: tuple[float, float]
    undistorted_pixel: tuple[float, float]
    point_world: tuple[float, float, float]
    camera_depth: float = Field(gt=0)
    colmap_reprojection_error: float = Field(ge=0)
    track_length: int = Field(gt=0)
    camera_model: Annotated[str, Field(min_length=1)]


class CameraUndistortionRecord(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    source_camera_model: Annotated[str, Field(min_length=1)]
    source_intrinsics: CameraIntrinsics
    source_distortion: list[float]
    undistorted_width: int = Field(gt=0)
    undistorted_height: int = Field(gt=0)
    undistorted_intrinsics: CameraIntrinsics
    roi_xywh: tuple[int, int, int, int]
    crop_policy: Literal["full_image_alpha_0"] = "full_image_alpha_0"
    map_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class SparseDepthObservationManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    observations: list[SparseDepthObservation]
    total_colmap_points: int = Field(ge=0)
    total_raw_observations: int = Field(ge=0)
    retained_observations: int = Field(ge=0)
    rejected_observations: int = Field(ge=0)
    filtering_configuration: dict[str, object]
    undistortion_records: list[CameraUndistortionRecord]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def observation_counts_match(self) -> Self:
        if self.retained_observations != len(self.observations):
            raise ValueError("retained sparse observation count must match records")
        if self.total_raw_observations != self.retained_observations + self.rejected_observations:
            raise ValueError("sparse observation accounting is inconsistent")
        return self


class AlignmentDatasetSplit(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    strategy: Literal["alternating_registered_frames_and_point_ids"]
    training_frame_ids: list[str]
    validation_frame_ids: list[str]
    training_point_ids: list[int]
    validation_point_ids: list[int]
    training_observation_count: int = Field(ge=0)
    validation_observation_count: int = Field(ge=0)
    split_seed: int

    @model_validator(mode="after")
    def split_is_disjoint(self) -> Self:
        if set(self.training_frame_ids) & set(self.validation_frame_ids):
            raise ValueError("alignment training and validation frames must be disjoint")
        if set(self.training_point_ids) & set(self.validation_point_ids):
            raise ValueError("alignment training and validation point IDs must be disjoint")
        return self


class AlignmentMetrics(StrictModel):
    observation_count: int = Field(ge=0)
    sparse_depth_residual_median: float | None = Field(default=None, ge=0)
    sparse_depth_residual_p75: float | None = Field(default=None, ge=0)
    sparse_depth_residual_p90: float | None = Field(default=None, ge=0)
    sparse_depth_residual_p95: float | None = Field(default=None, ge=0)
    log_depth_residual_median: float | None = Field(default=None, ge=0)
    inlier_fractions: dict[str, float]
    mesh_pixel_coverage: float = Field(ge=0, le=1)
    point_to_surface_median_scene_diagonal: float | None = Field(default=None, ge=0)
    point_to_surface_p90_scene_diagonal: float | None = Field(default=None, ge=0)
    point_to_plane_median_scene_diagonal: float | None = Field(default=None, ge=0)
    bad_frame_fraction: float = Field(ge=0, le=1)

    @field_validator("inlier_fractions")
    @classmethod
    def valid_alignment_inlier_fractions(cls, values: dict[str, float]) -> dict[str, float]:
        if any(value < 0 or value > 1 for value in values.values()):
            raise ValueError("alignment inlier fractions must be in [0, 1]")
        return values


class CameraAlignmentMetrics(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    camera_id: Annotated[str, Field(min_length=1)]
    valid_sparse_observations: int = Field(ge=0)
    mesh_pixel_coverage: float = Field(ge=0, le=1)
    baseline_median_residual: float | None = Field(default=None, ge=0)
    aligned_median_residual: float | None = Field(default=None, ge=0)
    baseline_p90_residual: float | None = Field(default=None, ge=0)
    aligned_p90_residual: float | None = Field(default=None, ge=0)
    baseline_inlier_fraction: float | None = Field(default=None, ge=0, le=1)
    aligned_inlier_fraction: float | None = Field(default=None, ge=0, le=1)
    visible_mesh_face_count: int = Field(ge=0)
    outlier: bool
    outlier_reason: str | None = None
    split: Literal["training", "validation"]


class ChunkAlignmentMetrics(StrictModel):
    chunk_id: Annotated[str, Field(min_length=1)]
    observation_count: int = Field(ge=0)
    baseline_median_residual: float | None = Field(default=None, ge=0)
    aligned_median_residual: float | None = Field(default=None, ge=0)
    aligned_p90_residual: float | None = Field(default=None, ge=0)
    aligned_inlier_fraction: float | None = Field(default=None, ge=0, le=1)


class AlignmentInitialization(StrictModel):
    initialization_id: Annotated[str, Field(min_length=1)]
    strategy: Literal[
        "identity",
        "robust_extent_scale",
        "centroid_alignment",
        "pca_axis_hypothesis",
    ]
    matrix: list[list[float]]
    initial_scale: float = Field(gt=0)
    initial_rotation_degrees: float = Field(ge=0)
    initial_translation_scene_diagonal_ratio: float = Field(ge=0)
    selected_for_optimization: bool
    rationale: Annotated[str, Field(min_length=1)]

    @field_validator("matrix")
    @classmethod
    def finite_initialization_matrix(cls, value: list[list[float]]) -> list[list[float]]:
        return TransformChainStage.finite_transform_stage_matrix(value)


class AlignmentIteration(StrictModel):
    candidate_id: Annotated[str, Field(min_length=1)]
    iteration: int = Field(ge=0)
    correspondence_count: int = Field(ge=0)
    inlier_count: int = Field(ge=0)
    loss: float = Field(ge=0)
    scale: float = Field(gt=0)
    rotation_degrees: float = Field(ge=0)
    translation_scene_diagonal_ratio: float = Field(ge=0)
    validation_point_to_surface_median: float | None = Field(default=None, ge=0)
    converged: bool


class AlignmentCandidate(StrictModel):
    candidate_id: Annotated[str, Field(min_length=1)]
    initialization_id: Annotated[str, Field(min_length=1)]
    matrix_original_mesh_to_aligned_colmap: list[list[float]]
    scale: float = Field(gt=0)
    rotation_degrees: float = Field(ge=0)
    translation_scene_diagonal_ratio: float = Field(ge=0)
    finite: bool
    hit_parameter_bound: bool
    correspondence_collapsed: bool
    training_metrics: AlignmentMetrics
    validation_metrics: AlignmentMetrics
    objective: float = Field(ge=0)
    selected: bool
    rejection_reason: str | None = None

    @field_validator("matrix_original_mesh_to_aligned_colmap")
    @classmethod
    def finite_candidate_matrix(cls, value: list[list[float]]) -> list[list[float]]:
        return TransformChainStage.finite_transform_stage_matrix(value)


class AlignmentCandidateManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    candidates: Annotated[list[AlignmentCandidate], Field(min_length=1)]

    @model_validator(mode="after")
    def one_selected_candidate(self) -> Self:
        if sum(candidate.selected for candidate in self.candidates) != 1:
            raise ValueError("exactly one alignment candidate must be selected")
        return self


class AlignmentIterationManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    iterations: list[AlignmentIteration]


class AlignmentTransform(StrictModel):
    matrix_original_mesh_to_aligned_colmap: list[list[float]]
    inverse_matrix: list[list[float]]
    scale: float = Field(gt=0)
    rotation_matrix: list[list[float]]
    rotation_axis_angle: tuple[float, float, float]
    rotation_degrees: float = Field(ge=0)
    translation: tuple[float, float, float]
    translation_scene_diagonal_ratio: float = Field(ge=0)
    determinant: float
    roundtrip_error: float = Field(ge=0)

    @field_validator("matrix_original_mesh_to_aligned_colmap", "inverse_matrix")
    @classmethod
    def finite_alignment_transform_matrix(cls, value: list[list[float]]) -> list[list[float]]:
        return TransformChainStage.finite_transform_stage_matrix(value)

    @field_validator("rotation_matrix")
    @classmethod
    def finite_rotation_matrix(cls, value: list[list[float]]) -> list[list[float]]:
        import math

        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError("alignment rotation must be a 3x3 matrix")
        if any(not math.isfinite(component) for row in value for component in row):
            raise ValueError("alignment rotation must contain finite values")
        return value


class CameraMeshAlignmentRequest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    run_id: Annotated[str, Field(min_length=1)]
    manifest_path: Literal["inputs/manifest.json"] = "inputs/manifest.json"
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_reconstruction_path: Literal["camera/reconstruction.json"] = "camera/reconstruction.json"
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    registered_frame_ids: Annotated[list[str], Field(min_length=1)]
    unregistered_frame_ids: list[str]
    coordinate_convention: CoordinateConvention
    camera_package_manifest_path: Literal["camera/genrecon_package/package_manifest.json"] = (
        "camera/genrecon_package/package_manifest.json"
    )
    camera_package_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    cameras_txt_path: Literal["camera/genrecon_package/cameras.txt"] = (
        "camera/genrecon_package/cameras.txt"
    )
    cameras_txt_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    images_txt_path: Literal["camera/genrecon_package/images.txt"] = (
        "camera/genrecon_package/images.txt"
    )
    images_txt_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    points3d_txt_path: Literal["camera/genrecon_package/points3D.txt"] = (
        "camera/genrecon_package/points3D.txt"
    )
    points3d_txt_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    global_reconstruction_path: Literal["reconstruction/global/metadata.json"] = (
        "reconstruction/global/metadata.json"
    )
    global_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    global_mesh_path: Literal["reconstruction/global/mesh.ply"] = "reconstruction/global/mesh.ply"
    global_mesh_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    global_worker_manifest_path: Literal["reconstruction/global/worker_manifest.json"] = (
        "reconstruction/global/worker_manifest.json"
    )
    global_worker_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    working_transform_path: str
    working_transform_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    chunk_transforms_path: str
    chunk_transforms_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    genrecon_camera_debug_path: str
    genrecon_camera_debug_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    working_mesh_path: str | None = None
    working_mesh_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    working_scene_path: str | None = None
    working_scene_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    audit_configuration: dict[str, object]
    sparse_observation_configuration: dict[str, object]
    mesh_sampling_configuration: dict[str, object]
    optimization_configuration: dict[str, object]
    acceptance_configuration: dict[str, object]
    output_directory: Literal["reconstruction/alignment"] = "reconstruction/alignment"
    seed: int

    @field_validator(
        "working_transform_path",
        "chunk_transforms_path",
        "genrecon_camera_debug_path",
        "working_mesh_path",
        "working_scene_path",
    )
    @classmethod
    def safe_alignment_request_paths(cls, value: str | None) -> str | None:
        return _relative_artifact_path(value) if value is not None else None

    @model_validator(mode="after")
    def consistent_alignment_request(self) -> Self:
        if set(self.registered_frame_ids) & set(self.unregistered_frame_ids):
            raise ValueError("alignment registration sets must not overlap")
        if (self.working_mesh_path is None) != (self.working_mesh_sha256 is None):
            raise ValueError("working mesh path and hash must be supplied together")
        if (self.working_scene_path is None) != (self.working_scene_sha256 is None):
            raise ValueError("working scene path and hash must be supplied together")
        return self


class CameraMeshAlignmentWorkerManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    worker_version: Annotated[str, Field(min_length=1)]
    backend: Literal["nvdiffrast_scipy", "fake"]
    python_version: Annotated[str, Field(min_length=1)]
    numpy_version: str | None = None
    scipy_version: str | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    nvdiffrast_version: str | None = None
    device: Annotated[str, Field(min_length=1)]
    device_name: str | None = None
    request_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_package_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    global_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    global_mesh_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    mesh_load_seconds: float = Field(ge=0)
    sparse_observation_seconds: float = Field(ge=0)
    baseline_render_seconds: float = Field(ge=0)
    correspondence_seconds: float = Field(ge=0)
    optimization_seconds: float = Field(ge=0)
    validation_render_seconds: float = Field(ge=0)
    preview_seconds: float = Field(ge=0)
    runtime_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    peak_host_memory_bytes: int | None = Field(default=None, ge=0)
    raw_output_paths: list[str]
    warnings: list[str] = Field(default_factory=list)

    @field_validator("raw_output_paths")
    @classmethod
    def safe_alignment_worker_outputs(cls, values: list[str]) -> list[str]:
        return [_relative_artifact_path(value) for value in values]


class CameraMeshAlignmentResult(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    status: Literal[
        "identity_already_consistent",
        "accepted_global_sim3",
        "rejected_no_validation_improvement",
        "rejected_implausible_transform",
        "global_sim3_insufficient",
        "transform_chain_bug_fixed",
        "generecon_geometry_inconsistent_with_colmap",
    ]
    accepted: bool
    transform: AlignmentTransform
    baseline_training_metrics: AlignmentMetrics
    aligned_training_metrics: AlignmentMetrics
    baseline_validation_metrics: AlignmentMetrics
    aligned_validation_metrics: AlignmentMetrics
    acceptance_checks: dict[str, bool]
    failure_reason: str | None = None
    coordinate_convention: CoordinateConvention
    scale_status: Literal[ScaleStatus.SCALE_AMBIGUOUS] = ScaleStatus.SCALE_AMBIGUOUS
    transform_chain_audit_path: Literal["reconstruction/alignment/transform_chain_audit.json"] = (
        "reconstruction/alignment/transform_chain_audit.json"
    )
    dataset_split_path: Literal["reconstruction/alignment/dataset_split.json"] = (
        "reconstruction/alignment/dataset_split.json"
    )
    candidate_id: Annotated[str, Field(min_length=1)]
    provenance: ProvenanceRecord
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def accepted_status_matches(self) -> Self:
        accepted_statuses = {
            "identity_already_consistent",
            "accepted_global_sim3",
            "transform_chain_bug_fixed",
        }
        if self.accepted != (self.status in accepted_statuses):
            raise ValueError("alignment acceptance must match its status")
        return self


class CameraMeshAlignmentDiagnostics(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    initializations: list[AlignmentInitialization]
    camera_metrics: list[CameraAlignmentMetrics]
    chunk_metrics: list[ChunkAlignmentMetrics]
    residual_is_locally_structured: bool
    candidate_solution_ambiguous: bool
    competing_candidate_ids: list[str]
    global_similarity_sufficient: bool
    transform_chain_consistent: bool
    camera_outlier_frame_ids: list[str]
    best_candidate_id: Annotated[str, Field(min_length=1)]
    diagnosis: Annotated[str, Field(min_length=1)]
    performance_seconds: dict[str, float]
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    peak_host_memory_bytes: int | None = Field(default=None, ge=0)
    warnings: list[str] = Field(default_factory=list)


class CameraMeshAlignmentPreviewManifest(StrictModel):
    transform_chain_comparison_path: Literal[
        "reconstruction/alignment/previews/transform_chain_comparison.png"
    ]
    baseline_depth_residual_path: Literal[
        "reconstruction/alignment/previews/baseline_depth_residual.png"
    ]
    aligned_depth_residual_path: Literal[
        "reconstruction/alignment/previews/aligned_depth_residual.png"
    ]
    baseline_vs_aligned_scatter_path: Literal[
        "reconstruction/alignment/previews/baseline_vs_aligned_scatter.png"
    ]
    per_camera_residuals_path: Literal["reconstruction/alignment/previews/per_camera_residuals.png"]
    per_chunk_residuals_path: Literal["reconstruction/alignment/previews/per_chunk_residuals.png"]
    sparse_points_and_mesh_before_path: Literal[
        "reconstruction/alignment/previews/sparse_points_and_mesh_before.png"
    ]
    sparse_points_and_mesh_after_path: Literal[
        "reconstruction/alignment/previews/sparse_points_and_mesh_after.png"
    ]
    heldout_validation_summary_path: Literal[
        "reconstruction/alignment/previews/heldout_validation_summary.png"
    ]


class ObjectLiftingAlignmentMetric(StrictModel):
    object_id: Annotated[str, Field(min_length=1)]
    baseline_status: Literal["accepted", "partial", "ambiguous", "unresolved"]
    aligned_status: Literal["accepted", "partial", "ambiguous", "unresolved"]
    baseline_accepted_faces: int = Field(ge=0)
    aligned_accepted_faces: int = Field(ge=0)
    baseline_ambiguous_faces: int = Field(ge=0)
    aligned_ambiguous_faces: int = Field(ge=0)
    baseline_components: int = Field(ge=0)
    aligned_components: int = Field(ge=0)
    baseline_surface_area: float = Field(ge=0)
    aligned_surface_area: float = Field(ge=0)
    baseline_precision: float = Field(ge=0, le=1)
    aligned_precision: float = Field(ge=0, le=1)
    baseline_recall: float = Field(ge=0, le=1)
    aligned_recall: float = Field(ge=0, le=1)
    baseline_iou: float = Field(ge=0, le=1)
    aligned_iou: float = Field(ge=0, le=1)
    baseline_association_confidence: float = Field(ge=0, le=1)
    aligned_association_confidence: float = Field(ge=0, le=1)
    baseline_supporting_views: int = Field(ge=0)
    aligned_supporting_views: int = Field(ge=0)


class ObjectLiftingAlignmentComparison(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    alignment_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    alignment_status: Annotated[str, Field(min_length=1)]
    alignment_accepted: bool
    objects: list[ObjectLiftingAlignmentMetric]
    baseline_scene_metrics: dict[str, float | int]
    aligned_scene_metrics: dict[str, float | int]
    conclusion: Annotated[str, Field(min_length=1)]
    warnings: list[str] = Field(default_factory=list)


class Phase4_2ConsistencyCheck(StrictModel):
    check_id: Annotated[str, Field(min_length=1)]
    passed: bool
    message: Annotated[str, Field(min_length=1)]


class Phase4_2ConsistencyReport(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    passed: bool
    checks: list[Phase4_2ConsistencyCheck]
    transform_chain_consistent: bool
    global_similarity_tested: Literal[True] = True
    global_similarity_accepted: bool
    global_similarity_sufficient: bool
    camera_poses_modified: Literal[False] = False
    mesh_topology_modified: Literal[False] = False
    metric_scale_known: Literal[False] = False
    canonical_gravity_alignment_known: Literal[False] = False
    hidden_surface_completion_implemented: Literal[False] = False
    sim_ready_scene_implemented: Literal[False] = False
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def phase4_2_summary_matches_checks(self) -> Self:
        if self.passed != all(check.passed for check in self.checks):
            raise ValueError("Phase 4.2 report passed must equal all checks")
        return self


class TrackObservation(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    bbox_xywh: tuple[int, int, int, int]
    mask_path: Annotated[str, Field(min_length=1)]
    confidence: ConfidenceRecord

    @field_validator("mask_path")
    @classmethod
    def validate_mask_path(cls, value: str) -> str:
        return _relative_artifact_path(value)

    @field_validator("bbox_xywh")
    @classmethod
    def valid_bbox(cls, value: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, width, height = value
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("bounding boxes require non-negative origins and positive dimensions")
        return value


class ObjectTrack(StrictModel):
    object_id: Annotated[str, Field(min_length=1)]
    name: Annotated[str, Field(min_length=1)]
    asset_type: AssetType
    observations: Annotated[list[TrackObservation], Field(min_length=1)]
    confidence: ConfidenceRecord
    provenance: ProvenanceRecord


class ObjectTracksArtifact(StrictModel):
    tracks: Annotated[list[ObjectTrack], Field(min_length=1)]
    provenance: ProvenanceRecord

    @model_validator(mode="after")
    def unique_tracks(self) -> Self:
        ids = [track.object_id for track in self.tracks]
        if len(ids) != len(set(ids)):
            raise ValueError("object track IDs must be unique")
        return self


class ReconstructedMaterial(StrictModel):
    name: Annotated[str, Field(min_length=1)]
    base_color_rgba: tuple[
        Annotated[float, Field(ge=0, le=1)],
        Annotated[float, Field(ge=0, le=1)],
        Annotated[float, Field(ge=0, le=1)],
        Annotated[float, Field(ge=0, le=1)],
    ]


class GlobalReconstructionArtifact(StrictModel):
    object_id: Literal["floor"] = "floor"
    name: Literal["floor"] = "floor"
    geometry_path: Annotated[str, Field(min_length=1)]
    collision_path: Annotated[str, Field(min_length=1)]
    format: Literal["obj"] = "obj"
    vertex_count: int = Field(gt=0)
    face_count: int = Field(gt=0)
    material: ReconstructedMaterial
    physics: PhysicsProperties
    confidence: ConfidenceRecord
    provenance: ProvenanceRecord

    @field_validator("geometry_path", "collision_path")
    @classmethod
    def validate_mesh_paths(cls, value: str) -> str:
        return _relative_artifact_path(value)


class ReconstructedPart(StrictModel):
    part_id: Annotated[str, Field(min_length=1)]
    name: Annotated[str, Field(min_length=1)]
    geometry_path: Annotated[str, Field(min_length=1)]
    collision_path: Annotated[str, Field(min_length=1)]
    format: Literal["obj"] = "obj"
    vertex_count: int = Field(gt=0)
    face_count: int = Field(gt=0)

    @field_validator("geometry_path", "collision_path")
    @classmethod
    def validate_mesh_paths(cls, value: str) -> str:
        return _relative_artifact_path(value)


class ReconstructedLink(StrictModel):
    link_id: Annotated[str, Field(min_length=1)]
    name: Annotated[str, Field(min_length=1)]
    part_ids: Annotated[list[str], Field(min_length=1)]


class ReconstructedJoint(StrictModel):
    joint_id: Annotated[str, Field(min_length=1)]
    parent_link_id: Annotated[str, Field(min_length=1)]
    child_link_id: Annotated[str, Field(min_length=1)]
    joint_type: Literal["fixed", "revolute", "prismatic"]
    axis_xyz: tuple[float, float, float]
    limits: tuple[float, float] | None = None


class ReconstructedArticulation(StrictModel):
    articulation_id: Annotated[str, Field(min_length=1)]
    links: Annotated[list[ReconstructedLink], Field(min_length=1)]
    joints: list[ReconstructedJoint]

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        link_ids = [link.link_id for link in self.links]
        joint_ids = [joint.joint_id for joint in self.joints]
        if len(link_ids) != len(set(link_ids)):
            raise ValueError("reconstructed link IDs must be unique")
        if len(joint_ids) != len(set(joint_ids)):
            raise ValueError("reconstructed joint IDs must be unique")
        known = set(link_ids)
        for joint in self.joints:
            missing = {joint.parent_link_id, joint.child_link_id} - known
            if missing:
                raise ValueError(f"reconstructed joint references unknown links: {sorted(missing)}")
        return self


class ObjectReconstructionResult(StrictModel):
    object_id: Annotated[str, Field(min_length=1)]
    name: Annotated[str, Field(min_length=1)]
    asset_type: AssetType
    parts: Annotated[list[ReconstructedPart], Field(min_length=1)]
    material: ReconstructedMaterial
    physics: PhysicsProperties
    articulation: ReconstructedArticulation | None = None
    confidence: ConfidenceRecord
    provenance: ProvenanceRecord

    @model_validator(mode="after")
    def validate_object_result(self) -> Self:
        part_ids = [part.part_id for part in self.parts]
        if len(part_ids) != len(set(part_ids)):
            raise ValueError("reconstructed part IDs must be unique")
        if self.asset_type is AssetType.ARTICULATED and self.articulation is None:
            raise ValueError("articulated reconstruction requires articulation metadata")
        if self.asset_type is not AssetType.ARTICULATED and self.articulation is not None:
            raise ValueError("non-articulated reconstruction cannot contain articulation metadata")
        if self.articulation is not None:
            linked_parts = {part for link in self.articulation.links for part in link.part_ids}
            missing = linked_parts - set(part_ids)
            if missing:
                raise ValueError(f"articulation links reference unknown parts: {sorted(missing)}")
        return self


class ObjectReconstructionArtifact(StrictModel):
    results: Annotated[list[ObjectReconstructionResult], Field(min_length=1)]
    provenance: ProvenanceRecord

    @model_validator(mode="after")
    def unique_results(self) -> Self:
        ids = [result.object_id for result in self.results]
        if len(ids) != len(set(ids)):
            raise ValueError("object reconstruction result IDs must be unique")
        return self


class CompiledScenePackage(StrictModel):
    source: Literal["mock"] = "mock"
    scene_ir_path: Annotated[str, Field(min_length=1)]
    exported_mesh_paths: list[str]
    simulator_outputs: list[str] = Field(default_factory=list)

    @field_validator("scene_ir_path")
    @classmethod
    def validate_scene_path(cls, value: str) -> str:
        return _relative_artifact_path(value)

    @field_validator("exported_mesh_paths", "simulator_outputs")
    @classmethod
    def validate_output_paths(cls, values: list[str]) -> list[str]:
        return [_relative_artifact_path(value) for value in values]


class ExportManifest(StrictModel):
    source: Literal["mock"] = "mock"
    compiled_package_path: Annotated[str, Field(min_length=1)]
    validation_report_path: Annotated[str, Field(min_length=1)]
    scene_ir_path: Annotated[str, Field(min_length=1)]

    @field_validator(
        "compiled_package_path",
        "validation_report_path",
        "scene_ir_path",
    )
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _relative_artifact_path(value)


class CommandResultArtifact(StrictModel):
    stage: Annotated[str, Field(min_length=1)]
    attempt: int = Field(gt=0)
    command: Annotated[list[str], Field(min_length=1)]
    return_code: int | None
    duration_s: float = Field(ge=0)
    timed_out: bool
    stdout_path: Annotated[str, Field(min_length=1)]
    stderr_path: Annotated[str, Field(min_length=1)]


# Phase 5A: measured dense geometry.  These models deliberately contain only
# serializable metadata so the lightweight core never needs a numerical runtime.
class DenseSparseModelFile(StrictModel):
    relative_path: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @field_validator("relative_path")
    @classmethod
    def safe_sparse_path(cls, value: str) -> str:
        return _relative_artifact_path(value)


class DenseMVSRequest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    run_id: Annotated[str, Field(min_length=1)]
    manifest_path: Literal["inputs/manifest.json"] = "inputs/manifest.json"
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    master_frame_order: Annotated[list[str], Field(min_length=1)]
    registered_frame_ids: Annotated[list[str], Field(min_length=1)]
    unregistered_frame_ids: list[str]
    normalized_frame_paths: dict[str, str]
    normalized_frame_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    camera_reconstruction_path: Literal["camera/reconstruction.json"] = "camera/reconstruction.json"
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    selected_sparse_model_files: Annotated[list[DenseSparseModelFile], Field(min_length=3)]
    official_colmap_repository: Literal["https://github.com/colmap/colmap"]
    official_colmap_version: Annotated[str, Field(min_length=1)]
    official_colmap_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    executable: Annotated[str, Field(min_length=1)]
    undistortion_configuration: dict[str, object]
    patchmatch_configuration: dict[str, object]
    fusion_configuration: dict[str, object]
    output_directory: Literal["reconstruction/dense"] = "reconstruction/dense"
    seed: int

    @field_validator("normalized_frame_paths")
    @classmethod
    def safe_dense_frame_paths(cls, value: dict[str, str]) -> dict[str, str]:
        return {frame_id: _relative_artifact_path(path) for frame_id, path in value.items()}

    @model_validator(mode="after")
    def consistent_dense_request(self) -> Self:
        master = set(self.master_frame_order)
        if len(master) != len(self.master_frame_order):
            raise ValueError("dense MVS master frame order must be unique")
        if set(self.registered_frame_ids) | set(self.unregistered_frame_ids) != master:
            raise ValueError("dense MVS registration sets must cover the master frames")
        if set(self.registered_frame_ids) & set(self.unregistered_frame_ids):
            raise ValueError("dense MVS registration sets must not overlap")
        if set(self.normalized_frame_paths) != master:
            raise ValueError("dense MVS frame paths must exactly cover the master frames")
        if set(self.normalized_frame_hashes) != master:
            raise ValueError("dense MVS frame hashes must exactly cover the master frames")
        ordered_registered = [
            frame_id
            for frame_id in self.master_frame_order
            if frame_id in self.registered_frame_ids
        ]
        if self.registered_frame_ids != ordered_registered:
            raise ValueError("dense MVS registered frames must follow manifest order")
        names = {
            PurePosixPath(item.relative_path).name for item in self.selected_sparse_model_files
        }
        if names != {"cameras.bin", "images.bin", "points3D.bin"}:
            raise ValueError("dense MVS requires exactly one complete selected sparse model")
        return self


class DenseFrameRecord(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    source_relative_path: str
    source_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    colmap_image_id: int = Field(gt=0)
    workspace_filename: str
    source_dimensions: tuple[int, int]
    dense_dimensions: tuple[int, int]
    dense_camera_id: int = Field(gt=0)
    dense_camera_model: Annotated[str, Field(min_length=1)]
    dense_intrinsics: tuple[float, float, float, float]

    @field_validator("source_relative_path", "workspace_filename")
    @classmethod
    def safe_dense_record_paths(cls, value: str) -> str:
        return _relative_artifact_path(value)


class DenseWorkspaceManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    selected_sparse_model_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    registered_frame_ids: Annotated[list[str], Field(min_length=1)]
    frames: Annotated[list[DenseFrameRecord], Field(min_length=1)]
    patch_match_config_path: str
    patch_match_config_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    workspace_path: Literal["reconstruction/dense/workspace"] = "reconstruction/dense/workspace"
    coordinate_convention: CoordinateConvention

    @field_validator("patch_match_config_path")
    @classmethod
    def safe_patchmatch_path(cls, value: str) -> str:
        return _relative_artifact_path(value)

    @model_validator(mode="after")
    def consistent_dense_frames(self) -> Self:
        if [frame.frame_id for frame in self.frames] != self.registered_frame_ids:
            raise ValueError("dense frame records must follow registered manifest order")
        if len(self.registered_frame_ids) != len(set(self.registered_frame_ids)):
            raise ValueError("dense registered frame IDs must be unique")
        return self


class DenseUndistortionRecord(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    source_camera_model: Annotated[str, Field(min_length=1)]
    source_intrinsics: tuple[float, float, float, float]
    source_distortion: list[float]
    source_dimensions: tuple[int, int]
    dense_camera_model: Literal["PINHOLE"]
    dense_intrinsics: tuple[float, float, float, float]
    dense_dimensions: tuple[int, int]
    roi_xywh: tuple[int, int, int, int] | None = None
    map_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    source_rgb_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    dense_rgb_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    rgb_remap_mean_absolute_error: float = Field(ge=0)
    mask_resampling: Literal["nearest"] = "nearest"


class DenseUndistortionManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    policy: Literal["official_colmap_image_undistorter"]
    records: Annotated[list[DenseUndistortionRecord], Field(min_length=1)]
    rgb_remap_tolerance: float = Field(ge=0)


class DenseDepthMapRecord(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    depth_path: str
    normal_path: str
    consistency_graph_path: str
    dimensions: tuple[int, int]
    depth_channels: Literal[1] = 1
    normal_channels: Literal[3] = 3
    positive_finite_depth_count: int = Field(ge=0)
    valid_depth_ratio: float = Field(ge=0, le=1)
    depth_percentiles: dict[str, float]
    finite_normal_ratio: float = Field(ge=0, le=1)
    consistency_valid_pixel_count: int = Field(ge=0)
    mean_consistency_source_count: float = Field(ge=0)
    median_consistency_source_count: float = Field(ge=0)
    source_view_ids: list[int]
    depth_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    normal_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    consistency_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    warnings: list[str] = Field(default_factory=list)

    @field_validator("depth_path", "normal_path", "consistency_graph_path")
    @classmethod
    def safe_dense_map_paths(cls, value: str) -> str:
        return _relative_artifact_path(value)


class DenseDepthManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    map_type: Literal["geometric"] = "geometric"
    records: list[DenseDepthMapRecord]
    failed_frame_ids: list[str] = Field(default_factory=list)


class DenseFusionArtifact(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    fused_point_cloud_path: Literal["reconstruction/dense/fused.ply"]
    fused_point_cloud_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    point_count: int = Field(gt=0)
    normal_count: int = Field(ge=0)
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    scene_diagonal_arbitrary_units: float = Field(gt=0)
    coordinate_convention: CoordinateConvention
    scale_status: ScaleStatus


class DenseMVSDiagnostics(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    registered_frame_count: int = Field(gt=0)
    successful_depth_map_count: int = Field(ge=0)
    failed_depth_map_count: int = Field(ge=0)
    fused_point_count: int = Field(gt=0)
    image_undistortion_seconds: float = Field(ge=0)
    patchmatch_seconds: float = Field(ge=0)
    fusion_seconds: float = Field(ge=0)
    total_runtime_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    peak_host_memory_bytes: int | None = Field(default=None, ge=0)
    warnings: list[str] = Field(default_factory=list)


class DenseMVSWorkerManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    worker_version: Annotated[str, Field(min_length=1)]
    official_colmap_repository: Literal["https://github.com/colmap/colmap"]
    official_colmap_version: Annotated[str, Field(min_length=1)]
    official_colmap_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    colmap_license: Annotated[str, Field(min_length=1)]
    build_configuration: dict[str, str]
    cuda_version: str | None = None
    compiler: str | None = None
    request_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    registered_frame_ids: Annotated[list[str], Field(min_length=1)]
    command_arguments: dict[str, list[str]]
    return_codes: dict[str, int]
    runtime_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    peak_host_memory_bytes: int | None = Field(default=None, ge=0)
    raw_output_paths: list[str]
    warnings: list[str] = Field(default_factory=list)

    @field_validator("raw_output_paths")
    @classmethod
    def safe_dense_worker_outputs(cls, values: list[str]) -> list[str]:
        return [_relative_artifact_path(value) for value in values]


class MeasuredObjectTrackRequest(StrictModel):
    object_id: Annotated[str, Field(min_length=1)]
    semantic_label: Annotated[str, Field(min_length=1)]
    prompt_id: Annotated[str, Field(min_length=1)]
    asset_type_hint: AssetType | None = None
    track_coverage: float = Field(ge=0, le=1)
    mask_paths_by_frame: dict[str, str]
    frame_scores: dict[str, float]

    @field_validator("mask_paths_by_frame")
    @classmethod
    def safe_measured_mask_paths(cls, value: dict[str, str]) -> dict[str, str]:
        return {frame_id: _relative_artifact_path(path) for frame_id, path in value.items()}


class MeasuredObjectGeometryRequest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    run_id: Annotated[str, Field(min_length=1)]
    manifest_path: Literal["inputs/manifest.json"] = "inputs/manifest.json"
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_reconstruction_path: Literal["camera/reconstruction.json"] = "camera/reconstruction.json"
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    segmentation_tracking_path: Literal["observations/object_tracks.json"] = (
        "observations/object_tracks.json"
    )
    segmentation_tracking_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    object_tracks: list[MeasuredObjectTrackRequest]
    dense_workspace_manifest_path: Literal["reconstruction/dense/workspace_manifest.json"]
    dense_workspace_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    undistortion_manifest_path: Literal["reconstruction/dense/undistortion_manifest.json"]
    undistortion_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    depth_manifest_path: Literal["reconstruction/dense/depth_manifest.json"]
    depth_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    backprojection_configuration: dict[str, object]
    consistency_configuration: dict[str, object]
    surfel_fusion_configuration: dict[str, object]
    observed_mesh_configuration: dict[str, object]
    reprojection_configuration: dict[str, object]
    coordinate_convention: CoordinateConvention
    output_directory: Literal["reconstruction/measured_objects"] = "reconstruction/measured_objects"
    seed: int


class MeasuredObjectObservation(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    registered: Literal[True] = True
    raw_sample_count: int = Field(ge=0)
    validated_sample_count: int = Field(ge=0)
    supporting_view_count: int = Field(ge=0)
    contradicting_view_count: int = Field(ge=0)
    depth_residual_median: float | None = Field(default=None, ge=0)
    mask_support_fraction: float = Field(ge=0, le=1)


class MeasuredPointCloudManifest(StrictModel):
    relative_path: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    point_count: int = Field(gt=0)
    has_normals: bool
    has_colors: bool

    @field_validator("relative_path")
    @classmethod
    def safe_measured_cloud_path(cls, value: str) -> str:
        return _relative_artifact_path(value)


class ObservedSurfaceMeshManifest(StrictModel):
    relative_path: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    vertex_count: int = Field(gt=0)
    face_count: int = Field(gt=0)
    surface_type: Literal["observed_depth_triangulation"]
    watertight: Literal[False] = False

    @field_validator("relative_path")
    @classmethod
    def safe_observed_mesh_path(cls, value: str) -> str:
        return _relative_artifact_path(value)


class MeasuredSurfelSpacingDiagnostics(StrictModel):
    method: Literal["coordinate_hash_kdtree_nearest_neighbor_v1"]
    source_point_count: int = Field(gt=0)
    sampled_point_count: int = Field(gt=0)
    nearest_neighbor_p10: float = Field(gt=0)
    nearest_neighbor_median: float = Field(gt=0)
    nearest_neighbor_p90: float = Field(gt=0)
    voxel_size: float = Field(gt=0)
    coordinate_hash_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class MeasuredObjectHypothesis(StrictModel):
    object_id: Annotated[str, Field(min_length=1)]
    semantic_label: Annotated[str, Field(min_length=1)]
    prompt_id: Annotated[str, Field(min_length=1)]
    asset_type_hint: AssetType | None = None
    status: Literal["accepted", "partial", "unresolved"]
    reason: str | None = None
    registered_mask_observations: int = Field(ge=0)
    observations_with_valid_dense_depth: int = Field(ge=0)
    raw_measured_sample_count: int = Field(ge=0)
    validated_sample_count: int = Field(ge=0)
    fused_surfel_count: int = Field(ge=0)
    supporting_view_count: int = Field(ge=0)
    point_cloud: MeasuredPointCloudManifest | None = None
    surfel_cloud: MeasuredPointCloudManifest | None = None
    observed_surface: ObservedSurfaceMeshManifest | None = None
    observations: list[MeasuredObjectObservation]
    depth_consistency: float = Field(ge=0, le=1)
    normal_consistency: float = Field(ge=0, le=1)
    reprojection_precision: float = Field(ge=0, le=1)
    reprojection_recall: float = Field(ge=0, le=1)
    reprojection_iou: float = Field(ge=0, le=1)
    visible_mask_coverage: float = Field(ge=0, le=1)
    connected_component_count: int = Field(ge=0)
    measurement_confidence: float = Field(ge=0, le=1)
    completeness_confidence: float = Field(default=0.0, ge=0, le=0)
    surfel_spacing: MeasuredSurfelSpacingDiagnostics | None = None
    geometry_source: Literal["measured"] = "measured"
    geometry_status: Literal["partial_measured"] = "partial_measured"
    hidden_surface_completion: Literal["not_implemented"] = "not_implemented"
    watertight: Literal[False] = False
    sim_ready: Literal[False] = False
    metric_scale_known: Literal[False] = False
    canonical_gravity_alignment_known: Literal[False] = False
    coordinate_convention: CoordinateConvention
    scale_status: ScaleStatus
    provenance: ProvenanceRecord
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_measured_geometry(self) -> Self:
        if self.status == "unresolved":
            if self.point_cloud is not None or self.surfel_cloud is not None:
                raise ValueError("unresolved measured objects cannot contain geometry")
            if not self.reason:
                raise ValueError("unresolved measured objects require a reason")
        elif self.point_cloud is None or self.surfel_cloud is None:
            raise ValueError("accepted or partial measured objects require point and surfel clouds")
        return self


class MeasuredObjectGeometryArtifact(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    segmentation_tracking_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    dense_workspace_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    undistortion_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    depth_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    hypotheses: list[MeasuredObjectHypothesis]
    coordinate_convention: CoordinateConvention
    scale_status: ScaleStatus
    generated_geometry_used_as_source: Literal[False] = False


class MeasuredObjectDiagnostics(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    track_count: int = Field(ge=0)
    accepted_object_count: int = Field(ge=0)
    partial_object_count: int = Field(ge=0)
    unresolved_object_count: int = Field(ge=0)
    raw_sample_count: int = Field(ge=0)
    validated_sample_count: int = Field(ge=0)
    fused_surfel_count: int = Field(ge=0)
    mask_mapping_seconds: float = Field(ge=0)
    backprojection_seconds: float = Field(ge=0)
    multiview_validation_seconds: float = Field(ge=0)
    surfel_fusion_seconds: float = Field(ge=0)
    observed_mesh_seconds: float = Field(ge=0)
    preview_seconds: float = Field(ge=0)
    total_runtime_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    peak_host_memory_bytes: int | None = Field(default=None, ge=0)
    warnings: list[str] = Field(default_factory=list)


class MeasuredObjectWorkerManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    worker_version: Annotated[str, Field(min_length=1)]
    backend: Literal["fake", "numpy_opencv"]
    request_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    segmentation_tracking_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    depth_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    runtime_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    peak_host_memory_bytes: int | None = Field(default=None, ge=0)
    raw_output_paths: list[str]
    warnings: list[str] = Field(default_factory=list)

    @field_validator("raw_output_paths")
    @classmethod
    def safe_measured_worker_outputs(cls, values: list[str]) -> list[str]:
        return [_relative_artifact_path(value) for value in values]


class MeasuredGeneratedObjectComparison(StrictModel):
    object_id: Annotated[str, Field(min_length=1)]
    point_to_genrecon_surface_median: float | None = Field(default=None, ge=0)
    rendered_depth_discrepancy: float | None = Field(default=None, ge=0)
    reprojection_precision: float = Field(ge=0, le=1)
    reprojection_recall: float = Field(ge=0, le=1)
    measured_surface_covered_by_genrecon: float | None = Field(default=None, ge=0, le=1)
    genrecon_hypothesis_covered_by_measurement: float | None = Field(default=None, ge=0, le=1)
    diagnosis: Literal[
        "not_computed",
        "consistent",
        "genrecon_missing_object_geometry",
        "genrecon_displaced_geometry",
        "genrecon_over_completed_geometry",
        "mvs_insufficient_geometry",
    ]


class MeasuredGeneratedComparisonArtifact(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    measured_geometry_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    global_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    objects: list[MeasuredGeneratedObjectComparison]
    diagnostic_only: Literal[True] = True


class Phase5AConsistencyReport(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    passed: bool
    checks: list[EndToEndConsistencyCheck]
    measured_dense_geometry_available: bool
    measured_object_geometry_available: bool
    generated_geometry_used_as_source: Literal[False] = False
    hidden_surface_completion_implemented: Literal[False] = False
    object_replacement_implemented: Literal[False] = False
    sim_ready_scene_implemented: Literal[False] = False
    metric_scale_known: Literal[False] = False
    canonical_gravity_alignment_known: Literal[False] = False
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def phase5a_summary_matches_checks(self) -> Self:
        if self.passed != all(check.passed for check in self.checks):
            raise ValueError("Phase 5A pass status must match its checks")
        return self


# Phase 5B: observation-grounded rigid visual completion. Generated assets are
# always distinct from their measured anchors and never imply physical validity.
class CompletionEligibilityStatus(StrEnum):
    ELIGIBLE_RIGID = "eligible_rigid"
    ELIGIBLE_STATIC = "eligible_static"
    DEFERRED_ARTICULATED = "deferred_articulated"
    DEFERRED_DEFORMABLE = "deferred_deformable"
    DEFERRED_FLUID = "deferred_fluid"
    DEFERRED_HUMAN = "deferred_human"
    DEFERRED_UNKNOWN = "deferred_unknown"


class CompletionBackend(StrEnum):
    SAM3D_OBJECTS = "sam3d_objects"
    TRELLIS2 = "trellis2"
    MEASURED_PARTIAL_BASELINE = "measured_partial_baseline"


class CompletionLicenseMode(StrEnum):
    RESEARCH_EVALUATION = "research_evaluation"
    PRODUCTION_CANDIDATE = "production_candidate"


class CandidateNativeFormat(StrEnum):
    MESH_PLY = "mesh_ply"
    MESH_GLB = "mesh_glb"
    PBR_GLB = "pbr_glb"
    GAUSSIAN_SPLAT_PLY = "gaussian_splat_ply"
    NATIVE_BACKEND_BUNDLE = "native_backend_bundle"


class CompletionLicenseRecord(StrictModel):
    backend: CompletionBackend
    code_license: Annotated[str, Field(min_length=1)]
    checkpoint_license: Annotated[str, Field(min_length=1)]
    dependency_licenses: dict[str, str] = Field(default_factory=dict)
    asset_license: Annotated[str, Field(min_length=1)]
    access_conditions: list[str] = Field(default_factory=list)
    commercial_use_review_status: Literal[
        "not_reviewed",
        "research_only",
        "approved_by_project_policy",
    ]
    research_evaluation_allowed: bool
    production_selectable: bool


class CompletionEligibilityRecord(StrictModel):
    object_id: Annotated[str, Field(min_length=1)]
    semantic_label: Annotated[str, Field(min_length=1)]
    asset_type_hint: AssetType | None = None
    status: CompletionEligibilityStatus
    explicitly_overridden: bool = False
    reason: Annotated[str, Field(min_length=1)]


class CompletionEligibilityArtifact(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    segmentation_tracking_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    measured_geometry_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    records: list[CompletionEligibilityRecord]

    @model_validator(mode="after")
    def unique_completion_objects(self) -> Self:
        ids = [record.object_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("completion eligibility object IDs must be unique")
        return self


class CompletionObjectEvidenceSplit(StrictModel):
    object_id: Annotated[str, Field(min_length=1)]
    generation_anchor_frames: list[str]
    registration_fitting_frames: list[str]
    heldout_validation_frames: list[str]
    degraded_split: bool = False
    limitation: str | None = None

    @model_validator(mode="after")
    def disjoint_completion_evidence(self) -> Self:
        groups = [
            self.generation_anchor_frames,
            self.registration_fitting_frames,
            self.heldout_validation_frames,
        ]
        if any(len(group) != len(set(group)) for group in groups):
            raise ValueError("completion evidence groups must not contain duplicate frames")
        if any(
            set(left) & set(right)
            for index, left in enumerate(groups)
            for right in groups[index + 1 :]
        ):
            raise ValueError("completion generation, fitting, and held-out frames must be disjoint")
        return self


class CompletionEvidenceSplit(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    split_version: Literal["disjoint_object_views_v1"] = "disjoint_object_views_v1"
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    objects: list[CompletionObjectEvidenceSplit]
    seed: int


class CompletionAnchorRecord(StrictModel):
    object_id: Annotated[str, Field(min_length=1)]
    frame_id: Annotated[str, Field(min_length=1)]
    rank: int = Field(gt=0)
    selection_score: float
    camera_direction: tuple[float, float, float]
    mask_bbox_xywh: tuple[int, int, int, int]
    mask_area_ratio: float = Field(gt=0, le=1)
    dense_valid_ratio: float = Field(ge=0, le=1)
    measured_sample_count: int = Field(ge=0)
    selection_reason: Annotated[str, Field(min_length=1)]
    crop_path: str
    crop_metadata_path: str
    crop_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    source_frame_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    source_mask_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    crop_to_source_transform: tuple[float, float, float, float, float, float, float, float, float]
    source_to_crop_transform: tuple[float, float, float, float, float, float, float, float, float]

    @field_validator("crop_path", "crop_metadata_path")
    @classmethod
    def safe_completion_crop_paths(cls, value: str) -> str:
        return _relative_artifact_path(value)


class CompletionCropManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    output_size: int = Field(gt=0)
    margin_ratio: float = Field(ge=0)
    padding_mode: Literal["transparent"] = "transparent"
    anchors: list[CompletionAnchorRecord]


class CompletionTrainingFrameRecord(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    raw_sample_count: int = Field(ge=0)
    backprojected_point_count: int = Field(ge=0)
    validated_point_count: int = Field(ge=0)
    maximum_supporting_views: int = Field(ge=0)
    median_relative_depth_residual: float | None = Field(default=None, ge=0)


class CompletionTrainingMeasuredGeometry(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    object_id: Annotated[str, Field(min_length=1)]
    training_frame_ids: list[str]
    heldout_frame_ids: list[str]
    raw_sample_count: int = Field(ge=0)
    boundary_rejected_count: int = Field(ge=0)
    invalid_geometry_rejected_count: int = Field(ge=0)
    sam_score_rejected_count: int = Field(ge=0)
    consistency_rejected_count: int = Field(ge=0)
    depth_discontinuity_rejected_count: int = Field(ge=0)
    multi_view_rejected_count: int = Field(ge=0)
    pre_cap_validated_point_count: int = Field(ge=0)
    validated_point_count: int = Field(ge=0)
    maximum_samples_per_object: int = Field(gt=0)
    sampling_cap_applied: bool
    supporting_fitting_views: list[str]
    point_cloud_path: str
    point_cloud_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    normal_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    renderer_control_mesh_path: str | None = None
    renderer_control_mesh_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    renderer_control_face_count: int = Field(default=0, ge=0)
    renderer_control_triangle_radius: float | None = Field(default=None, gt=0)
    phase5a_all_view_validated_point_count: int = Field(ge=0)
    phase5a_point_cloud_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    frame_records: list[CompletionTrainingFrameRecord]
    backprojection_configuration: dict[str, object]
    consistency_configuration: dict[str, object]

    @field_validator("point_cloud_path", "renderer_control_mesh_path")
    @classmethod
    def safe_training_point_cloud_path(cls, value: str) -> str:
        return _relative_artifact_path(value)

    @model_validator(mode="after")
    def fitting_and_heldout_are_disjoint(self) -> Self:
        if set(self.training_frame_ids) & set(self.heldout_frame_ids):
            raise ValueError("training measured geometry contains held-out frames")
        frame_total = sum(record.validated_point_count for record in self.frame_records)
        if self.pre_cap_validated_point_count != frame_total:
            raise ValueError("pre-cap training point count does not match frame records")
        expected_count = min(
            self.pre_cap_validated_point_count,
            self.maximum_samples_per_object,
        )
        if self.validated_point_count != expected_count:
            raise ValueError("post-cap training point count is inconsistent")
        if self.sampling_cap_applied != (
            self.pre_cap_validated_point_count > self.maximum_samples_per_object
        ):
            raise ValueError("training point sampling-cap status is inconsistent")
        return self


class CompletionTrainingEvidence(StrictModel):
    object_id: Annotated[str, Field(min_length=1)]
    training_frame_ids: list[str]
    heldout_frame_ids: list[str]
    training_points_path: str | None = None
    training_points_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    training_point_count: int = Field(ge=0)
    training_normals_available: bool = False
    training_geometry_manifest_path: str
    training_geometry_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    renderer_control_mesh_path: str | None = None
    renderer_control_mesh_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    heldout_measurement_manifest_path: str
    heldout_measurement_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @field_validator(
        "training_points_path",
        "training_geometry_manifest_path",
        "renderer_control_mesh_path",
        "heldout_measurement_manifest_path",
    )
    @classmethod
    def safe_completion_evidence_paths(cls, value: str | None) -> str | None:
        return _relative_artifact_path(value) if value is not None else None


class CompletionEvidencePackage(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    segmentation_tracking_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    dense_depth_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    measured_geometry_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    evidence_split_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    crop_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    objects: list[CompletionTrainingEvidence]
    coordinate_convention: CoordinateConvention
    scale_status: ScaleStatus


class CompletionEvidencePreparationRequest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    manifest_path: Literal["inputs/manifest.json"] = "inputs/manifest.json"
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_reconstruction_path: Literal["camera/reconstruction.json"]
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    segmentation_tracking_path: Literal["observations/object_tracks.json"]
    segmentation_tracking_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    dense_depth_manifest_path: Literal["reconstruction/dense/depth_manifest.json"]
    dense_depth_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    dense_undistortion_manifest_path: Literal["reconstruction/dense/undistortion_manifest.json"]
    dense_undistortion_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    measured_geometry_path: Literal["reconstruction/measured_objects/geometry_manifest.json"]
    measured_geometry_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    measured_geometry_request_path: Literal["reconstruction/measured_objects/request.json"]
    measured_geometry_request_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    evidence_split_path: Literal["reconstruction/completion/evidence_split.json"]
    evidence_split_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    crop_manifest_path: Literal["reconstruction/completion/crop_manifest.json"]
    crop_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    object_inputs: dict[str, dict[str, object]]
    backprojection_configuration: dict[str, object]
    consistency_configuration: dict[str, object]
    coordinate_convention: CoordinateConvention
    output_directory: Literal["reconstruction/completion/evidence"]
    seed: int


class CompletionWorkerManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    worker_name: Annotated[str, Field(min_length=1)]
    worker_version: Annotated[str, Field(min_length=1)]
    action: Annotated[str, Field(min_length=1)]
    backend: Annotated[str, Field(min_length=1)]
    request_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    official_repository: str | None = None
    official_code_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    checkpoint_repository: str | None = None
    checkpoint_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    checkpoint_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]] = Field(
        default_factory=dict
    )
    runtime_model_revisions: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]] = Field(
        default_factory=dict
    )
    runtime_model_hashes: dict[str, dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]] = (
        Field(default_factory=dict)
    )
    runtime_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    peak_host_memory_bytes: int | None = Field(default=None, ge=0)
    warnings: list[str] = Field(default_factory=list)


class ObjectCompletionCandidateRequest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    run_id: Annotated[str, Field(min_length=1)]
    object_id: Annotated[str, Field(min_length=1)]
    semantic_label: Annotated[str, Field(min_length=1)]
    asset_type_hint: AssetType | None = None
    eligibility_status: CompletionEligibilityStatus
    backend: CompletionBackend
    official_repository: Annotated[str, Field(min_length=1)]
    official_code_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    checkpoint_repository: Annotated[str, Field(min_length=1)]
    checkpoint_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    checkpoint_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    runtime_model_revisions: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]] = Field(
        default_factory=dict
    )
    runtime_model_hashes: dict[str, dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]] = (
        Field(default_factory=dict)
    )
    license_policy: CompletionLicenseRecord
    anchor_frame_id: Annotated[str, Field(min_length=1)]
    anchor_crop_path: str
    anchor_crop_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    anchor_crop_transform: tuple[float, float, float, float, float, float, float, float, float]
    source_frame_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    source_mask_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    generation_seed: int
    generation_configuration: dict[str, object]
    output_directory: str

    @field_validator("anchor_crop_path", "output_directory")
    @classmethod
    def safe_candidate_request_paths(cls, value: str) -> str:
        return _relative_artifact_path(value)


class CandidateNativeAsset(StrictModel):
    asset_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
    relative_path: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    format: CandidateNativeFormat
    size_bytes: int = Field(gt=0)
    role: Annotated[str, Field(min_length=1)]

    @field_validator("relative_path")
    @classmethod
    def safe_native_candidate_path(cls, value: str) -> str:
        return _relative_artifact_path(value)


class CandidateRenderCapability(StrictModel):
    renderer: Annotated[str, Field(min_length=1)]
    supports_rgba: bool
    supports_depth: bool
    supports_normals: bool = False
    camera_axes: Literal["x_right_y_down_z_forward"]


class CandidateBackendAnchorCamera(StrictModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    normalized_intrinsics: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]
    pixel_intrinsics: tuple[float, float, float, float]
    camera_axes: Literal["x_right_y_down_z_forward"]
    source: Literal["official_pointmap_intrinsics"]


class ObjectCompletionCandidate(StrictModel):
    candidate_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
    object_id: Annotated[str, Field(min_length=1)]
    semantic_label: Annotated[str, Field(min_length=1)]
    backend: CompletionBackend
    anchor_frame_id: Annotated[str, Field(min_length=1)]
    generation_seed: int
    native_assets: Annotated[list[CandidateNativeAsset], Field(min_length=1)]
    registration_asset_id: Annotated[str, Field(min_length=1)]
    registration_asset_path: str
    evaluation_asset_id: Annotated[str, Field(min_length=1)]
    evaluation_asset_path: str
    selection_asset_id: Annotated[str, Field(min_length=1)]
    selection_asset_path: str
    native_coordinate_convention: Annotated[str, Field(min_length=1)]
    native_bounds_min: tuple[float, float, float] | None = None
    native_bounds_max: tuple[float, float, float] | None = None
    native_center: tuple[float, float, float] | None = None
    native_scale: float | None = Field(default=None, gt=0)
    vertex_count: int | None = Field(default=None, ge=0)
    face_count: int | None = Field(default=None, ge=0)
    material_count: int | None = Field(default=None, ge=0)
    texture_count: int | None = Field(default=None, ge=0)
    gaussian_count: int | None = Field(default=None, ge=0)
    backend_predicted_layout: dict[str, object] = Field(default_factory=dict)
    backend_anchor_camera: CandidateBackendAnchorCamera | None = None
    render_capability: CandidateRenderCapability
    sampling_method: Annotated[str, Field(min_length=1)]
    generation_runtime_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    license_record: CompletionLicenseRecord
    provenance: ProvenanceRecord
    warnings: list[str] = Field(default_factory=list)

    @field_validator(
        "registration_asset_path",
        "evaluation_asset_path",
        "selection_asset_path",
    )
    @classmethod
    def safe_candidate_selected_asset_paths(cls, value: str) -> str:
        return _relative_artifact_path(value)

    @model_validator(mode="after")
    def declared_assets_exist(self) -> Self:
        assets = {asset.asset_id: asset.relative_path for asset in self.native_assets}
        declared = (
            (self.registration_asset_id, self.registration_asset_path),
            (self.evaluation_asset_id, self.evaluation_asset_path),
            (self.selection_asset_id, self.selection_asset_path),
        )
        if any(assets.get(asset_id) != path for asset_id, path in declared):
            raise ValueError("candidate registration/evaluation/selection assets must be native")
        return self


class CandidateGenerationManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    backend: CompletionBackend
    official_repository: Annotated[str, Field(min_length=1)]
    official_code_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    checkpoint_repository: Annotated[str, Field(min_length=1)]
    checkpoint_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    checkpoint_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    runtime_model_revisions: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]] = Field(
        default_factory=dict
    )
    runtime_model_hashes: dict[str, dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]] = (
        Field(default_factory=dict)
    )
    evidence_split_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    crop_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    requests: list[ObjectCompletionCandidateRequest]
    candidates: list[ObjectCompletionCandidate]
    failed_candidate_ids: list[str] = Field(default_factory=list)
    runtime_seconds: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class CandidateRegistrationRequest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    evidence_package_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    generation_manifest_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    candidate_ids: list[str]
    camera_reconstruction_path: Literal["camera/reconstruction.json"]
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    dense_undistortion_manifest_path: Literal["reconstruction/dense/undistortion_manifest.json"]
    dense_undistortion_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    fitting_inputs: dict[str, dict[str, object]]
    registration_configuration: dict[str, object]
    output_directory: Literal["reconstruction/completion"]
    seed: int


class CandidateTransformHypothesis(StrictModel):
    matrix_world_from_candidate: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]
    inverse_matrix: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]
    scale: float = Field(gt=0)
    rotation_determinant: float
    rotation_degrees: float = Field(ge=0, le=180)
    translation: tuple[float, float, float]
    measured_surface_median_residual: float = Field(ge=0)
    measured_surface_p90_residual: float = Field(ge=0)
    normal_agreement: float = Field(ge=-1, le=1)
    symmetry_ambiguous: bool = False
    fitting_refined: bool = False
    fitting_objective_before: float | None = Field(default=None, ge=0)
    fitting_objective_after: float | None = Field(default=None, ge=0)


class CandidateRegistrationArtifact(StrictModel):
    candidate_id: Annotated[str, Field(min_length=1)]
    object_id: Annotated[str, Field(min_length=1)]
    registration_asset_id: Annotated[str, Field(min_length=1)]
    registration_asset_path: str
    status: Literal["registered", "symmetry_ambiguous", "registration_failed"]
    frozen_transform: CandidateTransformHypothesis | None = None
    fitting_frame_ids: list[str]
    heldout_frame_ids: list[str]
    training_points_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    fitting_objective: float | None = Field(default=None, ge=0)
    failure_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("registration_asset_path")
    @classmethod
    def safe_registration_asset_path(cls, value: str) -> str:
        return _relative_artifact_path(value)

    @model_validator(mode="after")
    def registration_status_matches_transform(self) -> Self:
        if self.status == "registration_failed" and self.frozen_transform is not None:
            raise ValueError("failed registration cannot contain a frozen transform")
        if self.status != "registration_failed" and self.frozen_transform is None:
            raise ValueError("successful registration requires a frozen transform")
        if set(self.fitting_frame_ids) & set(self.heldout_frame_ids):
            raise ValueError("registration fitting and held-out frames must be disjoint")
        return self


class CandidateRegistrationManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    request_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    registrations: list[CandidateRegistrationArtifact]
    runtime_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    peak_host_memory_bytes: int | None = Field(default=None, ge=0)


class CandidateEvaluationRequest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    registration_manifest_path: Literal["reconstruction/completion/registration_manifest.json"]
    registration_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    evidence_package_path: Literal["reconstruction/completion/evidence/evidence_package.json"]
    evidence_package_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    evidence_split_path: Literal["reconstruction/completion/evidence_split.json"]
    evidence_split_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    generation_manifest_paths: dict[str, str]
    generation_manifest_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    segmentation_tracking_path: Literal["observations/object_tracks.json"]
    segmentation_tracking_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_reconstruction_path: Literal["camera/reconstruction.json"]
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    dense_depth_manifest_path: Literal["reconstruction/dense/depth_manifest.json"]
    dense_depth_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    dense_undistortion_manifest_path: Literal["reconstruction/dense/undistortion_manifest.json"]
    dense_undistortion_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    anchor_inputs: dict[str, dict[str, object]]
    fitting_inputs: dict[str, dict[str, object]]
    heldout_inputs: dict[str, dict[str, object]]
    evaluation_configuration: dict[str, object]
    output_directory: Literal["reconstruction/completion"]
    seed: int

    @field_validator("generation_manifest_paths")
    @classmethod
    def safe_evaluation_manifest_paths(cls, values: dict[str, str]) -> dict[str, str]:
        return {name: _relative_artifact_path(path) for name, path in values.items()}


class CandidateHeldoutMetrics(StrictModel):
    mask_precision: float = Field(ge=0, le=1)
    mask_recall: float = Field(ge=0, le=1)
    mask_iou: float = Field(ge=0, le=1)
    per_frame_iou: dict[str, float]
    dense_depth_relative_residual: float = Field(ge=0)
    depth_inlier_fraction: float = Field(ge=0, le=1)
    negative_space_violation_ratio: float = Field(ge=0, le=1)
    front_of_scene_violation_ratio: float = Field(ge=0, le=1)
    measured_point_to_candidate_median: float = Field(ge=0)
    measured_point_to_candidate_p90: float = Field(ge=0)
    normal_agreement: float = Field(ge=-1, le=1)
    candidate_visible_coverage: float = Field(ge=0, le=1)
    validation_view_count: int = Field(ge=0)
    visible_candidate_area: int = Field(ge=0)
    occluded_candidate_area: int = Field(ge=0)
    negative_space_violation_area: int = Field(ge=0)
    front_of_scene_violation_area: int = Field(ge=0)


class CandidateFrameRenderDiagnostic(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    raw_candidate_pixel_count: int = Field(ge=0)
    visible_pixel_count: int = Field(ge=0)
    occluded_pixel_count: int = Field(ge=0)
    negative_space_pixel_count: int = Field(ge=0)
    front_of_scene_pixel_count: int = Field(ge=0)
    candidate_depth_min: float | None = None
    candidate_depth_median: float | None = None
    candidate_depth_max: float | None = None
    scene_depth_min: float | None = None
    scene_depth_median: float | None = None
    scene_depth_max: float | None = None
    candidate_projected_bbox: tuple[int, int, int, int] | None = None
    target_mask_bbox: tuple[int, int, int, int] | None = None
    bbox_intersection: tuple[int, int, int, int] | None = None
    mask_area: int = Field(ge=0)
    candidate_area: int = Field(ge=0)
    mask_precision: float = Field(ge=0, le=1)
    mask_recall: float = Field(ge=0, le=1)
    mask_iou: float = Field(ge=0, le=1)


class CandidateSanityMetrics(StrictModel):
    frame_ids: list[str]
    transform_source: Annotated[str, Field(min_length=1)]
    mask_precision: float = Field(ge=0, le=1)
    mask_recall: float = Field(ge=0, le=1)
    mask_iou: float = Field(ge=0, le=1)
    dense_depth_relative_residual: float | None = Field(default=None, ge=0)
    depth_inlier_fraction: float | None = Field(default=None, ge=0, le=1)
    negative_space_violation_ratio: float = Field(ge=0, le=1)
    front_of_scene_violation_ratio: float = Field(ge=0, le=1)
    valid_candidate_pixel_count: int = Field(ge=0)
    per_frame: list[CandidateFrameRenderDiagnostic]


class CandidateFailureClassification(StrEnum):
    BACKEND_EXPORT_INVALID = "backend_export_invalid"
    NATIVE_RENDER_FAILED = "native_render_failed"
    EMPTY_CANDIDATE_RENDER = "empty_candidate_render"
    REGISTRATION_FAILED = "registration_failed"
    FITTING_VIEW_INCONSISTENT = "fitting_view_inconsistent"
    FITTING_OVERFIT_HELDOUT_FAILURE = "fitting_overfit_heldout_failure"
    HELDOUT_SHAPE_INCONSISTENT = "heldout_shape_inconsistent"
    NEGATIVE_SPACE_VIOLATION = "negative_space_violation"
    DEPTH_INCONSISTENT = "depth_inconsistent"
    LICENSE_BLOCKED = "license_blocked"
    PASSED = "passed"


class CandidateRepresentationParityView(StrictModel):
    view_id: Annotated[str, Field(min_length=1)]
    frame_id: Annotated[str, Field(min_length=1)]
    transform_source: Annotated[str, Field(min_length=1)]
    gaussian_valid_pixel_count: int = Field(ge=0)
    glb_valid_pixel_count: int = Field(ge=0)
    silhouette_iou: float = Field(ge=0, le=1)
    projected_bbox_iou: float = Field(ge=0, le=1)
    normalized_centroid_distance: float | None = Field(default=None, ge=0)
    gaussian_depth_available: bool
    glb_depth_available: bool
    gaussian_target_mask_precision: float | None = Field(default=None, ge=0, le=1)
    gaussian_target_mask_recall: float | None = Field(default=None, ge=0, le=1)
    gaussian_target_mask_iou: float | None = Field(default=None, ge=0, le=1)
    glb_target_mask_precision: float | None = Field(default=None, ge=0, le=1)
    glb_target_mask_recall: float | None = Field(default=None, ge=0, le=1)
    glb_target_mask_iou: float | None = Field(default=None, ge=0, le=1)


class CandidateRepresentationParityArtifact(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    candidate_id: Annotated[str, Field(min_length=1)]
    gaussian_asset_id: Annotated[str, Field(min_length=1)]
    gaussian_asset_path: str
    glb_asset_id: Annotated[str, Field(min_length=1)]
    glb_asset_path: str
    official_code_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    renderer: Literal["official_sam3d_gaussian_gsplat_and_nvdiffrast_glb"]
    views: Annotated[list[CandidateRepresentationParityView], Field(min_length=1)]
    minimum_silhouette_iou: float = Field(ge=0, le=1)
    minimum_bbox_iou: float = Field(ge=0, le=1)
    maximum_normalized_centroid_distance: float = Field(ge=0)
    accepted: bool
    failure_reasons: list[str]
    transform_transfer_permitted: bool
    warnings: list[str] = Field(default_factory=list)

    @field_validator("gaussian_asset_path", "glb_asset_path")
    @classmethod
    def safe_parity_asset_paths(cls, value: str) -> str:
        return _relative_artifact_path(value)

    @model_validator(mode="after")
    def parity_acceptance_matches_checks(self) -> Self:
        if self.accepted != (not self.failure_reasons):
            raise ValueError("representation parity acceptance must match its failed gates")
        if self.transform_transfer_permitted != self.accepted:
            raise ValueError("representation transform transfer requires accepted parity")
        return self


class CandidateCompletionGain(StrictModel):
    recall_gain_vs_measured_baseline: float = Field(ge=-1, le=1)
    iou_gain_vs_measured_baseline: float = Field(ge=-1, le=1)
    precision_change_vs_measured_baseline: float = Field(ge=-1, le=1)
    depth_residual_change: float
    visible_coverage_gain: float = Field(ge=-1, le=1)
    negative_space_change: float = Field(ge=-1, le=1)


class CandidateHeldoutEvaluation(StrictModel):
    candidate_id: Annotated[str, Field(min_length=1)]
    object_id: Annotated[str, Field(min_length=1)]
    backend: CompletionBackend
    registration_asset_id: Annotated[str, Field(min_length=1)]
    registration_asset_path: str
    evaluation_asset_id: Annotated[str, Field(min_length=1)]
    evaluation_asset_path: str
    selection_asset_id: Annotated[str, Field(min_length=1)]
    selection_asset_path: str
    transform_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    anchor_sanity: CandidateSanityMetrics
    fitting_metrics: CandidateSanityMetrics
    heldout_frame_ids: list[str]
    metrics: CandidateHeldoutMetrics
    measured_baseline_metrics: CandidateHeldoutMetrics
    completion_gain: CandidateCompletionGain
    passed_hard_gates: bool
    failed_gates: list[str]
    evaluation_runtime_seconds: float = Field(ge=0)
    license_record: CompletionLicenseRecord
    render_paths: dict[str, str] = Field(default_factory=dict)
    anchor_render_paths: dict[str, str] = Field(default_factory=dict)
    fitting_render_paths: dict[str, str] = Field(default_factory=dict)
    failure_classification: CandidateFailureClassification
    representation_parity_path: str | None = None
    representation_parity_accepted: bool = False

    @field_validator("render_paths", "anchor_render_paths", "fitting_render_paths")
    @classmethod
    def safe_candidate_render_paths(cls, values: dict[str, str]) -> dict[str, str]:
        return {frame_id: _relative_artifact_path(path) for frame_id, path in values.items()}

    @field_validator(
        "registration_asset_path",
        "evaluation_asset_path",
        "selection_asset_path",
        "representation_parity_path",
    )
    @classmethod
    def safe_evaluation_asset_paths(cls, value: str | None) -> str | None:
        return _relative_artifact_path(value) if value is not None else None

    @model_validator(mode="after")
    def representation_transfer_is_audited(self) -> Self:
        if self.registration_asset_id != self.evaluation_asset_id and not (
            self.representation_parity_accepted and self.representation_parity_path
        ):
            raise ValueError("evaluation asset differs from registration without accepted parity")
        if self.evaluation_asset_id != self.selection_asset_id and not (
            self.representation_parity_accepted and self.representation_parity_path
        ):
            raise ValueError("selection asset differs from evaluation without accepted parity")
        return self


class CandidateEvaluationManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    registration_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    evaluation_configuration: dict[str, object]
    evaluations: list[CandidateHeldoutEvaluation]
    transforms_frozen_before_heldout_evaluation: Literal[True] = True
    runtime_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    peak_host_memory_bytes: int | None = Field(default=None, ge=0)


class SelectedVisualCompletion(StrictModel):
    object_id: Annotated[str, Field(min_length=1)]
    status: Literal[
        "accepted_visual_completion",
        "ambiguous_multiple_candidates",
        "symmetry_ambiguous",
        "rejected_inconsistent",
        "unresolved_no_candidate",
        "deferred_object_type",
        "license_blocked",
        "backend_failed",
    ]
    best_research_candidate: str | None = None
    best_production_eligible_candidate: str | None = None
    selected_candidate: str | None = None
    measured_anchor_asset_path: str | None = None
    selected_native_asset_path: str | None = None
    selected_asset_id: str | None = None
    evaluated_asset_id: str | None = None
    representation_parity_path: str | None = None
    selection_rationale: list[str]
    geometry_status: Literal["complete_visual_candidate"] | None = None
    observation_grounded: Literal[True] = True
    physical_validation: Literal["not_implemented"] = "not_implemented"
    collision_ready: Literal[False] = False
    sim_ready: Literal[False] = False
    metric_scale_known: Literal[False] = False
    canonical_gravity_alignment_known: Literal[False] = False

    @field_validator(
        "measured_anchor_asset_path",
        "selected_native_asset_path",
        "representation_parity_path",
    )
    @classmethod
    def safe_selected_completion_paths(cls, value: str | None) -> str | None:
        return _relative_artifact_path(value) if value is not None else None

    @model_validator(mode="after")
    def selected_asset_is_the_evaluated_representation(self) -> Self:
        if self.selected_candidate is None:
            if any(
                value is not None
                for value in (
                    self.selected_native_asset_path,
                    self.selected_asset_id,
                    self.evaluated_asset_id,
                    self.representation_parity_path,
                )
            ):
                raise ValueError("unselected completion cannot name a native asset")
            return self
        if not self.selected_native_asset_path or not self.selected_asset_id:
            raise ValueError("selected completion requires an explicit native asset")
        if not self.evaluated_asset_id:
            raise ValueError("selected completion requires its evaluated asset ID")
        if (
            self.selected_asset_id != self.evaluated_asset_id
            and not self.representation_parity_path
        ):
            raise ValueError("selected asset differs from evaluated asset without parity")
        return self


class CandidateSelectionArtifact(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    evaluation_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    license_mode: CompletionLicenseMode
    ranking_policy: Literal["hard_gates_pareto_deterministic_v1"]
    objects: list[SelectedVisualCompletion]
    deterministic_selection_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class CompletionDiagnostics(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    eligible_object_count: int = Field(ge=0)
    deferred_object_count: int = Field(ge=0)
    candidate_count_by_backend: dict[str, int]
    registered_candidate_count: int = Field(ge=0)
    evaluated_candidate_count: int = Field(ge=0)
    passing_candidate_count: int = Field(ge=0)
    selected_research_count: int = Field(ge=0)
    selected_production_count: int = Field(ge=0)
    total_runtime_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    peak_host_memory_bytes: int | None = Field(default=None, ge=0)
    warnings: list[str] = Field(default_factory=list)


class Phase5BConsistencyReport(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    passed: bool
    checks: list[EndToEndConsistencyCheck]
    measured_anchor_preserved: Literal[True] = True
    generated_hidden_geometry_used: bool
    heldout_validation_used: Literal[True] = True
    articulated_completion_implemented: Literal[False] = False
    collision_generation_implemented: Literal[False] = False
    physical_validation_implemented: Literal[False] = False
    metric_scale_known: Literal[False] = False
    canonical_gravity_alignment_known: Literal[False] = False
    sim_ready_scene_implemented: Literal[False] = False
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def phase5b_summary_matches_checks(self) -> Self:
        if self.passed != all(check.passed for check in self.checks):
            raise ValueError("Phase 5B pass status must match its checks")
        return self


# Phase 5C: observation-grounded articulated visual reconstruction. State
# alignment and measured motion remain in arbitrary, unoriented COLMAP units.
class ArticulatedEligibilityStatus(StrEnum):
    ELIGIBLE_MULTI_STATE = "eligible_multi_state"
    ELIGIBLE_PRIOR_ONLY = "eligible_prior_only"
    DEFERRED_INSUFFICIENT_PARTS = "deferred_insufficient_parts"
    DEFERRED_INSUFFICIENT_STATES = "deferred_insufficient_states"
    DEFERRED_DEFORMABLE = "deferred_deformable"
    DEFERRED_HUMAN = "deferred_human"
    DEFERRED_COMPLEX_MECHANISM = "deferred_complex_mechanism"
    EXPLICIT_OVERRIDE = "explicit_override"


class ArticulationEvidenceLevel(StrEnum):
    SINGLE_STATE_PRIOR_ONLY = "single_state_prior_only"
    TWO_STATE_MOTION_SUPPORTED = "two_state_motion_supported"
    MULTI_STATE_HELDOUT_AVAILABLE = "multi_state_heldout_available"
    MULTI_STATE_HELDOUT_VALIDATED = "multi_state_heldout_validated"


class ArticulatedJointType(StrEnum):
    FIXED = "fixed"
    PRISMATIC = "prismatic"
    REVOLUTE = "revolute"
    CONTINUOUS_CANDIDATE = "continuous_candidate"
    UNKNOWN = "unknown"


class ArticulatedSourceFamily(StrEnum):
    MEASURED_MOTION = "measured_motion_analytic"
    ARTVIP = "artvip"
    PARTNET_MOBILITY = "partnet_mobility"
    PARTICULATE = "particulate"


class ArticulatedAssetSpace(StrEnum):
    REFERENCE_WORLD = "reference_world"
    CANDIDATE_BASE = "candidate_base"
    LINK_LOCAL = "link_local"


class ArticulatedLicenseMode(StrEnum):
    RESEARCH_EVALUATION = "research_evaluation"
    PRODUCTION_CANDIDATE = "production_candidate"


class ArticulatedCandidateStatus(StrEnum):
    ACCEPTED = "accepted_articulated_visual_candidate"
    AMBIGUOUS_JOINT_TYPE = "ambiguous_joint_type"
    AMBIGUOUS_LINK_ASSIGNMENT = "ambiguous_link_assignment"
    PRIOR_ONLY = "prior_only_unvalidated"
    TWO_STATE = "two_state_partially_validated"
    MULTI_STATE = "multi_state_validated"
    REJECTED_ALIGNMENT = "rejected_state_alignment"
    REJECTED_GEOMETRY = "rejected_geometry_mismatch"
    REJECTED_JOINT = "rejected_joint_constraint"
    REJECTED_HELDOUT = "rejected_heldout_state"
    LICENSE_BLOCKED = "license_blocked"
    UNRESOLVED = "unresolved_no_candidate"
    BACKEND_FAILED = "backend_failed"


class ArticulatedEligibilityRecord(StrictModel):
    articulated_object_id: Annotated[str, Field(min_length=1)]
    semantic_label: Annotated[str, Field(min_length=1)]
    asset_type_hint: AssetType | None = None
    state_count: int = Field(ge=0)
    movable_part_count: int = Field(ge=0)
    status: ArticulatedEligibilityStatus
    explicitly_overridden: bool = False
    reason: Annotated[str, Field(min_length=1)]


class ArticulatedEligibilityArtifact(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    phase5b_selection_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    records: list[ArticulatedEligibilityRecord]

    @model_validator(mode="after")
    def unique_articulated_objects(self) -> Self:
        identifiers = [record.articulated_object_id for record in self.records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("articulated eligibility object IDs must be unique")
        return self


class ArticulationBasePrompt(StrictModel):
    part_id: Annotated[str, Field(min_length=1)]
    prompt_id: Annotated[str, Field(min_length=1)]
    label: Annotated[str, Field(min_length=1)]


class ArticulationMovablePartPrompt(StrictModel):
    part_id: Annotated[str, Field(min_length=1)]
    prompt_id: Annotated[str, Field(min_length=1)]
    label: Annotated[str, Field(min_length=1)]
    expected_joint_hint: ArticulatedJointType = ArticulatedJointType.UNKNOWN
    parent_part_id: str | None = None
    handle_part_id: str | None = None
    include: bool = True


class ArticulationObjectPrompt(StrictModel):
    articulated_object_id: Annotated[str, Field(min_length=1)]
    semantic_label: Annotated[str, Field(min_length=1)]
    base: ArticulationBasePrompt
    movable_parts: Annotated[list[ArticulationMovablePartPrompt], Field(min_length=1)]
    excluded_prompt_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def stable_part_ids(self) -> Self:
        identifiers = [self.base.part_id, *(part.part_id for part in self.movable_parts)]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("articulation stable part IDs must be unique")
        return self


class ArticulationPartPromptManifest(StrictModel):
    schema_version: Literal["0.2.0"] = "0.2.0"
    objects: Annotated[list[ArticulationObjectPrompt], Field(min_length=1)]

    @model_validator(mode="after")
    def unique_prompt_objects(self) -> Self:
        identifiers = [item.articulated_object_id for item in self.objects]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("articulation prompt object IDs must be unique")
        return self


class ArticulationStateRecord(StrictModel):
    state_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
    run_dir: Annotated[str, Field(min_length=1)]
    semantic_state_label: Annotated[str, Field(min_length=1)]
    part_track_ids: dict[
        Annotated[str, Field(min_length=1)],
        Annotated[str, Field(min_length=1)],
    ]
    phase5a_consistency_passed: bool
    ingest_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    frame_sequence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    segmentation_tracking_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    dense_depth_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    measured_geometry_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    part_mask_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    measured_part_cloud_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    registered_frame_ids: list[str]
    camera_evidence_path: str
    segmentation_evidence_path: str
    undistortion_evidence_path: str
    depth_evidence_path: str
    dense_map_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]

    @model_validator(mode="after")
    def unique_part_track_mapping(self) -> Self:
        if len(self.part_track_ids) != len(set(self.part_track_ids.values())):
            raise ValueError("one state track cannot be assigned to multiple stable parts")
        return self

    @field_validator(
        "camera_evidence_path",
        "segmentation_evidence_path",
        "undistortion_evidence_path",
        "depth_evidence_path",
    )
    @classmethod
    def safe_articulation_state_evidence_paths(cls, value: str) -> str:
        return _relative_artifact_path(value)


class ArticulationCaptureManifest(StrictModel):
    schema_version: Literal["0.2.0"] = "0.2.0"
    articulated_object_id: Annotated[str, Field(min_length=1)]
    reference_state_id: Annotated[str, Field(min_length=1)]
    states: Annotated[list[ArticulationStateRecord], Field(min_length=1)]
    prompt_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    capture_state_count: int = Field(ge=1)
    capture_evidence_tier: ArticulationEvidenceLevel

    @model_validator(mode="after")
    def valid_capture_states(self) -> Self:
        identifiers = [state.state_id for state in self.states]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("articulation state IDs must be unique")
        if self.reference_state_id not in identifiers:
            raise ValueError("reference articulation state is not present")
        if not all(state.phase5a_consistency_passed for state in self.states):
            raise ValueError("every articulation state must pass Phase 5A")
        expected_tier = (
            ArticulationEvidenceLevel.SINGLE_STATE_PRIOR_ONLY
            if len(self.states) == 1
            else (
                ArticulationEvidenceLevel.TWO_STATE_MOTION_SUPPORTED
                if len(self.states) == 2
                else ArticulationEvidenceLevel.MULTI_STATE_HELDOUT_AVAILABLE
            )
        )
        if self.capture_state_count != len(self.states):
            raise ValueError("capture state count does not match state records")
        if self.capture_evidence_tier is not expected_tier:
            raise ValueError("capture evidence tier does not match state count")
        return self


class ArticulationStateTransform(StrictModel):
    state_id: Annotated[str, Field(min_length=1)]
    matrix_reference_from_state: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]
    inverse_matrix: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]
    scale: float = Field(gt=0)
    rotation_determinant: float
    translation: tuple[float, float, float]
    fitting_median_residual_scene_diagonal: float = Field(ge=0)
    fitting_p90_residual_scene_diagonal: float = Field(ge=0)
    heldout_static_depth_inlier_fraction: float = Field(ge=0, le=1)
    static_correspondence_count: int = Field(ge=0)
    excluded_movable_part_ids: list[str]
    accepted: bool
    failure_reason: str | None = None

    @model_validator(mode="after")
    def proper_invertible_sim3(self) -> Self:
        matrix = self.matrix_reference_from_state
        inverse = self.inverse_matrix
        if not all(math.isfinite(value) for value in (*matrix, *inverse)):
            raise ValueError("articulation state transform contains non-finite values")
        if any(
            abs(matrix[index] - expected) > 1e-6
            for index, expected in zip(
                (12, 13, 14, 15),
                (0.0, 0.0, 0.0, 1.0),
                strict=True,
            )
        ):
            raise ValueError("articulation state transform is not affine")
        determinant = (
            matrix[0] * (matrix[5] * matrix[10] - matrix[6] * matrix[9])
            - matrix[1] * (matrix[4] * matrix[10] - matrix[6] * matrix[8])
            + matrix[2] * (matrix[4] * matrix[9] - matrix[5] * matrix[8])
        )
        if determinant <= 0:
            raise ValueError("articulation state transform must be proper positive-scale Sim(3)")
        derived_scale = determinant ** (1.0 / 3.0)
        if abs(derived_scale - self.scale) > 1e-5 * max(1.0, self.scale):
            raise ValueError("articulation state transform scale is inconsistent")
        derived_rotation_determinant = determinant / (self.scale**3)
        if (
            abs(derived_rotation_determinant - 1.0) > 1e-5
            or abs(self.rotation_determinant - derived_rotation_determinant) > 1e-5
        ):
            raise ValueError("articulation state rotation must be proper")
        product = tuple(
            sum(matrix[row * 4 + inner] * inverse[inner * 4 + column] for inner in range(4))
            for row in range(4)
            for column in range(4)
        )
        identity = (
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        if (
            max(abs(actual - expected) for actual, expected in zip(product, identity, strict=True))
            > 1e-5
        ):
            raise ValueError("articulation state transform inverse fails round trip")
        if self.accepted == (self.failure_reason is not None):
            raise ValueError("state-alignment acceptance and failure reason disagree")
        return self


class ArticulationStateAlignmentArtifact(StrictModel):
    schema_version: Literal["0.2.0"] = "0.2.0"
    capture_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    reference_state_id: Annotated[str, Field(min_length=1)]
    transforms: Annotated[list[ArticulationStateTransform], Field(min_length=1)]
    capture_state_count: int = Field(ge=1)
    accepted_alignment_state_ids: list[str]
    aligned_state_count: int = Field(ge=0)
    static_evidence_only: Literal[True] = True
    source_states_unchanged: Literal[True] = True
    runtime_seconds: float = Field(ge=0)
    peak_host_memory_bytes: int | None = Field(default=None, ge=0)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_alignment_states(self) -> Self:
        identifiers = [item.state_id for item in self.transforms]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("state alignment transforms must be unique")
        accepted = [item.state_id for item in self.transforms if item.accepted]
        if self.capture_state_count != len(self.transforms):
            raise ValueError("alignment capture-state count does not match transforms")
        if self.accepted_alignment_state_ids != accepted:
            raise ValueError("accepted alignment state IDs do not match transforms")
        if self.aligned_state_count != len(accepted):
            raise ValueError("aligned state count does not match accepted transforms")
        reference = next(
            (item for item in self.transforms if item.state_id == self.reference_state_id),
            None,
        )
        if reference is None or not reference.accepted:
            raise ValueError("declared reference state must have an accepted identity transform")
        identity = (
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        if (
            max(
                abs(actual - expected)
                for actual, expected in zip(
                    reference.matrix_reference_from_state,
                    identity,
                    strict=True,
                )
            )
            > 1e-6
        ):
            raise ValueError("declared reference state transform must be identity")
        return self


class ArticulatedPartStateGeometry(StrictModel):
    state_id: Annotated[str, Field(min_length=1)]
    articulated_object_id: Annotated[str, Field(min_length=1)]
    part_id: Annotated[str, Field(min_length=1)]
    source_track_id: Annotated[str, Field(min_length=1)]
    prompt_id: Annotated[str, Field(min_length=1)]
    semantic_label: Annotated[str, Field(min_length=1)]
    measured_point_cloud_path: str
    measured_point_cloud_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    measured_surfel_path: str | None = None
    measured_surfel_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    point_count: int = Field(ge=0)
    normal_count: int = Field(ge=0)
    supporting_frame_ids: list[str]
    mask_paths: list[str]
    state_alignment_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    transformed_to_reference_frame: bool
    coordinate_convention: CoordinateConvention
    scale_status: ScaleStatus

    @field_validator("measured_point_cloud_path", "measured_surfel_path")
    @classmethod
    def safe_measured_state_geometry_paths(cls, value: str | None) -> str | None:
        return _relative_artifact_path(value) if value is not None else None

    @field_validator("mask_paths")
    @classmethod
    def safe_measured_state_mask_paths(cls, values: list[str]) -> list[str]:
        return [_relative_artifact_path(value) for value in values]


class ArticulatedPartStateGeometryManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    capture_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    geometries: list[ArticulatedPartStateGeometry]


class MeasuredJointState(StrictModel):
    state_id: Annotated[str, Field(min_length=1)]
    position: float
    part_registration_median_residual: float = Field(ge=0)
    part_coverage: float = Field(ge=0, le=1)
    supporting_point_count: int = Field(ge=0)
    state_confidence: float = Field(ge=0, le=1)


class MeasuredJointHypothesis(StrictModel):
    joint_id: Annotated[str, Field(min_length=1)]
    parent_part_id: Annotated[str, Field(min_length=1)]
    child_part_id: Annotated[str, Field(min_length=1)]
    joint_type: ArticulatedJointType
    axis: tuple[float, float, float] | None = None
    pivot: tuple[float, float, float] | None = None
    states: list[MeasuredJointState]
    observed_position_min: float | None = None
    observed_position_max: float | None = None
    candidate_limit_lower: float | None = None
    candidate_limit_upper: float | None = None
    limit_source: Literal["observed_range", "candidate_prior", "unknown"] = "observed_range"
    orthogonal_residual: float | None = Field(default=None, ge=0)
    rotation_leakage_degrees: float | None = Field(default=None, ge=0)
    axis_consistency_degrees: float | None = Field(default=None, ge=0)
    normalization_part_diagonal: float | None = Field(default=None, gt=0)
    fixed_translation_residual_arbitrary_units: float | None = Field(default=None, ge=0)
    fixed_translation_residual_part_diagonals: float | None = Field(default=None, ge=0)
    pivot_residual_arbitrary_units: float | None = Field(default=None, ge=0)
    pivot_residual_part_diagonals: float | None = Field(default=None, ge=0)
    confidence: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_measured_joint(self) -> Self:
        if self.joint_type in {
            ArticulatedJointType.PRISMATIC,
            ArticulatedJointType.REVOLUTE,
            ArticulatedJointType.CONTINUOUS_CANDIDATE,
        }:
            if self.axis is None:
                raise ValueError("moving joint requires an axis")
            norm = sum(value * value for value in self.axis) ** 0.5
            if abs(norm - 1.0) > 1e-5:
                raise ValueError("joint axis must be normalized")
        if (
            self.joint_type
            in {
                ArticulatedJointType.REVOLUTE,
                ArticulatedJointType.CONTINUOUS_CANDIDATE,
            }
            and self.pivot is None
        ):
            raise ValueError("revolute joint requires a pivot")
        if (self.observed_position_min is None) != (self.observed_position_max is None):
            raise ValueError("observed joint range must provide both endpoints")
        if (
            self.observed_position_min is not None
            and self.observed_position_max is not None
            and self.observed_position_min > self.observed_position_max
        ):
            raise ValueError("observed joint range is reversed")
        return self


class MeasuredPartMotionArtifact(StrictModel):
    schema_version: Literal["0.2.0"] = "0.2.0"
    capture_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    state_alignment_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    articulated_object_id: Annotated[str, Field(min_length=1)]
    reference_state_id: Annotated[str, Field(min_length=1)]
    capture_state_count: int = Field(ge=1)
    accepted_alignment_state_ids: list[str]
    effective_motion_evidence_level: ArticulationEvidenceLevel
    part_geometries: list[ArticulatedPartStateGeometry]
    joint_hypotheses: list[MeasuredJointHypothesis]
    base_link_fixed: bool
    runtime_seconds: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class ArticulationEvidenceSplit(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    articulated_object_id: Annotated[str, Field(min_length=1)]
    candidate_generation_states: list[str]
    kinematic_fitting_states: list[str]
    heldout_validation_states: list[str]
    heldout_views_by_state: dict[str, list[str]]
    seed: int

    @model_validator(mode="after")
    def disjoint_articulation_states(self) -> Self:
        groups = (
            self.candidate_generation_states,
            self.kinematic_fitting_states,
            self.heldout_validation_states,
        )
        if any(len(group) != len(set(group)) for group in groups):
            raise ValueError("articulation evidence groups contain duplicate states")
        if any(
            set(left) & set(right)
            for index, left in enumerate(groups)
            for right in groups[index + 1 :]
        ):
            raise ValueError(
                "articulation generation, fitting, and held-out states must be disjoint"
            )
        return self


class ArticulatedLicenseRecord(StrictModel):
    source_family: ArticulatedSourceFamily
    code_license: Annotated[str, Field(min_length=1)]
    checkpoint_license: Annotated[str, Field(min_length=1)]
    dependency_licenses: dict[str, str] = Field(default_factory=dict)
    asset_license: Annotated[str, Field(min_length=1)]
    training_data_notes: list[str] = Field(default_factory=list)
    commercial_review_status: Literal[
        "not_reviewed",
        "research_only",
        "approved_by_project_policy",
    ]
    research_evaluation_allowed: bool
    production_selectable: bool


class ArticulatedAssetIndexRecord(StrictModel):
    asset_id: Annotated[str, Field(min_length=1)]
    category: Annotated[str, Field(min_length=1)]
    link_count: int = Field(gt=0)
    joint_count: int = Field(ge=0)
    joint_types: list[ArticulatedJointType]
    visual_bounds: tuple[float, float, float]
    link_bounds: dict[str, tuple[float, float, float]]
    native_units: Annotated[str, Field(min_length=1)]
    native_up_axis: Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    visual_asset_paths: list[str]
    file_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    candidate_bundle_path: str | None = None
    candidate_bundle_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    license_record: ArticulatedLicenseRecord

    @field_validator("visual_asset_paths")
    @classmethod
    def safe_index_visual_paths(cls, values: list[str]) -> list[str]:
        return [_relative_artifact_path(value) for value in values]

    @field_validator("candidate_bundle_path")
    @classmethod
    def safe_index_bundle_path(cls, value: str | None) -> str | None:
        return _relative_artifact_path(value) if value is not None else None

    @model_validator(mode="after")
    def complete_candidate_bundle_identity(self) -> Self:
        if (self.candidate_bundle_path is None) != (self.candidate_bundle_sha256 is None):
            raise ValueError("articulated index candidate bundle requires path and hash")
        missing = set(self.visual_asset_paths) - set(self.file_hashes)
        if missing:
            raise ValueError(
                "articulated index is missing visual-asset hashes: " + ", ".join(sorted(missing))
            )
        return self


class ArticulatedAssetIndex(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    source_family: Literal["artvip", "partnet_mobility"]
    index_revision: Annotated[str, Field(min_length=1)]
    records: list[ArticulatedAssetIndexRecord]
    content_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ArticulatedRetrievalCandidate(StrictModel):
    candidate_id: Annotated[str, Field(min_length=1)]
    source_family: ArticulatedSourceFamily
    source_asset_id: Annotated[str, Field(min_length=1)]
    retrieval_score: float
    evidence_terms: dict[str, float]
    production_selectable: bool
    candidate_bundle_path: str | None = None
    candidate_bundle_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    visual_asset_paths: list[str] = Field(default_factory=list)
    visual_asset_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]] = Field(
        default_factory=dict
    )

    @field_validator("candidate_bundle_path")
    @classmethod
    def safe_retrieval_bundle_path(cls, value: str | None) -> str | None:
        return _relative_artifact_path(value) if value is not None else None

    @field_validator("visual_asset_paths")
    @classmethod
    def safe_retrieval_visual_paths(cls, values: list[str]) -> list[str]:
        return [_relative_artifact_path(value) for value in values]

    @model_validator(mode="after")
    def valid_retrieved_candidate_assets(self) -> Self:
        if (self.candidate_bundle_path is None) != (self.candidate_bundle_sha256 is None):
            raise ValueError("retrieval candidate bundle requires path and hash")
        if set(self.visual_asset_paths) != set(self.visual_asset_hashes):
            raise ValueError("retrieval visual asset paths and hashes do not match")
        return self


class ArticulatedRetrievalResult(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    articulated_object_id: Annotated[str, Field(min_length=1)]
    measured_motion_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    candidates: list[ArticulatedRetrievalCandidate]
    artvip_index_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    partnet_index_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    runtime_seconds: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class ParticulateCandidateRequest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    candidate_id: Annotated[str, Field(min_length=1)]
    articulated_object_id: Annotated[str, Field(min_length=1)]
    source_mesh_path: str
    source_mesh_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    source_backend: Annotated[str, Field(min_length=1)]
    source_representation: Annotated[str, Field(min_length=1)]
    source_license: ArticulatedLicenseRecord
    visual_completeness_status: Annotated[str, Field(min_length=1)]
    official_repository: Literal["https://github.com/RuiningLi/particulate"]
    official_code_commit: Literal["dee37a75c449f324d9989993461ee09eaccc1686"]
    checkpoint_repository: Literal["rayli/Particulate"]
    checkpoint_revision: Literal["096167e661feb92a443535d15916323ec8a01613"]
    checkpoint_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    runtime_model_revisions: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]]
    runtime_model_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    working_frame_hypothesis: Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    hypotheses_evaluated: list[Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]]
    hypothesis_selection_evidence: Annotated[str, Field(min_length=1)]
    generation_configuration: dict[str, object]
    output_directory: str
    seed: int

    @field_validator("source_mesh_path", "output_directory")
    @classmethod
    def safe_particulate_request_paths(cls, value: str) -> str:
        return _relative_artifact_path(value)


class ParticulateWorkerManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    worker_version: Annotated[str, Field(min_length=1)]
    request_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    official_repository: Literal["https://github.com/RuiningLi/particulate"]
    official_code_commit: Literal["dee37a75c449f324d9989993461ee09eaccc1686"]
    checkpoint_repository: Literal["rayli/Particulate"]
    checkpoint_revision: Literal["096167e661feb92a443535d15916323ec8a01613"]
    checkpoint_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    runtime_model_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    runtime_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    peak_host_memory_bytes: int | None = Field(default=None, ge=0)
    warnings: list[str] = Field(default_factory=list)


class ArticulatedLink(StrictModel):
    link_id: Annotated[str, Field(min_length=1)]
    name: Annotated[str, Field(min_length=1)]
    visual_asset_paths: list[str]
    visual_asset_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    visual_asset_spaces: dict[str, ArticulatedAssetSpace]
    visual_asset_transforms_candidate_base: dict[
        str,
        Annotated[tuple[float, ...], Field(min_length=16, max_length=16)],
    ]
    native_bounds_min: tuple[float, float, float]
    native_bounds_max: tuple[float, float, float]

    @field_validator("visual_asset_paths")
    @classmethod
    def safe_articulated_visual_paths(cls, values: list[str]) -> list[str]:
        return [_relative_artifact_path(value) for value in values]

    @model_validator(mode="after")
    def explicit_visual_asset_spaces(self) -> Self:
        paths = set(self.visual_asset_paths)
        if (
            paths != set(self.visual_asset_hashes)
            or paths != set(self.visual_asset_spaces)
            or paths != set(self.visual_asset_transforms_candidate_base)
        ):
            raise ValueError("articulated visual paths, hashes, spaces, and transforms must match")
        identity = (
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        for path in self.visual_asset_paths:
            matrix = self.visual_asset_transforms_candidate_base[path]
            if not all(math.isfinite(value) for value in matrix):
                raise ValueError("articulated visual asset transform must be finite")
            if any(abs(matrix[index] - identity[index]) > 1e-8 for index in (12, 13, 14, 15)):
                raise ValueError("articulated visual asset transform must be affine")
            linear = (
                (matrix[0], matrix[1], matrix[2]),
                (matrix[4], matrix[5], matrix[6]),
                (matrix[8], matrix[9], matrix[10]),
            )
            determinant = (
                linear[0][0] * (linear[1][1] * linear[2][2] - linear[1][2] * linear[2][1])
                - linear[0][1] * (linear[1][0] * linear[2][2] - linear[1][2] * linear[2][0])
                + linear[0][2] * (linear[1][0] * linear[2][1] - linear[1][1] * linear[2][0])
            )
            column_norms = [
                math.sqrt(sum(linear[row][column] ** 2 for row in range(3))) for column in range(3)
            ]
            if determinant <= 0 or min(column_norms) <= 1e-12:
                raise ValueError("articulated visual asset transform must be proper and invertible")
            if max(column_norms) - min(column_norms) > 1e-6 * max(column_norms):
                raise ValueError(
                    "articulated visual asset transform must use uniform positive scale"
                )
            for left in range(3):
                for right in range(left + 1, 3):
                    dot = sum(linear[row][left] * linear[row][right] for row in range(3))
                    if abs(dot) > 1e-6 * column_norms[left] * column_norms[right]:
                        raise ValueError(
                            "articulated visual asset transform rotation is not orthogonal"
                        )
            if self.visual_asset_spaces[path] is ArticulatedAssetSpace.CANDIDATE_BASE and any(
                abs(left - right) > 1e-8 for left, right in zip(matrix, identity, strict=True)
            ):
                raise ValueError("candidate-base visual assets require an identity transform")
            if self.visual_asset_spaces[path] is ArticulatedAssetSpace.REFERENCE_WORLD and any(
                abs(left - right) > 1e-8 for left, right in zip(matrix, identity, strict=True)
            ):
                raise ValueError(
                    "reference-world measured assets require an identity candidate baseline"
                )
        return self


class ArticulatedJoint(StrictModel):
    joint_id: Annotated[str, Field(min_length=1)]
    parent_link_id: Annotated[str, Field(min_length=1)]
    child_link_id: Annotated[str, Field(min_length=1)]
    joint_type: ArticulatedJointType
    axis: tuple[float, float, float]
    pivot: tuple[float, float, float] | None = None
    candidate_limit_lower: float | None = None
    candidate_limit_upper: float | None = None
    limit_source: Literal["candidate_prior", "observed_range", "unknown"]

    @model_validator(mode="after")
    def valid_candidate_joint(self) -> Self:
        norm = sum(value * value for value in self.axis) ** 0.5
        if abs(norm - 1.0) > 1e-5:
            raise ValueError("candidate joint axis must be normalized")
        if self.parent_link_id == self.child_link_id:
            raise ValueError("candidate joint cannot connect a link to itself")
        if (
            self.joint_type
            in {
                ArticulatedJointType.REVOLUTE,
                ArticulatedJointType.CONTINUOUS_CANDIDATE,
            }
            and self.pivot is None
        ):
            raise ValueError("candidate revolute joint requires a pivot")
        return self


class ArticulatedState(StrictModel):
    state_id: Annotated[str, Field(min_length=1)]
    joint_positions: dict[str, float]
    link_transforms: dict[
        str,
        tuple[
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
        ],
    ]


class ArticulatedCandidate(StrictModel):
    candidate_id: Annotated[str, Field(min_length=1)]
    articulated_object_id: Annotated[str, Field(min_length=1)]
    source_family: ArticulatedSourceFamily
    source_asset_id: Annotated[str, Field(min_length=1)]
    links: Annotated[list[ArticulatedLink], Field(min_length=1)]
    joints: list[ArticulatedJoint]
    states: list[ArticulatedState]
    native_coordinate_convention: Annotated[str, Field(min_length=1)]
    native_units: Annotated[str, Field(min_length=1)]
    native_output_paths: list[str] = Field(default_factory=list)
    native_output_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]] = Field(
        default_factory=dict
    )
    working_transform_source_to_particulate: (
        tuple[
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
        ]
        | None
    ) = None
    working_transform_particulate_to_source: (
        tuple[
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
        ]
        | None
    ) = None
    working_frame_hypothesis: str | None = None
    working_frame_hypotheses_evaluated: list[str] = Field(default_factory=list)
    working_frame_selection_evidence: str | None = None
    license_record: ArticulatedLicenseRecord
    production_selectable: bool
    provenance: ProvenanceRecord
    warnings: list[str] = Field(default_factory=list)

    @field_validator("native_output_paths")
    @classmethod
    def safe_articulated_native_output_paths(cls, values: list[str]) -> list[str]:
        return [_relative_artifact_path(value) for value in values]

    @model_validator(mode="after")
    def valid_candidate_graph(self) -> Self:
        if set(self.native_output_paths) != set(self.native_output_hashes):
            raise ValueError("articulated native-output paths and hashes do not match")
        asset_spaces = {space for link in self.links for space in link.visual_asset_spaces.values()}
        if self.source_family is ArticulatedSourceFamily.MEASURED_MOTION and asset_spaces != {
            ArticulatedAssetSpace.REFERENCE_WORLD
        }:
            raise ValueError("measured-motion candidate assets must remain in the reference world")
        if (
            self.source_family is not ArticulatedSourceFamily.MEASURED_MOTION
            and ArticulatedAssetSpace.REFERENCE_WORLD in asset_spaces
        ):
            raise ValueError(
                "generated or retrieved candidate visuals cannot declare reference-world space"
            )
        link_ids = [link.link_id for link in self.links]
        if len(link_ids) != len(set(link_ids)):
            raise ValueError("articulated candidate link IDs must be unique")
        known = set(link_ids)
        for joint in self.joints:
            if {joint.parent_link_id, joint.child_link_id} - known:
                raise ValueError("articulated joint references an unknown link")
        children: dict[str, str] = {}
        for joint in self.joints:
            if joint.child_link_id in children:
                raise ValueError("candidate link has multiple parent joints")
            children[joint.child_link_id] = joint.parent_link_id
        for child in children:
            visited: set[str] = set()
            current = child
            while current in children:
                if current in visited:
                    raise ValueError("articulated candidate joint graph contains a cycle")
                visited.add(current)
                current = children[current]
        return self


class ArticulatedCandidateManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    measured_motion_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    retrieval_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    candidates: list[ArticulatedCandidate]
    worker_manifests: list[ParticulateWorkerManifest] = Field(default_factory=list)
    failed_candidate_ids: list[str] = Field(default_factory=list)
    runtime_seconds: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class ArticulatedLinkAssignmentRecord(StrictModel):
    observed_part_id: Annotated[str, Field(min_length=1)]
    candidate_link_ids: list[str]
    assignment_confidence: float = Field(ge=0, le=1)
    evidence: dict[str, float]
    ambiguous: bool = False


class ArticulatedLinkAssignment(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    candidate_id: Annotated[str, Field(min_length=1)]
    assignments: list[ArticulatedLinkAssignmentRecord]
    unmatched_candidate_links: list[str]
    unmatched_observed_parts: list[str]


class ArticulatedLinkAssignmentManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    candidate_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    assignments: list[ArticulatedLinkAssignment]


class FittedArticulatedJoint(StrictModel):
    candidate_joint_id: Annotated[str, Field(min_length=1)]
    measured_joint_id: Annotated[str, Field(min_length=1)]
    parent_observed_part_id: Annotated[str, Field(min_length=1)]
    child_observed_part_id: Annotated[str, Field(min_length=1)]
    joint_type: ArticulatedJointType
    fitted_axis: tuple[float, float, float]
    fitted_pivot: tuple[float, float, float] | None = None
    axis_sign: Literal[-1, 1]
    axis_convention: Literal["oriented_toward_measured_axis"] = "oriented_toward_measured_axis"
    axis_sign_role: Literal["native_axis_flip_provenance_only"] = "native_axis_flip_provenance_only"
    q_scale: float
    q_scale_convention: Literal["candidate_q_per_measured_q"] = "candidate_q_per_measured_q"
    q_offset: float
    q_offset_fitted: bool = False
    q_offset_evidence_state_ids: list[str] = Field(default_factory=list)
    fitting_state_q: dict[str, float]
    axis_refinement_degrees: float = Field(ge=0)
    pivot_refinement_arbitrary_units: float | None = Field(default=None, ge=0)
    pivot_refinement_part_diagonals: float | None = Field(default=None, ge=0)
    fitting_residual_arbitrary_units: float = Field(ge=0)
    fitting_residual_part_diagonals: float = Field(ge=0)

    @model_validator(mode="after")
    def valid_fitted_joint(self) -> Self:
        norm = math.sqrt(sum(value * value for value in self.fitted_axis))
        if abs(norm - 1.0) > 1e-5:
            raise ValueError("fitted articulation axis must be normalized")
        if (
            self.joint_type
            in {
                ArticulatedJointType.REVOLUTE,
                ArticulatedJointType.CONTINUOUS_CANDIDATE,
            }
            and self.fitted_pivot is None
        ):
            raise ValueError("fitted revolute joint requires a pivot")
        if self.q_offset_fitted != bool(self.q_offset_evidence_state_ids):
            raise ValueError("q-offset fitting evidence is inconsistent")
        return self


class FittedArticulatedKinematicModel(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    candidate_id: Annotated[str, Field(min_length=1)]
    matrix_reference_world_from_candidate_base: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]
    scale: float = Field(gt=0)
    link_assignments: list[ArticulatedLinkAssignmentRecord]
    fitted_joints: list[FittedArticulatedJoint]
    generation_state_ids: list[str]
    fitting_state_ids: list[str]
    heldout_state_ids: list[str]
    fit_residual_arbitrary_units: float = Field(ge=0)
    fit_residual_scene_diagonals: float = Field(ge=0)
    ambiguity_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def canonical_joint_sign_convention(self) -> Self:
        for joint in self.fitted_joints:
            if joint.joint_type is ArticulatedJointType.PRISMATIC:
                expected = 1.0 / self.scale
            elif joint.joint_type in {
                ArticulatedJointType.REVOLUTE,
                ArticulatedJointType.CONTINUOUS_CANDIDATE,
            }:
                expected = 1.0
            else:
                expected = 0.0
            if abs(joint.q_scale - expected) > 1e-6 * max(1.0, abs(expected)):
                raise ValueError(
                    "fitted articulation q-scale violates the canonical axis convention"
                )
        return self


class ArticulationFittingArtifact(StrictModel):
    candidate_id: Annotated[str, Field(min_length=1)]
    status: Literal["fitted", "ambiguous", "failed"]
    matrix_reference_world_from_candidate_base: (
        tuple[
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
        ]
        | None
    ) = None
    scale: float | None = Field(default=None, gt=0)
    fitting_state_ids: list[str]
    heldout_state_ids: list[str]
    fitted_joint_positions: dict[str, dict[str, float]]
    joint_axis_signs: dict[str, Literal[-1, 1]]
    fitting_median_residual: float | None = Field(default=None, ge=0)
    fitting_part_iou: float | None = Field(default=None, ge=0, le=1)
    fitted_model: FittedArticulatedKinematicModel | None = None
    fitted_model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    structure_frozen_before_heldout: Literal[True] = True
    failure_reason: str | None = None

    @model_validator(mode="after")
    def fitting_excludes_heldout(self) -> Self:
        if set(self.fitting_state_ids) & set(self.heldout_state_ids):
            raise ValueError("articulation fitting and held-out states must be disjoint")
        if self.status == "failed" and self.matrix_reference_world_from_candidate_base is not None:
            raise ValueError("failed articulation fitting cannot have a base transform")
        if self.status != "failed" and self.matrix_reference_world_from_candidate_base is None:
            raise ValueError("successful articulation fitting requires a base transform")
        if self.status == "failed" and self.fitted_model is not None:
            raise ValueError("failed articulation fitting cannot contain a fitted model")
        if self.status != "failed" and (
            self.fitted_model is None or self.fitted_model_sha256 is None
        ):
            raise ValueError("successful articulation fitting requires a typed fitted model")
        return self


class ArticulationFittingManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    candidate_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    evidence_split_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    link_assignments: list[ArticulatedLinkAssignment]
    fittings: list[ArticulationFittingArtifact]
    runtime_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    peak_host_memory_bytes: int | None = Field(default=None, ge=0)


class ArticulationHeldoutViewEvaluation(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    camera_reconstruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    depth_path: str | None = None
    depth_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    valid_depth: bool
    target_mask_paths: dict[str, str]
    target_mask_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    target_masks_complete: bool
    required_link_ids: list[str]
    rendered_link_ids: list[str]
    missing_link_ids: list[str]
    usable: bool
    failure_reasons: list[str]
    render_path: str | None = None
    render_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    raw_candidate_pixel_count: int = Field(ge=0)
    visible_candidate_pixel_count: int = Field(ge=0)
    target_mask_pixel_count: int = Field(ge=0)

    @field_validator("depth_path", "render_path")
    @classmethod
    def safe_optional_heldout_paths(cls, value: str | None) -> str | None:
        return _relative_artifact_path(value) if value is not None else None

    @field_validator("target_mask_paths")
    @classmethod
    def safe_heldout_mask_paths(cls, values: dict[str, str]) -> dict[str, str]:
        return {part_id: _relative_artifact_path(path) for part_id, path in values.items()}

    @model_validator(mode="after")
    def heldout_view_identity_is_complete(self) -> Self:
        if set(self.target_mask_paths) != set(self.target_mask_hashes):
            raise ValueError("held-out target mask paths and hashes do not match")
        if (self.depth_path is None) != (self.depth_sha256 is None):
            raise ValueError("held-out depth path and hash must be paired")
        if (self.render_path is None) != (self.render_sha256 is None):
            raise ValueError("held-out render path and hash must be paired")
        if set(self.required_link_ids) != (
            set(self.rendered_link_ids) | set(self.missing_link_ids)
        ):
            raise ValueError("held-out link coverage is incomplete")
        if set(self.rendered_link_ids) & set(self.missing_link_ids):
            raise ValueError("held-out link cannot be both rendered and missing")
        if self.usable != (
            not self.failure_reasons
            and not self.missing_link_ids
            and self.valid_depth
            and self.target_masks_complete
            and self.visible_candidate_pixel_count > 0
            and self.target_mask_pixel_count > 0
            and self.render_path is not None
            and self.render_sha256 is not None
        ):
            raise ValueError("held-out view usability does not match its evidence")
        return self


class ArticulationStateEvaluation(StrictModel):
    state_id: Annotated[str, Field(min_length=1)]
    heldout: bool
    requested_heldout_view_count: int = Field(ge=0)
    usable_heldout_view_count: int = Field(ge=0)
    rendered_heldout_view_count: int = Field(ge=0)
    views_with_target_masks: int = Field(ge=0)
    views_with_valid_depth: int = Field(ge=0)
    base_mask_iou: float | None = Field(default=None, ge=0, le=1)
    movable_part_mask_iou: float | None = Field(default=None, ge=0, le=1)
    whole_object_mask_iou: float | None = Field(default=None, ge=0, le=1)
    per_link_depth_residual: dict[str, float | None]
    base_depth_residual: float | None = Field(default=None, ge=0)
    depth_inlier_fraction: float | None = Field(default=None, ge=0, le=1)
    negative_space_violation_ratio: float | None = Field(default=None, ge=0, le=1)
    front_of_scene_violation_ratio: float | None = Field(default=None, ge=0, le=1)
    scene_diagonal_arbitrary_units: float | None = Field(default=None, gt=0)
    base_point_residual_arbitrary_units: float | None = Field(default=None, ge=0)
    base_point_residual_scene_diagonals: float | None = Field(default=None, ge=0)
    base_motion_arbitrary_units: float | None = Field(default=None, ge=0)
    base_motion_scene_diagonals: float | None = Field(default=None, ge=0)
    movable_point_residual_arbitrary_units: float | None = Field(default=None, ge=0)
    joint_constraint_residual: float | None = Field(default=None, ge=0)
    prismatic_orthogonal_residual: float | None = Field(default=None, ge=0)
    prismatic_rotation_leakage_degrees: float | None = Field(default=None, ge=0)
    joint_q_residual: float | None = Field(default=None, ge=0)
    axis_error_degrees: float | None = Field(default=None, ge=0, le=180)
    pivot_residual_part_diagonals: float | None = Field(default=None, ge=0)
    inferred_joint_positions: dict[str, float]
    joint_position_source: Literal[
        "measured_geometry",
        "interpolated",
        "discrete_state",
    ]
    render_paths: dict[str, str] = Field(default_factory=dict)
    view_evaluations: list[ArticulationHeldoutViewEvaluation] = Field(default_factory=list)

    @field_validator("render_paths")
    @classmethod
    def safe_articulation_state_render_paths(cls, values: dict[str, str]) -> dict[str, str]:
        return {frame_id: _relative_artifact_path(path) for frame_id, path in values.items()}

    @model_validator(mode="after")
    def heldout_view_counts_match_provenance(self) -> Self:
        views = self.view_evaluations
        if self.requested_heldout_view_count != len(views):
            raise ValueError("requested held-out view count does not match provenance")
        if self.usable_heldout_view_count != sum(item.usable for item in views):
            raise ValueError("usable held-out view count does not match provenance")
        if self.rendered_heldout_view_count != sum(item.render_path is not None for item in views):
            raise ValueError("rendered held-out view count does not match provenance")
        if self.views_with_target_masks != sum(item.target_masks_complete for item in views):
            raise ValueError("target-mask held-out view count does not match provenance")
        if self.views_with_valid_depth != sum(item.valid_depth for item in views):
            raise ValueError("valid-depth held-out view count does not match provenance")
        if self.render_paths != {
            item.frame_id: item.render_path for item in views if item.render_path is not None
        }:
            raise ValueError("held-out render paths do not match per-view provenance")
        return self


class ArticulatedCandidateEvaluation(StrictModel):
    candidate_id: Annotated[str, Field(min_length=1)]
    status: ArticulatedCandidateStatus
    fitting_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    candidate_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    fitted_model_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    link_assignment_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    heldout_evidence_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    state_evaluations: list[ArticulationStateEvaluation]
    passed_hard_gates: bool
    failed_gates: list[str]
    heldout_state_validation_used: bool
    capture_state_count: int = Field(ge=1)
    accepted_alignment_state_ids: list[str]
    selected_candidate_validation_level: ArticulationEvidenceLevel
    link_assignment_confidence: float = Field(ge=0, le=1)
    runtime_seconds: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def evaluation_gate_status(self) -> Self:
        if self.passed_hard_gates == bool(self.failed_gates):
            raise ValueError("articulated evaluation gate status is inconsistent")
        heldout = [item for item in self.state_evaluations if item.heldout]
        if self.heldout_state_validation_used != bool(heldout):
            raise ValueError("held-out validation flag does not match state evaluations")
        if self.passed_hard_gates and (
            not heldout
            or any(
                item.usable_heldout_view_count <= 0
                or item.rendered_heldout_view_count <= 0
                or item.views_with_target_masks <= 0
                or item.views_with_valid_depth <= 0
                or item.base_mask_iou is None
                or item.movable_part_mask_iou is None
                or item.whole_object_mask_iou is None
                or item.depth_inlier_fraction is None
                or item.base_motion_scene_diagonals is None
                or item.joint_constraint_residual is None
                for item in heldout
            )
        ):
            raise ValueError("passing articulation evaluation lacks required held-out evidence")
        if (
            self.selected_candidate_validation_level
            is ArticulationEvidenceLevel.MULTI_STATE_HELDOUT_VALIDATED
            and not self.passed_hard_gates
        ):
            raise ValueError("only a passing candidate may be held-out validated")
        return self


class ArticulatedEvaluationManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    request_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    fitting_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    link_assignments_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    candidate_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    evidence_split_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    measured_states_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    state_alignment_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    measured_motion_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    evaluations: list[ArticulatedCandidateEvaluation]
    candidate_structures_frozen_before_heldout: Literal[True] = True
    runtime_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    peak_host_memory_bytes: int | None = Field(default=None, ge=0)


class SelectedArtifactReference(StrictModel):
    path: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @field_validator("path")
    @classmethod
    def safe_selected_artifact_path(cls, value: str) -> str:
        return _relative_artifact_path(value)


class ArticulatedSelectedIdentityManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    articulated_object_id: Annotated[str, Field(min_length=1)]
    candidate_id: Annotated[str, Field(min_length=1)]
    selected_candidate: SelectedArtifactReference
    fitted_kinematic_model: SelectedArtifactReference
    selected_link_assignment: SelectedArtifactReference
    selected_evaluation: SelectedArtifactReference


class ArticulatedKinematicBundle(StrictModel):
    schema_version: Literal["0.2.0"] = "0.2.0"
    articulated_object_id: Annotated[str, Field(min_length=1)]
    candidate_id: Annotated[str, Field(min_length=1)]
    selected_identity_manifest: SelectedArtifactReference
    selected_candidate: SelectedArtifactReference
    fitted_kinematic_model: SelectedArtifactReference
    selected_link_assignment: SelectedArtifactReference
    selected_evaluation: SelectedArtifactReference
    base_sim3: Annotated[tuple[float, ...], Field(min_length=16, max_length=16)]
    fitting_state_q: dict[str, dict[str, float]]
    heldout_inferred_q: dict[str, dict[str, float]]
    license_record: ArticulatedLicenseRecord
    measured_joint_hypotheses: list[MeasuredJointHypothesis]
    evidence_level: ArticulationEvidenceLevel
    coordinate_convention: CoordinateConvention
    scale_status: Literal["scale_ambiguous"] = "scale_ambiguous"
    physical_validation: Literal["not_implemented"] = "not_implemented"
    collision_ready: Literal[False] = False
    sim_ready: Literal[False] = False


class ArticulatedObjectSelection(StrictModel):
    articulated_object_id: Annotated[str, Field(min_length=1)]
    status: ArticulatedCandidateStatus
    capture_state_count: int = Field(ge=1)
    capture_evidence_tier: ArticulationEvidenceLevel
    accepted_alignment_state_ids: list[str]
    effective_motion_evidence_level: ArticulationEvidenceLevel
    selected_candidate_validation_level: ArticulationEvidenceLevel
    best_research_articulated_candidate: str | None = None
    best_production_eligible_articulated_candidate: str | None = None
    selected_candidate_id: str | None = None
    candidate_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    selected_candidate_path: str | None = None
    fitted_model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    fitted_model_path: str | None = None
    link_assignment_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    link_assignment_path: str | None = None
    evaluation_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    evaluation_path: str | None = None
    selected_candidate_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    selected_identity_manifest_path: str | None = None
    selected_identity_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    kinematic_bundle_path: str | None = None
    kinematic_bundle_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    selection_rationale: list[str]
    measured_geometry_retained: Literal[True] = True
    geometry_status: Literal["articulated_visual_candidate", "partial_measured"] | None = None
    completion_status: Literal["selected_by_multi_state_validation"] | None = None
    observation_grounded: Literal[True] = True
    physical_validation: Literal["not_implemented"] = "not_implemented"
    collision_ready: Literal[False] = False
    sim_ready: Literal[False] = False
    metric_scale_known: Literal[False] = False
    canonical_gravity_alignment_known: Literal[False] = False

    @model_validator(mode="after")
    def selected_identity_is_complete(self) -> Self:
        identity_values = (
            self.selected_candidate_path,
            self.fitted_model_path,
            self.link_assignment_path,
            self.evaluation_path,
            self.fitted_model_sha256,
            self.link_assignment_sha256,
            self.evaluation_sha256,
            self.selected_candidate_sha256,
            self.selected_identity_manifest_path,
            self.selected_identity_manifest_sha256,
            self.kinematic_bundle_path,
            self.kinematic_bundle_sha256,
        )
        if self.selected_candidate_id is None and any(
            value is not None for value in identity_values
        ):
            raise ValueError("unselected articulation cannot contain selected artifact hashes")
        if self.selected_candidate_id is not None and any(
            value is None for value in identity_values
        ):
            raise ValueError("selected articulation requires all fitted/evaluation identity hashes")
        if (
            self.completion_status == "selected_by_multi_state_validation"
            and self.selected_candidate_validation_level
            is not ArticulationEvidenceLevel.MULTI_STATE_HELDOUT_VALIDATED
        ):
            raise ValueError("visual completion requires multi-state held-out validation")
        return self

    @field_validator(
        "selected_candidate_path",
        "fitted_model_path",
        "link_assignment_path",
        "evaluation_path",
        "selected_identity_manifest_path",
        "kinematic_bundle_path",
    )
    @classmethod
    def safe_selected_paths(cls, value: str | None) -> str | None:
        return _relative_artifact_path(value) if value is not None else None


class ArticulatedCandidateSelection(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    license_mode: ArticulatedLicenseMode
    ranking_policy: Literal["hard_gates_heldout_pareto_deterministic_v1"]
    objects: list[ArticulatedObjectSelection]
    deterministic_selection_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ArticulationDiagnostics(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    capture_state_count: int = Field(ge=0)
    capture_evidence_tier: ArticulationEvidenceLevel
    accepted_alignment_state_ids: list[str]
    effective_motion_evidence_level: ArticulationEvidenceLevel
    selected_candidate_validation_level: ArticulationEvidenceLevel
    aligned_state_count: int = Field(ge=0)
    measured_part_count: int = Field(ge=0)
    joint_hypothesis_count: int = Field(ge=0)
    candidate_count_by_family: dict[str, int]
    fitted_candidate_count: int = Field(ge=0)
    evaluated_candidate_count: int = Field(ge=0)
    passing_candidate_count: int = Field(ge=0)
    total_runtime_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    peak_host_memory_bytes: int | None = Field(default=None, ge=0)
    warnings: list[str] = Field(default_factory=list)


class ArticulationPreviewManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    preview_paths: dict[str, str]
    deterministic: Literal[True] = True

    @field_validator("preview_paths")
    @classmethod
    def safe_articulation_preview_paths(cls, values: dict[str, str]) -> dict[str, str]:
        return {name: _relative_artifact_path(value) for name, value in values.items()}


class Phase5CConsistencyReport(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    passed: bool
    checks: list[EndToEndConsistencyCheck]
    multi_state_evidence_available: bool
    measured_joint_motion_available: bool
    heldout_state_validation_used: bool
    capture_state_count: int = Field(ge=1)
    capture_evidence_tier: ArticulationEvidenceLevel
    accepted_alignment_state_ids: list[str]
    effective_motion_evidence_level: ArticulationEvidenceLevel
    selected_candidate_validation_level: ArticulationEvidenceLevel
    physical_joint_validation_implemented: Literal[False] = False
    collision_generation_implemented: Literal[False] = False
    dynamics_identification_implemented: Literal[False] = False
    metric_scale_known: Literal[False] = False
    canonical_gravity_alignment_known: Literal[False] = False
    sim_ready_scene_implemented: Literal[False] = False
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def phase5c_summary_matches_checks(self) -> Self:
        if self.passed != all(check.passed for check in self.checks):
            raise ValueError("Phase 5C pass status must match its checks")
        return self
