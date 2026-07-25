from __future__ import annotations

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
    strategy: Annotated[str, Field(min_length=1)]
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
