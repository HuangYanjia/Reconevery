from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, Self

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
    WorldFrameStatus,
)


def _relative_artifact_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ValueError("artifact paths must be relative to the run directory")
    return value


def _relative_source_reference(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value == "":
        raise ValueError("source references must be relative to the configured input path")
    return value


class InputSourceType(StrEnum):
    MOCK = "mock"
    VIDEO = "video"
    GENERATED_TEST_IMAGE = "generated_test_image"
    IMAGE_DIRECTORY = "image_directory"


class FrameManifestEntry(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    relative_path: Annotated[str, Field(min_length=1)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    timestamp_s: float = Field(ge=0)
    source_type: InputSourceType
    source_file_reference: str | None = None
    original_frame_index: int | None = Field(default=None, ge=0)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _relative_artifact_path(value)

    @field_validator("source_file_reference")
    @classmethod
    def validate_source_reference(cls, value: str | None) -> str | None:
        return _relative_source_reference(value) if value is not None else None


class IngestManifest(StrictModel):
    source_type: InputSourceType
    frames: Annotated[list[FrameManifestEntry], Field(min_length=1)]
    source_input_reference: str | None = None
    source_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    ffmpeg_version: str | None = None
    ffprobe_version: str | None = None
    extraction_configuration: dict[str, Any] = Field(default_factory=dict)
    total_decoded_frames: int | None = Field(default=None, ge=0)
    selected_frames: int | None = Field(default=None, ge=0)
    dropped_frames: int | None = Field(default=None, ge=0)
    output_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]] = Field(
        default_factory=dict
    )
    frame_qa_path: str | None = None
    provenance: ProvenanceRecord

    @field_validator("frame_qa_path")
    @classmethod
    def validate_optional_artifact_path(cls, value: str | None) -> str | None:
        return _relative_artifact_path(value) if value is not None else None

    @field_validator("source_input_reference")
    @classmethod
    def validate_input_reference(cls, value: str | None) -> str | None:
        return _relative_source_reference(value) if value is not None else None

    @model_validator(mode="after")
    def unique_frames(self) -> Self:
        ids = [frame.frame_id for frame in self.frames]
        paths = [frame.relative_path for frame in self.frames]
        if len(ids) != len(set(ids)):
            raise ValueError("ingest frame IDs must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("ingest frame paths must be unique")
        if self.selected_frames is not None and self.selected_frames != len(self.frames):
            raise ValueError("selected_frames must equal the normalized manifest frame count")
        if (
            self.total_decoded_frames is not None
            and self.selected_frames is not None
            and self.dropped_frames is not None
            and self.total_decoded_frames != self.selected_frames + self.dropped_frames
        ):
            raise ValueError("decoded frame count must equal selected plus dropped frames")
        if self.output_hashes and self.output_hashes != {
            frame.relative_path: frame.sha256 for frame in self.frames
        }:
            raise ValueError("ingest output hashes must exactly match normalized frame entries")
        return self


class CameraReconstruction(StrictModel):
    camera_id: Annotated[str, Field(min_length=1)]
    model: Annotated[str, Field(min_length=1)] = "pinhole"
    intrinsics: CameraIntrinsics
    poses: Annotated[list[CameraPose], Field(min_length=1)]
    registered_frame_ids: list[str] = Field(default_factory=list)
    unregistered_frame_ids: list[str] = Field(default_factory=list)
    confidence: ConfidenceRecord
    coordinate_convention: CoordinateConvention
    scale_status: ScaleStatus = ScaleStatus.METRIC_SCALE_KNOWN
    world_frame_status: WorldFrameStatus = WorldFrameStatus.RECON2SIM_ALIGNED
    provenance: ProvenanceRecord

    @model_validator(mode="before")
    @classmethod
    def default_legacy_registration_ids(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "registered_frame_ids" in value:
            return value
        payload = dict(value)
        frame_ids: list[str] = []
        raw_poses = payload.get("poses", [])
        if isinstance(raw_poses, list):
            for pose in raw_poses:
                if isinstance(pose, CameraPose):
                    frame_ids.append(pose.frame_id)
                elif isinstance(pose, dict) and isinstance(pose.get("frame_id"), str):
                    frame_ids.append(pose["frame_id"])
        payload["registered_frame_ids"] = frame_ids
        return payload

    @model_validator(mode="after")
    def frame_registration_is_consistent(self) -> Self:
        pose_ids = [pose.frame_id for pose in self.poses]
        if len(pose_ids) != len(set(pose_ids)):
            raise ValueError("camera pose frame IDs must be unique")
        if len(self.registered_frame_ids) != len(set(self.registered_frame_ids)):
            raise ValueError("registered frame IDs must be unique")
        if len(self.unregistered_frame_ids) != len(set(self.unregistered_frame_ids)):
            raise ValueError("unregistered frame IDs must be unique")
        if set(self.registered_frame_ids) != set(pose_ids):
            raise ValueError("registered frame IDs must exactly match camera pose frame IDs")
        overlap = set(self.registered_frame_ids) & set(self.unregistered_frame_ids)
        if overlap:
            raise ValueError(
                f"frames cannot be both registered and unregistered: {sorted(overlap)}"
            )
        if (
            self.scale_status is ScaleStatus.SCALE_AMBIGUOUS
            and self.coordinate_convention.units != "arbitrary_scale"
        ):
            raise ValueError("scale_ambiguous cameras must use arbitrary_scale units")
        if (
            self.scale_status is not ScaleStatus.SCALE_AMBIGUOUS
            and self.coordinate_convention.units != "meters"
        ):
            raise ValueError("known or externally scaled cameras must use meter units")
        if (
            self.world_frame_status is WorldFrameStatus.COLMAP_UNALIGNED
            and self.coordinate_convention.world_axes != "colmap_arbitrary"
        ):
            raise ValueError("colmap_unaligned cameras must use colmap_arbitrary world axes")
        if (
            self.world_frame_status is WorldFrameStatus.RECON2SIM_ALIGNED
            and self.coordinate_convention.world_axes != "x_forward_y_left_z_up"
        ):
            raise ValueError("aligned cameras must use x_forward_y_left_z_up world axes")
        return self


class FrameQualityEntry(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    source_file_reference: Annotated[str, Field(min_length=1)]
    normalized_path: str | None = None
    rejected_path: str | None = None
    original_frame_index: int = Field(ge=0)
    blur_score: float = Field(ge=0)
    mean_brightness: float = Field(ge=0, le=255)
    intensity_variance: float = Field(ge=0)
    duplicate_score: float | None = Field(default=None, ge=0, le=1)
    is_duplicate: bool
    selected: bool
    rejection_reason: str | None = None

    @field_validator("source_file_reference")
    @classmethod
    def validate_source_path(cls, value: str) -> str:
        return _relative_source_reference(value)

    @field_validator("normalized_path", "rejected_path")
    @classmethod
    def validate_output_paths(cls, value: str | None) -> str | None:
        return _relative_artifact_path(value) if value is not None else None

    @model_validator(mode="after")
    def selection_has_consistent_paths(self) -> Self:
        if self.selected and self.normalized_path is None:
            raise ValueError("selected frame QA entries require normalized_path")
        if self.selected and self.rejection_reason is not None:
            raise ValueError("selected frame QA entries cannot have a rejection reason")
        if not self.selected and self.rejection_reason is None:
            raise ValueError("rejected frame QA entries require a rejection reason")
        return self


class FrameQualityReport(StrictModel):
    method: Literal["cpu_grayscale_statistics_v1"] = "cpu_grayscale_statistics_v1"
    thresholds: dict[str, float | bool]
    entries: Annotated[list[FrameQualityEntry], Field(min_length=1)]
    selected_count: int = Field(ge=0)
    dropped_count: int = Field(ge=0)
    provenance: ProvenanceRecord

    @model_validator(mode="after")
    def counts_match_entries(self) -> Self:
        selected = sum(entry.selected for entry in self.entries)
        if selected != self.selected_count or len(self.entries) - selected != self.dropped_count:
            raise ValueError("frame QA counts must match entry selection status")
        frame_ids = [entry.frame_id for entry in self.entries]
        original_indices = [entry.original_frame_index for entry in self.entries]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("frame QA frame IDs must be unique")
        if len(original_indices) != len(set(original_indices)):
            raise ValueError("frame QA original indices must be unique")
        return self


class ColmapModelDiagnostic(StrictModel):
    model_id: Annotated[str, Field(min_length=1)]
    registered_frames: int = Field(ge=0)
    registration_ratio: float = Field(ge=0, le=1)
    sparse_points: int = Field(ge=0)
    mean_track_length: float = Field(ge=0)
    mean_reprojection_error: float | None = Field(default=None, ge=0)
    selected: bool = False
    rejection_reason: str | None = None


class CameraDiagnostics(StrictModel):
    input_frame_count: int = Field(gt=0)
    selected_frame_count: int = Field(gt=0)
    registered_frames: int = Field(ge=0)
    registration_ratio: float = Field(ge=0, le=1)
    sparse_points: int = Field(ge=0)
    camera_model: str | None = None
    selected_model: str | None = None
    models: list[ColmapModelDiagnostic] = Field(default_factory=list)
    scale_status: ScaleStatus
    world_frame_status: WorldFrameStatus
    confidence_score: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


class ToolCommandRecord(StrictModel):
    name: Annotated[str, Field(min_length=1)]
    arguments: Annotated[list[str], Field(min_length=1)]
    return_code: int
    duration_s: float = Field(ge=0)
    stdout_path: Annotated[str, Field(min_length=1)]
    stderr_path: Annotated[str, Field(min_length=1)]

    @field_validator("stdout_path", "stderr_path")
    @classmethod
    def validate_log_paths(cls, value: str) -> str:
        return _relative_artifact_path(value)


class ColmapWorkspaceManifest(StrictModel):
    execution_mode: Literal["local", "docker"]
    colmap_version: Annotated[str, Field(min_length=1)]
    database_path: Annotated[str, Field(min_length=1)]
    sparse_model_paths: list[str]
    selected_model: str
    input_frame_hashes: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    configuration: dict[str, Any]
    commands: Annotated[list[ToolCommandRecord], Field(min_length=1)]
    provenance: ProvenanceRecord

    @field_validator("database_path", "sparse_model_paths")
    @classmethod
    def validate_workspace_paths(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, str):
            return _relative_artifact_path(value)
        return [_relative_artifact_path(path) for path in value]


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
    interrupted: bool = False
    stdout_path: Annotated[str, Field(min_length=1)]
    stderr_path: Annotated[str, Field(min_length=1)]
