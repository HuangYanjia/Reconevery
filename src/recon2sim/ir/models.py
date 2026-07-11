from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssetType(StrEnum):
    STATIC_STRUCTURE = "static_structure"
    RIGID = "rigid"
    ARTICULATED = "articulated"
    DEFORMABLE = "deformable"
    FLUID_OR_PARTICLE = "fluid_or_particle"
    IGNORE = "ignore"


class GeometrySourceType(StrEnum):
    MEASURED = "measured"
    RADIANCE_FIELD = "radiance_field"
    GENERATED = "generated"
    RETRIEVED = "retrieved"
    FUSED = "fused"
    MOCK = "mock"


class RelationType(StrEnum):
    SUPPORTED_BY = "supported_by"
    CONTAINS = "contains"
    INSIDE = "inside"
    ATTACHED_TO = "attached_to"
    PART_OF = "part_of"
    IN_CONTACT_WITH = "in_contact_with"
    ARTICULATES_RELATIVE_TO = "articulates_relative_to"
    OCCLUDES = "occludes"
    REACHABLE_BY = "reachable_by"


class Transform(StrictModel):
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


class ConfidenceRecord(StrictModel):
    score: float = Field(ge=0.0, le=1.0)
    method: str
    notes: str | None = None


class ProvenanceRecord(StrictModel):
    adapter_name: str
    adapter_version: str | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)
    input_artifact_paths: list[str] = Field(default_factory=list)
    output_artifact_paths: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    confidence: ConfidenceRecord
    source: GeometrySourceType


class SceneMetadata(StrictModel):
    scene_id: str
    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    units: Literal["meters"] = "meters"
    source: GeometrySourceType
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class CameraIntrinsics(StrictModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float
    distortion: list[float] = Field(default_factory=list)


class CameraPose(StrictModel):
    frame_id: str
    transform_world_from_camera: Transform
    confidence: ConfidenceRecord


class Camera(StrictModel):
    camera_id: str
    model: str
    intrinsics: CameraIntrinsics
    poses: list[CameraPose] = Field(default_factory=list)
    provenance: ProvenanceRecord


class ObjectObservation(StrictModel):
    object_id: str
    frame_id: str
    bbox_xywh: tuple[int, int, int, int]
    mask_path: str | None = None
    confidence: ConfidenceRecord


class FrameObservation(StrictModel):
    frame_id: str
    frame_path: str
    timestamp_s: float
    camera_id: str
    observations: list[ObjectObservation] = Field(default_factory=list)


class GeometryAsset(StrictModel):
    asset_id: str
    asset_type: AssetType
    uri: str
    format: str
    source: GeometrySourceType
    provenance: ProvenanceRecord


class MaterialAsset(StrictModel):
    material_id: str
    name: str
    base_color_rgba: tuple[float, float, float, float]
    uri: str | None = None
    provenance: ProvenanceRecord


class CollisionAsset(StrictModel):
    collision_id: str
    uri: str
    format: str
    source: GeometrySourceType
    provenance: ProvenanceRecord


class PhysicsProperties(StrictModel):
    mass_kg: float | None = Field(default=None, ge=0)
    friction: float | None = Field(default=None, ge=0)
    restitution: float | None = Field(default=None, ge=0, le=1)
    is_static: bool = False


class Link(StrictModel):
    link_id: str
    name: str
    transform: Transform = Field(default_factory=Transform)
    geometry_asset_ids: list[str] = Field(default_factory=list)


class Joint(StrictModel):
    joint_id: str
    parent_link_id: str
    child_link_id: str
    joint_type: Literal["fixed", "revolute", "prismatic"]
    axis_xyz: tuple[float, float, float] = (1.0, 0.0, 0.0)
    limits: tuple[float, float] | None = None


class Articulation(StrictModel):
    articulation_id: str
    links: list[Link]
    joints: list[Joint]


class ObjectInstance(StrictModel):
    object_id: str
    name: str
    asset_type: AssetType
    transform: Transform = Field(default_factory=Transform)
    geometry_asset_ids: list[str] = Field(default_factory=list)
    material_asset_ids: list[str] = Field(default_factory=list)
    collision_asset_ids: list[str] = Field(default_factory=list)
    physics: PhysicsProperties = Field(default_factory=PhysicsProperties)
    articulation: Articulation | None = None
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    confidence: ConfidenceRecord


class SceneRelation(StrictModel):
    relation_type: RelationType
    subject_id: str
    object_id: str
    confidence: ConfidenceRecord
    provenance: ProvenanceRecord


class ValidationIssue(StrictModel):
    severity: Literal["info", "warning", "error"]
    code: str
    message: str
    object_id: str | None = None


class ValidationReport(StrictModel):
    scene_id: str
    passed: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SceneIR(StrictModel):
    schema_version: str = "0.1.0"
    metadata: SceneMetadata
    cameras: list[Camera] = Field(default_factory=list)
    frames: list[FrameObservation] = Field(default_factory=list)
    objects: list[ObjectInstance] = Field(default_factory=list)
    geometry_assets: list[GeometryAsset] = Field(default_factory=list)
    material_assets: list[MaterialAsset] = Field(default_factory=list)
    collision_assets: list[CollisionAsset] = Field(default_factory=list)
    relations: list[SceneRelation] = Field(default_factory=list)
    validation: ValidationReport | None = None

    @field_validator("objects")
    @classmethod
    def unique_objects(cls, objects: list[ObjectInstance]) -> list[ObjectInstance]:
        ids = [obj.object_id for obj in objects]
        if len(ids) != len(set(ids)):
            raise ValueError("object_id values must be unique")
        return objects
