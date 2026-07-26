from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Identifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
ConfidenceScore = Annotated[float, Field(ge=0.0, le=1.0)]


def _duplicates(values: list[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ValueError("artifact paths must be non-empty paths relative to the run directory")
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AssetType(StrEnum):
    UNCLASSIFIED = "unclassified"
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


class ScaleStatus(StrEnum):
    METRIC_SCALE_KNOWN = "metric_scale_known"
    SCALE_AMBIGUOUS = "scale_ambiguous"
    EXTERNALLY_SCALED = "externally_scaled"


class WorldFrame(StrEnum):
    CANONICAL_X_FORWARD_Y_LEFT_Z_UP = "canonical_x_forward_y_left_z_up"
    COLMAP_ARBITRARY = "colmap_arbitrary"


class AlignmentStatus(StrEnum):
    CANONICAL = "canonical"
    GRAVITY_ALIGNED = "gravity_aligned"
    UNORIENTED = "unoriented"


class CameraAxes(StrEnum):
    UNSPECIFIED = "unspecified"
    X_RIGHT_Y_DOWN_Z_FORWARD = "x_right_y_down_z_forward"


class LinearUnits(StrEnum):
    METERS = "meters"
    ARBITRARY_UNITS = "arbitrary_units"


class TransformDirection(StrEnum):
    WORLD_FROM_CAMERA = "world_from_camera"


class CoordinateConvention(StrictModel):
    world_frame: WorldFrame = WorldFrame.CANONICAL_X_FORWARD_Y_LEFT_Z_UP
    alignment_status: AlignmentStatus = AlignmentStatus.CANONICAL
    camera_axes: CameraAxes = CameraAxes.UNSPECIFIED
    handedness: Literal["right"] = "right"
    linear_units: LinearUnits = LinearUnits.METERS
    scale_status: ScaleStatus = ScaleStatus.METRIC_SCALE_KNOWN
    quaternion_order: Literal["xyzw"] = "xyzw"
    transform_direction: TransformDirection = TransformDirection.WORLD_FROM_CAMERA

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_coordinate_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        legacy_world_axes = migrated.pop("world_axes", None)
        if legacy_world_axes is not None:
            if legacy_world_axes != "x_forward_y_left_z_up":
                raise ValueError(f"unsupported legacy world_axes value {legacy_world_axes!r}")
            migrated.setdefault(
                "world_frame",
                WorldFrame.CANONICAL_X_FORWARD_Y_LEFT_Z_UP,
            )
        legacy_units = migrated.pop("units", None)
        if legacy_units is not None:
            if "linear_units" in migrated and migrated["linear_units"] != legacy_units:
                raise ValueError("legacy units and linear_units disagree")
            migrated.setdefault("linear_units", legacy_units)
        legacy_direction = migrated.pop("camera_transform_direction", None)
        if legacy_direction is not None:
            if (
                "transform_direction" in migrated
                and migrated["transform_direction"] != legacy_direction
            ):
                raise ValueError(
                    "legacy camera_transform_direction and transform_direction disagree"
                )
            migrated.setdefault("transform_direction", legacy_direction)
        return migrated


class Transform(StrictModel):
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_metric_translation(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "translation_m" not in value:
            return value
        migrated = dict(value)
        legacy = migrated.pop("translation_m")
        if "translation" in migrated and migrated["translation"] != legacy:
            raise ValueError("legacy translation_m and translation disagree")
        migrated.setdefault("translation", legacy)
        return migrated

    @field_validator("rotation_xyzw")
    @classmethod
    def nonzero_quaternion(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        if sum(component * component for component in value) == 0:
            raise ValueError("rotation quaternion must be non-zero")
        return value

    @field_validator("scale")
    @classmethod
    def positive_scale(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(component <= 0 for component in value):
            raise ValueError("scale components must be positive")
        return value


class ConfidenceRecord(StrictModel):
    score: ConfidenceScore
    method: Annotated[str, Field(min_length=1)]
    notes: str | None = None


class ProvenanceRecord(StrictModel):
    adapter_name: Annotated[str, Field(min_length=1)]
    adapter_version: Annotated[str, Field(min_length=1)]
    configuration: dict[str, Any] = Field(default_factory=dict)
    input_artifact_paths: list[str] = Field(default_factory=list)
    output_artifact_paths: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    confidence: ConfidenceRecord
    source: GeometrySourceType

    @field_validator("input_artifact_paths", "output_artifact_paths")
    @classmethod
    def relative_artifact_paths(cls, values: list[str]) -> list[str]:
        return [_relative_path(value) for value in values]


class SceneMetadata(StrictModel):
    scene_id: Identifier
    name: Annotated[str, Field(min_length=1)]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    coordinate_convention: CoordinateConvention = Field(default_factory=CoordinateConvention)
    source: GeometrySourceType
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class CameraIntrinsics(StrictModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float = Field(ge=0)
    cy: float = Field(ge=0)
    distortion: list[float] = Field(default_factory=list)


class CameraPose(StrictModel):
    frame_id: Identifier
    transform_world_from_camera: Transform
    confidence: ConfidenceRecord


class Camera(StrictModel):
    camera_id: Identifier
    model: Annotated[str, Field(min_length=1)]
    intrinsics: CameraIntrinsics
    poses: list[CameraPose] = Field(default_factory=list)
    coordinate_convention: CoordinateConvention = Field(default_factory=CoordinateConvention)
    scale_status: ScaleStatus = ScaleStatus.METRIC_SCALE_KNOWN
    provenance: ProvenanceRecord

    @model_validator(mode="after")
    def unique_pose_frames(self) -> Self:
        duplicates = _duplicates([pose.frame_id for pose in self.poses])
        if duplicates:
            raise ValueError(
                f"camera {self.camera_id!r} has duplicate pose frames: {sorted(duplicates)}"
            )
        return self


class ObjectObservation(StrictModel):
    object_id: Identifier
    frame_id: Identifier
    bbox_xywh: tuple[int, int, int, int]
    mask_path: str
    confidence: ConfidenceRecord

    @field_validator("bbox_xywh")
    @classmethod
    def valid_bbox(cls, value: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, width, height = value
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("bounding boxes require non-negative origins and positive dimensions")
        return value

    @field_validator("mask_path")
    @classmethod
    def relative_mask_path(cls, value: str) -> str:
        return _relative_path(value)


class FrameObservation(StrictModel):
    frame_id: Identifier
    frame_path: str
    timestamp_s: float = Field(ge=0)
    camera_id: Identifier
    observations: list[ObjectObservation] = Field(default_factory=list)

    @field_validator("frame_path")
    @classmethod
    def relative_frame_path(cls, value: str) -> str:
        return _relative_path(value)

    @model_validator(mode="after")
    def observations_match_frame(self) -> Self:
        mismatches = [obs.frame_id for obs in self.observations if obs.frame_id != self.frame_id]
        if mismatches:
            raise ValueError(f"observations must refer to frame {self.frame_id!r}")
        return self


class GeometryAsset(StrictModel):
    asset_id: Identifier
    asset_type: AssetType
    uri: str
    format: Literal["obj", "glb", "ply"]
    source: GeometrySourceType
    coordinate_convention: CoordinateConvention | None = None
    scale_status: ScaleStatus | None = None
    geometry_status: Literal["partial_observation_supported"] | None = None
    completion_status: Literal["not_completed"] | None = None
    sim_ready: bool | None = None
    source_asset_id: Identifier | None = None
    alignment_transform_path: str | None = None
    geometry_alignment_status: (
        Literal[
            "identity_already_consistent",
            "accepted_global_sim3",
            "alignment_rejected",
        ]
        | None
    ) = None
    provenance: ProvenanceRecord

    @field_validator("uri", "alignment_transform_path")
    @classmethod
    def relative_geometry_path(cls, value: str | None) -> str | None:
        return _relative_path(value) if value is not None else None


class MaterialAsset(StrictModel):
    asset_id: Identifier
    name: Annotated[str, Field(min_length=1)]
    base_color_rgba: tuple[
        Annotated[float, Field(ge=0, le=1)],
        Annotated[float, Field(ge=0, le=1)],
        Annotated[float, Field(ge=0, le=1)],
        Annotated[float, Field(ge=0, le=1)],
    ]
    uri: str | None = None
    provenance: ProvenanceRecord

    @field_validator("uri")
    @classmethod
    def relative_optional_uri(cls, value: str | None) -> str | None:
        return _relative_path(value) if value is not None else None


class CollisionAsset(StrictModel):
    asset_id: Identifier
    uri: str
    format: Literal["obj", "glb", "ply"]
    source: GeometrySourceType
    provenance: ProvenanceRecord

    @field_validator("uri")
    @classmethod
    def relative_uri(cls, value: str) -> str:
        return _relative_path(value)


class PhysicsProperties(StrictModel):
    mass_kg: float | None = Field(default=None, ge=0)
    friction: float | None = Field(default=None, ge=0)
    restitution: float | None = Field(default=None, ge=0, le=1)
    is_static: bool = False


class Link(StrictModel):
    link_id: Identifier
    name: Annotated[str, Field(min_length=1)]
    transform: Transform = Field(default_factory=Transform)
    geometry_asset_ids: list[Identifier] = Field(default_factory=list)
    material_asset_ids: list[Identifier] = Field(default_factory=list)
    collision_asset_ids: list[Identifier] = Field(default_factory=list)


class Joint(StrictModel):
    joint_id: Identifier
    parent_link_id: Identifier
    child_link_id: Identifier
    joint_type: Literal["fixed", "revolute", "prismatic"]
    axis_xyz: tuple[float, float, float] = (1.0, 0.0, 0.0)
    limits: tuple[float, float] | None = None

    @field_validator("axis_xyz")
    @classmethod
    def nonzero_axis(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if all(component == 0 for component in value):
            raise ValueError("joint axis must be non-zero")
        return value

    @field_validator("limits")
    @classmethod
    def ordered_limits(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        if value is not None and value[0] > value[1]:
            raise ValueError("joint limits must be ordered lower, upper")
        return value


class Articulation(StrictModel):
    articulation_id: Identifier
    links: Annotated[list[Link], Field(min_length=1)]
    joints: list[Joint] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_link_graph(self) -> Self:
        link_ids = [link.link_id for link in self.links]
        duplicate_links = _duplicates(link_ids)
        duplicate_joints = _duplicates([joint.joint_id for joint in self.joints])
        if duplicate_links:
            raise ValueError(f"duplicate articulation link IDs: {sorted(duplicate_links)}")
        if duplicate_joints:
            raise ValueError(f"duplicate articulation joint IDs: {sorted(duplicate_joints)}")
        known_links = set(link_ids)
        for joint in self.joints:
            missing = {joint.parent_link_id, joint.child_link_id} - known_links
            if missing:
                raise ValueError(
                    f"joint {joint.joint_id!r} references unknown links: {sorted(missing)}"
                )
            if joint.parent_link_id == joint.child_link_id:
                raise ValueError(f"joint {joint.joint_id!r} cannot connect a link to itself")
        return self


class ObjectInstance(StrictModel):
    object_id: Identifier
    name: Annotated[str, Field(min_length=1)]
    asset_type: AssetType
    transform: Transform = Field(default_factory=Transform)
    geometry_asset_ids: list[Identifier] = Field(default_factory=list)
    material_asset_ids: list[Identifier] = Field(default_factory=list)
    collision_asset_ids: list[Identifier] = Field(default_factory=list)
    physics: PhysicsProperties = Field(default_factory=PhysicsProperties)
    articulation: Articulation | None = None
    geometry_status: Literal["partial_observation_supported"] | None = None
    completion_status: Literal["not_completed"] | None = None
    sim_ready: bool | None = None
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    confidence: ConfidenceRecord

    @model_validator(mode="after")
    def articulation_matches_type(self) -> Self:
        if self.asset_type is AssetType.ARTICULATED and self.articulation is None:
            raise ValueError(f"articulated object {self.object_id!r} must contain an articulation")
        if self.asset_type is not AssetType.ARTICULATED and self.articulation is not None:
            raise ValueError(
                f"non-articulated object {self.object_id!r} must not contain an articulation"
            )
        return self


class SceneRelation(StrictModel):
    relation_type: RelationType
    subject_id: Identifier
    object_id: Identifier
    confidence: ConfidenceRecord
    provenance: ProvenanceRecord


class ValidationIssue(StrictModel):
    severity: Literal["info", "warning", "error"]
    code: Identifier
    message: Annotated[str, Field(min_length=1)]
    object_id: Identifier | None = None


class ValidationReport(StrictModel):
    scene_id: Identifier
    passed: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SceneIR(StrictModel):
    schema_version: Literal["0.1.0", "0.1.1", "0.1.2", "0.1.3"] = "0.1.1"
    metadata: SceneMetadata
    cameras: list[Camera] = Field(default_factory=list)
    frames: list[FrameObservation] = Field(default_factory=list)
    objects: list[ObjectInstance] = Field(default_factory=list)
    geometry_assets: list[GeometryAsset] = Field(default_factory=list)
    material_assets: list[MaterialAsset] = Field(default_factory=list)
    collision_assets: list[CollisionAsset] = Field(default_factory=list)
    relations: list[SceneRelation] = Field(default_factory=list)
    validation: ValidationReport | None = None

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        object_ids = [obj.object_id for obj in self.objects]
        camera_ids = [camera.camera_id for camera in self.cameras]
        frame_ids = [frame.frame_id for frame in self.frames]
        asset_ids = [asset.asset_id for asset in self.geometry_assets]
        asset_ids += [asset.asset_id for asset in self.material_assets]
        asset_ids += [asset.asset_id for asset in self.collision_assets]
        link_ids = [
            link.link_id
            for obj in self.objects
            if obj.articulation is not None
            for link in obj.articulation.links
        ]
        joint_ids = [
            joint.joint_id
            for obj in self.objects
            if obj.articulation is not None
            for joint in obj.articulation.joints
        ]

        for label, values in (
            ("object", object_ids),
            ("camera", camera_ids),
            ("frame", frame_ids),
            ("asset", asset_ids),
            ("articulation link", link_ids),
            ("articulation joint", joint_ids),
        ):
            duplicates = _duplicates(values)
            if duplicates:
                raise ValueError(f"duplicate {label} IDs: {sorted(duplicates)}")

        known_objects = set(object_ids)
        known_cameras = set(camera_ids)
        known_frames = set(frame_ids)
        geometry_ids = {asset.asset_id for asset in self.geometry_assets}
        material_ids = {asset.asset_id for asset in self.material_assets}
        collision_ids = {asset.asset_id for asset in self.collision_assets}

        for obj in self.objects:
            self._check_asset_references(
                f"object {obj.object_id!r}",
                obj.geometry_asset_ids,
                obj.material_asset_ids,
                obj.collision_asset_ids,
                geometry_ids,
                material_ids,
                collision_ids,
            )
            if obj.articulation is not None:
                for link in obj.articulation.links:
                    self._check_asset_references(
                        f"link {link.link_id!r}",
                        link.geometry_asset_ids,
                        link.material_asset_ids,
                        link.collision_asset_ids,
                        geometry_ids,
                        material_ids,
                        collision_ids,
                    )

        for relation in self.relations:
            missing = {relation.subject_id, relation.object_id} - known_objects
            if missing:
                raise ValueError(
                    f"relation {relation.relation_type.value!r} references unknown objects: "
                    f"{sorted(missing)}"
                )

        for frame in self.frames:
            if frame.camera_id not in known_cameras:
                raise ValueError(
                    f"frame {frame.frame_id!r} references unknown camera {frame.camera_id!r}"
                )
            missing_objects = {obs.object_id for obs in frame.observations} - known_objects
            if missing_objects:
                raise ValueError(
                    f"frame {frame.frame_id!r} references unknown objects: "
                    f"{sorted(missing_objects)}"
                )

        for camera in self.cameras:
            missing_pose_frames = {pose.frame_id for pose in camera.poses} - known_frames
            if missing_pose_frames:
                raise ValueError(
                    f"camera {camera.camera_id!r} has poses for unknown frames: "
                    f"{sorted(missing_pose_frames)}"
                )
        return self

    @staticmethod
    def _check_asset_references(
        owner: str,
        geometry_references: list[str],
        material_references: list[str],
        collision_references: list[str],
        geometry_ids: set[str],
        material_ids: set[str],
        collision_ids: set[str],
    ) -> None:
        missing_geometry = set(geometry_references) - geometry_ids
        missing_material = set(material_references) - material_ids
        missing_collision = set(collision_references) - collision_ids
        if missing_geometry:
            raise ValueError(
                f"{owner} references unknown geometry assets: {sorted(missing_geometry)}"
            )
        if missing_material:
            raise ValueError(
                f"{owner} references unknown material assets: {sorted(missing_material)}"
            )
        if missing_collision:
            raise ValueError(
                f"{owner} references unknown collision assets: {sorted(missing_collision)}"
            )
