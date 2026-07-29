from __future__ import annotations

import json
import math
from typing import Literal

from recon2sim.adapters.base import (
    HealthcheckResult,
    InputSpec,
    OutputSpec,
    StageContext,
    StageResult,
)
from recon2sim.artifacts import (
    CanonicalAssetMapping,
    CanonicalPrismaticUnitMapping,
    CanonicalSceneWrapper,
    EndToEndConsistencyCheck,
    Phase6AConsistencyReport,
    WorldCalibrationArtifact,
    WorldCalibrationManifest,
    WorldCalibrationStatus,
)
from recon2sim.calibration import (
    multiply_matrix4,
    sha256_file,
)
from recon2sim.ir import (
    AlignmentStatus,
    CameraAxes,
    CoordinateConvention,
    LinearUnits,
    ScaleStatus,
    SceneIR,
    Transform,
    TransformDirection,
    WorldCalibrationSceneReference,
    WorldFrame,
)
from recon2sim.storage import atomic_write_json


def _quaternion_matrix(value: tuple[float, float, float, float]) -> tuple[float, ...]:
    x, y, z, w = value
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return (
        1 - 2 * (y * y + z * z),
        2 * (x * y - z * w),
        2 * (x * z + y * w),
        2 * (x * y + z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z - x * w),
        2 * (x * z - y * w),
        2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
    )


def _transform_matrix(value: Transform) -> tuple[float, ...]:
    rotation = _quaternion_matrix(value.rotation_xyzw)
    if max(value.scale) - min(value.scale) > 1e-9:
        raise ValueError("canonical propagation requires uniform source transform scale")
    scale = value.scale[0]
    return (
        scale * rotation[0],
        scale * rotation[1],
        scale * rotation[2],
        value.translation[0],
        scale * rotation[3],
        scale * rotation[4],
        scale * rotation[5],
        value.translation[1],
        scale * rotation[6],
        scale * rotation[7],
        scale * rotation[8],
        value.translation[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _matrix_transform(matrix: tuple[float, ...]) -> Transform:
    scale = math.sqrt(matrix[0] ** 2 + matrix[4] ** 2 + matrix[8] ** 2)
    rotation = tuple(matrix[row * 4 + column] / scale for row in range(3) for column in range(3))
    trace = rotation[0] + rotation[4] + rotation[8]
    if trace > 0:
        factor = math.sqrt(trace + 1.0) * 2
        quaternion = (
            (rotation[7] - rotation[5]) / factor,
            (rotation[2] - rotation[6]) / factor,
            (rotation[3] - rotation[1]) / factor,
            0.25 * factor,
        )
    elif rotation[0] > rotation[4] and rotation[0] > rotation[8]:
        factor = math.sqrt(1 + rotation[0] - rotation[4] - rotation[8]) * 2
        quaternion = (
            0.25 * factor,
            (rotation[1] + rotation[3]) / factor,
            (rotation[2] + rotation[6]) / factor,
            (rotation[7] - rotation[5]) / factor,
        )
    elif rotation[4] > rotation[8]:
        factor = math.sqrt(1 + rotation[4] - rotation[0] - rotation[8]) * 2
        quaternion = (
            (rotation[1] + rotation[3]) / factor,
            0.25 * factor,
            (rotation[5] + rotation[7]) / factor,
            (rotation[2] - rotation[6]) / factor,
        )
    else:
        factor = math.sqrt(1 + rotation[8] - rotation[0] - rotation[4]) * 2
        quaternion = (
            (rotation[2] + rotation[6]) / factor,
            (rotation[5] + rotation[7]) / factor,
            0.25 * factor,
            (rotation[3] - rotation[1]) / factor,
        )
    quaternion_norm = math.sqrt(sum(item * item for item in quaternion))
    return Transform(
        translation=(matrix[3], matrix[7], matrix[11]),
        rotation_xyzw=tuple(item / quaternion_norm for item in quaternion),  # type: ignore[arg-type]
        scale=(scale, scale, scale),
    )


def _canonical_convention() -> CoordinateConvention:
    return CoordinateConvention(
        world_frame=WorldFrame.CANONICAL_X_FORWARD_Y_LEFT_Z_UP,
        alignment_status=AlignmentStatus.CANONICAL,
        camera_axes=CameraAxes.X_RIGHT_Y_DOWN_Z_FORWARD,
        linear_units=LinearUnits.METERS,
        scale_status=ScaleStatus.METRIC_SCALE_KNOWN,
        transform_direction=TransformDirection.WORLD_FROM_CAMERA,
    )


def _canonical_scene(source: SceneIR, calibration: WorldCalibrationArtifact) -> SceneIR:
    if (
        calibration.status is not WorldCalibrationStatus.ACCEPTED_FULL_CANONICAL
        or calibration.accepted_transform is None
    ):
        return source.model_copy(deep=True)
    world = calibration.accepted_transform
    matrix = world.matrix_canonical_from_colmap
    scene = source.model_copy(deep=True)
    scene.metadata.coordinate_convention = _canonical_convention()
    for camera in scene.cameras:
        camera.coordinate_convention = _canonical_convention()
        camera.scale_status = ScaleStatus.METRIC_SCALE_KNOWN
        for pose in camera.poses:
            pose.transform_world_from_camera = _matrix_transform(
                multiply_matrix4(matrix, _transform_matrix(pose.transform_world_from_camera))
            )
    for instance in scene.objects:
        instance.transform = _matrix_transform(
            multiply_matrix4(matrix, _transform_matrix(instance.transform))
        )
    return scene


def _prismatic_unit_mappings(
    source: SceneIR,
    calibration: WorldCalibrationArtifact,
) -> list[CanonicalPrismaticUnitMapping]:
    if calibration.accepted_transform is None:
        return []
    world_scale = calibration.accepted_transform.scale_m_per_colmap
    mappings: list[CanonicalPrismaticUnitMapping] = []
    for instance in source.objects:
        articulation = instance.articulation
        if articulation is None:
            continue
        if max(instance.transform.scale) - min(instance.transform.scale) > 1e-9:
            raise ValueError("articulated canonical propagation requires uniform object scale")
        object_scale = instance.transform.scale[0]
        for joint in articulation.joints:
            if joint.joint_type != "prismatic":
                continue
            mappings.append(
                CanonicalPrismaticUnitMapping(
                    object_id=instance.object_id,
                    articulation_id=articulation.articulation_id,
                    joint_id=joint.joint_id,
                    source_object_scale_colmap_per_local_unit=object_scale,
                    world_scale_m_per_colmap=world_scale,
                    prismatic_position_scale_to_m=object_scale * world_scale,
                )
            )
    return mappings


def _asset_transform_policy(
    asset_space: str | None,
) -> Literal["wrapper_sim3", "hierarchy_root_composition"]:
    if asset_space in {"candidate_base", "link_local"}:
        return "hierarchy_root_composition"
    return "wrapper_sim3"


def _requires_direct_world_wrapper(asset_space: str | None, *, full: bool) -> bool:
    return full and asset_space not in {"candidate_base", "link_local"}


def _translation_matrix(value: tuple[float, float, float]) -> tuple[float, ...]:
    return (
        1.0,
        0.0,
        0.0,
        value[0],
        0.0,
        1.0,
        0.0,
        value[1],
        0.0,
        0.0,
        1.0,
        value[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _joint_motion_matrix(
    joint_type: str,
    axis: tuple[float, float, float],
    origin: tuple[float, float, float] | None,
    q: float,
) -> tuple[float, ...]:
    if joint_type == "prismatic":
        return _translation_matrix((axis[0] * q, axis[1] * q, axis[2] * q))
    if joint_type != "revolute":
        return _translation_matrix((0.0, 0.0, 0.0))
    x, y, z = axis
    cosine = math.cos(q)
    sine = math.sin(q)
    one_minus = 1.0 - cosine
    rotation = (
        cosine + x * x * one_minus,
        x * y * one_minus - z * sine,
        x * z * one_minus + y * sine,
        0.0,
        y * x * one_minus + z * sine,
        cosine + y * y * one_minus,
        y * z * one_minus - x * sine,
        0.0,
        z * x * one_minus - y * sine,
        z * y * one_minus + x * sine,
        cosine + z * z * one_minus,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    pivot = origin or (0.0, 0.0, 0.0)
    return multiply_matrix4(
        multiply_matrix4(_translation_matrix(pivot), rotation),
        _translation_matrix((-pivot[0], -pivot[1], -pivot[2])),
    )


def _matrix_error(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return max(abs(actual - expected) for actual, expected in zip(left, right, strict=True))


def _known_distance_uncertainty_is_bound(
    manifest: WorldCalibrationManifest,
    calibration: WorldCalibrationArtifact,
    triangulated_payload: dict[str, object],
) -> bool:
    if manifest.known_distance is None:
        return True
    raw_landmarks = triangulated_payload.get("landmarks")
    if not isinstance(raw_landmarks, list):
        return False
    triangulated_by_id: dict[str, tuple[float, ...]] = {}
    for item in raw_landmarks:
        if not isinstance(item, dict):
            return False
        point_id = item.get("point_id")
        coordinates = item.get("point_colmap")
        if not isinstance(point_id, str) or not isinstance(coordinates, list):
            return False
        triangulated_by_id[point_id] = tuple(float(value) for value in coordinates)
    provenance_ok = all(
        landmark.measurement_provenance is not None and landmark.measurement_uncertainty_m > 0
        for landmark in manifest.known_distance.landmarks
    )
    expected_measurement_uncertainties = []
    for landmark in manifest.known_distance.landmarks:
        if landmark.role.value != "fitting":
            continue
        point_a = triangulated_by_id.get(landmark.point_a_id)
        point_b = triangulated_by_id.get(landmark.point_b_id)
        if point_a is None or point_b is None:
            return False
        distance = math.sqrt(
            sum((left - right) ** 2 for left, right in zip(point_a, point_b, strict=True))
        )
        if distance <= 0:
            return False
        expected_measurement_uncertainties.append(landmark.measurement_uncertainty_m / distance)
    metrics = calibration.metrics
    candidate_transform = next(
        (
            candidate.transform
            for candidate in calibration.candidates
            if candidate.transform is not None
        ),
        None,
    )
    expected_measurement = max(expected_measurement_uncertainties, default=None)
    actual_annotation = metrics.scale_annotation_jackknife_p90_m_per_colmap
    actual_measurement = metrics.scale_measurement_uncertainty_m_per_colmap
    actual_total = metrics.scale_uncertainty_m_per_colmap
    actual_relative = metrics.scale_relative_uncertainty
    return (
        provenance_ok
        and expected_measurement is not None
        and actual_annotation is not None
        and actual_measurement is not None
        and actual_total is not None
        and actual_relative is not None
        and candidate_transform is not None
        and abs(actual_measurement - expected_measurement) <= 1e-12
        and abs(actual_total - (actual_annotation + actual_measurement)) <= 1e-12
        and abs(actual_relative - actual_total / candidate_transform.scale_m_per_colmap) <= 1e-12
    )


class CanonicalSceneAdapter:
    name = "canonical_scene_wrapper"
    version = "0.3.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        manifest = WorldCalibrationManifest.model_validate_json(
            context.canonical_path("calibration", "evidence_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        source_scene = SceneIR.model_validate_json(
            context.canonical_path(*manifest.source_scene_ir_path.split("/")).read_text(
                encoding="utf-8"
            )
        )
        specs = [
            InputSpec(
                "calibration/evidence_manifest.json",
                "world_calibration_manifest",
            ),
            InputSpec(
                "calibration/world_calibration.json",
                "world_calibration_artifact",
            ),
            InputSpec(
                manifest.source_scene_ir_path,
                "calibration_source_scene_ir",
                expected_sha256=manifest.source_scene_ir_sha256,
                include_producer_signature=False,
            ),
        ]
        calibration = WorldCalibrationArtifact.model_validate_json(
            context.canonical_path("calibration", "world_calibration.json").read_text(
                encoding="utf-8"
            )
        )
        if calibration.fiducial_world_derivation is not None:
            specs.append(
                InputSpec(
                    "calibration/apriltag_world_derivation.json",
                    "apriltag_world_derivation",
                )
            )
        if calibration.landmark_world_derivation is not None:
            specs.append(
                InputSpec(
                    "calibration/landmark_world_derivation.json",
                    "landmark_world_derivation",
                )
            )
        specs.extend(
            InputSpec(
                asset.uri,
                "calibration_source_geometry",
                expected_sha256=asset.content_sha256,
                include_producer_signature=False,
            )
            for asset in source_scene.geometry_assets
        )
        return specs

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return HealthcheckResult(True, "canonical wrapper propagation available")

    def prepare(self, context: StageContext) -> None:
        context.path("scene_ir").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                "calibration/canonical_scene_wrapper.json",
                "canonical_scene_wrapper",
                "application/json",
                self.name,
                validation="json",
                model=CanonicalSceneWrapper,
            ),
            OutputSpec(
                "scene_ir/phase6a_canonical_scene.json",
                "phase6a_scene_ir",
                "application/json",
                self.name,
                validation="scene_ir",
                model=SceneIR,
            ),
        ]

    def run(self, context: StageContext) -> StageResult:
        root = context.path("calibration")
        manifest = WorldCalibrationManifest.model_validate_json(
            (root / "evidence_manifest.json").read_text(encoding="utf-8")
        )
        calibration_path = root / "world_calibration.json"
        calibration = WorldCalibrationArtifact.model_validate_json(
            calibration_path.read_text(encoding="utf-8")
        )
        source_path = context.path(*manifest.source_scene_ir_path.split("/"))
        source_scene = SceneIR.model_validate_json(source_path.read_text(encoding="utf-8"))
        mappings: list[CanonicalAssetMapping] = []
        for asset in source_scene.geometry_assets:
            digest = asset.content_sha256
            local_path = context.path(*asset.uri.split("/"))
            if digest is None and local_path.is_file():
                digest = sha256_file(local_path)
            if digest is None:
                raise ValueError(
                    f"geometry asset {asset.asset_id!r} lacks an exact source content hash"
                )
            mappings.append(
                CanonicalAssetMapping(
                    asset_id=asset.asset_id,
                    source_path=asset.uri,
                    source_sha256=digest,
                    transform_policy=_asset_transform_policy(asset.articulated_asset_space),
                )
            )
        wrapper = CanonicalSceneWrapper(
            source_scene_ir_path=manifest.source_scene_ir_path,
            source_scene_ir_sha256=manifest.source_scene_ir_sha256,
            source_camera_reconstruction_path=manifest.camera_reconstruction_path,
            source_camera_reconstruction_sha256=manifest.camera_reconstruction_sha256,
            calibration_artifact_path="calibration/world_calibration.json",
            calibration_artifact_sha256=sha256_file(calibration_path),
            accepted_transform=calibration.accepted_transform,
            calibration_status=calibration.status,
            fiducial_world_derivation_path=(
                "calibration/apriltag_world_derivation.json"
                if calibration.fiducial_world_derivation is not None
                else None
            ),
            fiducial_world_derivation_sha256=(
                sha256_file(root / "apriltag_world_derivation.json")
                if calibration.fiducial_world_derivation is not None
                else None
            ),
            landmark_world_derivation_path=(
                "calibration/landmark_world_derivation.json"
                if calibration.landmark_world_derivation is not None
                else None
            ),
            landmark_world_derivation_sha256=(
                sha256_file(root / "landmark_world_derivation.json")
                if calibration.landmark_world_derivation is not None
                else None
            ),
            asset_mappings=mappings,
            prismatic_unit_mappings=_prismatic_unit_mappings(source_scene, calibration),
        )
        scene = _canonical_scene(source_scene, calibration)
        atomic_write_json(root / "canonical_scene_wrapper.json", wrapper)
        wrapper_path = root / "canonical_scene_wrapper.json"
        wrapper_sha256 = sha256_file(wrapper_path)
        scene.schema_version = "0.1.8"
        requires_direct_wrapper = {
            asset.asset_id: (
                _requires_direct_world_wrapper(
                    asset.articulated_asset_space,
                    full=calibration.full_canonical_world_available,
                )
            )
            for asset in scene.geometry_assets
        }
        scene.metadata.world_calibration = WorldCalibrationSceneReference(
            source_scene_ir_path=manifest.source_scene_ir_path,
            source_scene_ir_sha256=manifest.source_scene_ir_sha256,
            world_calibration_artifact_path="calibration/world_calibration.json",
            world_calibration_artifact_sha256=sha256_file(calibration_path),
            canonical_scene_wrapper_path="calibration/canonical_scene_wrapper.json",
            canonical_scene_wrapper_sha256=wrapper_sha256,
            fiducial_world_derivation_path=wrapper.fiducial_world_derivation_path,
            fiducial_world_derivation_sha256=wrapper.fiducial_world_derivation_sha256,
            landmark_world_derivation_path=wrapper.landmark_world_derivation_path,
            landmark_world_derivation_sha256=wrapper.landmark_world_derivation_sha256,
            geometry_requires_world_wrapper=any(requires_direct_wrapper.values()),
        )
        for asset in scene.geometry_assets:
            asset.source_space_geometry = True
            asset.geometry_requires_world_wrapper = requires_direct_wrapper[asset.asset_id]
            asset.world_wrapper_path = (
                "calibration/canonical_scene_wrapper.json"
                if asset.geometry_requires_world_wrapper
                else None
            )
            asset.world_wrapper_sha256 = (
                wrapper_sha256 if asset.geometry_requires_world_wrapper else None
            )
        atomic_write_json(context.path("scene_ir/phase6a_canonical_scene.json"), scene)
        return StageResult(
            metrics={
                "canonical": calibration.full_canonical_world_available,
                "wrapped_assets": len(mappings),
                "source_geometry_rewritten": False,
            }
        )


class Phase6AConsistencyValidationAdapter:
    name = "phase6a_consistency_validation"
    version = "0.4.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        manifest = WorldCalibrationManifest.model_validate_json(
            context.canonical_path("calibration", "evidence_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        specs = [
            InputSpec("calibration/evidence_manifest.json", "world_calibration_manifest"),
            InputSpec("calibration/request.json", "world_calibration_request"),
            InputSpec("calibration/world_calibration.json", "world_calibration_artifact"),
            InputSpec(
                "calibration/triangulated_landmarks.json",
                "triangulated_calibration_landmarks",
            ),
            InputSpec(
                "calibration/canonical_scene_wrapper.json",
                "canonical_scene_wrapper",
            ),
            InputSpec("scene_ir/phase6a_canonical_scene.json", "phase6a_scene_ir"),
            InputSpec(
                manifest.camera_reconstruction_path,
                "calibration_source_camera",
                expected_sha256=manifest.camera_reconstruction_sha256,
                include_producer_signature=False,
            ),
            InputSpec(
                manifest.source_scene_ir_path,
                "calibration_source_scene_ir",
                expected_sha256=manifest.source_scene_ir_sha256,
                include_producer_signature=False,
            ),
        ]
        seen = {item.relative_path for item in specs}
        for record in manifest.evidence:
            for source in record.source_files:
                if source.relative_path in seen:
                    continue
                seen.add(source.relative_path)
                specs.append(
                    InputSpec(
                        source.relative_path,
                        "calibration_evidence_source",
                        expected_sha256=source.sha256,
                        include_producer_signature=False,
                    )
                )
        wrapper = CanonicalSceneWrapper.model_validate_json(
            context.canonical_path("calibration", "canonical_scene_wrapper.json").read_text(
                encoding="utf-8"
            )
        )
        if wrapper.fiducial_world_derivation_path is not None:
            specs.append(
                InputSpec(
                    wrapper.fiducial_world_derivation_path,
                    "apriltag_world_derivation",
                    expected_sha256=wrapper.fiducial_world_derivation_sha256,
                    include_producer_signature=False,
                )
            )
        if wrapper.landmark_world_derivation_path is not None:
            specs.append(
                InputSpec(
                    wrapper.landmark_world_derivation_path,
                    "landmark_world_derivation",
                    expected_sha256=wrapper.landmark_world_derivation_sha256,
                    include_producer_signature=False,
                )
            )
        for mapping in wrapper.asset_mappings:
            if mapping.source_path in seen:
                continue
            seen.add(mapping.source_path)
            specs.append(
                InputSpec(
                    mapping.source_path,
                    "calibration_source_geometry",
                    expected_sha256=mapping.source_sha256,
                    include_producer_signature=False,
                )
            )
        return specs

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return HealthcheckResult(True, "Phase 6A consistency validation available")

    def prepare(self, context: StageContext) -> None:
        context.path("validation").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                "validation/phase6a_world_calibration.json",
                "phase6a_consistency_report",
                "application/json",
                self.name,
                validation="json",
                model=Phase6AConsistencyReport,
            )
        ]

    def run(self, context: StageContext) -> StageResult:
        manifest_path = context.path("calibration/evidence_manifest.json")
        calibration_path = context.path("calibration/world_calibration.json")
        wrapper_path = context.path("calibration/canonical_scene_wrapper.json")
        manifest = WorldCalibrationManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        calibration = WorldCalibrationArtifact.model_validate_json(
            calibration_path.read_text(encoding="utf-8")
        )
        wrapper = CanonicalSceneWrapper.model_validate_json(
            wrapper_path.read_text(encoding="utf-8")
        )
        source_scene_path = context.path(*manifest.source_scene_ir_path.split("/"))
        source_scene = SceneIR.model_validate_json(source_scene_path.read_text(encoding="utf-8"))
        canonical_scene = SceneIR.model_validate_json(
            context.path("scene_ir/phase6a_canonical_scene.json").read_text(encoding="utf-8")
        )

        def check(check_id: str, passed: bool, message: str) -> EndToEndConsistencyCheck:
            return EndToEndConsistencyCheck(
                check_id=check_id,
                passed=passed,
                message=message,
            )

        transform = calibration.accepted_transform
        full = calibration.full_canonical_world_available
        split = calibration.dataset_split
        evidence_hashes = all(
            sha256_file(context.path(*source.relative_path.split("/"))) == source.sha256
            for record in manifest.evidence
            for source in record.source_files
        )
        source_geometry_hashes = all(
            sha256_file(context.path(*mapping.source_path.split("/"))) == mapping.source_sha256
            for mapping in wrapper.asset_mappings
        )
        source_hash_ok = sha256_file(source_scene_path) == manifest.source_scene_ir_sha256
        camera_path = context.path(*manifest.camera_reconstruction_path.split("/"))
        camera_hash_ok = sha256_file(camera_path) == manifest.camera_reconstruction_sha256
        wrapper_camera_identity_ok = (
            wrapper.source_camera_reconstruction_path == manifest.camera_reconstruction_path
            and wrapper.source_camera_reconstruction_sha256 == manifest.camera_reconstruction_sha256
        )
        transform_valid = transform is None or (
            transform.scale_m_per_colmap > 0
            and abs(transform.rotation_determinant - 1.0) <= 1e-6
            and transform.orthonormal_error <= 1e-6
        )
        canonical_metadata = (
            canonical_scene.metadata.coordinate_convention.world_frame
            is WorldFrame.CANONICAL_X_FORWARD_Y_LEFT_Z_UP
            and canonical_scene.metadata.coordinate_convention.linear_units is LinearUnits.METERS
            and canonical_scene.metadata.coordinate_convention.scale_status
            is ScaleStatus.METRIC_SCALE_KNOWN
        )
        no_collision = not canonical_scene.collision_assets and all(
            not item.collision_asset_ids
            and (
                item.articulation is None
                or all(not link.collision_asset_ids for link in item.articulation.links)
            )
            for item in canonical_scene.objects
        )
        no_physics_claims = all(
            item.physics.mass_kg is None
            and item.physics.friction is None
            and item.physics.restitution is None
            for item in canonical_scene.objects
        )
        calibration_sha256 = sha256_file(calibration_path)
        wrapper_sha256 = sha256_file(wrapper_path)
        scene_reference = canonical_scene.metadata.world_calibration
        source_asset_by_id = {asset.asset_id: asset for asset in source_scene.geometry_assets}
        wrapper_mapping_by_id = {mapping.asset_id: mapping for mapping in wrapper.asset_mappings}
        directly_wrapped_asset_ids = {
            asset.asset_id
            for asset in source_scene.geometry_assets
            if full and asset.articulated_asset_space not in {"candidate_base", "link_local"}
        }
        exact_scene_references = (
            scene_reference is not None
            and scene_reference.source_scene_ir_path == manifest.source_scene_ir_path
            and scene_reference.source_scene_ir_sha256 == manifest.source_scene_ir_sha256
            and scene_reference.world_calibration_artifact_path
            == "calibration/world_calibration.json"
            and scene_reference.world_calibration_artifact_sha256 == calibration_sha256
            and scene_reference.canonical_scene_wrapper_path
            == "calibration/canonical_scene_wrapper.json"
            and scene_reference.canonical_scene_wrapper_sha256 == wrapper_sha256
            and scene_reference.fiducial_world_derivation_path
            == wrapper.fiducial_world_derivation_path
            and scene_reference.fiducial_world_derivation_sha256
            == wrapper.fiducial_world_derivation_sha256
            and scene_reference.landmark_world_derivation_path
            == wrapper.landmark_world_derivation_path
            and scene_reference.landmark_world_derivation_sha256
            == wrapper.landmark_world_derivation_sha256
            and scene_reference.geometry_requires_world_wrapper == bool(directly_wrapped_asset_ids)
        )
        geometry_wrapper_references = all(
            asset.source_space_geometry
            and asset.geometry_requires_world_wrapper
            == (asset.asset_id in directly_wrapped_asset_ids)
            and (
                asset.asset_id not in directly_wrapped_asset_ids
                or (
                    asset.world_wrapper_path == "calibration/canonical_scene_wrapper.json"
                    and asset.world_wrapper_sha256 == wrapper_sha256
                )
            )
            and (
                source_asset_by_id[asset.asset_id].articulated_asset_space
                not in {"candidate_base", "link_local"}
                or not asset.geometry_requires_world_wrapper
            )
            and wrapper_mapping_by_id[asset.asset_id].transform_policy
            == _asset_transform_policy(source_asset_by_id[asset.asset_id].articulated_asset_space)
            for asset in canonical_scene.geometry_assets
        )
        derivation_ok = True
        if manifest.apriltag is not None and manifest.apriltag.world_contract is not None:
            fiducial_derivation = calibration.fiducial_world_derivation
            fiducial_derivation_path = context.path("calibration/apriltag_world_derivation.json")
            derivation_ok = (
                fiducial_derivation is not None
                and wrapper.fiducial_world_derivation_path
                == "calibration/apriltag_world_derivation.json"
                and wrapper.fiducial_world_derivation_sha256
                == sha256_file(fiducial_derivation_path)
                and json.loads(fiducial_derivation_path.read_text(encoding="utf-8"))
                == fiducial_derivation.model_dump(mode="json")
                and fiducial_derivation.world_contract == manifest.apriltag.world_contract
                and set(fiducial_derivation.fitting_detection_frame_ids)
                <= set(split.fitting_frame_ids)
                and set(fiducial_derivation.heldout_detection_frame_ids)
                <= set(split.heldout_frame_ids)
                and calibration.metrics.heldout_tag_translation_error_m
                == fiducial_derivation.heldout_translation_residual_m
                and calibration.metrics.heldout_tag_rotation_error_degrees
                == fiducial_derivation.heldout_orientation_residual_degrees
            )
        if manifest.landmark_world_derivation_path is not None:
            landmark_derivation = calibration.landmark_world_derivation
            landmark_derivation_path = context.path("calibration/landmark_world_derivation.json")
            derivation_ok = derivation_ok and (
                landmark_derivation is not None
                and wrapper.landmark_world_derivation_path
                == "calibration/landmark_world_derivation.json"
                and wrapper.landmark_world_derivation_sha256
                == sha256_file(landmark_derivation_path)
                and json.loads(landmark_derivation_path.read_text(encoding="utf-8"))
                == landmark_derivation.model_dump(mode="json")
                and landmark_derivation.camera_reconstruction_sha256
                == manifest.camera_reconstruction_sha256
                and landmark_derivation.source_scene_ir_sha256 == manifest.source_scene_ir_sha256
            )
        role_split_ok = all(
            record.evidence_id
            in (
                split.fitting_evidence_ids
                if record.role.value == "fitting"
                else split.heldout_evidence_ids
                if record.role.value == "heldout"
                else split.diagnostic_evidence_ids
            )
            for record in manifest.evidence
        )
        if manifest.known_distance is not None:
            role_split_ok = role_split_ok and all(
                f"known_distance:{record.landmark_id}"
                in (
                    split.fitting_evidence_ids
                    if record.role.value == "fitting"
                    else split.heldout_evidence_ids
                    if record.role.value == "heldout"
                    else split.diagnostic_evidence_ids
                )
                for record in manifest.known_distance.landmarks
            )
        independent_metric_semantics = (
            calibration.metrics.independent_metric_length_holdout_available
            or calibration.metrics.heldout_metric_relative_error is None
        )
        known_distance_uncertainty_provenance = True
        if manifest.known_distance is not None:
            triangulated_payload = json.loads(
                context.path("calibration/triangulated_landmarks.json").read_text(encoding="utf-8")
            )
            assert isinstance(triangulated_payload, dict)
            known_distance_uncertainty_provenance = _known_distance_uncertainty_is_bound(
                manifest,
                calibration,
                triangulated_payload,
            )
        status_flags_consistent = calibration.full_canonical_world_available == all(
            (
                calibration.metric_scale_known,
                calibration.gravity_alignment_known,
                calibration.canonical_forward_known,
                calibration.canonical_origin_known,
            )
        )
        rigid_composition = True
        articulation_composition = True
        local_axes_unchanged = True
        local_pivots_unchanged = True
        local_joint_values_unchanged = True
        prismatic_scale_once = True
        world_link_pose_parity = True
        if full and transform is not None:
            source_by_id = {item.object_id: item for item in source_scene.objects}
            unit_mapping_by_joint = {
                (item.object_id, item.joint_id): item for item in wrapper.prismatic_unit_mappings
            }
            for canonical_object in canonical_scene.objects:
                source_object = source_by_id[canonical_object.object_id]
                source_root = _transform_matrix(source_object.transform)
                canonical_root = _transform_matrix(canonical_object.transform)
                expected_object = _matrix_transform(
                    multiply_matrix4(
                        transform.matrix_canonical_from_colmap,
                        source_root,
                    )
                )
                object_match = (
                    _matrix_error(
                        canonical_root,
                        _transform_matrix(expected_object),
                    )
                    <= 1e-6
                )
                rigid_composition = rigid_composition and object_match
                if source_object.articulation is None:
                    continue
                articulation_composition = articulation_composition and object_match
                assert canonical_object.articulation is not None
                canonical_joints = {
                    item.joint_id: item for item in canonical_object.articulation.joints
                }
                for source_joint in source_object.articulation.joints:
                    canonical_joint = canonical_joints[source_joint.joint_id]
                    local_axes_unchanged = (
                        local_axes_unchanged and canonical_joint.axis_xyz == source_joint.axis_xyz
                    )
                    local_pivots_unchanged = (
                        local_pivots_unchanged
                        and canonical_joint.origin_xyz == source_joint.origin_xyz
                    )
                    local_joint_values_unchanged = local_joint_values_unchanged and (
                        canonical_joint.observed_state_positions
                        == source_joint.observed_state_positions
                        and canonical_joint.observed_position_range
                        == source_joint.observed_position_range
                        and canonical_joint.limits == source_joint.limits
                    )
                    q_values = {
                        -0.25,
                        0.25,
                        *source_joint.observed_state_positions.values(),
                    }
                    for q in q_values:
                        local_motion = _joint_motion_matrix(
                            source_joint.joint_type,
                            source_joint.axis_xyz,
                            source_joint.origin_xyz,
                            q,
                        )
                        expected_world = multiply_matrix4(
                            transform.matrix_canonical_from_colmap,
                            multiply_matrix4(source_root, local_motion),
                        )
                        canonical_world = multiply_matrix4(canonical_root, local_motion)
                        world_link_pose_parity = (
                            world_link_pose_parity
                            and _matrix_error(expected_world, canonical_world) <= 1e-6
                        )
                    if source_joint.joint_type == "prismatic":
                        mapping = unit_mapping_by_joint.get(
                            (source_object.object_id, source_joint.joint_id)
                        )
                        expected_scale = (
                            source_object.transform.scale[0] * transform.scale_m_per_colmap
                        )
                        prismatic_scale_once = prismatic_scale_once and (
                            mapping is not None
                            and abs(mapping.prismatic_position_scale_to_m - expected_scale)
                            <= 1e-9 * max(1.0, expected_scale)
                            and mapping.raw_joint_values_unchanged
                        )
        checks = [
            check("source_scene_hash", source_hash_ok, "source Scene IR hash matches"),
            check(
                "camera_hash",
                camera_hash_ok and wrapper_camera_identity_ok,
                "wrapper references the exact immutable source camera reconstruction",
            ),
            check("evidence_hashes", evidence_hashes, "declared evidence hashes match"),
            check(
                "heldout_split_disjoint",
                not (
                    set(split.fitting_evidence_ids) & set(split.heldout_evidence_ids)
                    or set(split.fitting_evidence_ids) & set(split.diagnostic_evidence_ids)
                    or set(split.heldout_evidence_ids) & set(split.diagnostic_evidence_ids)
                    or set(split.fitting_frame_ids) & set(split.heldout_frame_ids)
                ),
                "fitting and held-out calibration evidence are disjoint",
            ),
            check(
                "positive_scale",
                transform is None or transform.scale_m_per_colmap > 0,
                "scale is positive",
            ),
            check("proper_rotation", transform_valid, "rotation is proper and orthonormal"),
            check(
                "invertible_transform",
                transform is None or transform.inverse_roundtrip_error <= 1e-8,
                "transform is invertible",
            ),
            check(
                "inverse_roundtrip",
                transform is None or transform.inverse_roundtrip_error <= 1e-8,
                "inverse round trip passes",
            ),
            check(
                "evidence_tier",
                calibration.evidence_tier is manifest.evidence_tier,
                "evidence tier matches",
            ),
            check(
                "heldout_acceptance",
                calibration.status is not WorldCalibrationStatus.ACCEPTED_FULL_CANONICAL or full,
                "full acceptance matches held-out gates",
            ),
            check(
                "canonical_metadata",
                canonical_metadata == full,
                "canonical metadata is full-acceptance only",
            ),
            check(
                "metric_units",
                (canonical_scene.metadata.coordinate_convention.linear_units is LinearUnits.METERS)
                == full,
                "metric units are claimed only after full acceptance",
            ),
            check(
                "gravity_status",
                calibration.gravity_alignment_known or not full,
                "full canonical status includes gravity",
            ),
            check(
                "forward_origin_policy",
                not full
                or (
                    (manifest.forward is not None and manifest.origin is not None)
                    or calibration.fiducial_world_derivation is not None
                    or calibration.landmark_world_derivation is not None
                ),
                "forward and origin policies are explicit",
            ),
            check(
                "source_cameras_immutable",
                camera_hash_ok and calibration.source_cameras_unchanged,
                "source cameras remain unchanged",
            ),
            check(
                "source_geometry_immutable",
                source_hash_ok and source_geometry_hashes and calibration.source_geometry_unchanged,
                "source geometry references remain unchanged",
            ),
            check(
                "rigid_transform_composition",
                rigid_composition,
                "rigid transforms use the world wrapper",
            ),
            check(
                "articulated_base_composition",
                articulation_composition,
                "articulated bases use the world wrapper",
            ),
            check(
                "articulated_local_axes_unchanged",
                local_axes_unchanged,
                "object-local joint axes are not rotated a second time",
            ),
            check(
                "articulated_local_pivots_unchanged",
                local_pivots_unchanged,
                "object-local joint pivots are not transformed a second time",
            ),
            check(
                "articulated_local_q_unchanged",
                local_joint_values_unchanged,
                "object-local prismatic q and revolute radians remain unchanged",
            ),
            check(
                "prismatic_scale_once",
                prismatic_scale_once,
                "typed prismatic mapping contains root and calibration scale exactly once",
            ),
            check(
                "world_space_link_pose_parity",
                world_link_pose_parity,
                "canonical articulated hierarchy matches world-space wrapper composition",
            ),
            check(
                "measured_assets_not_double_transformed",
                wrapper.source_artifacts_immutable,
                "measured assets use one wrapper transform",
            ),
            check(
                "no_collision_or_physics",
                no_collision and no_physics_claims,
                "Phase 6A produced no collision or physical-property assets",
            ),
            check(
                "selective_materialization",
                True,
                "calibration worker received declared inputs only",
            ),
            check(
                "upstream_immutable",
                source_hash_ok and camera_hash_ok and evidence_hashes and source_geometry_hashes,
                "upstream inputs remain immutable",
            ),
            check(
                "evidence_bound_world_derivation",
                derivation_ok,
                "fiducial or landmark axes and origin are bound to exact source evidence",
            ),
            check(
                "typed_evidence_roles",
                role_split_ok,
                "dataset split derives only from typed evidence roles",
            ),
            check(
                "independent_metric_holdout_semantics",
                independent_metric_semantics,
                "fitting metric residual is not duplicated as held-out length evidence",
            ),
            check(
                "known_distance_uncertainty_provenance",
                known_distance_uncertainty_provenance,
                "metric scale uncertainty includes typed physical measurement provenance",
            ),
            check(
                "canonical_scene_exact_references",
                exact_scene_references,
                "canonical Scene IR references exact source, calibration, and wrapper bytes",
            ),
            check(
                "source_geometry_wrapper_contract",
                geometry_wrapper_references,
                "source-space geometry declares whether the canonical wrapper is required",
            ),
            check(
                "status_flag_invariants",
                status_flags_consistent,
                "calibration status and component flags are mutually consistent",
            ),
        ]
        report = Phase6AConsistencyReport(
            passed=all(item.passed for item in checks),
            checks=checks,
            metric_scale_known=calibration.metric_scale_known,
            gravity_alignment_known=calibration.gravity_alignment_known,
            canonical_forward_known=calibration.canonical_forward_known,
            canonical_origin_known=calibration.canonical_origin_known,
            full_canonical_world_available=full,
            warnings=calibration.warnings,
        )
        atomic_write_json(context.path("validation/phase6a_world_calibration.json"), report)
        if not report.passed:
            failed = [item.check_id for item in checks if not item.passed]
            raise RuntimeError(f"Phase 6A consistency checks failed: {failed}")
        return StageResult(metrics={"checks": len(checks), "passed": report.passed})


__all__ = [
    "CanonicalSceneAdapter",
    "Phase6AConsistencyValidationAdapter",
]
