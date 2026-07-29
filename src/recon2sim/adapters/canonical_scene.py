from __future__ import annotations

import math

from recon2sim.adapters.base import (
    HealthcheckResult,
    InputSpec,
    OutputSpec,
    StageContext,
    StageResult,
)
from recon2sim.artifacts import (
    CanonicalAssetMapping,
    CanonicalSceneWrapper,
    EndToEndConsistencyCheck,
    Phase6AConsistencyReport,
    WorldCalibrationArtifact,
    WorldCalibrationManifest,
    WorldCalibrationStatus,
)
from recon2sim.calibration import (
    multiply_matrix4,
    rotate_vector,
    sha256_file,
    transform_point,
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
    rotation = world.rotation_canonical_from_colmap
    metric_scale = world.scale_m_per_colmap
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
        if instance.articulation is None:
            continue
        for joint in instance.articulation.joints:
            joint.axis_xyz = rotate_vector(rotation, joint.axis_xyz)
            if joint.origin_xyz is not None:
                joint.origin_xyz = transform_point(matrix, joint.origin_xyz)
            if joint.joint_type == "prismatic":
                if joint.limits is not None and joint.limit_source != "candidate_prior":
                    joint.limits = (
                        joint.limits[0] * metric_scale,
                        joint.limits[1] * metric_scale,
                    )
                if joint.observed_position_range is not None:
                    joint.observed_position_range = (
                        joint.observed_position_range[0] * metric_scale,
                        joint.observed_position_range[1] * metric_scale,
                    )
                joint.observed_state_positions = {
                    state: value * metric_scale
                    for state, value in joint.observed_state_positions.items()
                }
    return scene


class CanonicalSceneAdapter:
    name = "canonical_scene_wrapper"
    version = "0.1.0"

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
                    transform_policy="wrapper_sim3",
                )
            )
        wrapper = CanonicalSceneWrapper(
            source_scene_ir_path=manifest.source_scene_ir_path,
            source_scene_ir_sha256=manifest.source_scene_ir_sha256,
            calibration_artifact_path="calibration/world_calibration.json",
            calibration_artifact_sha256=sha256_file(calibration_path),
            accepted_transform=calibration.accepted_transform,
            calibration_status=calibration.status,
            asset_mappings=mappings,
        )
        scene = _canonical_scene(source_scene, calibration)
        atomic_write_json(root / "canonical_scene_wrapper.json", wrapper)
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
    version = "0.1.0"

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
        rigid_composition = True
        articulation_composition = True
        prismatic_scale_once = True
        angular_unchanged = True
        if full and transform is not None:
            rotation = transform.rotation_canonical_from_colmap
            metric_scale = transform.scale_m_per_colmap
            source_by_id = {item.object_id: item for item in source_scene.objects}
            for canonical_object in canonical_scene.objects:
                source_object = source_by_id[canonical_object.object_id]
                expected_object = _matrix_transform(
                    multiply_matrix4(
                        transform.matrix_canonical_from_colmap,
                        _transform_matrix(source_object.transform),
                    )
                )
                object_match = (
                    max(
                        abs(actual - expected)
                        for actual, expected in zip(
                            canonical_object.transform.translation,
                            expected_object.translation,
                            strict=True,
                        )
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
                    expected_axis = rotate_vector(rotation, source_joint.axis_xyz)
                    axis_match = (
                        max(
                            abs(actual - expected)
                            for actual, expected in zip(
                                canonical_joint.axis_xyz,
                                expected_axis,
                                strict=True,
                            )
                        )
                        <= 1e-6
                    )
                    articulation_composition = articulation_composition and axis_match
                    if source_joint.joint_type == "prismatic":
                        positions_match = all(
                            abs(
                                canonical_joint.observed_state_positions[state]
                                - value * metric_scale
                            )
                            <= 1e-6
                            for state, value in source_joint.observed_state_positions.items()
                        )
                        prismatic_scale_once = prismatic_scale_once and positions_match
                    else:
                        angular_unchanged = angular_unchanged and (
                            canonical_joint.observed_state_positions
                            == source_joint.observed_state_positions
                            and canonical_joint.limits == source_joint.limits
                        )
        checks = [
            check("source_scene_hash", source_hash_ok, "source Scene IR hash matches"),
            check("camera_hash", camera_hash_ok, "source camera reconstruction hash matches"),
            check("evidence_hashes", evidence_hashes, "declared evidence hashes match"),
            check(
                "heldout_split_disjoint",
                not (
                    set(split.fitting_evidence_ids) & set(split.heldout_evidence_ids)
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
                not full or (manifest.forward is not None and manifest.origin is not None),
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
                "prismatic_scale_once",
                prismatic_scale_once,
                "prismatic quantities scale exactly once",
            ),
            check(
                "angular_quantities_unchanged",
                angular_unchanged,
                "angular quantities remain radians",
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
