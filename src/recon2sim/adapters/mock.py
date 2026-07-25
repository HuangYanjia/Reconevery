from __future__ import annotations

import hashlib
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from recon2sim.adapters.base import HealthcheckResult, OutputSpec, StageContext, StageResult
from recon2sim.artifacts import (
    CameraReconstruction,
    CompiledScenePackage,
    ExportManifest,
    FrameManifestEntry,
    GlobalReconstructionArtifact,
    IngestManifest,
    InputSourceType,
    ObjectReconstructionArtifact,
    ObjectReconstructionResult,
    ObjectTrack,
    ObjectTracksArtifact,
    ReconstructedArticulation,
    ReconstructedJoint,
    ReconstructedLink,
    ReconstructedMaterial,
    ReconstructedPart,
    TrackObservation,
)
from recon2sim.images import png_dimensions, write_solid_png
from recon2sim.ir import (
    Articulation,
    AssetType,
    Camera,
    CameraIntrinsics,
    CameraPose,
    CollisionAsset,
    ConfidenceRecord,
    CoordinateConvention,
    FrameObservation,
    GeometryAsset,
    GeometrySourceType,
    Joint,
    Link,
    MaterialAsset,
    ObjectInstance,
    ObjectObservation,
    PhysicsProperties,
    ProvenanceRecord,
    RelationType,
    SceneIR,
    SceneMetadata,
    SceneRelation,
    Transform,
    ValidationIssue,
    ValidationReport,
)
from recon2sim.storage import atomic_write_json, atomic_write_text


def _read_model[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    if not path.is_file():
        raise FileNotFoundError(f"required upstream artifact is missing: {path}")
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance(
    adapter: str,
    config: dict[str, Any],
    inputs: list[str],
    outputs: list[str],
    *,
    score: float = 0.9,
) -> ProvenanceRecord:
    return ProvenanceRecord(
        adapter_name=adapter,
        adapter_version="0.1.0",
        configuration=config,
        input_artifact_paths=inputs,
        output_artifact_paths=outputs,
        timestamp=datetime.now(UTC),
        confidence=ConfidenceRecord(score=score, method="deterministic_mock"),
        source=GeometrySourceType.MOCK,
    )


def _write_obj(path: Path, name: str, *, offset: float = 0.0) -> tuple[int, int]:
    atomic_write_text(
        path,
        "\n".join(
            [
                f"# deterministic mock mesh for {name}",
                f"o {name}",
                f"v {offset:.3f} 0.000 0.000",
                f"v {offset + 1.0:.3f} 0.000 0.000",
                f"v {offset:.3f} 1.000 0.000",
                f"v {offset:.3f} 0.000 1.000",
                "f 1 2 3",
                "f 1 2 4",
                "",
            ]
        ),
    )
    return 4, 2


def _json_spec(
    path: str,
    artifact_type: str,
    model: type[BaseModel],
    schema_identifier: str,
) -> OutputSpec:
    return OutputSpec(
        relative_path=path,
        artifact_type=artifact_type,
        media_type="application/json",
        source_type="mock",
        validation="json",
        schema_identifier=schema_identifier,
        model=model,
    )


def _obj_spec(path: str, artifact_type: str) -> OutputSpec:
    return OutputSpec(
        relative_path=path,
        artifact_type=artifact_type,
        media_type="model/obj",
        source_type="mock",
        validation="obj",
    )


def _png_spec(path: str, artifact_type: str) -> OutputSpec:
    return OutputSpec(
        relative_path=path,
        artifact_type=artifact_type,
        media_type="image/png",
        source_type="mock",
        validation="png",
    )


class MockAdapter:
    name = "mock"
    version = "0.1.1"

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return HealthcheckResult(True, "deterministic mock adapter ready")

    def prepare(self, context: StageContext) -> None:
        context.run_dir.mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return []


class MockIngestAdapter(MockAdapter):
    name = "mock_ingest"

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            _json_spec(
                "inputs/manifest.json",
                "ingest_manifest",
                IngestManifest,
                "recon2sim/ingest-manifest/0.1.0",
            )
        ]

    def run(self, context: StageContext) -> StageResult:
        sources = sorted((context.input_dir / "frames").glob("*.png"))
        if not sources:
            sources = sorted(context.input_dir.glob("*.png"))
        if not sources:
            raise FileNotFoundError(
                f"mock ingest requires at least one PNG in {context.input_dir / 'frames'}"
            )

        source_type = InputSourceType(
            str(context.config.adapter.config.get("source_type", "generated_test_image"))
        )
        frame_entries: list[FrameManifestEntry] = []
        outputs: list[OutputSpec] = []
        for index, source in enumerate(sources):
            png_dimensions(source)
            relative_path = f"frames/frame_{index:03d}.png"
            destination = context.path(relative_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            width, height = png_dimensions(destination)
            frame_entries.append(
                FrameManifestEntry(
                    frame_id=f"frame_{index:03d}",
                    relative_path=relative_path,
                    sha256=_sha256(destination),
                    width=width,
                    height=height,
                    timestamp_s=index / 30.0,
                    source_type=source_type,
                )
            )
            outputs.append(_png_spec(relative_path, "input_frame"))

        manifest_path = "inputs/manifest.json"
        provenance = _provenance(
            self.name,
            context.config.adapter.config,
            [],
            [manifest_path, *[frame.relative_path for frame in frame_entries]],
        )
        manifest = IngestManifest(
            source_type=source_type,
            frames=frame_entries,
            provenance=provenance,
        )
        atomic_write_json(context.path(manifest_path), manifest)
        return StageResult(outputs=outputs, metrics={"frame_count": len(frame_entries)})


class MockCameraRecoveryAdapter(MockAdapter):
    name = "mock_camera_recovery"

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            _json_spec(
                "camera/reconstruction.json",
                "camera_reconstruction",
                CameraReconstruction,
                "recon2sim/camera-reconstruction/0.1.0",
            )
        ]

    def run(self, context: StageContext) -> StageResult:
        manifest = _read_model(context.path("inputs", "manifest.json"), IngestManifest)
        for frame in manifest.frames:
            frame_path = context.path(frame.relative_path)
            width, height = png_dimensions(frame_path)
            if (width, height) != (frame.width, frame.height) or _sha256(
                frame_path
            ) != frame.sha256:
                raise ValueError(
                    f"ingested frame no longer matches manifest: {frame.relative_path}"
                )

        first = manifest.frames[0]
        intrinsics = CameraIntrinsics(
            width=first.width,
            height=first.height,
            fx=float(max(first.width, first.height)),
            fy=float(max(first.width, first.height)),
            cx=first.width / 2.0,
            cy=first.height / 2.0,
        )
        poses = [
            CameraPose(
                frame_id=frame.frame_id,
                transform_world_from_camera=Transform(translation=(index * 0.05, -1.5, 1.0)),
                confidence=ConfidenceRecord(score=0.94, method="deterministic_mock"),
            )
            for index, frame in enumerate(manifest.frames)
        ]
        output_path = "camera/reconstruction.json"
        provenance = _provenance(
            self.name,
            context.config.adapter.config,
            ["inputs/manifest.json", *[frame.relative_path for frame in manifest.frames]],
            [output_path],
            score=0.94,
        )
        reconstruction = CameraReconstruction(
            camera_id="cam0",
            intrinsics=intrinsics,
            poses=poses,
            confidence=ConfidenceRecord(score=0.94, method="deterministic_mock"),
            coordinate_convention=CoordinateConvention(),
            provenance=provenance,
        )
        atomic_write_json(context.path(output_path), reconstruction)
        return StageResult(metrics={"pose_count": len(poses)})


class MockSegmentationTrackingAdapter(MockAdapter):
    name = "mock_segmentation_tracking"

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            _json_spec(
                "observations/object_tracks.json",
                "object_tracks",
                ObjectTracksArtifact,
                "recon2sim/object-tracks/0.1.0",
            )
        ]

    def run(self, context: StageContext) -> StageResult:
        manifest = _read_model(context.path("inputs", "manifest.json"), IngestManifest)
        camera = _read_model(context.path("camera", "reconstruction.json"), CameraReconstruction)
        frame_ids = {frame.frame_id for frame in manifest.frames}
        pose_ids = {pose.frame_id for pose in camera.poses}
        if not pose_ids or not pose_ids <= frame_ids:
            raise ValueError(
                "camera reconstruction poses must be a non-empty subset of ingest manifest frames"
            )

        track_specs = [
            ("table", "table", AssetType.STATIC_STRUCTURE, (150, 110, 70)),
            ("cup", "cup", AssetType.RIGID, (70, 130, 210)),
            ("cabinet", "cabinet", AssetType.ARTICULATED, (180, 120, 65)),
        ]
        output_path = "observations/object_tracks.json"
        mask_paths: list[str] = []
        tracks: list[ObjectTrack] = []
        for object_index, (object_id, name, asset_type, color) in enumerate(track_specs):
            observations: list[TrackObservation] = []
            for frame in manifest.frames:
                mask_path = f"observations/masks/{frame.frame_id}_{object_id}.png"
                write_solid_png(context.path(mask_path), frame.width, frame.height, color)
                mask_paths.append(mask_path)
                bbox_width = max(1, frame.width // 3)
                bbox_height = max(1, frame.height // 3)
                x = min(object_index * 2, frame.width - bbox_width)
                y = min(object_index * 2, frame.height - bbox_height)
                observations.append(
                    TrackObservation(
                        frame_id=frame.frame_id,
                        bbox_xywh=(x, y, bbox_width, bbox_height),
                        mask_path=mask_path,
                        confidence=ConfidenceRecord(
                            score=0.88 - object_index * 0.02,
                            method="deterministic_mock",
                        ),
                    )
                )
            track_provenance = _provenance(
                self.name,
                context.config.adapter.config,
                ["inputs/manifest.json", "camera/reconstruction.json"],
                [output_path, *mask_paths],
                score=0.88,
            )
            tracks.append(
                ObjectTrack(
                    object_id=object_id,
                    name=name,
                    asset_type=asset_type,
                    observations=observations,
                    confidence=ConfidenceRecord(score=0.88, method="deterministic_mock"),
                    provenance=track_provenance,
                )
            )

        provenance = _provenance(
            self.name,
            context.config.adapter.config,
            ["inputs/manifest.json", "camera/reconstruction.json"],
            [output_path, *mask_paths],
            score=0.88,
        )
        artifact = ObjectTracksArtifact(tracks=tracks, provenance=provenance)
        atomic_write_json(context.path(output_path), artifact)
        outputs = [_png_spec(path, "object_mask") for path in mask_paths]
        return StageResult(
            outputs=outputs,
            metrics={"track_count": len(tracks), "mask_count": len(mask_paths)},
        )


class MockGlobalReconstructionAdapter(MockAdapter):
    name = "mock_global_reconstruction"

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            _obj_spec("reconstruction/global/floor.obj", "global_visual_mesh"),
            _json_spec(
                "reconstruction/global/metadata.json",
                "global_reconstruction_metadata",
                GlobalReconstructionArtifact,
                "recon2sim/global-reconstruction/0.1.0",
            ),
        ]

    def run(self, context: StageContext) -> StageResult:
        camera = _read_model(context.path("camera", "reconstruction.json"), CameraReconstruction)
        if not camera.poses:
            raise ValueError("global reconstruction requires at least one camera pose")
        mesh_path = "reconstruction/global/floor.obj"
        metadata_path = "reconstruction/global/metadata.json"
        vertex_count, face_count = _write_obj(context.path(mesh_path), "floor")
        provenance = _provenance(
            self.name,
            context.config.adapter.config,
            ["camera/reconstruction.json"],
            [mesh_path, metadata_path],
            score=0.86,
        )
        metadata = GlobalReconstructionArtifact(
            geometry_path=mesh_path,
            collision_path=mesh_path,
            vertex_count=vertex_count,
            face_count=face_count,
            material=ReconstructedMaterial(
                name="mock floor material", base_color_rgba=(0.45, 0.45, 0.45, 1.0)
            ),
            physics=PhysicsProperties(is_static=True),
            confidence=ConfidenceRecord(score=0.86, method="deterministic_mock"),
            provenance=provenance,
        )
        atomic_write_json(context.path(metadata_path), metadata)
        return StageResult(metrics={"camera_pose_count": len(camera.poses)})


class MockObjectReconstructionAdapter(MockAdapter):
    name = "mock_object_reconstruction"

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            _json_spec(
                "reconstruction/objects/results.json",
                "object_reconstruction_results",
                ObjectReconstructionArtifact,
                "recon2sim/object-reconstruction/0.1.0",
            )
        ]

    def run(self, context: StageContext) -> StageResult:
        tracks = _read_model(
            context.path("observations", "object_tracks.json"), ObjectTracksArtifact
        )
        output_path = "reconstruction/objects/results.json"
        mesh_outputs: list[OutputSpec] = []
        all_mesh_paths: list[str] = []
        results: list[ObjectReconstructionResult] = []
        for object_index, track in enumerate(tracks.tracks):
            part_names = ["body", "drawer"] if track.object_id == "cabinet" else ["body"]
            parts: list[ReconstructedPart] = []
            for part_index, part_name in enumerate(part_names):
                part_id = f"{track.object_id}_{part_name}"
                geometry_path = f"reconstruction/objects/{part_id}.obj"
                collision_path = f"reconstruction/objects/{part_id}_collision.obj"
                vertex_count, face_count = _write_obj(
                    context.path(geometry_path), part_id, offset=part_index * 0.2
                )
                _write_obj(
                    context.path(collision_path), f"{part_id}_collision", offset=part_index * 0.2
                )
                all_mesh_paths.extend([geometry_path, collision_path])
                mesh_outputs.extend(
                    [
                        _obj_spec(geometry_path, "object_visual_mesh"),
                        _obj_spec(collision_path, "object_collision_mesh"),
                    ]
                )
                parts.append(
                    ReconstructedPart(
                        part_id=part_id,
                        name=part_name,
                        geometry_path=geometry_path,
                        collision_path=collision_path,
                        vertex_count=vertex_count,
                        face_count=face_count,
                    )
                )

            articulation: ReconstructedArticulation | None = None
            if track.asset_type is AssetType.ARTICULATED:
                if track.object_id != "cabinet" or len(parts) != 2:
                    raise ValueError(
                        "the Phase 0.1 mock only supports the canonical cabinet articulation"
                    )
                articulation = ReconstructedArticulation(
                    articulation_id="cabinet_articulation",
                    links=[
                        ReconstructedLink(
                            link_id="cabinet_body", name="cabinet body", part_ids=[parts[0].part_id]
                        ),
                        ReconstructedLink(
                            link_id="cabinet_drawer", name="drawer", part_ids=[parts[1].part_id]
                        ),
                    ],
                    joints=[
                        ReconstructedJoint(
                            joint_id="cabinet_drawer_slide",
                            parent_link_id="cabinet_body",
                            child_link_id="cabinet_drawer",
                            joint_type="prismatic",
                            axis_xyz=(1.0, 0.0, 0.0),
                            limits=(0.0, 0.4),
                        )
                    ],
                )

            result_provenance = _provenance(
                self.name,
                context.config.adapter.config,
                ["observations/object_tracks.json"],
                [output_path, *all_mesh_paths],
                score=0.84,
            )
            is_static = track.asset_type in {AssetType.STATIC_STRUCTURE, AssetType.ARTICULATED}
            results.append(
                ObjectReconstructionResult(
                    object_id=track.object_id,
                    name=track.name,
                    asset_type=track.asset_type,
                    parts=parts,
                    material=ReconstructedMaterial(
                        name=f"mock {track.name} material",
                        base_color_rgba=(
                            0.25 + object_index * 0.15,
                            0.4,
                            0.65 - object_index * 0.1,
                            1.0,
                        ),
                    ),
                    physics=PhysicsProperties(
                        mass_kg=None if is_static else 0.2,
                        friction=0.5,
                        restitution=0.1,
                        is_static=is_static,
                    ),
                    articulation=articulation,
                    confidence=ConfidenceRecord(score=0.84, method="deterministic_mock"),
                    provenance=result_provenance,
                )
            )

        provenance = _provenance(
            self.name,
            context.config.adapter.config,
            ["observations/object_tracks.json"],
            [output_path, *all_mesh_paths],
            score=0.84,
        )
        artifact = ObjectReconstructionArtifact(results=results, provenance=provenance)
        atomic_write_json(context.path(output_path), artifact)
        return StageResult(
            outputs=mesh_outputs,
            metrics={"object_result_count": len(results), "mesh_count": len(mesh_outputs)},
        )


class MockSceneIRAssemblyAdapter(MockAdapter):
    name = "mock_scene_ir_assembly"

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                relative_path="scene_ir/scene.json",
                artifact_type="canonical_scene_ir",
                media_type="application/json",
                source_type="mock",
                validation="scene_ir",
                schema_identifier="recon2sim/scene-ir/0.1.0",
                model=SceneIR,
            )
        ]

    def run(self, context: StageContext) -> StageResult:
        manifest = _read_model(context.path("inputs", "manifest.json"), IngestManifest)
        camera_data = _read_model(
            context.path("camera", "reconstruction.json"), CameraReconstruction
        )
        tracks_data = _read_model(
            context.path("observations", "object_tracks.json"), ObjectTracksArtifact
        )
        global_data = _read_model(
            context.path("reconstruction", "global", "metadata.json"),
            GlobalReconstructionArtifact,
        )
        object_data = _read_model(
            context.path("reconstruction", "objects", "results.json"),
            ObjectReconstructionArtifact,
        )
        track_ids = {track.object_id for track in tracks_data.tracks}
        result_ids = {result.object_id for result in object_data.results}
        if track_ids != result_ids:
            raise ValueError(
                "object reconstruction results must correspond one-to-one with object tracks"
            )

        input_paths = [
            "inputs/manifest.json",
            "camera/reconstruction.json",
            "observations/object_tracks.json",
            "reconstruction/global/metadata.json",
            "reconstruction/objects/results.json",
        ]
        output_path = "scene_ir/scene.json"
        provenance = _provenance(
            self.name,
            context.config.adapter.config,
            input_paths,
            [output_path],
            score=0.9,
        )

        camera = Camera(
            camera_id=camera_data.camera_id,
            model=camera_data.model,
            intrinsics=camera_data.intrinsics,
            poses=camera_data.poses,
            coordinate_convention=camera_data.coordinate_convention,
            scale_status=camera_data.scale_status,
            provenance=camera_data.provenance,
        )
        observations_by_frame: dict[str, list[ObjectObservation]] = {
            frame.frame_id: [] for frame in manifest.frames
        }
        for track in tracks_data.tracks:
            for observation in track.observations:
                if observation.frame_id not in observations_by_frame:
                    raise ValueError(
                        f"track {track.object_id!r} references unknown frame "
                        f"{observation.frame_id!r}"
                    )
                observations_by_frame[observation.frame_id].append(
                    ObjectObservation(
                        object_id=track.object_id,
                        frame_id=observation.frame_id,
                        bbox_xywh=observation.bbox_xywh,
                        mask_path=observation.mask_path,
                        confidence=observation.confidence,
                    )
                )
        frames = [
            FrameObservation(
                frame_id=frame.frame_id,
                frame_path=frame.relative_path,
                timestamp_s=frame.timestamp_s,
                camera_id=camera_data.camera_id,
                observations=sorted(
                    observations_by_frame[frame.frame_id], key=lambda item: item.object_id
                ),
            )
            for frame in manifest.frames
        ]

        geometry_assets: list[GeometryAsset] = []
        material_assets: list[MaterialAsset] = []
        collision_assets: list[CollisionAsset] = []
        objects: list[ObjectInstance] = []

        floor_geometry_id = "geom_floor"
        floor_material_id = "mat_floor"
        floor_collision_id = "coll_floor"
        geometry_assets.append(
            GeometryAsset(
                asset_id=floor_geometry_id,
                asset_type=AssetType.STATIC_STRUCTURE,
                uri=global_data.geometry_path,
                format=global_data.format,
                source=GeometrySourceType.MOCK,
                provenance=global_data.provenance,
            )
        )
        material_assets.append(
            MaterialAsset(
                asset_id=floor_material_id,
                name=global_data.material.name,
                base_color_rgba=global_data.material.base_color_rgba,
                provenance=global_data.provenance,
            )
        )
        collision_assets.append(
            CollisionAsset(
                asset_id=floor_collision_id,
                uri=global_data.collision_path,
                format=global_data.format,
                source=GeometrySourceType.MOCK,
                provenance=global_data.provenance,
            )
        )
        objects.append(
            ObjectInstance(
                object_id=global_data.object_id,
                name=global_data.name,
                asset_type=AssetType.STATIC_STRUCTURE,
                geometry_asset_ids=[floor_geometry_id],
                material_asset_ids=[floor_material_id],
                collision_asset_ids=[floor_collision_id],
                physics=global_data.physics,
                provenance=[global_data.provenance, provenance],
                confidence=global_data.confidence,
            )
        )

        for result in object_data.results:
            material_id = f"mat_{result.object_id}"
            material_assets.append(
                MaterialAsset(
                    asset_id=material_id,
                    name=result.material.name,
                    base_color_rgba=result.material.base_color_rgba,
                    provenance=result.provenance,
                )
            )
            geometry_by_part: dict[str, str] = {}
            collision_by_part: dict[str, str] = {}
            for part in result.parts:
                geometry_id = f"geom_{part.part_id}"
                collision_id = f"coll_{part.part_id}"
                geometry_by_part[part.part_id] = geometry_id
                collision_by_part[part.part_id] = collision_id
                geometry_assets.append(
                    GeometryAsset(
                        asset_id=geometry_id,
                        asset_type=result.asset_type,
                        uri=part.geometry_path,
                        format=part.format,
                        source=GeometrySourceType.MOCK,
                        provenance=result.provenance,
                    )
                )
                collision_assets.append(
                    CollisionAsset(
                        asset_id=collision_id,
                        uri=part.collision_path,
                        format=part.format,
                        source=GeometrySourceType.MOCK,
                        provenance=result.provenance,
                    )
                )

            articulation: Articulation | None = None
            if result.articulation is not None:
                articulation = Articulation(
                    articulation_id=result.articulation.articulation_id,
                    links=[
                        Link(
                            link_id=link.link_id,
                            name=link.name,
                            geometry_asset_ids=[geometry_by_part[part] for part in link.part_ids],
                            material_asset_ids=[material_id],
                            collision_asset_ids=[collision_by_part[part] for part in link.part_ids],
                        )
                        for link in result.articulation.links
                    ],
                    joints=[
                        Joint(
                            joint_id=joint.joint_id,
                            parent_link_id=joint.parent_link_id,
                            child_link_id=joint.child_link_id,
                            joint_type=joint.joint_type,
                            axis_xyz=joint.axis_xyz,
                            limits=joint.limits,
                        )
                        for joint in result.articulation.joints
                    ],
                )
            objects.append(
                ObjectInstance(
                    object_id=result.object_id,
                    name=result.name,
                    asset_type=result.asset_type,
                    geometry_asset_ids=list(geometry_by_part.values()),
                    material_asset_ids=[material_id],
                    collision_asset_ids=list(collision_by_part.values()),
                    physics=result.physics,
                    articulation=articulation,
                    provenance=[result.provenance, provenance],
                    confidence=result.confidence,
                )
            )

        known_objects = {obj.object_id for obj in objects}
        relation_specs = [
            ("table", "floor"),
            ("cup", "table"),
            ("cabinet", "floor"),
        ]
        relations = [
            SceneRelation(
                relation_type=RelationType.SUPPORTED_BY,
                subject_id=subject,
                object_id=object_id,
                confidence=ConfidenceRecord(score=0.9, method="deterministic_mock"),
                provenance=provenance,
            )
            for subject, object_id in relation_specs
            if {subject, object_id} <= known_objects
        ]

        scene = SceneIR(
            metadata=SceneMetadata(
                scene_id="tabletop_demo",
                name="Phase 0.1 mock tabletop",
                coordinate_convention=camera_data.coordinate_convention,
                source=GeometrySourceType.MOCK,
                provenance=[
                    manifest.provenance,
                    camera_data.provenance,
                    tracks_data.provenance,
                    global_data.provenance,
                    object_data.provenance,
                    provenance,
                ],
            ),
            cameras=[camera],
            frames=frames,
            objects=objects,
            geometry_assets=geometry_assets,
            material_assets=material_assets,
            collision_assets=collision_assets,
            relations=relations,
        )
        atomic_write_json(context.path(output_path), scene)
        return StageResult(
            metrics={
                "camera_count": len(scene.cameras),
                "frame_count": len(scene.frames),
                "object_count": len(scene.objects),
            }
        )


class MockSceneCompilerAdapter(MockAdapter):
    name = "mock_scene_compiler"

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            _json_spec(
                "compiled/scene_package/package.json",
                "compiled_scene_package",
                CompiledScenePackage,
                "recon2sim/compiled-scene-package/0.1.0",
            ),
            _obj_spec("compiled/scene_package/mock_scene.obj", "compiled_visual_mesh"),
        ]

    def run(self, context: StageContext) -> StageResult:
        scene = _read_model(context.path("scene_ir", "scene.json"), SceneIR)
        mesh_path = "compiled/scene_package/mock_scene.obj"
        package_path = "compiled/scene_package/package.json"
        _write_obj(context.path(mesh_path), f"compiled_{scene.metadata.scene_id}")
        package = CompiledScenePackage(
            scene_ir_path="scene_ir/scene.json",
            exported_mesh_paths=[mesh_path],
            simulator_outputs=[],
        )
        atomic_write_json(context.path(package_path), package)
        return StageResult(metrics={"object_count": len(scene.objects)})


class MockPhysicsValidatorAdapter(MockAdapter):
    name = "mock_physics_validator"

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            _json_spec(
                "validation/report.json",
                "validation_report",
                ValidationReport,
                "recon2sim/validation-report/0.1.0",
            )
        ]

    def run(self, context: StageContext) -> StageResult:
        scene = _read_model(context.path("scene_ir", "scene.json"), SceneIR)
        _read_model(context.path("compiled", "scene_package", "package.json"), CompiledScenePackage)
        report = ValidationReport(
            scene_id=scene.metadata.scene_id,
            passed=True,
            issues=[
                ValidationIssue(
                    severity="warning",
                    code="MOCK_LOW_FIDELITY_COLLISION",
                    message="Mock collision meshes validate contracts, not physical fidelity.",
                    object_id="cup",
                )
            ],
        )
        atomic_write_json(context.path("validation", "report.json"), report)
        return StageResult(metrics={"issue_count": len(report.issues)})


class MockExportAdapter(MockAdapter):
    name = "mock_export"

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            _json_spec(
                "export_manifest.json",
                "export_manifest",
                ExportManifest,
                "recon2sim/export-manifest/0.1.0",
            )
        ]

    def run(self, context: StageContext) -> StageResult:
        _read_model(context.path("compiled", "scene_package", "package.json"), CompiledScenePackage)
        report = _read_model(context.path("validation", "report.json"), ValidationReport)
        if not report.passed:
            raise ValueError("cannot export a scene whose validation report did not pass")
        manifest = ExportManifest(
            compiled_package_path="compiled/scene_package/package.json",
            validation_report_path="validation/report.json",
            scene_ir_path="scene_ir/scene.json",
        )
        atomic_write_json(context.path("export_manifest.json"), manifest)
        return StageResult(metrics={"validation_passed": report.passed})
