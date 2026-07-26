from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from recon2sim.adapters.base import (
    HealthcheckResult,
    InputSpec,
    OutputSpec,
    StageContext,
    StageResult,
)
from recon2sim.adapters.ingest import (
    ProcessExecutionError,
    allowed_environment,
    resolve_executable,
    run_process,
)
from recon2sim.artifacts import (
    CameraReconstruction,
    DenseDepthManifest,
    DenseFusionArtifact,
    DenseWorkspaceManifest,
    EndToEndConsistencyCheck,
    IngestManifest,
    MeasuredGeneratedComparisonArtifact,
    MeasuredGeneratedObjectComparison,
    MeasuredObjectDiagnostics,
    MeasuredObjectGeometryArtifact,
    MeasuredObjectGeometryRequest,
    MeasuredObjectTrackRequest,
    MeasuredObjectWorkerManifest,
    Phase5AConsistencyReport,
    SegmentationTrackingArtifact,
)
from recon2sim.dense_mvs import ply_counts, sha256_file
from recon2sim.genrecon import coordinate_metadata_is_raw_colmap
from recon2sim.ir import (
    AssetType,
    Camera,
    FrameObservation,
    GeometryAsset,
    GeometrySourceType,
    ObjectInstance,
    ScaleStatus,
    SceneIR,
    SceneMetadata,
    StrictModel,
)
from recon2sim.storage import atomic_write_json

MEASURED_GEOMETRY_WORKER_VERSION = "0.1.0"


def _resolve_python(value: str) -> str | None:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        return str(candidate.absolute()) if candidate.is_file() else None
    return shutil.which(value)


class MeasuredGeometryAdapterConfig(StrictModel):
    execution_mode: Literal["local_worker", "docker", "fake_worker"]
    worker_python: str = "python"
    worker_module: str = "measured_geometry_worker"
    worker_script: str | None = None
    docker_executable: str = "docker"
    docker_image: str = "reconevery/measured-geometry:phase5a"
    minimum_consistent_source_views: int = Field(default=2, ge=1)
    minimum_sam_frame_score: float = Field(default=0.5, ge=0, le=1)
    exclude_mask_boundary_by_default: bool = True
    maximum_relative_depth_discontinuity: float = Field(default=0.03, gt=0)
    minimum_supporting_views: int = Field(default=2, ge=1)
    maximum_relative_depth_residual: float = Field(default=0.03, gt=0)
    maximum_contradicting_views: int = Field(default=1, ge=0)
    voxel_size_source: Literal["median_sample_spacing", "scene_diagonal"] = "median_sample_spacing"
    voxel_size_multiplier: float = Field(default=1.5, gt=0)
    pixel_stride: int = Field(default=1, gt=0)
    maximum_samples_per_object: int = Field(default=2_000_000, gt=0)
    reprojection_splat_radius_pixels: int = Field(default=1, ge=0, le=4)
    minimum_accepted_reprojection_iou: float = Field(default=0.1, ge=0, le=1)
    fake_mode: str = "success"

    @model_validator(mode="after")
    def validate_execution(self) -> MeasuredGeometryAdapterConfig:
        if self.execution_mode == "fake_worker":
            if self.worker_script is None:
                raise ValueError("fake measured-geometry execution requires worker_script")
            return self
        if self.execution_mode == "local_worker":
            python = _resolve_python(self.worker_python)
            if python is None:
                raise ValueError(f"configured worker Python {self.worker_python!r} was not found")
            root = Path(python).absolute().parent.parent
            if not (root / "pyvenv.cfg").is_file() and not (root / "conda-meta").is_dir():
                raise ValueError("measured-geometry worker must use an isolated environment")
            if root.resolve() == Path(sys.prefix).resolve():
                raise ValueError("measured-geometry worker must not use the core environment")
        return self


class MeasuredObjectGeometryAdapter:
    name = "measured_object_geometry"
    version = "0.1.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        tracks = SegmentationTrackingArtifact.model_validate_json(
            context.canonical_path("observations", "object_tracks.json").read_text(encoding="utf-8")
        )
        depth = DenseDepthManifest.model_validate_json(
            context.canonical_path("reconstruction", "dense", "depth_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        specs = [
            InputSpec("inputs/manifest.json", "ingest_manifest"),
            InputSpec("camera/reconstruction.json", "camera_reconstruction"),
            InputSpec("observations/object_tracks.json", "segmentation_tracking"),
            InputSpec("reconstruction/dense/workspace_manifest.json", "dense_workspace_manifest"),
            InputSpec(
                "reconstruction/dense/undistortion_manifest.json",
                "dense_undistortion_manifest",
            ),
            InputSpec("reconstruction/dense/depth_manifest.json", "dense_depth_manifest"),
            InputSpec("reconstruction/dense/fusion.json", "dense_fusion_artifact"),
            InputSpec(
                "reconstruction/dense/fused.ply",
                "dense_fused_point_cloud",
                required=False,
                materialization_mode="reflink_or_copy",
            ),
        ]
        mask_paths = sorted(
            {observation.mask_path for track in tracks.tracks for observation in track.observations}
        )
        specs.extend(InputSpec(path, "canonical_object_mask") for path in mask_paths)
        for record in depth.records:
            specs.extend(
                [
                    InputSpec(record.depth_path, "dense_mvs_workspace_file"),
                    InputSpec(record.normal_path, "dense_mvs_workspace_file"),
                    InputSpec(record.consistency_graph_path, "dense_mvs_workspace_file"),
                ]
            )
        return specs

    def prepare(self, context: StageContext) -> None:
        for path in (
            context.path("reconstruction", "measured_objects", "objects"),
            context.path("reconstruction", "measured_objects", "raw", "logs"),
            context.path("reconstruction", "measured_objects", "previews", "objects"),
        ):
            path.mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        root = "reconstruction/measured_objects"
        outputs = [
            OutputSpec(
                f"{root}/request.json",
                "measured_object_geometry_request",
                "application/json",
                "measured_geometry",
                validation="json",
                model=MeasuredObjectGeometryRequest,
            ),
            OutputSpec(
                f"{root}/worker_manifest.json",
                "measured_object_worker_manifest",
                "application/json",
                "measured_geometry",
                validation="json",
                model=MeasuredObjectWorkerManifest,
            ),
            OutputSpec(
                f"{root}/geometry_manifest.json",
                "measured_object_geometry",
                "application/json",
                "measured_geometry",
                validation="json",
                model=MeasuredObjectGeometryArtifact,
            ),
            OutputSpec(
                f"{root}/diagnostics.json",
                "measured_object_diagnostics",
                "application/json",
                "measured_geometry",
                validation="json",
                model=MeasuredObjectDiagnostics,
            ),
            OutputSpec(
                "scene_ir/phase5a_scene.json",
                "scene_ir",
                "application/json",
                "measured_geometry",
                validation="scene_ir",
                model=SceneIR,
            ),
        ]
        for name in (
            "measured_object_contact_sheet",
            "depth_mask_contact_sheet",
            "reprojection_contact_sheet",
            "object_point_clouds",
        ):
            outputs.append(
                OutputSpec(
                    f"{root}/previews/{name}.png",
                    "measured_object_preview",
                    "image/png",
                    "measured_geometry",
                    validation="png",
                )
            )
        return outputs

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        if context is None:
            return HealthcheckResult(False, "measured-geometry healthcheck requires --config")
        try:
            config = MeasuredGeometryAdapterConfig.model_validate(context.config.adapter.config)
        except ValueError as exc:
            return HealthcheckResult(False, f"invalid measured-geometry configuration: {exc}")
        with tempfile.TemporaryDirectory(prefix="reconevery-measured-health-") as temp:
            path = Path(temp) / "config.json"
            atomic_write_json(path, {"worker_version": MEASURED_GEOMETRY_WORKER_VERSION})
            command = (
                self._docker_healthcheck_command(config, path)
                if config.execution_mode == "docker"
                else self._local_command(config, "healthcheck", path)
            )
            if isinstance(command, str):
                return HealthcheckResult(False, command)
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=min(context.config.adapter.timeout_s, 120),
                    check=False,
                    env=allowed_environment(context),
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return HealthcheckResult(False, f"measured-geometry healthcheck failed: {exc}")
            output = result.stdout.strip() or result.stderr.strip()
            if config.execution_mode == "docker" and result.returncode == 0:
                docker = resolve_executable(config.docker_executable)
                assert docker is not None
                try:
                    image_id = subprocess.run(
                        [docker, "image", "inspect", "--format", "{{.Id}}", config.docker_image],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    ).stdout.strip()
                except (OSError, subprocess.TimeoutExpired):
                    image_id = "unavailable"
                output = f"image_id={image_id} {output}"
            return HealthcheckResult(result.returncode == 0, output or "worker unavailable")

    @staticmethod
    def _docker_healthcheck_command(
        config: MeasuredGeometryAdapterConfig, config_path: Path
    ) -> list[str] | str:
        docker = resolve_executable(config.docker_executable)
        if docker is None:
            return "Docker executable was not found"
        try:
            daemon = subprocess.run(
                [docker, "version"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"Docker daemon check failed: {exc}"
        if daemon.returncode != 0:
            return f"Docker daemon is unavailable: {(daemon.stderr or daemon.stdout).strip()}"
        try:
            inspect = subprocess.run(
                [docker, "image", "inspect", "--format", "{{.Id}}", config.docker_image],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"Docker image check failed: {exc}"
        if inspect.returncode != 0:
            return f"Docker image {config.docker_image!r} is unavailable"
        return [
            docker,
            "run",
            "--rm",
            "-v",
            f"{config_path.parent.resolve()}:/health:ro",
            config.docker_image,
            "healthcheck",
            "--config",
            "/health/config.json",
        ]

    def run(self, context: StageContext) -> StageResult:
        config = MeasuredGeometryAdapterConfig.model_validate(context.config.adapter.config)
        manifest_path = context.path("inputs", "manifest.json")
        camera_path = context.path("camera", "reconstruction.json")
        tracks_path = context.path("observations", "object_tracks.json")
        dense_root = context.path("reconstruction", "dense")
        manifest = IngestManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        camera = CameraReconstruction.model_validate_json(camera_path.read_text(encoding="utf-8"))
        tracks = SegmentationTrackingArtifact.model_validate_json(
            tracks_path.read_text(encoding="utf-8")
        )
        workspace = DenseWorkspaceManifest.model_validate_json(
            (dense_root / "workspace_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.frame_sequence_digest is None:
            raise ValueError("measured geometry requires a frame-sequence digest")
        if (
            camera.frame_sequence_digest != manifest.frame_sequence_digest
            or tracks.frame_sequence_digest != manifest.frame_sequence_digest
            or workspace.frame_sequence_digest != manifest.frame_sequence_digest
        ):
            raise ValueError("measured geometry inputs do not share a frame lineage")
        if not coordinate_metadata_is_raw_colmap(camera.coordinate_convention):
            raise ValueError("measured geometry requires raw arbitrary COLMAP coordinates")
        request = MeasuredObjectGeometryRequest(
            run_id=context.canonical_run_dir.name,
            manifest_sha256=sha256_file(manifest_path),
            frame_sequence_digest=manifest.frame_sequence_digest,
            camera_reconstruction_sha256=sha256_file(camera_path),
            segmentation_tracking_sha256=sha256_file(tracks_path),
            object_tracks=[
                MeasuredObjectTrackRequest(
                    object_id=track.object_id,
                    semantic_label=track.semantic_label,
                    prompt_id=track.prompt_id,
                    asset_type_hint=track.asset_type_hint,
                    track_coverage=track.coverage_ratio,
                    mask_paths_by_frame={
                        observation.frame_id: observation.mask_path
                        for observation in track.observations
                        if observation.frame_id in camera.registered_frame_ids
                    },
                    frame_scores={
                        observation.frame_id: observation.frame_score
                        for observation in track.observations
                        if observation.frame_id in camera.registered_frame_ids
                    },
                )
                for track in tracks.tracks
            ],
            dense_workspace_manifest_path="reconstruction/dense/workspace_manifest.json",
            dense_workspace_manifest_sha256=sha256_file(dense_root / "workspace_manifest.json"),
            undistortion_manifest_path="reconstruction/dense/undistortion_manifest.json",
            undistortion_manifest_sha256=sha256_file(dense_root / "undistortion_manifest.json"),
            depth_manifest_path="reconstruction/dense/depth_manifest.json",
            depth_manifest_sha256=sha256_file(dense_root / "depth_manifest.json"),
            backprojection_configuration={
                "minimum_consistent_source_views": config.minimum_consistent_source_views,
                "minimum_sam_frame_score": config.minimum_sam_frame_score,
                "exclude_mask_boundary": config.exclude_mask_boundary_by_default,
                "maximum_relative_depth_discontinuity": (
                    config.maximum_relative_depth_discontinuity
                ),
                "pixel_stride": config.pixel_stride,
                "maximum_samples_per_object": config.maximum_samples_per_object,
                "fake_mode": config.fake_mode,
            },
            consistency_configuration={
                "minimum_supporting_views": config.minimum_supporting_views,
                "maximum_relative_depth_residual": config.maximum_relative_depth_residual,
                "maximum_contradicting_views": config.maximum_contradicting_views,
            },
            surfel_fusion_configuration={
                "voxel_size_source": config.voxel_size_source,
                "voxel_size_multiplier": config.voxel_size_multiplier,
            },
            observed_mesh_configuration={
                "enabled": False,
                "close_holes": False,
                "watertight": False,
                "reason": "surfel-only Phase 5A path; observed triangulation is optional",
            },
            reprojection_configuration={
                "visibility_aware": True,
                "splat_radius_pixels": config.reprojection_splat_radius_pixels,
                "minimum_accepted_iou": config.minimum_accepted_reprojection_iou,
            },
            coordinate_convention=camera.coordinate_convention,
            seed=context.seed,
        )
        request_path = context.path("reconstruction", "measured_objects", "request.json")
        atomic_write_json(request_path, request)
        try:
            run_process(
                self._inference_command(context, config),
                context=context,
                name="measured_geometry_worker",
                log_directory="reconstruction/measured_objects/raw/logs",
            )
        except ProcessExecutionError as exc:
            stderr = exc.result.stderr.lower()
            if "out of memory" in stderr:
                raise RuntimeError(
                    "measured geometry worker ran out of memory; increase pixel_stride "
                    "or lower maximum_samples_per_object"
                ) from exc
            raise RuntimeError(str(exc)) from exc
        root = context.path("reconstruction", "measured_objects")
        worker = self._model(root / "worker_manifest.json", MeasuredObjectWorkerManifest)
        artifact = self._model(root / "geometry_manifest.json", MeasuredObjectGeometryArtifact)
        diagnostics = self._model(root / "diagnostics.json", MeasuredObjectDiagnostics)
        expected = {
            "request_sha256": sha256_file(request_path),
            "manifest_sha256": request.manifest_sha256,
            "frame_sequence_digest": request.frame_sequence_digest,
            "camera_reconstruction_sha256": request.camera_reconstruction_sha256,
            "segmentation_tracking_sha256": request.segmentation_tracking_sha256,
            "depth_manifest_sha256": request.depth_manifest_sha256,
        }
        for key, value in expected.items():
            if getattr(worker, key) != value:
                raise RuntimeError(f"measured-geometry worker {key} does not match request")
            if key != "request_sha256" and hasattr(artifact, key):
                if getattr(artifact, key) != value:
                    raise RuntimeError(f"measured geometry {key} does not match request")
        if artifact.coordinate_convention != camera.coordinate_convention:
            raise RuntimeError("measured geometry changed coordinate semantics")
        if artifact.generated_geometry_used_as_source:
            raise RuntimeError("canonical measured geometry may not use generated geometry")
        for hypothesis in artifact.hypotheses:
            for cloud in (hypothesis.point_cloud, hypothesis.surfel_cloud):
                if cloud is None:
                    continue
                path = context.path(*Path(cloud.relative_path).parts)
                points, _ = ply_counts(path)
                if points != cloud.point_count or sha256_file(path) != cloud.sha256:
                    raise RuntimeError(f"measured cloud is inconsistent for {hypothesis.object_id}")
            if hypothesis.observed_surface is not None:
                path = context.path(*Path(hypothesis.observed_surface.relative_path).parts)
                vertices, faces = ply_counts(path)
                if (
                    vertices != hypothesis.observed_surface.vertex_count
                    or faces != hypothesis.observed_surface.face_count
                    or sha256_file(path) != hypothesis.observed_surface.sha256
                ):
                    raise RuntimeError(
                        f"observed surface is inconsistent for {hypothesis.object_id}"
                    )
        canonical_scene = context.canonical_path("scene_ir", "scene.json")
        scene = (
            SceneIR.model_validate_json(canonical_scene.read_text(encoding="utf-8"))
            if canonical_scene.is_file()
            else self._base_scene(manifest, camera)
        )
        atomic_write_json(
            context.path("scene_ir", "phase5a_scene.json"),
            self._integrate_scene(scene, artifact),
        )
        if diagnostics.track_count != len(tracks.tracks):
            raise RuntimeError("measured diagnostics do not cover all SAM tracks")
        fixed = {
            "request.json",
            "worker_manifest.json",
            "geometry_manifest.json",
            "diagnostics.json",
            "previews/measured_object_contact_sheet.png",
            "previews/depth_mask_contact_sheet.png",
            "previews/reprojection_contact_sheet.png",
            "previews/object_point_clouds.png",
        }
        dynamic = [
            OutputSpec(
                path.relative_to(context.run_dir).as_posix(),
                "measured_object_geometry_file",
                "model/ply" if path.suffix == ".ply" else "application/octet-stream",
                "measured_geometry",
            )
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.relative_to(root).as_posix() not in fixed
        ]
        return StageResult(
            outputs=dynamic,
            metrics={
                "accepted_objects": diagnostics.accepted_object_count,
                "partial_objects": diagnostics.partial_object_count,
                "unresolved_objects": diagnostics.unresolved_object_count,
                "surfels": diagnostics.fused_surfel_count,
            },
        )

    @staticmethod
    def _model(path: Path, model: Any) -> Any:
        if not path.is_file():
            raise RuntimeError(f"measured-geometry worker omitted {path.name}")
        try:
            return model.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise RuntimeError(
                f"measured-geometry worker produced malformed {path.name}: {exc}"
            ) from exc

    @staticmethod
    def _base_scene(manifest: IngestManifest, camera: CameraReconstruction) -> SceneIR:
        return SceneIR(
            metadata=SceneMetadata(
                scene_id="measured_scene",
                name="Measured partial object geometry",
                coordinate_convention=camera.coordinate_convention,
                source=GeometrySourceType.MEASURED,
                provenance=[camera.provenance],
            ),
            cameras=[
                Camera(
                    camera_id=camera.camera_id,
                    model=camera.model,
                    intrinsics=camera.intrinsics,
                    poses=camera.poses,
                    coordinate_convention=camera.coordinate_convention,
                    scale_status=camera.scale_status,
                    provenance=camera.provenance,
                )
            ],
            frames=[
                FrameObservation(
                    frame_id=frame.frame_id,
                    frame_path=frame.relative_path,
                    timestamp_s=frame.timestamp_s,
                    camera_id=camera.camera_id,
                )
                for frame in manifest.frames
            ],
        )

    @staticmethod
    def _integrate_scene(scene: SceneIR, artifact: MeasuredObjectGeometryArtifact) -> SceneIR:
        asset_ids = {f"measured_partial_{item.object_id}" for item in artifact.hypotheses}
        scene.geometry_assets = [
            asset for asset in scene.geometry_assets if asset.asset_id not in asset_ids
        ]
        objects = {item.object_id: item for item in scene.objects}
        for hypothesis in artifact.hypotheses:
            if hypothesis.status == "unresolved" or hypothesis.point_cloud is None:
                continue
            asset_id = f"measured_partial_{hypothesis.object_id}"
            scene.geometry_assets.append(
                GeometryAsset(
                    asset_id=asset_id,
                    asset_type=AssetType.UNCLASSIFIED,
                    uri=hypothesis.point_cloud.relative_path,
                    format="ply",
                    source=GeometrySourceType.MEASURED,
                    coordinate_convention=hypothesis.coordinate_convention,
                    scale_status=ScaleStatus.SCALE_AMBIGUOUS,
                    geometry_status="partial_measured",
                    completion_status="not_completed",
                    sim_ready=False,
                    provenance=hypothesis.provenance,
                )
            )
            prior = objects.get(hypothesis.object_id)
            geometry_ids = list(prior.geometry_asset_ids) if prior is not None else []
            if asset_id not in geometry_ids:
                geometry_ids.append(asset_id)
            objects[hypothesis.object_id] = ObjectInstance(
                object_id=hypothesis.object_id,
                name=hypothesis.semantic_label,
                asset_type=AssetType.UNCLASSIFIED,
                geometry_asset_ids=geometry_ids,
                geometry_status="partial_measured",
                completion_status="not_completed",
                sim_ready=False,
                confidence=(
                    prior.confidence if prior is not None else hypothesis.provenance.confidence
                ),
                provenance=[
                    *(prior.provenance if prior is not None else []),
                    hypothesis.provenance,
                ],
            )
        scene.objects = sorted(objects.values(), key=lambda item: item.object_id)
        scene.schema_version = "0.1.4"
        return SceneIR.model_validate(scene.model_dump(mode="json"))

    def _inference_command(
        self, context: StageContext, config: MeasuredGeometryAdapterConfig
    ) -> list[str]:
        request = Path("reconstruction/measured_objects/request.json")
        if config.execution_mode != "docker":
            command = self._local_command(config, "infer", request)
            if isinstance(command, str):
                raise RuntimeError(command)
            return [
                *command,
                "--input-root",
                str(context.run_dir.resolve()),
                "--output-dir",
                str(context.path("reconstruction", "measured_objects").resolve()),
            ]
        docker = resolve_executable(config.docker_executable)
        if docker is None:
            raise RuntimeError("Docker executable was not found")
        user = (
            ["--user", f"{os.getuid()}:{os.getgid()}"]
            if hasattr(os, "getuid") and hasattr(os, "getgid")
            else []
        )
        return [
            docker,
            "run",
            "--rm",
            *user,
            "-v",
            f"{context.run_dir.resolve()}:/workspace:rw",
            "-w",
            "/workspace",
            config.docker_image,
            "infer",
            "--request",
            f"/workspace/{request.as_posix()}",
            "--input-root",
            "/workspace",
            "--output-dir",
            "/workspace/reconstruction/measured_objects",
        ]

    @staticmethod
    def _local_command(
        config: MeasuredGeometryAdapterConfig, action: str, path: Path
    ) -> list[str] | str:
        python = _resolve_python(config.worker_python)
        if python is None:
            return f"configured worker Python {config.worker_python!r} was not found"
        option = "--config" if action == "healthcheck" else "--request"
        if config.execution_mode == "fake_worker":
            assert config.worker_script is not None
            script = Path(config.worker_script)
            if not script.is_absolute():
                script = Path.cwd() / script
            if not script.is_file():
                return f"fake measured-geometry worker does not exist: {script}"
            return [python, str(script.resolve()), action, option, str(path)]
        return [python, "-m", config.worker_module, action, option, str(path)]


class MeasuredGeneratedComparisonAdapter:
    name = "measured_generated_comparison"
    version = "0.1.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        return [
            InputSpec(
                "reconstruction/measured_objects/geometry_manifest.json",
                "measured_object_geometry",
            ),
            InputSpec("reconstruction/global/metadata.json", "global_scene_reconstruction"),
            InputSpec(
                "reconstruction/object_surfaces/evidence_manifest.json",
                "object_surface_evidence",
                required=False,
            ),
        ]

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return HealthcheckResult(True, "lightweight measured/generated comparison available")

    def prepare(self, context: StageContext) -> None:
        context.path("reconstruction", "measured_generated").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                "reconstruction/measured_generated/comparison.json",
                "measured_generated_comparison",
                "application/json",
                "measured_generated_comparison",
                validation="json",
                model=MeasuredGeneratedComparisonArtifact,
            )
        ]

    def run(self, context: StageContext) -> StageResult:
        measured_path = context.path("reconstruction", "measured_objects", "geometry_manifest.json")
        global_path = context.path("reconstruction", "global", "metadata.json")
        measured = MeasuredObjectGeometryArtifact.model_validate_json(
            measured_path.read_text(encoding="utf-8")
        )
        comparison = MeasuredGeneratedComparisonArtifact(
            measured_geometry_sha256=sha256_file(measured_path),
            global_reconstruction_sha256=sha256_file(global_path),
            objects=[
                MeasuredGeneratedObjectComparison(
                    object_id=item.object_id,
                    reprojection_precision=item.reprojection_precision,
                    reprojection_recall=item.reprojection_recall,
                    measured_surface_covered_by_genrecon=None,
                    genrecon_hypothesis_covered_by_measurement=None,
                    diagnosis="not_computed",
                )
                for item in measured.hypotheses
            ],
        )
        atomic_write_json(
            context.path("reconstruction", "measured_generated", "comparison.json"),
            comparison,
        )
        return StageResult(metrics={"compared_objects": len(comparison.objects)})


class Phase5AConsistencyValidationAdapter:
    name = "phase5a_consistency_validation"
    version = "0.1.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        return [
            InputSpec("inputs/manifest.json", "ingest_manifest"),
            InputSpec("camera/reconstruction.json", "camera_reconstruction"),
            InputSpec("observations/object_tracks.json", "segmentation_tracking"),
            InputSpec("reconstruction/dense/workspace_manifest.json", "dense_workspace_manifest"),
            InputSpec("reconstruction/dense/depth_manifest.json", "dense_depth_manifest"),
            InputSpec("reconstruction/dense/fusion.json", "dense_fusion_artifact"),
            InputSpec(
                "reconstruction/measured_objects/geometry_manifest.json",
                "measured_object_geometry",
            ),
            InputSpec("scene_ir/phase5a_scene.json", "scene_ir"),
        ]

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return HealthcheckResult(True, "Phase 5A consistency validator available")

    def prepare(self, context: StageContext) -> None:
        context.path("validation").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                "validation/phase5a_measured_geometry.json",
                "phase5a_consistency_report",
                "application/json",
                "validation",
                validation="json",
                model=Phase5AConsistencyReport,
            )
        ]

    def run(self, context: StageContext) -> StageResult:
        manifest_path = context.path("inputs", "manifest.json")
        camera_path = context.path("camera", "reconstruction.json")
        tracks_path = context.path("observations", "object_tracks.json")
        manifest = IngestManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        camera = CameraReconstruction.model_validate_json(camera_path.read_text(encoding="utf-8"))
        workspace = DenseWorkspaceManifest.model_validate_json(
            context.path("reconstruction", "dense", "workspace_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        depth = DenseDepthManifest.model_validate_json(
            context.path("reconstruction", "dense", "depth_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        fusion = DenseFusionArtifact.model_validate_json(
            context.path("reconstruction", "dense", "fusion.json").read_text(encoding="utf-8")
        )
        measured = MeasuredObjectGeometryArtifact.model_validate_json(
            context.path("reconstruction", "measured_objects", "geometry_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        scene = SceneIR.model_validate_json(
            context.path("scene_ir", "phase5a_scene.json").read_text(encoding="utf-8")
        )
        frame_order = [frame.frame_id for frame in manifest.frames]
        checks = [
            EndToEndConsistencyCheck(
                check_id="manifest_lineage",
                passed=(
                    measured.manifest_sha256 == sha256_file(manifest_path)
                    and workspace.manifest_sha256 == sha256_file(manifest_path)
                ),
                message="dense and measured artifacts reference the ingest manifest",
            ),
            EndToEndConsistencyCheck(
                check_id="frame_sequence_digest",
                passed=(
                    manifest.frame_sequence_digest
                    == measured.frame_sequence_digest
                    == workspace.frame_sequence_digest
                ),
                message="dense and measured artifacts retain ordered frame hashes",
            ),
            EndToEndConsistencyCheck(
                check_id="camera_lineage",
                passed=(
                    measured.camera_reconstruction_sha256
                    == sha256_file(camera_path)
                    == workspace.camera_reconstruction_sha256
                ),
                message="dense and measured artifacts reference real registered cameras",
            ),
            EndToEndConsistencyCheck(
                check_id="registered_manifest_order",
                passed=workspace.registered_frame_ids
                == [item for item in frame_order if item in camera.registered_frame_ids],
                message="dense frames are registered frames in manifest order",
            ),
            EndToEndConsistencyCheck(
                check_id="dense_map_dimensions",
                passed=all(
                    record.dimensions[0] > 0 and record.dimensions[1] > 0
                    for record in depth.records
                ),
                message="depth, normal, and consistency records have valid dimensions",
            ),
            EndToEndConsistencyCheck(
                check_id="measured_not_generated",
                passed=not measured.generated_geometry_used_as_source,
                message="canonical measured geometry does not depend on GenRecon",
            ),
            EndToEndConsistencyCheck(
                check_id="coordinate_semantics",
                passed=(
                    measured.coordinate_convention
                    == camera.coordinate_convention
                    == fusion.coordinate_convention
                    and coordinate_metadata_is_raw_colmap(camera.coordinate_convention)
                ),
                message="all geometry remains arbitrary, unoriented COLMAP coordinates",
            ),
            EndToEndConsistencyCheck(
                check_id="finite_positive_measured_geometry",
                passed=all(
                    item.status == "unresolved"
                    or (
                        item.point_cloud is not None
                        and item.point_cloud.point_count > 0
                        and item.supporting_view_count >= 1
                    )
                    for item in measured.hypotheses
                ),
                message="non-empty hypotheses contain finite typed measured points",
            ),
            EndToEndConsistencyCheck(
                check_id="honest_capability_boundary",
                passed=all(
                    item.completeness_confidence == 0 and not item.watertight and not item.sim_ready
                    for item in measured.hypotheses
                ),
                message="no completion, watertight, metric, collision, or simulation claim",
            ),
            EndToEndConsistencyCheck(
                check_id="scene_ir_measured_assets",
                passed=all(
                    asset.source is GeometrySourceType.MEASURED
                    and asset.geometry_status == "partial_measured"
                    and not asset.sim_ready
                    for asset in scene.geometry_assets
                    if asset.asset_id.startswith("measured_partial_")
                ),
                message="Scene IR records measured partial assets without physics claims",
            ),
            EndToEndConsistencyCheck(
                check_id="segmentation_lineage",
                passed=measured.segmentation_tracking_sha256 == sha256_file(tracks_path),
                message="measured objects reference canonical SAM tracks",
            ),
        ]
        report = Phase5AConsistencyReport(
            passed=all(check.passed for check in checks),
            checks=checks,
            measured_dense_geometry_available=fusion.point_count > 0,
            measured_object_geometry_available=any(
                item.status != "unresolved" for item in measured.hypotheses
            ),
        )
        atomic_write_json(context.path("validation", "phase5a_measured_geometry.json"), report)
        if not report.passed:
            raise RuntimeError(
                "Phase 5A consistency failed: "
                + ", ".join(check.check_id for check in checks if not check.passed)
            )
        return StageResult(
            metrics={
                "checks": len(checks),
                "measured_objects": sum(
                    item.status != "unresolved" for item in measured.hypotheses
                ),
            }
        )
