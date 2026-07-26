from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal, cast

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
    CameraMeshAlignmentArtifact,
    CameraReconstruction,
    GlobalSceneReconstructionArtifact,
    IngestManifest,
    ObjectSurfaceDiagnostics,
    ObjectSurfaceEvidenceArtifact,
    ObjectSurfaceLiftingRequest,
    ObjectSurfaceMethodComparison,
    ObjectSurfacePreviewManifest,
    ObjectSurfaceTrackRequest,
    ObjectSurfaceWorkerManifest,
    Phase4ConsistencyCheck,
    Phase4ConsistencyReport,
    SegmentationTrackingArtifact,
)
from recon2sim.genrecon import sha256_file
from recon2sim.ir import (
    AssetType,
    GeometryAsset,
    GeometrySourceType,
    ObjectInstance,
    ScaleStatus,
    SceneIR,
    StrictModel,
)
from recon2sim.object_lifting import (
    coordinate_metadata_is_raw_colmap,
    read_compact_face_ids,
    validate_surface_mesh,
)
from recon2sim.storage import atomic_write_json

OBJECT_LIFTING_WORKER_VERSION = "0.1.1"


def _resolve_python(value: str) -> str | None:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        return str(candidate.absolute()) if candidate.is_file() else None
    return shutil.which(value)


class ObjectLiftingAdapterConfig(StrictModel):
    execution_mode: Literal["local_worker", "docker", "fake_worker"]
    worker_python: str = "python"
    worker_module: str = "object_lifting_worker"
    worker_script: str | None = None
    docker_executable: str = "docker"
    docker_image: str = "reconevery/object-lifting:phase4"
    device: Literal["cuda", "cpu"] = "cuda"
    lifting_method: Literal["exact_face_vote_v1", "surface_sample_fusion_v2"] = (
        "surface_sample_fusion_v2"
    )
    raster_scale: float = Field(default=0.5, gt=0, le=1)
    face_chunk_size: int = Field(default=1_000_000, gt=0)
    near_plane_strategy: Literal["camera_relative"] = "camera_relative"
    far_plane_strategy: Literal["scene_bounds"] = "scene_bounds"
    mask_core_erosion_pixels: int = Field(default=2, ge=0)
    mask_boundary_width_pixels: int = Field(default=3, ge=0)
    mask_exclusion_dilation_pixels: int = Field(default=2, ge=0)
    core_positive_weight: float = Field(default=1.0, ge=0)
    boundary_positive_weight: float = Field(default=0.25, ge=0)
    exterior_negative_weight: float = Field(default=1.0, ge=0)
    min_visible_pixels_per_face: int = Field(default=2, gt=0)
    min_positive_pixels_per_face: int = Field(default=2, gt=0)
    min_supporting_views: int = Field(default=2, gt=0)
    accepted_face_score: float = Field(default=0.65, ge=0, le=1)
    ambiguous_face_score: float = Field(default=0.40, ge=0, le=1)
    instance_score_margin: float = Field(default=0.05, ge=0, le=1)
    sample_voxel_edge_multiplier: float = Field(default=4.0, gt=0)
    sample_min_supporting_views: int = Field(default=2, gt=0)
    sample_min_positive_weight: float = Field(default=2.0, gt=0)
    sample_negative_margin_multiplier: float = Field(default=2.0, ge=0)
    min_component_faces: int = Field(default=4, gt=0)
    min_relative_component_area: float = Field(default=0.01, ge=0, le=1)
    seam_diagnostic_enabled: bool = True
    seam_centroid_distance_multiplier: float = Field(default=3.0, gt=0)
    seam_endpoint_distance_multiplier: float = Field(default=2.0, gt=0)
    seam_normal_cosine: float = Field(default=0.95, ge=-1, le=1)
    accepted_min_reprojection_iou: float = Field(default=0.10, ge=0, le=1)
    partial_min_reprojection_iou: float = Field(default=0.01, ge=0, le=1)
    accepted_max_ambiguity_ratio: float = Field(default=0.50, ge=0, le=1)
    alignment_depth_inlier_threshold: float = Field(default=0.10, gt=0)
    alignment_min_inlier_fraction: float = Field(default=0.30, ge=0, le=1)
    seed: int = 42
    fake_mode: str = "success"

    @model_validator(mode="after")
    def validate_execution(self) -> ObjectLiftingAdapterConfig:
        if self.accepted_face_score < self.ambiguous_face_score:
            raise ValueError("accepted_face_score must be at least ambiguous_face_score")
        if self.accepted_min_reprojection_iou < self.partial_min_reprojection_iou:
            raise ValueError(
                "accepted_min_reprojection_iou must be at least partial_min_reprojection_iou"
            )
        if self.execution_mode == "fake_worker":
            if self.worker_script is None:
                raise ValueError("fake_worker execution requires worker_script")
            if self.device != "cpu":
                raise ValueError("fake_worker must use device=cpu")
            return self
        if self.device != "cuda":
            raise ValueError("real object lifting requires device=cuda")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None or visible.strip().lower() in {"", "-1", "none", "void"}:
            raise ValueError("real object lifting requires CUDA_VISIBLE_DEVICES")
        if self.execution_mode == "local_worker":
            python = _resolve_python(self.worker_python)
            if python is None:
                raise ValueError(f"configured worker Python {self.worker_python!r} was not found")
            executable = Path(python).absolute()
            root = executable.parent.parent
            if not (root / "pyvenv.cfg").is_file() and not (root / "conda-meta").is_dir():
                raise ValueError(
                    "local object-lifting worker_python must be in an isolated environment"
                )
            if root == Path(sys.prefix).resolve():
                raise ValueError("object-lifting worker must not use the core environment")
        return self


class ObjectSurfaceLiftingAdapter:
    name = "object_surface_lifting"
    version = "0.1.1"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        tracks_path = context.canonical_path("observations", "object_tracks.json")
        if not tracks_path.is_file():
            raise FileNotFoundError("object lifting requires canonical SAM object tracks")
        tracks = SegmentationTrackingArtifact.model_validate_json(
            tracks_path.read_text(encoding="utf-8")
        )
        specs = [
            InputSpec("inputs/manifest.json", "ingest_manifest"),
            InputSpec("camera/reconstruction.json", "camera_reconstruction"),
            InputSpec(
                "camera/genrecon_package/package_manifest.json",
                "genrecon_camera_package_manifest",
            ),
            InputSpec(
                "camera/genrecon_package/images.txt",
                "genrecon_colmap_text",
            ),
            InputSpec(
                "camera/genrecon_package/points3D.txt",
                "genrecon_colmap_text",
            ),
            InputSpec(
                "camera/genrecon_package/registered_frames.json",
                "genrecon_registered_frames",
            ),
            InputSpec("observations/object_tracks.json", "segmentation_tracking"),
            InputSpec(
                "reconstruction/global/metadata.json",
                "global_scene_reconstruction",
            ),
            InputSpec(
                "reconstruction/global/mesh.ply",
                "global_scene_mesh",
                materialization_mode="reflink_or_copy",
            ),
            InputSpec("scene_ir/scene.json", "scene_ir"),
        ]
        mask_paths = sorted(
            {observation.mask_path for track in tracks.tracks for observation in track.observations}
        )
        specs.extend(InputSpec(path, "canonical_object_mask") for path in mask_paths)
        return specs

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        if context is None:
            return HealthcheckResult(
                False,
                "object-lifting healthcheck requires --config to verify its worker",
            )
        try:
            config = ObjectLiftingAdapterConfig.model_validate(context.config.adapter.config)
        except ValueError as exc:
            return HealthcheckResult(False, f"invalid object-lifting configuration: {exc}")
        payload = {
            "worker_version": OBJECT_LIFTING_WORKER_VERSION,
            "device": config.device,
            "backend": "fake" if config.execution_mode == "fake_worker" else "nvdiffrast",
        }
        with tempfile.TemporaryDirectory(prefix="reconevery-object-lifting-health-") as temp:
            path = Path(temp) / "worker_config.json"
            atomic_write_json(path, payload)
            if config.execution_mode == "docker":
                return self._docker_healthcheck(context, config, path)
            command = self._local_command(config, "healthcheck", path)
            if isinstance(command, str):
                return HealthcheckResult(False, command)
            try:
                result = subprocess.run(
                    command,
                    cwd=Path.cwd(),
                    env=allowed_environment(context),
                    capture_output=True,
                    text=True,
                    timeout=min(context.config.adapter.timeout_s, 120),
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return HealthcheckResult(False, f"object-lifting healthcheck failed: {exc}")
            output = result.stdout.strip() or result.stderr.strip()
            if result.returncode != 0:
                return HealthcheckResult(False, output or "object-lifting healthcheck failed")
            return HealthcheckResult(True, output or "object-lifting worker available")

    def _docker_healthcheck(
        self,
        context: StageContext,
        config: ObjectLiftingAdapterConfig,
        config_path: Path,
    ) -> HealthcheckResult:
        docker = resolve_executable(config.docker_executable)
        if docker is None:
            return HealthcheckResult(False, "Docker executable was not found")
        for command in (
            [docker, "version", "--format", "{{.Server.Version}}"],
            [docker, "image", "inspect", "--format", "{{.Id}}", config.docker_image],
        ):
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode != 0:
                return HealthcheckResult(False, result.stderr.strip() or "Docker check failed")
        command = [
            docker,
            "run",
            "--rm",
            "--gpus",
            "all",
            *self._docker_user_arguments(),
            "-v",
            f"{config_path.parent}:/workspace:ro",
            "--entrypoint",
            "python",
            config.docker_image,
            "-m",
            config.worker_module,
            "healthcheck",
            "--config",
            "/workspace/worker_config.json",
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=min(context.config.adapter.timeout_s, 180),
            check=False,
        )
        if result.returncode != 0:
            return HealthcheckResult(
                False,
                result.stderr.strip() or "in-container object-lifting healthcheck failed",
            )
        return HealthcheckResult(True, result.stdout.strip())

    def prepare(self, context: StageContext) -> None:
        context.path("reconstruction", "object_surfaces", "raw", "logs").mkdir(
            parents=True, exist_ok=True
        )
        context.path("reconstruction", "object_surfaces", "objects").mkdir(
            parents=True, exist_ok=True
        )
        context.path("reconstruction", "object_surfaces", "previews", "objects").mkdir(
            parents=True, exist_ok=True
        )

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        root = "reconstruction/object_surfaces"
        return [
            OutputSpec(
                f"{root}/request.json",
                "object_surface_lifting_request",
                "application/json",
                "object_lifting",
                validation="json",
                schema_identifier="recon2sim/object-surface-lifting-request/0.1.0",
                model=ObjectSurfaceLiftingRequest,
            ),
            OutputSpec(
                f"{root}/worker_manifest.json",
                "object_surface_worker_manifest",
                "application/json",
                "object_lifting",
                validation="json",
                schema_identifier="recon2sim/object-surface-worker-manifest/0.1.0",
                model=ObjectSurfaceWorkerManifest,
            ),
            OutputSpec(
                f"{root}/evidence_manifest.json",
                "object_surface_evidence",
                "application/json",
                "object_lifting",
                validation="json",
                schema_identifier="recon2sim/object-surface-evidence/0.1.0",
                model=ObjectSurfaceEvidenceArtifact,
            ),
            OutputSpec(
                f"{root}/face_assignment_manifest.json",
                "object_surface_face_assignment",
                "application/json",
                "object_lifting",
                validation="json",
                model=ObjectSurfaceEvidenceArtifact,
            ),
            OutputSpec(
                f"{root}/diagnostics.json",
                "object_surface_diagnostics",
                "application/json",
                "object_lifting",
                validation="json",
                schema_identifier="recon2sim/object-surface-diagnostics/0.1.0",
                model=ObjectSurfaceDiagnostics,
            ),
            OutputSpec(
                f"{root}/method_comparison.json",
                "object_surface_method_comparison",
                "application/json",
                "object_lifting",
                validation="json",
                schema_identifier="recon2sim/object-surface-method-comparison/0.1.0",
                model=ObjectSurfaceMethodComparison,
            ),
            OutputSpec(
                f"{root}/camera_mesh_alignment.json",
                "camera_mesh_alignment",
                "application/json",
                "object_lifting",
                validation="json",
                schema_identifier="recon2sim/camera-mesh-alignment/0.1.0",
                model=CameraMeshAlignmentArtifact,
            ),
            OutputSpec(
                f"{root}/preview_manifest.json",
                "object_surface_preview_manifest",
                "application/json",
                "object_lifting",
                validation="json",
                model=ObjectSurfacePreviewManifest,
            ),
            *[
                OutputSpec(
                    f"{root}/previews/{name}.png",
                    "object_surface_preview",
                    "image/png",
                    "object_lifting",
                    validation="png",
                )
                for name in (
                    "global_face_assignment",
                    "object_surface_contact_sheet",
                    "reprojection_contact_sheet",
                    "conflict_heatmap",
                    "global_mesh_depth_contact_sheet",
                    "global_mesh_edge_overlay",
                    "sparse_point_vs_mesh_depth",
                    "surface_sample_fusion",
                )
            ],
            OutputSpec(
                "scene_ir/phase4_scene.json",
                "scene_ir",
                "application/json",
                "object_lifting",
                validation="scene_ir",
                schema_identifier="recon2sim/scene-ir/0.1.2",
                model=SceneIR,
            ),
        ]

    def run(self, context: StageContext) -> StageResult:
        config = ObjectLiftingAdapterConfig.model_validate(context.config.adapter.config)
        manifest_path = context.path("inputs", "manifest.json")
        camera_path = context.path("camera", "reconstruction.json")
        tracks_path = context.path("observations", "object_tracks.json")
        global_path = context.path("reconstruction", "global", "metadata.json")
        mesh_path = context.path("reconstruction", "global", "mesh.ply")
        glb_path = context.canonical_path("reconstruction", "global", "scene.glb")
        if not glb_path.is_file():
            raise FileNotFoundError("canonical Phase 3 scene.glb is missing")
        manifest = IngestManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        camera = CameraReconstruction.model_validate_json(camera_path.read_text(encoding="utf-8"))
        tracks = SegmentationTrackingArtifact.model_validate_json(
            tracks_path.read_text(encoding="utf-8")
        )
        global_scene = GlobalSceneReconstructionArtifact.model_validate_json(
            global_path.read_text(encoding="utf-8")
        )
        self._validate_upstream_lineage(manifest, camera, tracks, global_scene, camera_path)
        request = self._request(
            context,
            config,
            manifest,
            camera,
            tracks,
            global_scene,
            manifest_path,
            camera_path,
            tracks_path,
            global_path,
            mesh_path,
        )
        request_path = context.path("reconstruction", "object_surfaces", "request.json")
        atomic_write_json(request_path, request)
        command = self._inference_command(context, config)
        try:
            run_process(
                command,
                context=context,
                name="object_lifting_worker",
                log_directory="reconstruction/object_surfaces/raw/logs",
            )
        except ProcessExecutionError as exc:
            raise self._worker_failure(exc) from exc
        root = context.path("reconstruction", "object_surfaces")
        worker = self._load_model(
            root / "worker_manifest.json",
            ObjectSurfaceWorkerManifest,
            "object-lifting worker manifest",
        )
        evidence = self._load_model(
            root / "evidence_manifest.json",
            ObjectSurfaceEvidenceArtifact,
            "object-surface evidence",
        )
        diagnostics = self._load_model(
            root / "diagnostics.json",
            ObjectSurfaceDiagnostics,
            "object-surface diagnostics",
        )
        comparison = self._load_model(
            root / "method_comparison.json",
            ObjectSurfaceMethodComparison,
            "object-surface method comparison",
        )
        alignment = self._load_model(
            root / "camera_mesh_alignment.json",
            CameraMeshAlignmentArtifact,
            "camera/mesh alignment diagnostics",
        )
        previews = self._load_model(
            root / "preview_manifest.json",
            ObjectSurfacePreviewManifest,
            "object-surface previews",
        )
        self._validate_worker_output(
            context,
            config,
            request,
            worker,
            evidence,
            previews,
            comparison,
            alignment,
        )
        if diagnostics.track_count != len(tracks.tracks):
            raise RuntimeError("object-lifting diagnostics track count is inconsistent")
        shutil.copy2(root / "evidence_manifest.json", root / "face_assignment_manifest.json")
        scene = SceneIR.model_validate_json(
            context.path("scene_ir", "scene.json").read_text(encoding="utf-8")
        )
        atomic_write_json(
            context.path("scene_ir", "phase4_scene.json"),
            self._integrate_scene_ir(scene, evidence),
        )
        dynamic = [
            OutputSpec(
                path.relative_to(context.run_dir).as_posix(),
                self._dynamic_artifact_type(path),
                self._media_type(path),
                "object_lifting",
                validation="png" if path.suffix.lower() == ".png" else "exists",
            )
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.relative_to(root).as_posix()
            not in {
                "request.json",
                "worker_manifest.json",
                "evidence_manifest.json",
                "face_assignment_manifest.json",
                "diagnostics.json",
                "method_comparison.json",
                "camera_mesh_alignment.json",
                "preview_manifest.json",
                "previews/global_face_assignment.png",
                "previews/object_surface_contact_sheet.png",
                "previews/reprojection_contact_sheet.png",
                "previews/conflict_heatmap.png",
                "previews/global_mesh_depth_contact_sheet.png",
                "previews/global_mesh_edge_overlay.png",
                "previews/sparse_point_vs_mesh_depth.png",
                "previews/surface_sample_fusion.png",
            }
        ]
        return StageResult(
            outputs=dynamic,
            metrics={
                "tracks": diagnostics.track_count,
                "accepted_objects": diagnostics.accepted_object_count,
                "ambiguous_objects": diagnostics.ambiguous_object_count,
                "unresolved_objects": diagnostics.unresolved_object_count,
                "accepted_faces": diagnostics.accepted_face_count,
                "processed_cameras": diagnostics.processed_camera_count,
            },
        )

    @staticmethod
    def _load_model(path: Path, model: Any, label: str) -> Any:
        if not path.is_file():
            raise RuntimeError(f"worker completed without {label}: {path.name}")
        try:
            return model.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise RuntimeError(f"{label} is malformed: {exc}") from exc

    @staticmethod
    def _validate_upstream_lineage(
        manifest: IngestManifest,
        camera: CameraReconstruction,
        tracks: SegmentationTrackingArtifact,
        global_scene: GlobalSceneReconstructionArtifact,
        camera_path: Path,
    ) -> None:
        digest = manifest.frame_sequence_digest
        if digest is None:
            raise ValueError("object lifting requires a frame-sequence digest")
        if camera.frame_sequence_digest != digest or tracks.frame_sequence_digest != digest:
            raise ValueError("camera or segmentation frame lineage does not match ingest")
        if global_scene.frame_sequence_digest != digest:
            raise ValueError("global reconstruction frame lineage does not match ingest")
        if global_scene.camera_reconstruction_sha256 != sha256_file(camera_path):
            raise ValueError("global reconstruction references a different camera reconstruction")
        if not coordinate_metadata_is_raw_colmap(camera.coordinate_convention):
            raise ValueError("object lifting requires raw arbitrary COLMAP coordinate semantics")
        if camera.coordinate_convention != global_scene.coordinate_convention:
            raise ValueError("camera and global mesh coordinate metadata disagree")
        registered = set(camera.registered_frame_ids)
        for track in tracks.tracks:
            for observation in track.observations:
                if observation.camera_pose_available != (observation.frame_id in registered):
                    raise ValueError(
                        f"SAM camera-pose availability is wrong for {observation.frame_id}"
                    )

    @staticmethod
    def _request(
        context: StageContext,
        config: ObjectLiftingAdapterConfig,
        manifest: IngestManifest,
        camera: CameraReconstruction,
        tracks: SegmentationTrackingArtifact,
        global_scene: GlobalSceneReconstructionArtifact,
        manifest_path: Path,
        camera_path: Path,
        tracks_path: Path,
        global_path: Path,
        mesh_path: Path,
    ) -> ObjectSurfaceLiftingRequest:
        camera_package_root = context.path("camera", "genrecon_package")
        return ObjectSurfaceLiftingRequest(
            run_id=context.canonical_run_dir.name,
            manifest_sha256=sha256_file(manifest_path),
            frame_sequence_digest=cast(str, manifest.frame_sequence_digest),
            master_frame_order=[frame.frame_id for frame in manifest.frames],
            normalized_frame_paths={
                frame.frame_id: frame.relative_path for frame in manifest.frames
            },
            normalized_frame_hashes={frame.frame_id: frame.sha256 for frame in manifest.frames},
            camera_reconstruction_sha256=sha256_file(camera_path),
            camera_package_manifest_sha256=sha256_file(
                camera_package_root / "package_manifest.json"
            ),
            camera_package_images_sha256=sha256_file(camera_package_root / "images.txt"),
            camera_package_points3d_sha256=sha256_file(camera_package_root / "points3D.txt"),
            camera_package_registered_frames_sha256=sha256_file(
                camera_package_root / "registered_frames.json"
            ),
            registered_frame_ids=camera.registered_frame_ids,
            unregistered_frame_ids=camera.unregistered_frame_ids,
            coordinate_convention=camera.coordinate_convention,
            segmentation_tracking_sha256=sha256_file(tracks_path),
            object_tracks=[
                ObjectSurfaceTrackRequest(
                    object_id=track.object_id,
                    semantic_label=track.semantic_label,
                    prompt_id=track.prompt_id,
                    asset_type_hint=track.asset_type_hint,
                    track_coverage=track.coverage_ratio,
                    mask_paths_by_frame={
                        observation.frame_id: observation.mask_path
                        for observation in track.observations
                    },
                    frame_scores={
                        observation.frame_id: observation.frame_score
                        for observation in track.observations
                    },
                )
                for track in tracks.tracks
            ],
            global_reconstruction_sha256=sha256_file(global_path),
            global_mesh_sha256=sha256_file(mesh_path),
            lifting_method=config.lifting_method,
            rasterization_configuration={
                "backend": "fake" if config.execution_mode == "fake_worker" else "nvdiffrast",
                "raster_scale": config.raster_scale,
                "face_chunk_size": config.face_chunk_size,
                "near_plane_strategy": config.near_plane_strategy,
                "far_plane_strategy": config.far_plane_strategy,
                "global_vertex_count": global_scene.mesh.vertex_count,
                "global_face_count": global_scene.mesh.face_count,
            },
            mask_processing_configuration={
                "mask_core_erosion_pixels": config.mask_core_erosion_pixels,
                "mask_boundary_width_pixels": config.mask_boundary_width_pixels,
                "mask_exclusion_dilation_pixels": config.mask_exclusion_dilation_pixels,
                "resampling": "nearest",
            },
            face_evidence_configuration={
                "core_positive_weight": config.core_positive_weight,
                "boundary_positive_weight": config.boundary_positive_weight,
                "exterior_negative_weight": config.exterior_negative_weight,
                "min_visible_pixels_per_face": config.min_visible_pixels_per_face,
                "min_positive_pixels_per_face": config.min_positive_pixels_per_face,
                "min_supporting_views": config.min_supporting_views,
                "accepted_face_score": config.accepted_face_score,
                "ambiguous_face_score": config.ambiguous_face_score,
                "instance_score_margin": config.instance_score_margin,
                "formula": "positive/(positive+negative+epsilon)",
            },
            surface_sample_configuration={
                "sample_voxel_edge_multiplier": config.sample_voxel_edge_multiplier,
                "sample_min_supporting_views": config.sample_min_supporting_views,
                "sample_min_positive_weight": config.sample_min_positive_weight,
                "sample_negative_margin_multiplier": config.sample_negative_margin_multiplier,
                "accepted_patch_score": config.accepted_face_score,
                "ambiguous_patch_score": config.ambiguous_face_score,
                "mapping": "direct_samples_to_original_global_faces",
            },
            surface_extraction_configuration={
                "min_component_faces": config.min_component_faces,
                "min_relative_component_area": config.min_relative_component_area,
                "seam_diagnostic_enabled": config.seam_diagnostic_enabled,
                "seam_centroid_distance_multiplier": config.seam_centroid_distance_multiplier,
                "seam_endpoint_distance_multiplier": config.seam_endpoint_distance_multiplier,
                "seam_normal_cosine": config.seam_normal_cosine,
                "accepted_min_reprojection_iou": config.accepted_min_reprojection_iou,
                "partial_min_reprojection_iou": config.partial_min_reprojection_iou,
                "accepted_max_ambiguity_ratio": config.accepted_max_ambiguity_ratio,
                "alignment_depth_inlier_threshold": config.alignment_depth_inlier_threshold,
                "alignment_min_inlier_fraction": config.alignment_min_inlier_fraction,
                "preserve_original_global_face_ids": True,
                "fake_mode": config.fake_mode,
            },
            seed=config.seed,
        )

    def _inference_command(
        self,
        context: StageContext,
        config: ObjectLiftingAdapterConfig,
    ) -> list[str]:
        request = Path("reconstruction/object_surfaces/request.json")
        if config.execution_mode != "docker":
            command = self._local_command(config, "infer", request)
            if isinstance(command, str):
                raise RuntimeError(command)
            command.extend(
                [
                    "--input-root",
                    str(context.run_dir.resolve()),
                    "--output-dir",
                    "reconstruction/object_surfaces",
                ]
            )
            return command
        docker = resolve_executable(config.docker_executable)
        if docker is None:
            raise RuntimeError("Docker executable was not found")
        return [
            docker,
            "run",
            "--rm",
            "--gpus",
            "all",
            *self._docker_user_arguments(),
            "-v",
            f"{context.run_dir.resolve()}:/workspace:rw",
            "-w",
            "/workspace",
            "--entrypoint",
            "python",
            config.docker_image,
            "-m",
            config.worker_module,
            "infer",
            "--request",
            "/workspace/reconstruction/object_surfaces/request.json",
            "--input-root",
            "/workspace",
            "--output-dir",
            "/workspace/reconstruction/object_surfaces",
        ]

    @staticmethod
    def _local_command(
        config: ObjectLiftingAdapterConfig,
        action: str,
        path: Path,
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
                return f"fake object-lifting worker does not exist: {script}"
            return [python, str(script.resolve()), action, option, str(path)]
        return [python, "-m", config.worker_module, action, option, str(path)]

    @staticmethod
    def _docker_user_arguments() -> list[str]:
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            return ["--user", f"{os.getuid()}:{os.getgid()}"]
        return []

    @staticmethod
    def _worker_failure(exc: ProcessExecutionError) -> RuntimeError:
        stderr = exc.result.stderr.lower()
        if "out of memory" in stderr or "cuda oom" in stderr:
            return RuntimeError(
                "object-lifting rasterization ran out of GPU memory; lower raster_scale, "
                "or face_chunk_size"
            )
        if "unsupported camera model" in stderr:
            return RuntimeError(exc.result.stderr.strip())
        if "rasterizer" in stderr or "nvdiffrast" in stderr:
            return RuntimeError(
                "object-lifting rasterizer failed; verify nvdiffrast and CUDA compatibility"
            )
        return RuntimeError(str(exc))

    @staticmethod
    def _validate_worker_output(
        context: StageContext,
        config: ObjectLiftingAdapterConfig,
        request: ObjectSurfaceLiftingRequest,
        worker: ObjectSurfaceWorkerManifest,
        evidence: ObjectSurfaceEvidenceArtifact,
        previews: ObjectSurfacePreviewManifest,
        comparison: ObjectSurfaceMethodComparison,
        alignment: CameraMeshAlignmentArtifact,
    ) -> None:
        expected = {
            "request_sha256": sha256_file(
                context.path("reconstruction", "object_surfaces", "request.json")
            ),
            "manifest_sha256": request.manifest_sha256,
            "frame_sequence_digest": request.frame_sequence_digest,
            "camera_reconstruction_sha256": request.camera_reconstruction_sha256,
            "segmentation_tracking_sha256": request.segmentation_tracking_sha256,
            "global_reconstruction_sha256": request.global_reconstruction_sha256,
            "global_mesh_sha256": request.global_mesh_sha256,
        }
        for field, value in expected.items():
            if getattr(worker, field) != value:
                raise RuntimeError(f"object-lifting worker {field} does not match its request")
            if field != "request_sha256" and hasattr(evidence, field):
                if getattr(evidence, field) != value:
                    raise RuntimeError(f"object-surface evidence {field} is inconsistent")
        if evidence.coordinate_convention != request.coordinate_convention:
            raise RuntimeError("object-surface evidence changed coordinate semantics")
        if not coordinate_metadata_is_raw_colmap(evidence.coordinate_convention):
            raise RuntimeError("object-surface output does not retain raw COLMAP semantics")
        if evidence.partition.global_face_count != worker.global_face_count:
            raise RuntimeError("global face counts disagree across object-lifting outputs")
        if comparison.selected_method != request.lifting_method:
            raise RuntimeError("method comparison selected method does not match request")
        expected_method_metrics = {
            (track.object_id, method)
            for track in request.object_tracks
            for method in ("exact_face_vote_v1", "surface_sample_fusion_v2")
        }
        actual_method_metrics = {(item.object_id, item.method) for item in comparison.metrics}
        if actual_method_metrics != expected_method_metrics:
            raise RuntimeError(
                "method comparison must contain exact-face and surface-sample metrics "
                "for every object"
            )
        if alignment.frame_sequence_digest != request.frame_sequence_digest:
            raise RuntimeError("camera/mesh alignment frame lineage does not match request")
        if alignment.camera_reconstruction_sha256 != request.camera_reconstruction_sha256:
            raise RuntimeError("camera/mesh alignment references the wrong cameras")
        if alignment.global_mesh_sha256 != request.global_mesh_sha256:
            raise RuntimeError("camera/mesh alignment references the wrong mesh")
        object_ids = {track.object_id for track in request.object_tracks}
        if {hypothesis.object_id for hypothesis in evidence.hypotheses} != object_ids:
            raise RuntimeError("worker hypotheses do not exactly cover canonical SAM tracks")
        for hypothesis in evidence.hypotheses:
            if hypothesis.global_mesh_sha256 != request.global_mesh_sha256:
                raise RuntimeError(f"object {hypothesis.object_id!r} references the wrong mesh")
            validate_surface_mesh(context.run_dir, hypothesis)
            if any(
                frame_id not in request.registered_frame_ids
                for frame_id in hypothesis.supporting_registered_frame_ids
            ):
                raise RuntimeError(f"object {hypothesis.object_id!r} uses unregistered 3D evidence")
        preview_paths = [
            previews.global_face_assignment_path,
            previews.object_surface_contact_sheet_path,
            previews.reprojection_contact_sheet_path,
            previews.conflict_heatmap_path,
            previews.global_mesh_depth_contact_sheet_path,
            previews.global_mesh_edge_overlay_path,
            previews.sparse_point_vs_mesh_depth_path,
            previews.surface_sample_fusion_path,
            *previews.object_preview_paths.values(),
        ]
        for relative_path in preview_paths:
            if not context.path(*Path(relative_path).parts).is_file():
                raise RuntimeError(f"object-lifting preview is missing: {relative_path}")
        root = context.path("reconstruction", "object_surfaces")
        for relative_path in worker.raw_output_paths:
            path = context.path(*Path(relative_path).parts)
            try:
                path.resolve().relative_to(root.resolve())
            except ValueError as exc:
                raise RuntimeError(
                    f"object-lifting output escapes the attempt: {relative_path}"
                ) from exc
            if not path.is_file():
                raise RuntimeError(f"worker declared a missing output: {relative_path}")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise RuntimeError(f"object-lifting output must not contain symlinks: {path}")
        if config.execution_mode != "fake_worker" and worker.backend != "nvdiffrast":
            raise RuntimeError("real object lifting did not report nvdiffrast execution")

    @staticmethod
    def _integrate_scene_ir(
        scene: SceneIR,
        evidence: ObjectSurfaceEvidenceArtifact,
    ) -> SceneIR:
        phase4_asset_ids = {
            f"partial_surface_{hypothesis.object_id}" for hypothesis in evidence.hypotheses
        }
        scene.geometry_assets = [
            asset for asset in scene.geometry_assets if asset.asset_id not in phase4_asset_ids
        ]
        existing = {obj.object_id: obj for obj in scene.objects}
        for hypothesis in evidence.hypotheses:
            if hypothesis.status == "unresolved" or hypothesis.surface_mesh_path is None:
                continue
            asset_id = f"partial_surface_{hypothesis.object_id}"
            # A segmentation hint is not a measured physical classification.
            asset_type = AssetType.UNCLASSIFIED
            scene.geometry_assets.append(
                GeometryAsset(
                    asset_id=asset_id,
                    asset_type=asset_type,
                    uri=hypothesis.surface_mesh_path,
                    format="ply",
                    source=GeometrySourceType.FUSED,
                    coordinate_convention=hypothesis.coordinate_convention,
                    scale_status=ScaleStatus.SCALE_AMBIGUOUS,
                    geometry_status="partial_observation_supported",
                    completion_status="not_completed",
                    sim_ready=False,
                    provenance=hypothesis.provenance,
                )
            )
            object_instance = ObjectInstance(
                object_id=hypothesis.object_id,
                name=hypothesis.semantic_label,
                asset_type=asset_type,
                geometry_asset_ids=[asset_id],
                geometry_status="partial_observation_supported",
                completion_status="not_completed",
                sim_ready=False,
                provenance=[hypothesis.provenance],
                confidence=hypothesis.confidence,
            )
            existing[hypothesis.object_id] = object_instance
        scene.objects = sorted(existing.values(), key=lambda item: item.object_id)
        scene.schema_version = "0.1.2"
        return SceneIR.model_validate(scene.model_dump(mode="json"))

    @staticmethod
    def _dynamic_artifact_type(path: Path) -> str:
        if path.name.endswith("_face_ids.bin"):
            return "compact_global_face_ids"
        if path.name == "surface_mesh.ply":
            return "partial_object_surface_mesh"
        if path.name == "surface_points.ply":
            return "partial_object_surface_points"
        if path.suffix.lower() == ".npz":
            return "object_face_evidence"
        if path.suffix.lower() == ".png":
            return "object_surface_preview"
        return "object_lifting_raw_output"

    @staticmethod
    def _media_type(path: Path) -> str:
        return {
            ".json": "application/json",
            ".png": "image/png",
            ".ply": "model/ply",
            ".glb": "model/gltf-binary",
            ".npz": "application/zip",
            ".bin": "application/octet-stream",
        }.get(path.suffix.lower(), "application/octet-stream")


class Phase4ConsistencyValidationAdapter:
    name = "phase4_consistency_validation"
    version = "0.1.0"

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return HealthcheckResult(True, "Phase 4 consistency validator available")

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        evidence_path = context.canonical_path(
            "reconstruction", "object_surfaces", "evidence_manifest.json"
        )
        evidence = ObjectSurfaceEvidenceArtifact.model_validate_json(
            evidence_path.read_text(encoding="utf-8")
        )
        specs = [
            InputSpec("inputs/manifest.json", "ingest_manifest"),
            InputSpec("camera/reconstruction.json", "camera_reconstruction"),
            InputSpec("observations/object_tracks.json", "segmentation_tracking"),
            InputSpec(
                "reconstruction/global/metadata.json",
                "global_scene_reconstruction",
            ),
            InputSpec(
                "reconstruction/global/mesh.ply",
                "global_scene_mesh",
                materialization_mode="reference_only",
            ),
            InputSpec(
                "reconstruction/object_surfaces/evidence_manifest.json",
                "object_surface_evidence",
            ),
            InputSpec("scene_ir/phase4_scene.json", "scene_ir"),
        ]
        for hypothesis in evidence.hypotheses:
            specs.extend(
                [
                    InputSpec(
                        hypothesis.accepted_global_face_ids.relative_path,
                        "compact_global_face_ids",
                    ),
                    InputSpec(
                        hypothesis.ambiguous_global_face_ids.relative_path,
                        "compact_global_face_ids",
                    ),
                    InputSpec(hypothesis.face_evidence_path, "object_face_evidence"),
                ]
            )
            if hypothesis.surface_mesh_path:
                specs.append(InputSpec(hypothesis.surface_mesh_path, "partial_object_surface_mesh"))
        return specs

    def prepare(self, context: StageContext) -> None:
        context.path("validation").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                "validation/phase4_object_surface_consistency.json",
                "phase4_consistency_report",
                "application/json",
                "object_lifting",
                validation="json",
                schema_identifier="recon2sim/phase4-consistency/0.1.0",
                model=Phase4ConsistencyReport,
            )
        ]

    def run(self, context: StageContext) -> StageResult:
        manifest_path = context.path("inputs", "manifest.json")
        camera_path = context.path("camera", "reconstruction.json")
        tracks_path = context.path("observations", "object_tracks.json")
        global_path = context.path("reconstruction", "global", "metadata.json")
        evidence_path = context.path("reconstruction", "object_surfaces", "evidence_manifest.json")
        manifest = IngestManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        camera = CameraReconstruction.model_validate_json(camera_path.read_text(encoding="utf-8"))
        tracks = SegmentationTrackingArtifact.model_validate_json(
            tracks_path.read_text(encoding="utf-8")
        )
        global_scene = GlobalSceneReconstructionArtifact.model_validate_json(
            global_path.read_text(encoding="utf-8")
        )
        evidence = ObjectSurfaceEvidenceArtifact.model_validate_json(
            evidence_path.read_text(encoding="utf-8")
        )
        scene = SceneIR.model_validate_json(
            context.path("scene_ir", "phase4_scene.json").read_text(encoding="utf-8")
        )
        checks: list[Phase4ConsistencyCheck] = []

        def check(check_id: str, passed: bool, message: str) -> None:
            checks.append(Phase4ConsistencyCheck(check_id=check_id, passed=passed, message=message))

        digest = manifest.frame_sequence_digest
        check(
            "manifest_sha",
            evidence.manifest_sha256 == sha256_file(manifest_path),
            "manifest hash",
        )
        check(
            "frame_sequence_digest",
            digest is not None
            and evidence.frame_sequence_digest
            == camera.frame_sequence_digest
            == tracks.frame_sequence_digest
            == global_scene.frame_sequence_digest
            == digest,
            "Phase 1-4 frame-sequence digest",
        )
        check(
            "camera_sha",
            evidence.camera_reconstruction_sha256 == sha256_file(camera_path),
            "camera reconstruction hash",
        )
        check(
            "segmentation_sha",
            evidence.segmentation_tracking_sha256 == sha256_file(tracks_path),
            "segmentation artifact hash",
        )
        check(
            "global_reconstruction_sha",
            evidence.global_reconstruction_sha256 == sha256_file(global_path),
            "global reconstruction hash",
        )
        mesh_path = context.canonical_path("reconstruction", "global", "mesh.ply")
        check(
            "global_mesh_sha",
            evidence.global_mesh_sha256 == sha256_file(mesh_path),
            "global mesh hash",
        )
        known_frames = {frame.frame_id for frame in manifest.frames}
        supporting = {
            frame_id
            for hypothesis in evidence.hypotheses
            for frame_id in hypothesis.supporting_frame_ids
        }
        registered_support = {
            frame_id
            for hypothesis in evidence.hypotheses
            for frame_id in hypothesis.supporting_registered_frame_ids
        }
        check("support_frames_exist", supporting <= known_frames, "all supporting frames exist")
        check(
            "three_dimensional_support_registered",
            registered_support <= set(camera.registered_frame_ids),
            "all 3D-support frames are registered",
        )
        face_arrays_valid = True
        surfaces_valid = True
        try:
            for hypothesis in evidence.hypotheses:
                read_compact_face_ids(
                    context.run_dir,
                    hypothesis.accepted_global_face_ids,
                    global_face_count=evidence.partition.global_face_count,
                )
                read_compact_face_ids(
                    context.run_dir,
                    hypothesis.ambiguous_global_face_ids,
                    global_face_count=evidence.partition.global_face_count,
                )
                validate_surface_mesh(context.run_dir, hypothesis)
        except (OSError, ValueError):
            face_arrays_valid = False
            surfaces_valid = False
        check("face_index_manifests", face_arrays_valid, "face-index data and bounds")
        check("surface_face_mapping", surfaces_valid, "surface meshes match original face IDs")
        raw_coordinates = coordinate_metadata_is_raw_colmap(evidence.coordinate_convention)
        check("coordinate_semantics", raw_coordinates, "raw COLMAP arbitrary coordinates retained")
        check(
            "no_metric_or_gravity_claim",
            not evidence.metric_scale_known
            and not evidence.canonical_gravity_alignment_known
            and evidence.scale_status is ScaleStatus.SCALE_AMBIGUOUS,
            "no metric scale or gravity alignment is claimed",
        )
        check(
            "no_collision_assets",
            not scene.collision_assets
            and all(not obj.collision_asset_ids for obj in scene.objects),
            "Phase 4 creates no collision assets",
        )
        check(
            "no_completion_claim",
            evidence.hidden_surface_completion == "not_implemented",
            "hidden surface completion is not implemented",
        )
        forbidden_clean = self._lifting_materialization_is_selective(context.canonical_run_dir)
        check(
            "selective_materialization",
            forbidden_clean,
            "lifting attempt excludes raw COLMAP/SAM/GenRecon model workspaces",
        )
        scene_assets = {asset.uri for asset in scene.geometry_assets}
        expected_assets = {
            hypothesis.surface_mesh_path
            for hypothesis in evidence.hypotheses
            if hypothesis.surface_mesh_path is not None and hypothesis.status != "unresolved"
        }
        check(
            "scene_ir_partial_surfaces",
            expected_assets <= scene_assets
            and all(
                asset.source is GeometrySourceType.FUSED
                and asset.geometry_status == "partial_observation_supported"
                and asset.sim_ready is False
                for asset in scene.geometry_assets
                if asset.uri in expected_assets
            ),
            "Scene IR references fused partial surface assets honestly",
        )
        report = Phase4ConsistencyReport(
            passed=all(item.passed for item in checks),
            checks=checks,
            manifest_sha256=sha256_file(manifest_path),
            frame_sequence_digest=cast(str, digest),
            real_2d_tracks_lifted_to_global_3d=any(
                hypothesis.accepted_global_face_ids.count > 0 for hypothesis in evidence.hypotheses
            ),
            warnings=evidence.warnings,
        )
        atomic_write_json(
            context.path("validation", "phase4_object_surface_consistency.json"),
            report,
        )
        if not report.passed:
            failed = [item.check_id for item in report.checks if not item.passed]
            raise RuntimeError(f"Phase 4 consistency validation failed: {failed}")
        return StageResult(
            metrics={
                "checks": len(checks),
                "lifted_objects": sum(
                    hypothesis.accepted_global_face_ids.count > 0
                    for hypothesis in evidence.hypotheses
                ),
            }
        )

    @staticmethod
    def _lifting_materialization_is_selective(run_dir: Path) -> bool:
        path = run_dir / "manifest.json"
        if not path.is_file():
            return False
        payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        stages = cast(dict[str, Any], payload.get("stages", {}))
        stage = cast(dict[str, Any], stages.get("object_surface_lifting", {}))
        attempts = cast(list[dict[str, Any]], stage.get("attempts", []))
        if not attempts:
            return False
        materialized = {
            str(item.get("relative_path", ""))
            for item in cast(list[dict[str, Any]], attempts[-1].get("materialized_inputs", []))
        }
        forbidden = (
            "camera/colmap/",
            "observations/raw/",
            "reconstruction/global/raw/",
            "reconstruction/global/checkpoint",
        )
        return not any(path.startswith(prefix) for path in materialized for prefix in forbidden)
