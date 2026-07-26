from __future__ import annotations

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
from recon2sim.alignment import validate_similarity_transform
from recon2sim.artifacts import (
    AlignmentCandidateManifest,
    AlignmentDatasetSplit,
    AlignmentIterationManifest,
    CameraMeshAlignmentDiagnostics,
    CameraMeshAlignmentPreviewManifest,
    CameraMeshAlignmentRequest,
    CameraMeshAlignmentResult,
    CameraMeshAlignmentWorkerManifest,
    CameraReconstruction,
    GenReconCameraPackageManifest,
    GlobalSceneReconstructionArtifact,
    IngestManifest,
    ObjectLiftingAlignmentComparison,
    ObjectSurfaceEvidenceArtifact,
    ObjectSurfaceLiftingRequest,
    Phase4_2ConsistencyCheck,
    Phase4_2ConsistencyReport,
    SparseDepthObservationManifest,
    TransformChainAudit,
)
from recon2sim.genrecon import sha256_file
from recon2sim.ir import GeometryAsset, GeometrySourceType, ScaleStatus, SceneIR, StrictModel
from recon2sim.object_lifting import (
    coordinate_metadata_is_raw_colmap,
    read_compact_face_ids,
)
from recon2sim.storage import atomic_write_json

ALIGNMENT_WORKER_VERSION = "0.1.0"


def _resolve_python(value: str) -> str | None:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        return str(candidate.absolute()) if candidate.is_file() else None
    return shutil.which(value)


class CameraMeshAlignmentAdapterConfig(StrictModel):
    execution_mode: Literal["local_worker", "docker", "fake_worker"]
    worker_python: str = "python"
    worker_module: str = "alignment_worker"
    worker_script: str | None = None
    docker_executable: str = "docker"
    docker_image: str = "reconevery/alignment:phase4.2"
    device: Literal["cuda", "cpu"] = "cuda"
    raster_scale: float = Field(default=0.25, gt=0, le=1)
    face_chunk_size: int = Field(default=1_000_000, gt=0)
    include_working_mesh_audit: bool = True
    include_working_scene_audit: bool = False
    roundtrip_tolerance: float = Field(default=1e-6, gt=0)
    max_colmap_reprojection_error: float = Field(default=2.0, ge=0)
    min_track_length: int = Field(default=3, gt=0)
    minimum_camera_depth: float = Field(default=1e-8, gt=0)
    require_inside_undistorted_image: bool = True
    maximum_sample_vertices: int = Field(default=200_000, gt=0)
    maximum_sample_face_centroids: int = Field(default=100_000, gt=0)
    maximum_iterations: int = Field(default=8, gt=0)
    minimum_correspondences: int = Field(default=50, ge=3)
    correspondence_mad_multiplier: float = Field(default=2.5, gt=0)
    cauchy_scale: float = Field(default=0.05, gt=0)
    convergence_tolerance: float = Field(default=1e-6, gt=0)
    min_scale: float = Field(default=0.25, gt=0)
    max_scale: float = Field(default=4.0, gt=0)
    max_rotation_degrees_from_identity: float = Field(default=60.0, ge=0, le=180)
    max_translation_scene_diagonals: float = Field(default=1.0, gt=0)
    minimum_validation_observations: int = Field(default=100, gt=0)
    minimum_median_residual_relative_improvement: float = Field(default=0.20, ge=0, le=1)
    minimum_p90_residual_relative_improvement: float = Field(default=0.10, ge=0, le=1)
    minimum_inlier_fraction_absolute_improvement: float = Field(default=0.05, ge=0, le=1)
    minimum_mesh_coverage_ratio_vs_baseline: float = Field(default=0.90, ge=0, le=1)
    maximum_bad_frame_fraction: float = Field(default=0.30, ge=0, le=1)
    bad_frame_threshold: float = Field(default=0.50, gt=0)
    identity_median_residual_threshold: float = Field(default=0.10, gt=0)
    identity_inlier_fraction_threshold: float = Field(default=0.30, ge=0, le=1)
    sufficient_inlier_fraction_threshold: float = Field(default=0.30, ge=0, le=1)
    fake_mode: str = "success_full_sim3"
    seed: int = 42

    @model_validator(mode="after")
    def valid_execution(self) -> CameraMeshAlignmentAdapterConfig:
        if self.min_scale >= self.max_scale:
            raise ValueError("alignment min_scale must be below max_scale")
        if self.execution_mode == "fake_worker":
            if self.worker_script is None:
                raise ValueError("fake alignment worker requires worker_script")
            if self.device != "cpu":
                raise ValueError("fake alignment worker must use device=cpu")
            return self
        if self.device != "cuda":
            raise ValueError("real camera/mesh alignment requires device=cuda")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None or visible.strip().lower() in {"", "-1", "none", "void"}:
            raise ValueError("real camera/mesh alignment requires CUDA_VISIBLE_DEVICES")
        if self.execution_mode == "local_worker":
            python = _resolve_python(self.worker_python)
            if python is None:
                raise ValueError(
                    f"configured alignment worker Python {self.worker_python!r} missing"
                )
            # Inspect the configured environment path, not the interpreter symlink target.
            root = Path(python).absolute().parent.parent
            if not (root / "pyvenv.cfg").is_file() and not (root / "conda-meta").is_dir():
                raise ValueError("alignment worker_python must be in an isolated environment")
            if root == Path(sys.prefix).resolve():
                raise ValueError("alignment worker must not use the core environment")
        return self


class CameraMeshAlignmentAdapter:
    name = "camera_mesh_alignment"
    version = "0.1.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        config = CameraMeshAlignmentAdapterConfig.model_validate(context.config.adapter.config)
        specs = [
            InputSpec("inputs/manifest.json", "ingest_manifest"),
            InputSpec("camera/reconstruction.json", "camera_reconstruction"),
            InputSpec(
                "camera/genrecon_package/package_manifest.json",
                "genrecon_camera_package_manifest",
            ),
            InputSpec("camera/genrecon_package/cameras.txt", "genrecon_colmap_text"),
            InputSpec("camera/genrecon_package/images.txt", "genrecon_colmap_text"),
            InputSpec("camera/genrecon_package/points3D.txt", "genrecon_colmap_text"),
            InputSpec(
                "reconstruction/global/metadata.json",
                "global_scene_reconstruction",
            ),
            InputSpec(
                "reconstruction/global/mesh.ply",
                "global_scene_mesh",
                materialization_mode="reflink_or_copy",
            ),
            InputSpec(
                "reconstruction/global/worker_manifest.json",
                "genrecon_worker_manifest",
            ),
            InputSpec(
                "reconstruction/global/raw/working_transform.json",
                "genrecon_raw_output",
            ),
            InputSpec(
                "reconstruction/global/raw/chunk_transforms.json",
                "genrecon_raw_output",
            ),
            InputSpec(
                "reconstruction/global/raw/cameras.json",
                "genrecon_raw_output",
            ),
            InputSpec("scene_ir/scene.json", "scene_ir"),
        ]
        if config.include_working_mesh_audit:
            specs.append(
                InputSpec(
                    "reconstruction/global/raw/mesh_working.ply",
                    "genrecon_raw_output",
                    required=False,
                    materialization_mode="reflink_or_copy",
                )
            )
        if config.include_working_scene_audit:
            specs.append(
                InputSpec(
                    "reconstruction/global/raw/scene_working.glb",
                    "genrecon_raw_output",
                    required=False,
                    materialization_mode="reflink_or_copy",
                )
            )
        return specs

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        if context is None:
            return HealthcheckResult(False, "alignment healthcheck requires --config")
        try:
            config = CameraMeshAlignmentAdapterConfig.model_validate(context.config.adapter.config)
        except ValueError as exc:
            return HealthcheckResult(False, f"invalid alignment configuration: {exc}")
        payload = {
            "worker_version": ALIGNMENT_WORKER_VERSION,
            "device": config.device,
            "backend": ("fake" if config.execution_mode == "fake_worker" else "nvdiffrast_scipy"),
        }
        with tempfile.TemporaryDirectory(prefix="reconevery-alignment-health-") as temporary:
            config_path = Path(temporary) / "worker_config.json"
            atomic_write_json(config_path, payload)
            if config.execution_mode == "docker":
                return self._docker_healthcheck(context, config, config_path)
            command = self._local_command(config, "healthcheck", config_path)
            if isinstance(command, str):
                return HealthcheckResult(False, command)
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=min(context.config.adapter.timeout_s, 180),
                    check=False,
                    env=allowed_environment(context),
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return HealthcheckResult(False, f"alignment healthcheck failed: {exc}")
            output = result.stdout.strip() or result.stderr.strip()
            return HealthcheckResult(
                result.returncode == 0,
                output or f"alignment healthcheck exited {result.returncode}",
            )

    def _docker_healthcheck(
        self,
        context: StageContext,
        config: CameraMeshAlignmentAdapterConfig,
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
                command, capture_output=True, text=True, timeout=30, check=False
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
            "python3.10",
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
            timeout=min(context.config.adapter.timeout_s, 300),
            check=False,
        )
        if result.returncode != 0:
            return HealthcheckResult(
                False,
                result.stderr.strip() or "in-container alignment healthcheck failed",
            )
        return HealthcheckResult(True, result.stdout.strip())

    def prepare(self, context: StageContext) -> None:
        context.path("reconstruction", "alignment", "raw", "logs").mkdir(
            parents=True, exist_ok=True
        )
        context.path("reconstruction", "alignment", "previews").mkdir(parents=True, exist_ok=True)
        context.path("scene_ir").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        root = "reconstruction/alignment"
        outputs = [
            OutputSpec(
                f"{root}/request.json",
                "camera_mesh_alignment_request",
                "application/json",
                "alignment",
                validation="json",
                schema_identifier="recon2sim/camera-mesh-alignment-request/0.1.0",
                model=CameraMeshAlignmentRequest,
            ),
            OutputSpec(
                f"{root}/transform_chain_audit.json",
                "transform_chain_audit",
                "application/json",
                "alignment",
                validation="json",
                schema_identifier="recon2sim/transform-chain-audit/0.1.0",
                model=TransformChainAudit,
            ),
            OutputSpec(
                f"{root}/sparse_observations.json",
                "sparse_depth_observations",
                "application/json",
                "alignment",
                validation="json",
                schema_identifier="recon2sim/sparse-depth-observations/0.1.0",
                model=SparseDepthObservationManifest,
            ),
            OutputSpec(
                f"{root}/dataset_split.json",
                "alignment_dataset_split",
                "application/json",
                "alignment",
                validation="json",
                schema_identifier="recon2sim/alignment-dataset-split/0.1.0",
                model=AlignmentDatasetSplit,
            ),
            OutputSpec(
                f"{root}/worker_manifest.json",
                "camera_mesh_alignment_worker_manifest",
                "application/json",
                "alignment",
                validation="json",
                schema_identifier="recon2sim/camera-mesh-alignment-worker/0.1.0",
                model=CameraMeshAlignmentWorkerManifest,
            ),
            OutputSpec(
                f"{root}/alignment.json",
                "camera_mesh_alignment_result",
                "application/json",
                "alignment",
                validation="json",
                schema_identifier="recon2sim/camera-mesh-alignment-result/0.1.0",
                model=CameraMeshAlignmentResult,
            ),
            OutputSpec(
                f"{root}/diagnostics.json",
                "camera_mesh_alignment_diagnostics",
                "application/json",
                "alignment",
                validation="json",
                schema_identifier="recon2sim/camera-mesh-alignment-diagnostics/0.1.0",
                model=CameraMeshAlignmentDiagnostics,
            ),
            OutputSpec(
                f"{root}/candidates.json",
                "camera_mesh_alignment_candidates",
                "application/json",
                "alignment",
                validation="json",
                model=AlignmentCandidateManifest,
            ),
            OutputSpec(
                f"{root}/iterations.json",
                "camera_mesh_alignment_iterations",
                "application/json",
                "alignment",
                validation="json",
                model=AlignmentIterationManifest,
            ),
            OutputSpec(
                f"{root}/preview_manifest.json",
                "camera_mesh_alignment_preview_manifest",
                "application/json",
                "alignment",
                validation="json",
                model=CameraMeshAlignmentPreviewManifest,
            ),
            OutputSpec(
                "scene_ir/alignment_scene.json",
                "scene_ir",
                "application/json",
                "alignment",
                validation="scene_ir",
                schema_identifier="recon2sim/scene-ir/0.1.3",
                model=SceneIR,
            ),
        ]
        outputs.extend(
            OutputSpec(
                f"{root}/previews/{name}.png",
                "camera_mesh_alignment_preview",
                "image/png",
                "alignment",
                validation="png",
            )
            for name in (
                "transform_chain_comparison",
                "baseline_depth_residual",
                "aligned_depth_residual",
                "baseline_vs_aligned_scatter",
                "per_camera_residuals",
                "per_chunk_residuals",
                "sparse_points_and_mesh_before",
                "sparse_points_and_mesh_after",
                "heldout_validation_summary",
            )
        )
        return outputs

    def run(self, context: StageContext) -> StageResult:
        config = CameraMeshAlignmentAdapterConfig.model_validate(context.config.adapter.config)
        manifest_path = context.path("inputs", "manifest.json")
        camera_path = context.path("camera", "reconstruction.json")
        package_root = context.path("camera", "genrecon_package")
        package_path = package_root / "package_manifest.json"
        global_root = context.path("reconstruction", "global")
        global_path = global_root / "metadata.json"
        mesh_path = global_root / "mesh.ply"
        worker_path = global_root / "worker_manifest.json"
        working_transform_path = global_root / "raw" / "working_transform.json"
        chunk_transforms_path = global_root / "raw" / "chunk_transforms.json"
        camera_debug_path = global_root / "raw" / "cameras.json"
        manifest = IngestManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        camera = CameraReconstruction.model_validate_json(camera_path.read_text(encoding="utf-8"))
        package = GenReconCameraPackageManifest.model_validate_json(
            package_path.read_text(encoding="utf-8")
        )
        global_scene = GlobalSceneReconstructionArtifact.model_validate_json(
            global_path.read_text(encoding="utf-8")
        )
        self._validate_lineage(manifest, camera, package, global_scene, camera_path)
        working_mesh = global_root / "raw" / "mesh_working.ply"
        working_scene = global_root / "raw" / "scene_working.glb"
        request = CameraMeshAlignmentRequest(
            run_id=context.canonical_run_dir.name,
            manifest_sha256=sha256_file(manifest_path),
            frame_sequence_digest=cast(str, manifest.frame_sequence_digest),
            camera_reconstruction_sha256=sha256_file(camera_path),
            registered_frame_ids=camera.registered_frame_ids,
            unregistered_frame_ids=camera.unregistered_frame_ids,
            coordinate_convention=camera.coordinate_convention,
            camera_package_sha256=sha256_file(package_path),
            cameras_txt_sha256=sha256_file(package_root / "cameras.txt"),
            images_txt_sha256=sha256_file(package_root / "images.txt"),
            points3d_txt_sha256=sha256_file(package_root / "points3D.txt"),
            global_reconstruction_sha256=sha256_file(global_path),
            global_mesh_sha256=sha256_file(mesh_path),
            global_worker_manifest_sha256=sha256_file(worker_path),
            working_transform_path=working_transform_path.relative_to(context.run_dir).as_posix(),
            working_transform_sha256=sha256_file(working_transform_path),
            chunk_transforms_path=chunk_transforms_path.relative_to(context.run_dir).as_posix(),
            chunk_transforms_sha256=sha256_file(chunk_transforms_path),
            genrecon_camera_debug_path=camera_debug_path.relative_to(context.run_dir).as_posix(),
            genrecon_camera_debug_sha256=sha256_file(camera_debug_path),
            working_mesh_path=(
                working_mesh.relative_to(context.run_dir).as_posix()
                if working_mesh.is_file()
                else None
            ),
            working_mesh_sha256=sha256_file(working_mesh) if working_mesh.is_file() else None,
            working_scene_path=(
                working_scene.relative_to(context.run_dir).as_posix()
                if working_scene.is_file()
                else None
            ),
            working_scene_sha256=sha256_file(working_scene) if working_scene.is_file() else None,
            audit_configuration={
                "roundtrip_tolerance": config.roundtrip_tolerance,
                "require_pre_post_equivalence": True,
            },
            sparse_observation_configuration={
                "max_colmap_reprojection_error": config.max_colmap_reprojection_error,
                "min_track_length": config.min_track_length,
                "minimum_camera_depth": config.minimum_camera_depth,
                "require_inside_undistorted_image": config.require_inside_undistorted_image,
                "raster_scale": config.raster_scale,
                "undistortion_policy": "opencv_optimal_new_camera_matrix_alpha_0_full_image",
            },
            mesh_sampling_configuration={
                "maximum_sample_vertices": config.maximum_sample_vertices,
                "maximum_sample_face_centroids": config.maximum_sample_face_centroids,
                "face_chunk_size": config.face_chunk_size,
            },
            optimization_configuration={
                "maximum_iterations": config.maximum_iterations,
                "minimum_correspondences": config.minimum_correspondences,
                "correspondence_mad_multiplier": config.correspondence_mad_multiplier,
                "cauchy_scale": config.cauchy_scale,
                "convergence_tolerance": config.convergence_tolerance,
                "min_scale": config.min_scale,
                "max_scale": config.max_scale,
                "max_rotation_degrees_from_identity": (config.max_rotation_degrees_from_identity),
                "max_translation_scene_diagonals": (config.max_translation_scene_diagonals),
                "fake_mode": config.fake_mode,
            },
            acceptance_configuration={
                "minimum_validation_observations": config.minimum_validation_observations,
                "minimum_median_residual_relative_improvement": (
                    config.minimum_median_residual_relative_improvement
                ),
                "minimum_p90_residual_relative_improvement": (
                    config.minimum_p90_residual_relative_improvement
                ),
                "minimum_inlier_fraction_absolute_improvement": (
                    config.minimum_inlier_fraction_absolute_improvement
                ),
                "minimum_mesh_coverage_ratio_vs_baseline": (
                    config.minimum_mesh_coverage_ratio_vs_baseline
                ),
                "maximum_bad_frame_fraction": config.maximum_bad_frame_fraction,
                "bad_frame_threshold": config.bad_frame_threshold,
                "identity_median_residual_threshold": (config.identity_median_residual_threshold),
                "identity_inlier_fraction_threshold": (config.identity_inlier_fraction_threshold),
                "sufficient_inlier_fraction_threshold": (
                    config.sufficient_inlier_fraction_threshold
                ),
            },
            seed=config.seed,
        )
        request_path = context.path("reconstruction", "alignment", "request.json")
        atomic_write_json(request_path, request)
        command = self._inference_command(context, config)
        try:
            run_process(
                command,
                context=context,
                name="alignment_worker",
                log_directory="reconstruction/alignment/raw/logs",
            )
        except ProcessExecutionError as exc:
            raise self._worker_failure(exc) from exc
        root = context.path("reconstruction", "alignment")
        audit = self._load(root / "transform_chain_audit.json", TransformChainAudit)
        sparse = self._load(root / "sparse_observations.json", SparseDepthObservationManifest)
        split = self._load(root / "dataset_split.json", AlignmentDatasetSplit)
        worker = self._load(root / "worker_manifest.json", CameraMeshAlignmentWorkerManifest)
        alignment = self._load(root / "alignment.json", CameraMeshAlignmentResult)
        diagnostics = self._load(root / "diagnostics.json", CameraMeshAlignmentDiagnostics)
        candidates = self._load(root / "candidates.json", AlignmentCandidateManifest)
        self._load(root / "iterations.json", AlignmentIterationManifest)
        previews = self._load(root / "preview_manifest.json", CameraMeshAlignmentPreviewManifest)
        self._validate_outputs(
            context,
            config,
            request,
            audit,
            sparse,
            split,
            worker,
            alignment,
            diagnostics,
            candidates,
            previews,
        )
        source_scene = SceneIR.model_validate_json(
            context.path("scene_ir", "scene.json").read_text(encoding="utf-8")
        )
        atomic_write_json(
            context.path("scene_ir", "alignment_scene.json"),
            self._alignment_scene(source_scene, alignment),
        )
        return StageResult(
            metrics={
                "accepted": alignment.accepted,
                "status": alignment.status,
                "training_observations": split.training_observation_count,
                "validation_observations": split.validation_observation_count,
                "scale": alignment.transform.scale,
                "rotation_degrees": alignment.transform.rotation_degrees,
                "bad_cameras": len(diagnostics.camera_outlier_frame_ids),
            }
        )

    @staticmethod
    def _validate_lineage(
        manifest: IngestManifest,
        camera: CameraReconstruction,
        package: GenReconCameraPackageManifest,
        global_scene: GlobalSceneReconstructionArtifact,
        camera_path: Path,
    ) -> None:
        digest = manifest.frame_sequence_digest
        if digest is None:
            raise ValueError("alignment requires a frame-sequence digest")
        if not (
            camera.frame_sequence_digest
            == package.frame_sequence_digest
            == global_scene.frame_sequence_digest
            == digest
        ):
            raise ValueError("camera/mesh alignment upstream frame lineage disagrees")
        if package.camera_reconstruction_sha256 != sha256_file(camera_path):
            raise ValueError("camera package references a different camera reconstruction")
        if global_scene.camera_reconstruction_sha256 != sha256_file(camera_path):
            raise ValueError("global mesh references a different camera reconstruction")
        if package.registered_frame_ids != camera.registered_frame_ids:
            raise ValueError("camera package registration order disagrees with typed cameras")
        if not coordinate_metadata_is_raw_colmap(camera.coordinate_convention):
            raise ValueError("alignment requires raw arbitrary COLMAP coordinate semantics")
        if global_scene.coordinate_convention != camera.coordinate_convention:
            raise ValueError("global mesh coordinate metadata disagrees with cameras")

    @staticmethod
    def _load(path: Path, model: Any) -> Any:
        if not path.is_file():
            raise RuntimeError(f"alignment worker omitted required output {path.name}")
        try:
            return model.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise RuntimeError(f"alignment worker output {path.name} is malformed: {exc}") from exc

    @staticmethod
    def _validate_outputs(
        context: StageContext,
        config: CameraMeshAlignmentAdapterConfig,
        request: CameraMeshAlignmentRequest,
        audit: TransformChainAudit,
        sparse: SparseDepthObservationManifest,
        split: AlignmentDatasetSplit,
        worker: CameraMeshAlignmentWorkerManifest,
        alignment: CameraMeshAlignmentResult,
        diagnostics: CameraMeshAlignmentDiagnostics,
        candidates: AlignmentCandidateManifest,
        previews: CameraMeshAlignmentPreviewManifest,
    ) -> None:
        expected = {
            "request_sha256": sha256_file(
                context.path("reconstruction", "alignment", "request.json")
            ),
            "manifest_sha256": request.manifest_sha256,
            "frame_sequence_digest": request.frame_sequence_digest,
            "camera_reconstruction_sha256": request.camera_reconstruction_sha256,
            "camera_package_sha256": request.camera_package_sha256,
            "global_reconstruction_sha256": request.global_reconstruction_sha256,
            "global_mesh_sha256": request.global_mesh_sha256,
        }
        for field, value in expected.items():
            if getattr(worker, field) != value:
                raise RuntimeError(f"alignment worker {field} does not match request")
        if alignment.coordinate_convention != request.coordinate_convention:
            raise RuntimeError("alignment output changed coordinate semantics")
        if not coordinate_metadata_is_raw_colmap(alignment.coordinate_convention):
            raise RuntimeError("alignment output does not preserve raw COLMAP semantics")
        validate_similarity_transform(alignment.transform)
        if alignment.accepted and alignment.status not in {
            "identity_already_consistent",
            "accepted_global_sim3",
            "transform_chain_bug_fixed",
        }:
            raise RuntimeError("alignment accepted flag is inconsistent with its status")
        if alignment.status == "accepted_global_sim3":
            required_gates = {
                "transform_chain_consistent",
                "minimum_validation_observations",
                "median_residual_improvement",
                "p90_residual_improvement",
                "inlier_fraction_improvement",
                "mesh_coverage_preserved",
                "bad_frame_fraction",
                "finite_transform",
                "positive_scale",
                "proper_rotation",
                "scale_plausible",
                "rotation_plausible",
                "translation_plausible",
                "point_surface_not_degraded",
            }
            if not required_gates <= set(alignment.acceptance_checks) or not all(
                alignment.acceptance_checks[key] for key in required_gates
            ):
                raise RuntimeError("core rejected an accepted Sim(3) with failed held-out gates")
        if audit.status == "transform_chain_bug" and alignment.accepted:
            raise RuntimeError("a transform-chain bug cannot be hidden by an accepted Sim(3)")
        if diagnostics.transform_chain_consistent != (audit.status == "consistent"):
            raise RuntimeError("alignment diagnostics contradict transform-chain audit")
        if (
            split.training_observation_count + split.validation_observation_count
            > (sparse.retained_observations)
            and sparse.retained_observations > 0
        ):
            raise RuntimeError("alignment split counts exceed retained observations")
        selected_candidates = {
            candidate.candidate_id for candidate in candidates.candidates if candidate.selected
        }
        if selected_candidates != {alignment.candidate_id}:
            raise RuntimeError("alignment result does not reference the selected candidate")
        known_frames = set(request.registered_frame_ids)
        if any(item.frame_id not in known_frames for item in sparse.observations):
            raise RuntimeError("sparse alignment observation references an unregistered frame")
        preview_paths = list(previews.model_dump(mode="json").values())
        for relative_path in preview_paths:
            if not context.path(*Path(relative_path).parts).is_file():
                raise RuntimeError(f"alignment preview is missing: {relative_path}")
        root = context.path("reconstruction", "alignment")
        for relative_path in worker.raw_output_paths:
            path = context.path(*Path(relative_path).parts)
            try:
                path.resolve().relative_to(root.resolve())
            except ValueError as exc:
                raise RuntimeError(f"alignment output escapes attempt: {relative_path}") from exc
            if not path.is_file():
                raise RuntimeError(f"alignment worker declared missing output: {relative_path}")
        if any(path.is_symlink() for path in root.rglob("*")):
            raise RuntimeError("alignment worker outputs must not contain symlinks")
        if config.execution_mode != "fake_worker" and worker.backend != "nvdiffrast_scipy":
            raise RuntimeError("real alignment did not report nvdiffrast/scipy execution")

    @staticmethod
    def _alignment_scene(scene: SceneIR, alignment: CameraMeshAlignmentResult) -> SceneIR:
        scene.geometry_assets = [
            asset for asset in scene.geometry_assets if not asset.asset_id.startswith("aligned_")
        ]
        if alignment.accepted:
            wrappers = []
            for asset in scene.geometry_assets:
                if asset.asset_id not in {"global_scene_pbr", "global_scene_mesh"}:
                    continue
                wrappers.append(
                    GeometryAsset(
                        asset_id=f"aligned_{asset.asset_id}",
                        asset_type=asset.asset_type,
                        uri=asset.uri,
                        format=asset.format,
                        source=GeometrySourceType.FUSED,
                        coordinate_convention=alignment.coordinate_convention,
                        scale_status=ScaleStatus.SCALE_AMBIGUOUS,
                        source_asset_id=asset.asset_id,
                        alignment_transform_path="reconstruction/alignment/alignment.json",
                        geometry_alignment_status=(
                            "identity_already_consistent"
                            if alignment.status == "identity_already_consistent"
                            else "accepted_global_sim3"
                        ),
                        sim_ready=False,
                        provenance=alignment.provenance,
                    )
                )
            scene.geometry_assets.extend(wrappers)
        scene.metadata.provenance.append(alignment.provenance)
        scene.schema_version = "0.1.3"
        return SceneIR.model_validate(scene.model_dump(mode="json"))

    def _inference_command(
        self,
        context: StageContext,
        config: CameraMeshAlignmentAdapterConfig,
    ) -> list[str]:
        request = Path("reconstruction/alignment/request.json")
        if config.execution_mode != "docker":
            command = self._local_command(config, "infer", request)
            if isinstance(command, str):
                raise RuntimeError(command)
            command.extend(
                [
                    "--input-root",
                    str(context.run_dir.resolve()),
                    "--output-dir",
                    str(context.path("reconstruction", "alignment").resolve()),
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
            "python3.10",
            config.docker_image,
            "-m",
            config.worker_module,
            "infer",
            "--request",
            "/workspace/reconstruction/alignment/request.json",
            "--input-root",
            "/workspace",
            "--output-dir",
            "/workspace/reconstruction/alignment",
        ]

    @staticmethod
    def _local_command(
        config: CameraMeshAlignmentAdapterConfig,
        action: str,
        path: Path,
    ) -> list[str] | str:
        python = _resolve_python(config.worker_python)
        if python is None:
            return f"configured alignment worker Python {config.worker_python!r} was not found"
        option = "--config" if action == "healthcheck" else "--request"
        if config.execution_mode == "fake_worker":
            assert config.worker_script is not None
            script = Path(config.worker_script)
            if not script.is_absolute():
                script = Path.cwd() / script
            if not script.is_file():
                return f"fake alignment worker does not exist: {script}"
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
                "camera/mesh alignment ran out of GPU memory; lower raster_scale or face_chunk_size"
            )
        if "insufficient" in stderr and "observation" in stderr:
            return RuntimeError(
                "camera/mesh alignment has insufficient filtered, disjoint sparse observations"
            )
        return RuntimeError(str(exc))


class Phase4_2ConsistencyValidationAdapter:
    name = "phase4_2_consistency_validation"
    version = "0.1.0"

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return HealthcheckResult(True, "Phase 4.2 consistency validator available")

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
            InputSpec(
                "camera/genrecon_package/package_manifest.json",
                "genrecon_camera_package_manifest",
            ),
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
                "reconstruction/alignment/request.json",
                "camera_mesh_alignment_request",
            ),
            InputSpec(
                "reconstruction/alignment/transform_chain_audit.json",
                "transform_chain_audit",
            ),
            InputSpec(
                "reconstruction/alignment/sparse_observations.json",
                "sparse_depth_observations",
            ),
            InputSpec(
                "reconstruction/alignment/dataset_split.json",
                "alignment_dataset_split",
            ),
            InputSpec(
                "reconstruction/alignment/alignment.json",
                "camera_mesh_alignment_result",
            ),
            InputSpec(
                "reconstruction/alignment/diagnostics.json",
                "camera_mesh_alignment_diagnostics",
            ),
            InputSpec(
                "reconstruction/alignment/object_lifting_comparison.json",
                "object_lifting_alignment_comparison",
            ),
            InputSpec(
                "reconstruction/object_surfaces/request.json",
                "object_surface_lifting_request",
            ),
            InputSpec(
                "reconstruction/object_surfaces/evidence_manifest.json",
                "object_surface_evidence",
            ),
            InputSpec("scene_ir/alignment_scene.json", "scene_ir"),
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
                ]
            )
            if hypothesis.surface_mesh_path is not None:
                specs.append(
                    InputSpec(
                        hypothesis.surface_mesh_path,
                        "partial_object_surface_mesh",
                    )
                )
        return specs

    def prepare(self, context: StageContext) -> None:
        context.path("validation").mkdir(parents=True, exist_ok=True)
        context.path("scene_ir").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                "validation/phase4_2_camera_mesh_alignment.json",
                "phase4_2_consistency_report",
                "application/json",
                "alignment",
                validation="json",
                schema_identifier="recon2sim/phase4-2-consistency/0.1.0",
                model=Phase4_2ConsistencyReport,
            ),
            OutputSpec(
                "scene_ir/phase4_2_scene.json",
                "scene_ir",
                "application/json",
                "alignment",
                validation="scene_ir",
                schema_identifier="recon2sim/scene-ir/0.1.3",
                model=SceneIR,
            ),
        ]

    def run(self, context: StageContext) -> StageResult:
        manifest_path = context.path("inputs", "manifest.json")
        camera_path = context.path("camera", "reconstruction.json")
        package_path = context.path("camera", "genrecon_package", "package_manifest.json")
        global_path = context.path("reconstruction", "global", "metadata.json")
        request_path = context.path("reconstruction", "alignment", "request.json")
        alignment_path = context.path("reconstruction", "alignment", "alignment.json")
        evidence_path = context.path("reconstruction", "object_surfaces", "evidence_manifest.json")
        manifest = IngestManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        camera = CameraReconstruction.model_validate_json(camera_path.read_text(encoding="utf-8"))
        package = GenReconCameraPackageManifest.model_validate_json(
            package_path.read_text(encoding="utf-8")
        )
        global_scene = GlobalSceneReconstructionArtifact.model_validate_json(
            global_path.read_text(encoding="utf-8")
        )
        request = CameraMeshAlignmentRequest.model_validate_json(
            request_path.read_text(encoding="utf-8")
        )
        audit = TransformChainAudit.model_validate_json(
            context.path("reconstruction", "alignment", "transform_chain_audit.json").read_text(
                encoding="utf-8"
            )
        )
        sparse = SparseDepthObservationManifest.model_validate_json(
            context.path("reconstruction", "alignment", "sparse_observations.json").read_text(
                encoding="utf-8"
            )
        )
        split = AlignmentDatasetSplit.model_validate_json(
            context.path("reconstruction", "alignment", "dataset_split.json").read_text(
                encoding="utf-8"
            )
        )
        alignment = CameraMeshAlignmentResult.model_validate_json(
            alignment_path.read_text(encoding="utf-8")
        )
        diagnostics = CameraMeshAlignmentDiagnostics.model_validate_json(
            context.path("reconstruction", "alignment", "diagnostics.json").read_text(
                encoding="utf-8"
            )
        )
        comparison = ObjectLiftingAlignmentComparison.model_validate_json(
            context.path("reconstruction", "alignment", "object_lifting_comparison.json").read_text(
                encoding="utf-8"
            )
        )
        lifting_request = ObjectSurfaceLiftingRequest.model_validate_json(
            context.path("reconstruction", "object_surfaces", "request.json").read_text(
                encoding="utf-8"
            )
        )
        evidence = ObjectSurfaceEvidenceArtifact.model_validate_json(
            evidence_path.read_text(encoding="utf-8")
        )
        phase4_scene = SceneIR.model_validate_json(
            context.path("scene_ir", "phase4_scene.json").read_text(encoding="utf-8")
        )
        alignment_scene = SceneIR.model_validate_json(
            context.path("scene_ir", "alignment_scene.json").read_text(encoding="utf-8")
        )
        checks: list[Phase4_2ConsistencyCheck] = []

        def check(check_id: str, passed: bool, message: str) -> None:
            checks.append(
                Phase4_2ConsistencyCheck(
                    check_id=check_id,
                    passed=passed,
                    message=message,
                )
            )

        manifest_hash = sha256_file(manifest_path)
        camera_hash = sha256_file(camera_path)
        package_hash = sha256_file(package_path)
        global_hash = sha256_file(global_path)
        mesh_hash = sha256_file(context.canonical_path("reconstruction", "global", "mesh.ply"))
        digest = manifest.frame_sequence_digest
        check(
            "manifest_lineage",
            digest is not None
            and request.manifest_sha256 == manifest_hash
            and request.frame_sequence_digest
            == camera.frame_sequence_digest
            == package.frame_sequence_digest
            == global_scene.frame_sequence_digest
            == digest,
            "alignment uses the Phase 1-4 ordered observation lineage",
        )
        check(
            "camera_hash",
            request.camera_reconstruction_sha256
            == evidence.camera_reconstruction_sha256
            == camera_hash,
            "camera reconstruction bytes remain unchanged",
        )
        check(
            "camera_package_hash",
            request.camera_package_sha256 == package_hash,
            "minimal camera package hash matches",
        )
        check(
            "global_reconstruction_hash",
            request.global_reconstruction_sha256 == global_hash,
            "global reconstruction metadata hash matches",
        )
        check(
            "global_mesh_hash",
            request.global_mesh_sha256 == evidence.global_mesh_sha256 == mesh_hash,
            "original Phase 3 global mesh bytes remain unchanged",
        )
        check(
            "transform_chain_inputs",
            bool(audit.stages)
            and max(
                audit.colmap_working_roundtrip_error,
                audit.camera_basis_roundtrip_error,
                audit.sampled_mesh_roundtrip_error,
            )
            >= 0,
            "transform-chain audit covers explicit coordinate stages",
        )
        known_registered = set(camera.registered_frame_ids)
        check(
            "sparse_registered_cameras",
            all(item.frame_id in known_registered for item in sparse.observations),
            "sparse observations reference registered cameras only",
        )
        training_frames = set(split.training_frame_ids)
        validation_frames = set(split.validation_frame_ids)
        training_points = set(split.training_point_ids)
        validation_points = set(split.validation_point_ids)
        check(
            "heldout_frame_split",
            not training_frames & validation_frames,
            "training and validation camera sets are disjoint",
        )
        check(
            "heldout_point_split",
            not training_points & validation_points,
            "training and validation sparse-point IDs are disjoint",
        )
        transform_valid = True
        try:
            validate_similarity_transform(alignment.transform)
        except ValueError:
            transform_valid = False
        check(
            "transform_invertible",
            transform_valid,
            "candidate Sim(3) is finite, proper, positive-scale, and invertible",
        )
        gates_valid = (
            not alignment.accepted
            or alignment.status == "identity_already_consistent"
            or all(alignment.acceptance_checks.values())
        )
        check(
            "heldout_acceptance_gates",
            gates_valid,
            "accepted non-identity alignment passes recorded held-out gates",
        )
        check(
            "coordinate_semantics",
            coordinate_metadata_is_raw_colmap(alignment.coordinate_convention)
            and alignment.coordinate_convention == camera.coordinate_convention,
            "alignment preserves arbitrary, unoriented, scale-ambiguous COLMAP semantics",
        )
        check(
            "camera_poses_not_rewritten",
            request.camera_reconstruction_sha256 == camera_hash,
            "alignment never rewrites canonical camera poses",
        )
        check(
            "mesh_topology_not_rewritten",
            evidence.partition.global_face_count == global_scene.mesh.face_count,
            "object lifting retains original global face topology",
        )
        expected_alignment_sha = sha256_file(alignment_path)
        check(
            "object_lifting_alignment_reference",
            lifting_request.alignment_sha256 == expected_alignment_sha
            and evidence.alignment_sha256 == expected_alignment_sha
            and comparison.alignment_sha256 == expected_alignment_sha
            and lifting_request.alignment_accepted == alignment.accepted
            and evidence.alignment_accepted == alignment.accepted,
            "aligned object lifting references the exact typed alignment artifact",
        )
        face_ids_valid = True
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
        except (OSError, ValueError):
            face_ids_valid = False
        check(
            "original_face_ids",
            face_ids_valid,
            "all compact face IDs remain inside original global mesh bounds",
        )
        check(
            "no_metric_or_gravity_claim",
            alignment.scale_status is ScaleStatus.SCALE_AMBIGUOUS
            and evidence.metric_scale_known is False
            and evidence.canonical_gravity_alignment_known is False,
            "fitted similarity does not claim metric scale or gravity alignment",
        )
        check(
            "no_collision_or_physics",
            not phase4_scene.collision_assets
            and all(not item.collision_asset_ids for item in phase4_scene.objects),
            "Phase 4.2 emits no collision or physics assets",
        )
        check(
            "no_hidden_completion",
            evidence.hidden_surface_completion == "not_implemented",
            "hidden-surface completion remains unimplemented",
        )
        check(
            "selective_materialization",
            self._attempt_inputs_are_selective(context.run_dir),
            "validator attempt contains only declared canonical inputs",
        )
        final_scene = self._merge_scenes(phase4_scene, alignment_scene, alignment)
        check(
            "scene_ir_alignment",
            (
                not alignment.accepted
                or any(
                    asset.alignment_transform_path == "reconstruction/alignment/alignment.json"
                    for asset in final_scene.geometry_assets
                )
            )
            and all(not asset.sim_ready for asset in final_scene.geometry_assets),
            "Scene IR references alignment without simulation-ready claims",
        )
        report = Phase4_2ConsistencyReport(
            passed=all(item.passed for item in checks),
            checks=checks,
            transform_chain_consistent=(
                audit.status == "consistent" and diagnostics.transform_chain_consistent
            ),
            global_similarity_accepted=alignment.accepted,
            global_similarity_sufficient=diagnostics.global_similarity_sufficient,
            warnings=[*alignment.warnings, *diagnostics.warnings],
        )
        atomic_write_json(
            context.path("validation", "phase4_2_camera_mesh_alignment.json"),
            report,
        )
        atomic_write_json(
            context.path("scene_ir", "phase4_2_scene.json"),
            final_scene,
        )
        if not report.passed:
            failed = [item.check_id for item in report.checks if not item.passed]
            raise RuntimeError(f"Phase 4.2 consistency validation failed: {failed}")
        return StageResult(
            metrics={
                "checks": len(checks),
                "alignment_accepted": alignment.accepted,
                "alignment_sufficient": diagnostics.global_similarity_sufficient,
            }
        )

    @staticmethod
    def _merge_scenes(
        phase4_scene: SceneIR,
        alignment_scene: SceneIR,
        alignment: CameraMeshAlignmentResult,
    ) -> SceneIR:
        wrappers = {
            asset.asset_id: asset
            for asset in alignment_scene.geometry_assets
            if asset.alignment_transform_path is not None
        }
        existing = {asset.asset_id: asset for asset in phase4_scene.geometry_assets}
        if alignment.accepted:
            existing.update(wrappers)
        phase4_scene.geometry_assets = sorted(
            existing.values(),
            key=lambda asset: asset.asset_id,
        )
        phase4_scene.metadata.provenance.append(alignment.provenance)
        phase4_scene.schema_version = "0.1.3"
        return SceneIR.model_validate(phase4_scene.model_dump(mode="json"))

    @staticmethod
    def _attempt_inputs_are_selective(run_dir: Path) -> bool:
        forbidden = (
            "camera/colmap/database.db",
            "camera/colmap/sparse",
            "camera/colmap/logs",
            "observations/raw",
            "reconstruction/global/raw",
        )
        return all(not run_dir.joinpath(*Path(path).parts).exists() for path in forbidden)
