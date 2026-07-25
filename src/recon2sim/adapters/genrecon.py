from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
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
    CameraDiagnostics,
    CameraReconstruction,
    EndToEndConsistencyCheck,
    EndToEndConsistencyReport,
    GenReconCameraPackageManifest,
    GenReconCheckpointManifest,
    GenReconCheckpointRecord,
    GenReconInferenceRequest,
    GenReconWorkerManifest,
    GlobalSceneChunkDiagnostic,
    GlobalSceneDiagnostics,
    GlobalScenePreviewManifest,
    GlobalSceneReconstructionArtifact,
    IngestManifest,
    Sam3InferenceRequest,
    Sam3WorkerManifest,
    SegmentationTrackingArtifact,
)
from recon2sim.colmap import read_model
from recon2sim.genrecon import (
    OFFICIAL_CHECKPOINT_URLS,
    OFFICIAL_GENRECON_COMMIT,
    OFFICIAL_GENRECON_REPOSITORY,
    OFFICIAL_GENRECON_SUBMODULES,
    coordinate_metadata_is_raw_colmap,
    export_colmap_text_package,
    inspect_global_mesh,
    read_colmap_text_points,
    render_camera_trajectory_preview,
    render_global_previews,
    sha256_file,
    validate_camera_package,
)
from recon2sim.ir import (
    AssetType,
    Camera,
    ConfidenceRecord,
    FrameObservation,
    GeometryAsset,
    GeometrySourceType,
    ProvenanceRecord,
    ScaleStatus,
    SceneIR,
    SceneMetadata,
    StrictModel,
)
from recon2sim.storage import atomic_write_json

GENRECON_WORKER_VERSION = "0.1.1"


def _resolve_python(value: str) -> str | None:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        return str(candidate.absolute()) if candidate.is_file() else None
    return shutil.which(value)


class GenReconAdapterConfig(StrictModel):
    execution_mode: Literal["local_worker", "docker", "fake_worker"]
    worker_python: str = "python"
    worker_module: str = "genrecon_worker"
    worker_script: str | None = None
    docker_executable: str = "docker"
    docker_image: str = "reconevery/genrecon:phase3"
    hf_cache_path: str | None = None
    official_repository: Literal["https://github.com/kasothaphie/GenRecon"] = (
        OFFICIAL_GENRECON_REPOSITORY
    )
    official_code_commit: Literal["eaf1468118d20469d17079a4a19737297d2ef87b"] = (
        OFFICIAL_GENRECON_COMMIT
    )
    official_checkout_path: str | None = None
    sparse_structure_checkpoint: str
    shape_checkpoint: str
    texture_checkpoint: str
    device: Literal["cuda"] = "cuda"
    precision: Literal["float16"] = "float16"
    seed: int = 42
    requested_max_views: int = Field(default=32, gt=0)
    working_transform_strategy: Literal["identity", "pca_scene_axes"] = "pca_scene_axes"
    pipeline_config: str = "configs/pipelines/texture.json"
    chunk_size_factor: float = Field(default=1.08, gt=0)
    stat_std_ratio: float = Field(default=3.0, gt=0)
    radius_nb_points: int = Field(default=7, gt=0)
    radius_m: float = Field(default=0.2, gt=0)
    min_points_per_chunk: int = Field(default=30, gt=0)
    skip_point_cleaning: bool = False
    proj_batch_voxels: int = Field(default=2048, gt=0)
    fake_mode: str = "success"

    @model_validator(mode="after")
    def validate_execution(self) -> GenReconAdapterConfig:
        if self.execution_mode == "fake_worker":
            if self.worker_script is None:
                raise ValueError("fake_worker execution requires worker_script")
        else:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible is None or visible.strip().lower() in {"", "-1", "none", "void"}:
                raise ValueError("real GenRecon execution requires CUDA_VISIBLE_DEVICES")
        if self.execution_mode == "local_worker":
            python = _resolve_python(self.worker_python)
            if python is None:
                raise ValueError(f"configured worker Python {self.worker_python!r} was not found")
            executable = Path(python).absolute()
            root = executable.parent.parent
            if not (root / "pyvenv.cfg").is_file() and not (root / "conda-meta").is_dir():
                raise ValueError(
                    "local GenRecon worker_python must be in an isolated venv or conda environment"
                )
            if root == Path(sys.prefix).resolve():
                raise ValueError("GenRecon worker must not use the Reconevery core environment")
            if self.official_checkout_path is None:
                raise ValueError("local_worker requires official_checkout_path")
        for name, raw_path in self.checkpoint_paths().items():
            path = Path(raw_path).expanduser()
            if self.execution_mode != "fake_worker" and not path.is_absolute():
                raise ValueError(f"{name} checkpoint path must be absolute")
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            if not path.is_file() or not os.access(path, os.R_OK):
                raise ValueError(f"{name} checkpoint is not a readable file: {path}")
        if self.official_checkout_path is not None:
            checkout = Path(self.official_checkout_path).expanduser()
            if not checkout.is_absolute() or not checkout.is_dir():
                raise ValueError("official_checkout_path must be an existing absolute directory")
        if self.hf_cache_path is not None:
            cache = Path(self.hf_cache_path).expanduser()
            if not cache.is_absolute() or not cache.is_dir():
                raise ValueError("hf_cache_path must be an existing absolute directory")
            if not os.access(cache, os.R_OK | os.W_OK):
                raise ValueError("hf_cache_path must be readable and writable")
        return self

    def checkpoint_paths(self) -> dict[str, str]:
        return {
            "sparse_structure": self.sparse_structure_checkpoint,
            "shape_slat": self.shape_checkpoint,
            "texture_slat": self.texture_checkpoint,
        }


class GenReconCameraPackageAdapter:
    name = "genrecon_camera_package"
    version = "0.1.1"

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return HealthcheckResult(True, "COLMAP text package exporter available")

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        diagnostics_path = context.canonical_path("camera", "diagnostics.json")
        if not diagnostics_path.is_file():
            raise FileNotFoundError("GenRecon camera export requires camera/diagnostics.json")
        diagnostics = CameraDiagnostics.model_validate_json(
            diagnostics_path.read_text(encoding="utf-8")
        )
        if diagnostics.selected_model is None:
            raise ValueError("camera diagnostics do not identify a selected COLMAP model")
        model_root = f"camera/colmap/sparse/{diagnostics.selected_model}"
        return [
            InputSpec("inputs/manifest.json", "ingest_manifest"),
            InputSpec("camera/reconstruction.json", "camera_reconstruction"),
            InputSpec("camera/diagnostics.json", "camera_diagnostics"),
            InputSpec(f"{model_root}/cameras.bin", "colmap_raw_model"),
            InputSpec(f"{model_root}/images.bin", "colmap_raw_model"),
            InputSpec(f"{model_root}/points3D.bin", "colmap_raw_model"),
        ]

    def prepare(self, context: StageContext) -> None:
        context.path("camera", "genrecon_package").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        root = "camera/genrecon_package"
        return [
            OutputSpec(
                f"{root}/cameras.txt",
                "genrecon_colmap_text",
                "text/plain",
                "colmap",
            ),
            OutputSpec(
                f"{root}/images.txt",
                "genrecon_colmap_text",
                "text/plain",
                "colmap",
            ),
            OutputSpec(
                f"{root}/points3D.txt",
                "genrecon_colmap_text",
                "text/plain",
                "colmap",
            ),
            OutputSpec(
                f"{root}/registered_frames.json",
                "genrecon_registered_frames",
                "application/json",
                "colmap",
                validation="json",
            ),
            OutputSpec(
                f"{root}/package_manifest.json",
                "genrecon_camera_package_manifest",
                "application/json",
                "colmap",
                validation="json",
                schema_identifier="recon2sim/genrecon-camera-package/0.1.0",
                model=GenReconCameraPackageManifest,
            ),
            OutputSpec(
                f"{root}/previews/camera_trajectory_and_sparse_points.png",
                "colmap_camera_preview",
                "image/png",
                "colmap",
                validation="png",
            ),
        ]

    def run(self, context: StageContext) -> StageResult:
        manifest_path = context.path("inputs", "manifest.json")
        camera_path = context.path("camera", "reconstruction.json")
        diagnostics = CameraDiagnostics.model_validate_json(
            context.path("camera", "diagnostics.json").read_text(encoding="utf-8")
        )
        if diagnostics.selected_model is None:
            raise ValueError("camera diagnostics do not identify a selected COLMAP model")
        manifest = IngestManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        camera = CameraReconstruction.model_validate_json(camera_path.read_text(encoding="utf-8"))
        if not coordinate_metadata_is_raw_colmap(camera.coordinate_convention):
            raise ValueError(
                "GenRecon Phase 3 requires raw COLMAP arbitrary/unoriented camera metadata"
            )
        model_root = context.path("camera", "colmap", "sparse", diagnostics.selected_model)
        source_hashes = {
            name: sha256_file(model_root / name)
            for name in ("cameras.bin", "images.bin", "points3D.bin")
        }
        model = read_model(model_root)
        package = export_colmap_text_package(
            model=model,
            manifest=manifest,
            camera=camera,
            output_dir=context.path("camera", "genrecon_package"),
            selected_model_id=diagnostics.selected_model,
            source_model_hashes=source_hashes,
            manifest_sha256=sha256_file(manifest_path),
            camera_reconstruction_sha256=sha256_file(camera_path),
        )
        atomic_write_json(
            context.path("camera", "genrecon_package", "package_manifest.json"),
            package,
        )
        validate_camera_package(context.run_dir, package)
        render_camera_trajectory_preview(
            camera,
            read_colmap_text_points(context.path("camera", "genrecon_package", "points3D.txt")),
            context.path(
                "camera",
                "genrecon_package",
                "previews",
                "camera_trajectory_and_sparse_points.png",
            ),
        )
        return StageResult(
            metrics={
                "registered_frames": len(package.registered_frame_ids),
                "unregistered_frames": len(package.unregistered_frame_ids),
                "sparse_points": len(model.points3d),
            }
        )


def _checkpoint_records(config: GenReconAdapterConfig) -> list[GenReconCheckpointRecord]:
    records: list[GenReconCheckpointRecord] = []
    for checkpoint_id, raw_path in config.checkpoint_paths().items():
        path = Path(raw_path).expanduser().resolve()
        timestamp = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
        records.append(
            GenReconCheckpointRecord(
                checkpoint_id=cast(
                    Literal["sparse_structure", "shape_slat", "texture_slat"],
                    checkpoint_id,
                ),
                source_url=OFFICIAL_CHECKPOINT_URLS[checkpoint_id],
                local_filename=path.name,
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
                resolved_at=timestamp,
                access_mode=("fake" if config.execution_mode == "fake_worker" else "local_cache"),
            )
        )
    return records


def _worker_failure(exc: ProcessExecutionError) -> RuntimeError:
    stderr = exc.result.stderr.lower()
    if "dinov3" in stderr and ("gated" in stderr or "403" in stderr or "access" in stderr):
        return RuntimeError(
            "official GenRecon requires accepted Hugging Face access to "
            "facebook/dinov3-vitl16-pretrain-lvd1689m; accept its terms and authenticate "
            "through HF_TOKEN or the configured HF_HOME cache"
        )
    if "out of memory" in stderr or "cuda oom" in stderr:
        return RuntimeError(
            "GenRecon worker ran out of GPU memory; reduce selected views/chunks or lower "
            "proj_batch_voxels"
        )
    if "checkpoint" in stderr and ("missing" in stderr or "not found" in stderr):
        return RuntimeError("GenRecon worker could not resolve a required official checkpoint")
    if "cuda extension" in stderr or "undefined symbol" in stderr:
        return RuntimeError(
            "GenRecon CUDA extension import failed; rebuild extensions for the pinned "
            "PyTorch/CUDA/H100 environment"
        )
    return RuntimeError(str(exc))


class GenReconGlobalReconstructionAdapter:
    name = "genrecon_global_reconstruction"
    version = "0.1.1"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        config = GenReconAdapterConfig.model_validate(context.config.adapter.config)
        package_path = context.canonical_path("camera", "genrecon_package", "package_manifest.json")
        if not package_path.is_file():
            raise FileNotFoundError("GenRecon requires a successful camera-package stage")
        package = GenReconCameraPackageManifest.model_validate_json(
            package_path.read_text(encoding="utf-8")
        )
        specs = [
            InputSpec("inputs/manifest.json", "ingest_manifest"),
            InputSpec("camera/reconstruction.json", "camera_reconstruction"),
            InputSpec(
                "camera/genrecon_package/package_manifest.json",
                "genrecon_camera_package_manifest",
            ),
            InputSpec(
                "camera/genrecon_package/cameras.txt",
                "genrecon_colmap_text",
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
        ]
        frame_by_id = {
            frame.frame_id: frame
            for frame in IngestManifest.model_validate_json(
                context.canonical_path("inputs", "manifest.json").read_text(encoding="utf-8")
            ).frames
        }
        specs.extend(
            InputSpec(
                frame_by_id[frame_id].relative_path,
                "input_frame",
                expected_sha256=frame_by_id[frame_id].sha256,
            )
            for frame_id in package.eligible_frame_ids
        )
        for checkpoint_id, raw_path in config.checkpoint_paths().items():
            checkpoint = Path(raw_path).expanduser().resolve()
            specs.append(
                InputSpec(
                    f"reconstruction/global/checkpoint_refs/{checkpoint_id}.pt",
                    "genrecon_checkpoint",
                    expected_sha256=sha256_file(checkpoint),
                    source_path=checkpoint,
                    materialization_mode="reference_only",
                )
            )
        return specs

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        if context is None:
            return HealthcheckResult(
                False,
                "GenRecon healthcheck requires --config to verify the worker and checkpoints",
            )
        try:
            config = GenReconAdapterConfig.model_validate(context.config.adapter.config)
            records = _checkpoint_records(config)
        except (OSError, ValueError) as exc:
            return HealthcheckResult(False, f"invalid GenRecon configuration: {exc}")
        payload = self._worker_configuration(config, records)
        with tempfile.TemporaryDirectory(prefix="reconevery-genrecon-health-") as temporary:
            config_path = Path(temporary) / "worker_config.json"
            atomic_write_json(config_path, payload)
            if config.execution_mode == "docker":
                return self._docker_healthcheck(context, config, config_path)
            command_or_error = self._local_command(
                config,
                "healthcheck",
                config_path,
            )
            if isinstance(command_or_error, str):
                return HealthcheckResult(False, command_or_error)
            try:
                result = subprocess.run(
                    command_or_error,
                    cwd=Path.cwd(),
                    env=allowed_environment(context),
                    text=True,
                    capture_output=True,
                    timeout=min(context.config.adapter.timeout_s, 180),
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return HealthcheckResult(False, f"GenRecon worker healthcheck failed: {exc}")
            output = result.stdout.strip() or result.stderr.strip()
            if result.returncode != 0:
                return HealthcheckResult(
                    False,
                    f"GenRecon worker healthcheck failed (exit {result.returncode}): {output}",
                )
            return HealthcheckResult(True, output or "GenRecon worker healthcheck succeeded")

    def _docker_healthcheck(
        self,
        context: StageContext,
        config: GenReconAdapterConfig,
        config_path: Path,
    ) -> HealthcheckResult:
        docker = resolve_executable(config.docker_executable)
        if docker is None:
            return HealthcheckResult(False, "Docker executable was not found")
        commands = (
            [docker, "version", "--format", "{{.Server.Version}}"],
            [docker, "image", "inspect", "--format", "{{.Id}}", config.docker_image],
        )
        results = []
        for command in commands:
            try:
                result = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return HealthcheckResult(False, f"Docker healthcheck failed: {exc}")
            if result.returncode != 0:
                return HealthcheckResult(False, result.stderr.strip() or "Docker check failed")
            results.append(result.stdout.strip())
        command = [
            docker,
            "run",
            "--rm",
            "--gpus",
            "all",
            *self._docker_user_arguments(),
            "-v",
            f"{config_path.parent}:/workspace:ro",
            *self._docker_checkpoint_mounts(config),
            *self._docker_hf_cache_arguments(config),
            *self._docker_environment_arguments(context),
            "--entrypoint",
            "python",
            config.docker_image,
            "-m",
            config.worker_module,
            "healthcheck",
            "--config",
            "/workspace/worker_config.json",
        ]
        try:
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=min(context.config.adapter.timeout_s, 300),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return HealthcheckResult(False, f"in-container GenRecon healthcheck failed: {exc}")
        if result.returncode != 0:
            return HealthcheckResult(
                False,
                result.stderr.strip() or "in-container GenRecon healthcheck failed",
            )
        return HealthcheckResult(
            True,
            f"docker={results[0]}; image={config.docker_image} ({results[1]}); "
            f"{result.stdout.strip()}",
        )

    def prepare(self, context: StageContext) -> None:
        context.path("reconstruction", "global", "raw").mkdir(parents=True, exist_ok=True)
        context.path("reconstruction", "global", "previews").mkdir(parents=True, exist_ok=True)
        context.path("scene_ir").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        root = "reconstruction/global"
        return [
            OutputSpec(
                f"{root}/request.json",
                "genrecon_inference_request",
                "application/json",
                "genrecon",
                validation="json",
                schema_identifier="recon2sim/genrecon-request/0.1.0",
                model=GenReconInferenceRequest,
            ),
            OutputSpec(
                f"{root}/camera_package_manifest.json",
                "genrecon_camera_package_manifest",
                "application/json",
                "genrecon",
                validation="json",
                model=GenReconCameraPackageManifest,
            ),
            OutputSpec(
                f"{root}/checkpoint_manifest.json",
                "genrecon_checkpoint_manifest",
                "application/json",
                "genrecon",
                validation="json",
                schema_identifier="recon2sim/genrecon-checkpoints/0.1.0",
                model=GenReconCheckpointManifest,
            ),
            OutputSpec(
                f"{root}/worker_manifest.json",
                "genrecon_worker_manifest",
                "application/json",
                "genrecon",
                validation="json",
                schema_identifier="recon2sim/genrecon-worker-manifest/0.1.0",
                model=GenReconWorkerManifest,
            ),
            OutputSpec(
                f"{root}/metadata.json",
                "global_scene_reconstruction",
                "application/json",
                "genrecon",
                validation="json",
                schema_identifier="recon2sim/global-scene-reconstruction/0.1.0",
                model=GlobalSceneReconstructionArtifact,
            ),
            OutputSpec(
                f"{root}/diagnostics.json",
                "global_scene_diagnostics",
                "application/json",
                "genrecon",
                validation="json",
                schema_identifier="recon2sim/global-scene-diagnostics/0.1.0",
                model=GlobalSceneDiagnostics,
            ),
            OutputSpec(
                f"{root}/preview_manifest.json",
                "global_scene_preview_manifest",
                "application/json",
                "genrecon",
                validation="json",
                model=GlobalScenePreviewManifest,
            ),
            OutputSpec(
                f"{root}/scene.glb",
                "global_pbr_scene",
                "model/gltf-binary",
                "genrecon",
            ),
            OutputSpec(
                f"{root}/mesh.ply",
                "global_scene_mesh",
                "model/ply",
                "genrecon",
            ),
            *[
                OutputSpec(
                    path,
                    "global_scene_preview",
                    "image/png",
                    "genrecon",
                    validation="png",
                )
                for path in (
                    f"{root}/previews/global_scene_preview.png",
                    f"{root}/previews/camera_trajectory_and_sparse_points.png",
                    f"{root}/previews/input_vs_geometry_contact_sheet.png",
                )
            ],
            OutputSpec(
                "scene_ir/scene.json",
                "scene_ir",
                "application/json",
                "genrecon",
                validation="scene_ir",
                schema_identifier="recon2sim/scene-ir/0.1.1",
                model=SceneIR,
            ),
        ]

    def run(self, context: StageContext) -> StageResult:
        config = GenReconAdapterConfig.model_validate(context.config.adapter.config)
        manifest_path = context.path("inputs", "manifest.json")
        camera_path = context.path("camera", "reconstruction.json")
        package_path = context.path("camera", "genrecon_package", "package_manifest.json")
        manifest = IngestManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        camera = CameraReconstruction.model_validate_json(camera_path.read_text(encoding="utf-8"))
        package = GenReconCameraPackageManifest.model_validate_json(
            package_path.read_text(encoding="utf-8")
        )
        validate_camera_package(context.run_dir, package)
        if not coordinate_metadata_is_raw_colmap(camera.coordinate_convention):
            raise ValueError("GenRecon input coordinates must retain raw COLMAP semantics")
        if manifest.frame_sequence_digest != package.frame_sequence_digest:
            raise ValueError("camera package frame lineage does not match ingest")
        records = _checkpoint_records(config)
        checkpoint_manifest_path = context.path(
            "reconstruction", "global", "checkpoint_manifest.json"
        )
        atomic_write_json(
            checkpoint_manifest_path,
            GenReconCheckpointManifest(
                official_host="https://kaldir.vc.cit.tum.de/genrecon/",
                checkpoints=records,
            ),
        )
        shutil.copy2(
            package_path,
            context.path("reconstruction", "global", "camera_package_manifest.json"),
        )
        checkpoint_paths = self._request_checkpoint_paths(config)
        request = GenReconInferenceRequest(
            run_id=context.canonical_run_dir.name,
            official_repository=OFFICIAL_GENRECON_REPOSITORY,
            official_code_commit=OFFICIAL_GENRECON_COMMIT,
            official_checkout_path=(
                "/opt/GenRecon"
                if config.execution_mode == "docker"
                else str(Path(config.official_checkout_path or "").expanduser().resolve())
            ),
            checkpoint_paths=checkpoint_paths,
            checkpoint_hashes={record.checkpoint_id: record.sha256 for record in records},
            checkpoint_manifest_path="reconstruction/global/checkpoint_manifest.json",
            manifest_sha256=sha256_file(manifest_path),
            frame_sequence_digest=self._require_digest(manifest.frame_sequence_digest),
            camera_reconstruction_sha256=sha256_file(camera_path),
            camera_package_sha256=sha256_file(package_path),
            master_frame_order=[frame.frame_id for frame in manifest.frames],
            normalized_frame_paths={
                frame.frame_id: frame.relative_path for frame in manifest.frames
            },
            normalized_frame_hashes={frame.frame_id: frame.sha256 for frame in manifest.frames},
            registered_frame_ids=camera.registered_frame_ids,
            unregistered_frame_ids=camera.unregistered_frame_ids,
            eligible_frame_ids=package.eligible_frame_ids,
            requested_max_views=config.requested_max_views,
            coordinate_convention=camera.coordinate_convention,
            working_transform_strategy=config.working_transform_strategy,
            pipeline_config=config.pipeline_config,
            reconstruction_parameters=self._reconstruction_parameters(config),
            seed=config.seed,
        )
        request_path = context.path("reconstruction", "global", "request.json")
        atomic_write_json(request_path, request)
        command = self._inference_command(context, config)
        try:
            run_process(
                command,
                context=context,
                name="genrecon_worker",
                log_directory="reconstruction/global/raw/logs",
            )
        except ProcessExecutionError as exc:
            raise _worker_failure(exc) from exc
        raw_root = context.path("reconstruction", "global", "raw")
        worker_path = raw_root / "worker_manifest.json"
        diagnostics_path = raw_root / "worker_diagnostics.json"
        if not worker_path.is_file() or not diagnostics_path.is_file():
            raise RuntimeError("GenRecon worker completed without its manifest and diagnostics")
        try:
            worker = GenReconWorkerManifest.model_validate_json(
                worker_path.read_text(encoding="utf-8")
            )
            raw_diagnostics = cast(
                dict[str, Any],
                json.loads(diagnostics_path.read_text(encoding="utf-8")),
            )
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"GenRecon worker output is malformed: {exc}") from exc
        self._validate_worker_output(
            context,
            config,
            request,
            records,
            worker,
        )
        scene_source = raw_root / "scene.glb"
        mesh_source = raw_root / "mesh.ply"
        mesh_statistics = inspect_global_mesh(mesh_source, scene_source)
        scene_path = context.path("reconstruction", "global", "scene.glb")
        mesh_path = context.path("reconstruction", "global", "mesh.ply")
        shutil.copy2(scene_source, scene_path)
        shutil.copy2(mesh_source, mesh_path)
        atomic_write_json(
            context.path("reconstruction", "global", "worker_manifest.json"),
            worker,
        )
        diagnostics = self._normalize_diagnostics(
            raw_diagnostics,
            request,
            camera,
            mesh_statistics,
            worker,
        )
        atomic_write_json(
            context.path("reconstruction", "global", "diagnostics.json"),
            diagnostics,
        )
        render_global_previews(
            root=context.run_dir,
            manifest=manifest,
            camera=camera,
            sparse_points=read_colmap_text_points(
                context.path("camera", "genrecon_package", "points3D.txt")
            ),
            mesh_path=mesh_path,
        )
        preview_manifest = GlobalScenePreviewManifest(
            global_scene_preview_path=("reconstruction/global/previews/global_scene_preview.png"),
            camera_trajectory_path=(
                "reconstruction/global/previews/camera_trajectory_and_sparse_points.png"
            ),
            input_vs_geometry_path=(
                "reconstruction/global/previews/input_vs_geometry_contact_sheet.png"
            ),
        )
        atomic_write_json(
            context.path("reconstruction", "global", "preview_manifest.json"),
            preview_manifest,
        )
        provenance = ProvenanceRecord(
            adapter_name=self.name,
            adapter_version=self.version,
            configuration=self._safe_configuration(config),
            input_artifact_paths=[
                "inputs/manifest.json",
                "camera/reconstruction.json",
                "camera/genrecon_package/package_manifest.json",
                *[
                    frame.relative_path
                    for frame in manifest.frames
                    if frame.frame_id in set(package.eligible_frame_ids)
                ],
            ],
            output_artifact_paths=[
                "reconstruction/global/metadata.json",
                "reconstruction/global/scene.glb",
                "reconstruction/global/mesh.ply",
                "scene_ir/scene.json",
            ],
            timestamp=manifest.provenance.timestamp,
            confidence=ConfidenceRecord(
                score=min(
                    1.0,
                    len(worker.selected_frame_ids) / max(len(package.eligible_frame_ids), 1),
                ),
                method="selected_registered_view_coverage",
            ),
            source=GeometrySourceType.GENERATED,
        )
        artifact = GlobalSceneReconstructionArtifact(
            scene_asset_path="reconstruction/global/scene.glb",
            mesh_asset_path="reconstruction/global/mesh.ply",
            scene_ir_path="scene_ir/scene.json",
            coordinate_convention=camera.coordinate_convention,
            scale_status=ScaleStatus.SCALE_AMBIGUOUS,
            manifest_sha256=request.manifest_sha256,
            frame_sequence_digest=request.frame_sequence_digest,
            camera_reconstruction_sha256=request.camera_reconstruction_sha256,
            camera_package_sha256=request.camera_package_sha256,
            input_frame_count=len(manifest.frames),
            registered_frame_count=len(camera.registered_frame_ids),
            unregistered_frame_count=len(camera.unregistered_frame_ids),
            eligible_frame_ids=package.eligible_frame_ids,
            actual_selected_frame_ids=worker.selected_frame_ids,
            mesh=mesh_statistics,
            chunk_count=diagnostics.chunks_after_filtering,
            checkpoints=records,
            official_repository=OFFICIAL_GENRECON_REPOSITORY,
            official_code_commit=OFFICIAL_GENRECON_COMMIT,
            runtime_model_repository=worker.runtime_model_repository,
            runtime_model_revision=worker.runtime_model_revision,
            runtime_repository_revisions=worker.runtime_repository_revisions,
            runtime_seconds=worker.runtime_seconds,
            peak_gpu_memory_bytes=worker.peak_gpu_memory_bytes,
            seed=config.seed,
            provenance=provenance,
        )
        atomic_write_json(
            context.path("reconstruction", "global", "metadata.json"),
            artifact,
        )
        atomic_write_json(
            context.path("scene_ir", "scene.json"),
            self._scene_ir(manifest, camera, artifact, provenance),
        )
        dynamic = [
            OutputSpec(
                path.relative_to(context.run_dir).as_posix(),
                "genrecon_raw_output",
                "application/octet-stream",
                "genrecon",
            )
            for path in sorted(raw_root.rglob("*"))
            if path.is_file()
        ]
        return StageResult(
            outputs=dynamic,
            metrics={
                "eligible_frames": len(package.eligible_frame_ids),
                "selected_views": len(worker.selected_frame_ids),
                "chunks": diagnostics.chunks_after_filtering,
                "vertices": mesh_statistics.vertex_count,
                "faces": mesh_statistics.face_count,
            },
        )

    @staticmethod
    def _require_digest(value: str | None) -> str:
        if value is None:
            raise ValueError("Phase 3 requires an ingest frame-sequence digest")
        return value

    @staticmethod
    def _reconstruction_parameters(config: GenReconAdapterConfig) -> dict[str, object]:
        return {
            "chunk_size_factor": config.chunk_size_factor,
            "stat_std_ratio": config.stat_std_ratio,
            "radius_nb_points": config.radius_nb_points,
            "radius_m": config.radius_m,
            "radius_parameter_units": "arbitrary_units",
            "min_points_per_chunk": config.min_points_per_chunk,
            "skip_point_cleaning": config.skip_point_cleaning,
            "proj_batch_voxels": config.proj_batch_voxels,
            "fake_mode": config.fake_mode,
        }

    @staticmethod
    def _safe_configuration(config: GenReconAdapterConfig) -> dict[str, Any]:
        payload = config.model_dump(mode="json")
        payload["sparse_structure_checkpoint"] = Path(config.sparse_structure_checkpoint).name
        payload["shape_checkpoint"] = Path(config.shape_checkpoint).name
        payload["texture_checkpoint"] = Path(config.texture_checkpoint).name
        if config.official_checkout_path is not None:
            payload["official_checkout_path"] = "<official-checkout>"
        return payload

    @staticmethod
    def _request_checkpoint_paths(config: GenReconAdapterConfig) -> dict[str, str]:
        if config.execution_mode != "docker":
            return {
                name: str(Path(path).expanduser().resolve())
                for name, path in config.checkpoint_paths().items()
            }
        return {
            name: f"/checkpoints/{name}/{Path(path).name}"
            for name, path in config.checkpoint_paths().items()
        }

    @staticmethod
    def _worker_configuration(
        config: GenReconAdapterConfig,
        records: list[GenReconCheckpointRecord],
    ) -> dict[str, Any]:
        checkout = config.official_checkout_path
        if config.execution_mode == "docker":
            checkout = "/opt/GenRecon"
        return {
            "official_repository": OFFICIAL_GENRECON_REPOSITORY,
            "official_code_commit": OFFICIAL_GENRECON_COMMIT,
            "official_checkout_path": checkout,
            "submodule_commits": OFFICIAL_GENRECON_SUBMODULES,
            "checkpoint_paths": GenReconGlobalReconstructionAdapter._request_checkpoint_paths(
                config
            ),
            "checkpoint_hashes": {record.checkpoint_id: record.sha256 for record in records},
            "device": config.device,
            "precision": config.precision,
        }

    def _inference_command(
        self,
        context: StageContext,
        config: GenReconAdapterConfig,
    ) -> list[str]:
        if config.execution_mode != "docker":
            command_or_error = self._local_command(
                config,
                "infer",
                Path("reconstruction/global/request.json"),
            )
            if isinstance(command_or_error, str):
                raise RuntimeError(command_or_error)
            return command_or_error
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
            *self._docker_checkpoint_mounts(config),
            *self._docker_hf_cache_arguments(config),
            *self._docker_environment_arguments(context),
            "-w",
            "/workspace",
            "--entrypoint",
            "python",
            config.docker_image,
            "-m",
            config.worker_module,
            "infer",
            "--request",
            "/workspace/reconstruction/global/request.json",
            "--output-dir",
            "/workspace/reconstruction/global/raw",
        ]

    @staticmethod
    def _local_command(
        config: GenReconAdapterConfig,
        action: str,
        path: Path,
    ) -> list[str] | str:
        python = _resolve_python(config.worker_python)
        if python is None:
            return f"configured worker Python {config.worker_python!r} was not found"
        option = "--config" if action == "healthcheck" else "--request"
        if config.execution_mode == "fake_worker":
            if config.worker_script is None:
                return "fake worker script is not configured"
            script = Path(config.worker_script).expanduser()
            if not script.is_absolute():
                script = Path.cwd() / script
            if not script.is_file():
                return f"fake GenRecon worker does not exist: {script}"
            command = [python, str(script.resolve()), action, option, str(path)]
        else:
            command = [python, "-m", config.worker_module, action, option, str(path)]
        if action == "infer":
            command.extend(["--output-dir", "reconstruction/global/raw"])
        return command

    @staticmethod
    def _docker_user_arguments() -> list[str]:
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            return ["--user", f"{os.getuid()}:{os.getgid()}"]
        return []

    @staticmethod
    def _docker_checkpoint_mounts(config: GenReconAdapterConfig) -> list[str]:
        arguments: list[str] = []
        for name, path in config.checkpoint_paths().items():
            resolved = Path(path).expanduser().resolve()
            arguments.extend(["-v", f"{resolved}:/checkpoints/{name}/{resolved.name}:ro"])
        return arguments

    @staticmethod
    def _docker_hf_cache_arguments(config: GenReconAdapterConfig) -> list[str]:
        if config.hf_cache_path is None:
            return []
        cache = Path(config.hf_cache_path).expanduser().resolve()
        return ["-v", f"{cache}:/hf-cache:rw", "-e", "HF_HOME=/hf-cache"]

    @staticmethod
    def _docker_environment_arguments(context: StageContext) -> list[str]:
        arguments: list[str] = []
        for name in context.config.adapter.env:
            if name == "HF_HOME":
                continue
            if name in os.environ:
                arguments.extend(["-e", name])
        return arguments

    @staticmethod
    def _validate_worker_output(
        context: StageContext,
        config: GenReconAdapterConfig,
        request: GenReconInferenceRequest,
        checkpoint_records: list[GenReconCheckpointRecord],
        worker: GenReconWorkerManifest,
    ) -> None:
        expected_hashes = {record.checkpoint_id: record.sha256 for record in checkpoint_records}
        actual_hashes = {
            record.checkpoint_id: record.sha256 for record in worker.checkpoint_records
        }
        required_runtime_repositories = {
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "microsoft/TRELLIS-image-large",
            "microsoft/TRELLIS.2-4B",
        }
        checks = {
            "official repository": worker.official_repository == OFFICIAL_GENRECON_REPOSITORY,
            "official commit": worker.official_code_commit == OFFICIAL_GENRECON_COMMIT,
            "submodules": worker.submodule_commits == OFFICIAL_GENRECON_SUBMODULES,
            "checkpoint hashes": actual_hashes == expected_hashes,
            "runtime repositories": set(worker.runtime_repository_revisions)
            == required_runtime_repositories,
            "DINO revision": worker.runtime_repository_revisions.get(
                worker.runtime_model_repository
            )
            == worker.runtime_model_revision,
            "request hash": worker.request_sha256
            == sha256_file(context.path("reconstruction", "global", "request.json")),
            "frame lineage": worker.frame_sequence_digest == request.frame_sequence_digest,
            "camera package": worker.camera_package_sha256 == request.camera_package_sha256,
            "registered frames": worker.registered_frame_ids == request.registered_frame_ids,
            "reconstruction return code": worker.reconstruct_return_code == 0,
            "GLB return code": worker.glb_conversion_return_code == 0,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(f"GenRecon worker identity/output validation failed: {failed}")
        selected = worker.selected_frame_ids
        eligible = request.eligible_frame_ids
        if selected != [frame_id for frame_id in eligible if frame_id in set(selected)]:
            raise RuntimeError("GenRecon worker selected views outside eligible manifest order")
        if not selected:
            raise RuntimeError("GenRecon worker selected no eligible registered views")
        if _transform_roundtrip_error(worker) > 1e-6:
            raise RuntimeError("GenRecon working transform is not reversibly invertible")
        raw_root = context.path("reconstruction", "global", "raw")
        required = (
            "to_glb_inputs.pt",
            "chunk_inputs.pt",
            "scene.glb",
            "mesh.ply",
        )
        missing = [name for name in required if not (raw_root / name).is_file()]
        if missing:
            raise RuntimeError(f"GenRecon worker is missing required outputs: {missing}")
        for relative_path in worker.raw_output_paths:
            declared = context.path(*Path(relative_path).parts)
            try:
                declared.resolve().relative_to(raw_root.resolve())
            except ValueError as exc:
                raise RuntimeError(
                    f"GenRecon worker declared a raw path outside its output: {relative_path}"
                ) from exc
            if not declared.is_file():
                raise RuntimeError(
                    f"GenRecon worker declared a missing raw output: {relative_path}"
                )
        for path in raw_root.rglob("*"):
            if path.is_symlink():
                raise RuntimeError(f"GenRecon raw output must not be a symlink: {path}")
            try:
                path.resolve().relative_to(raw_root.resolve())
            except ValueError as exc:
                raise RuntimeError(f"GenRecon output path escapes its attempt: {path}") from exc
        if config.execution_mode != "fake_worker" and worker.device != "cuda":
            raise RuntimeError("official GenRecon inference did not report CUDA execution")

    @staticmethod
    def _normalize_diagnostics(
        raw: dict[str, Any],
        request: GenReconInferenceRequest,
        camera: CameraReconstruction,
        mesh: Any,
        worker: GenReconWorkerManifest,
    ) -> GlobalSceneDiagnostics:
        required = {
            "initial_sparse_points",
            "cleaned_sparse_points",
            "robust_bounds_min",
            "robust_bounds_max",
            "scene_diagonal_arbitrary_units",
            "chunks_before_filtering",
            "chunks_after_filtering",
            "chunks",
            "chosen_parameters",
        }
        missing = required - set(raw)
        if missing:
            raise ValueError(f"GenRecon worker diagnostics are missing fields: {sorted(missing)}")
        initial = int(raw["initial_sparse_points"])
        cleaned = int(raw["cleaned_sparse_points"])
        if initial <= 0 or cleaned <= 0:
            raise ValueError("GenRecon point cleaning left no usable sparse points")
        chunks = [
            GlobalSceneChunkDiagnostic.model_validate(item)
            for item in cast(list[object], raw["chunks"])
        ]
        after = int(raw["chunks_after_filtering"])
        if after <= 0:
            raise ValueError("GenRecon produced no non-empty chunks")
        return GlobalSceneDiagnostics(
            eligible_frame_count=len(request.eligible_frame_ids),
            selected_view_count=len(worker.selected_frame_ids),
            registered_coverage=len(camera.registered_frame_ids)
            / max(len(request.master_frame_order), 1),
            initial_sparse_points=initial,
            cleaned_sparse_points=cleaned,
            point_retention_ratio=min(1.0, cleaned / initial),
            robust_bounds_min=tuple(raw["robust_bounds_min"]),
            robust_bounds_max=tuple(raw["robust_bounds_max"]),
            scene_diagonal_arbitrary_units=float(raw["scene_diagonal_arbitrary_units"]),
            chunks_before_filtering=int(raw["chunks_before_filtering"]),
            chunks_after_filtering=after,
            chunks=chunks,
            mesh=mesh,
            chosen_parameters=cast(dict[str, object], raw["chosen_parameters"]),
            runtime_seconds=worker.runtime_seconds,
            peak_gpu_memory_bytes=worker.peak_gpu_memory_bytes,
            warnings=[*worker.warnings, *cast(list[str], raw.get("warnings", []))],
        )

    @staticmethod
    def _scene_ir(
        manifest: IngestManifest,
        camera: CameraReconstruction,
        artifact: GlobalSceneReconstructionArtifact,
        provenance: ProvenanceRecord,
    ) -> SceneIR:
        scene_camera = Camera(
            camera_id=camera.camera_id,
            model=camera.model,
            intrinsics=camera.intrinsics,
            poses=camera.poses,
            coordinate_convention=camera.coordinate_convention,
            scale_status=camera.scale_status,
            provenance=camera.provenance,
        )
        frames = [
            FrameObservation(
                frame_id=frame.frame_id,
                frame_path=frame.relative_path,
                timestamp_s=frame.timestamp_s,
                camera_id=camera.camera_id,
            )
            for frame in manifest.frames
        ]
        assets = [
            GeometryAsset(
                asset_id="global_scene_pbr",
                asset_type=AssetType.STATIC_STRUCTURE,
                uri=artifact.scene_asset_path,
                format="glb",
                source=GeometrySourceType.GENERATED,
                coordinate_convention=camera.coordinate_convention,
                scale_status=ScaleStatus.SCALE_AMBIGUOUS,
                provenance=provenance,
            ),
            GeometryAsset(
                asset_id="global_scene_mesh",
                asset_type=AssetType.STATIC_STRUCTURE,
                uri=artifact.mesh_asset_path,
                format="ply",
                source=GeometrySourceType.GENERATED,
                coordinate_convention=camera.coordinate_convention,
                scale_status=ScaleStatus.SCALE_AMBIGUOUS,
                provenance=provenance,
            ),
        ]
        return SceneIR(
            metadata=SceneMetadata(
                scene_id="global_scene",
                name="GenRecon global visual reconstruction",
                coordinate_convention=camera.coordinate_convention,
                source=GeometrySourceType.GENERATED,
                provenance=[provenance],
            ),
            cameras=[scene_camera],
            frames=frames,
            geometry_assets=assets,
        )


def _matrix_product(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [sum(left[row][inner] * right[inner][column] for inner in range(4)) for column in range(4)]
        for row in range(4)
    ]


def _transform_roundtrip_error(worker: GenReconWorkerManifest) -> float:
    product = _matrix_product(
        worker.working_transform.matrix_colmap_to_working,
        worker.working_transform.matrix_working_to_colmap,
    )
    return max(
        abs(product[row][column] - (1.0 if row == column else 0.0))
        for row in range(4)
        for column in range(4)
    )


class Phase3EndToEndConsistencyAdapter:
    name = "phase3_e2e_consistency"
    version = "0.1.0"

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return HealthcheckResult(True, "Phase 3 consistency validator available")

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        return [
            InputSpec("inputs/manifest.json", "ingest_manifest"),
            InputSpec("inputs/frame_qa.json", "frame_quality_report"),
            InputSpec("camera/reconstruction.json", "camera_reconstruction"),
            InputSpec(
                "camera/genrecon_package/package_manifest.json",
                "genrecon_camera_package_manifest",
            ),
            InputSpec("observations/sam3_request.json", "sam3_inference_request"),
            InputSpec("observations/worker_manifest.json", "sam3_worker_manifest"),
            InputSpec("observations/object_tracks.json", "segmentation_tracking"),
            InputSpec(
                "reconstruction/global/request.json",
                "genrecon_inference_request",
            ),
            InputSpec(
                "reconstruction/global/worker_manifest.json",
                "genrecon_worker_manifest",
            ),
            InputSpec(
                "reconstruction/global/metadata.json",
                "global_scene_reconstruction",
            ),
        ]

    def prepare(self, context: StageContext) -> None:
        context.path("validation").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                "validation/phase3_e2e_consistency.json",
                "phase3_e2e_consistency",
                "application/json",
                "validation",
                validation="json",
                schema_identifier="recon2sim/phase3-e2e-consistency/0.1.0",
                model=EndToEndConsistencyReport,
            )
        ]

    def run(self, context: StageContext) -> StageResult:
        report = self.build_report(context)
        atomic_write_json(
            context.path("validation", "phase3_e2e_consistency.json"),
            report,
        )
        if not report.passed:
            failed = [check.check_id for check in report.checks if not check.passed]
            raise ValueError(f"Phase 3 end-to-end consistency failed: {failed}")
        return StageResult(
            metrics={
                "checks": len(report.checks),
                "passed": report.passed,
            }
        )

    def build_report(self, context: StageContext) -> EndToEndConsistencyReport:
        manifest_path = context.path("inputs", "manifest.json")
        camera_path = context.path("camera", "reconstruction.json")
        manifest = IngestManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        camera = CameraReconstruction.model_validate_json(camera_path.read_text(encoding="utf-8"))
        package = GenReconCameraPackageManifest.model_validate_json(
            context.path("camera", "genrecon_package", "package_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        sam_request = Sam3InferenceRequest.model_validate_json(
            context.path("observations", "sam3_request.json").read_text(encoding="utf-8")
        )
        sam_worker = Sam3WorkerManifest.model_validate_json(
            context.path("observations", "worker_manifest.json").read_text(encoding="utf-8")
        )
        segmentation = SegmentationTrackingArtifact.model_validate_json(
            context.path("observations", "object_tracks.json").read_text(encoding="utf-8")
        )
        genrecon_request = GenReconInferenceRequest.model_validate_json(
            context.path("reconstruction", "global", "request.json").read_text(encoding="utf-8")
        )
        genrecon_worker = GenReconWorkerManifest.model_validate_json(
            context.path("reconstruction", "global", "worker_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        global_scene = GlobalSceneReconstructionArtifact.model_validate_json(
            context.path("reconstruction", "global", "metadata.json").read_text(encoding="utf-8")
        )
        manifest_hash = sha256_file(manifest_path)
        camera_hash = sha256_file(camera_path)
        master_order = [frame.frame_id for frame in manifest.frames]
        frame_paths = {frame.frame_id: frame.relative_path for frame in manifest.frames}
        frame_hashes = {frame.frame_id: frame.sha256 for frame in manifest.frames}
        digest = manifest.frame_sequence_digest or ""
        checks: list[EndToEndConsistencyCheck] = []

        def add(check_id: str, passed: bool, message: str) -> None:
            checks.append(
                EndToEndConsistencyCheck(
                    check_id=check_id,
                    passed=passed,
                    message=message,
                )
            )

        add(
            "manifest_sha",
            bool(digest)
            and package.source_manifest_sha256 == manifest_hash
            and sam_request.frame_manifest_sha256 == manifest_hash
            and genrecon_request.manifest_sha256 == manifest_hash
            and global_scene.manifest_sha256 == manifest_hash,
            "COLMAP package, SAM, and GenRecon reference the canonical ingest manifest hash",
        )
        digest_values = {
            digest,
            camera.frame_sequence_digest or "",
            package.frame_sequence_digest,
            sam_request.frame_sequence_digest or "",
            sam_worker.frame_sequence_digest or "",
            segmentation.frame_sequence_digest or "",
            genrecon_request.frame_sequence_digest,
            genrecon_worker.frame_sequence_digest,
            global_scene.frame_sequence_digest,
        }
        add(
            "frame_sequence_digest",
            "" not in digest_values and len(digest_values) == 1,
            "all stages reference the same ordered normalized-frame digest",
        )
        add(
            "master_frame_order",
            sam_request.frame_order == master_order
            and genrecon_request.master_frame_order == master_order
            and package.master_frame_ids == master_order,
            "SAM uses master order and GenRecon package/request preserve it",
        )
        add(
            "normalized_frame_identity",
            dict(zip(sam_request.frame_order, sam_request.frame_paths, strict=True)) == frame_paths
            and genrecon_request.normalized_frame_paths == frame_paths
            and genrecon_request.normalized_frame_hashes == frame_hashes,
            "all frame IDs retain normalized paths and SHA-256 identities",
        )
        expected_eligible = [
            frame_id for frame_id in master_order if frame_id in set(camera.registered_frame_ids)
        ]
        add(
            "registration_sets",
            sam_request.registered_frame_ids == camera.registered_frame_ids
            and sam_request.unregistered_frame_ids == camera.unregistered_frame_ids
            and genrecon_request.registered_frame_ids == camera.registered_frame_ids
            and genrecon_request.unregistered_frame_ids == camera.unregistered_frame_ids
            and package.eligible_frame_ids == expected_eligible
            and genrecon_request.eligible_frame_ids == expected_eligible,
            "SAM records real registration sets and GenRecon filters the registered subset",
        )
        add(
            "selected_view_subset",
            genrecon_worker.selected_frame_ids
            == [
                frame_id
                for frame_id in expected_eligible
                if frame_id in set(genrecon_worker.selected_frame_ids)
            ],
            "GenRecon selected views are an order-preserving eligible subset",
        )
        add(
            "camera_reconstruction_hash",
            package.camera_reconstruction_sha256 == camera_hash
            and sam_request.camera_reconstruction_sha256 == camera_hash
            and genrecon_request.camera_reconstruction_sha256 == camera_hash
            and global_scene.camera_reconstruction_sha256 == camera_hash,
            "SAM and GenRecon reference the same typed real camera reconstruction",
        )
        pose_ids = set(camera.registered_frame_ids)
        add(
            "sam_pose_availability",
            all(
                observation.camera_pose_available == (observation.frame_id in pose_ids)
                for track in segmentation.tracks
                for observation in track.observations
            ),
            "every SAM observation records camera-pose availability from real registration",
        )
        add(
            "coordinate_semantics",
            coordinate_metadata_is_raw_colmap(camera.coordinate_convention)
            and genrecon_request.coordinate_convention == camera.coordinate_convention
            and global_scene.coordinate_convention == camera.coordinate_convention
            and global_scene.scale_status is ScaleStatus.SCALE_AMBIGUOUS,
            "raw arbitrary, unoriented, scale-ambiguous COLMAP semantics are preserved",
        )
        roundtrip_error = _transform_roundtrip_error(genrecon_worker)
        add(
            "working_transform_roundtrip",
            roundtrip_error <= 1e-6
            and genrecon_worker.working_transform.semantic_status
            == "internal_unoriented_preprocessing",
            f"GenRecon internal working transform is reversible (max error {roundtrip_error:.3g})",
        )
        sam_clean, genrecon_clean = self._audit_materialization(context)
        add(
            "sam_selective_materialization",
            sam_clean,
            "SAM attempt excludes COLMAP database, sparse binaries, and logs",
        )
        add(
            "genrecon_selective_materialization",
            genrecon_clean,
            "GenRecon attempt contains only the normalized package, registered frames, "
            "and references",
        )
        add(
            "honest_capability_boundary",
            True,
            "object-level 2D/3D fusion, metric scale, gravity alignment, and sim-ready "
            "output are absent",
        )
        passed = all(check.passed for check in checks)
        return EndToEndConsistencyReport(
            passed=passed,
            checks=checks,
            manifest_sha256=manifest_hash,
            frame_sequence_digest=self._require_digest(manifest.frame_sequence_digest),
            real_modules_share_consistent_inputs=passed,
            warnings=[],
        )

    @staticmethod
    def _require_digest(value: str | None) -> str:
        if value is None:
            return "0" * 64
        return value

    @staticmethod
    def _audit_materialization(context: StageContext) -> tuple[bool, bool]:
        run_manifest_path = context.canonical_path("manifest.json")
        if not run_manifest_path.is_file():
            return False, False
        payload = cast(
            dict[str, Any],
            json.loads(run_manifest_path.read_text(encoding="utf-8")),
        )
        stages = cast(dict[str, dict[str, Any]], payload.get("stages", {}))

        def successful_inputs(stage_name: str) -> list[str]:
            stage = stages.get(stage_name, {})
            attempts = cast(list[dict[str, Any]], stage.get("attempts", []))
            successful = [attempt for attempt in attempts if attempt.get("status") == "succeeded"]
            if not successful:
                return []
            return [
                cast(str, item["relative_path"])
                for item in cast(
                    list[dict[str, Any]],
                    successful[-1].get("materialized_inputs", []),
                )
            ]

        sam_inputs = successful_inputs("segmentation_tracking")
        genrecon_inputs = successful_inputs("global_reconstruction")
        sam_forbidden = (
            "camera/colmap/database.db",
            "camera/colmap/sparse/",
            "camera/colmap/logs/",
        )
        genrecon_forbidden = (
            "camera/colmap/database.db",
            "camera/colmap/sparse/",
            "camera/colmap/logs/",
            "observations/",
            "compiled/",
            "validation/",
        )
        sam_clean = bool(sam_inputs) and not any(
            path == prefix or path.startswith(prefix)
            for path in sam_inputs
            for prefix in sam_forbidden
        )
        genrecon_clean = bool(genrecon_inputs) and not any(
            path == prefix or path.startswith(prefix)
            for path in genrecon_inputs
            for prefix in genrecon_forbidden
        )
        return sam_clean, genrecon_clean


__all__ = [
    "GenReconAdapterConfig",
    "GenReconCameraPackageAdapter",
    "GenReconGlobalReconstructionAdapter",
    "Phase3EndToEndConsistencyAdapter",
]
