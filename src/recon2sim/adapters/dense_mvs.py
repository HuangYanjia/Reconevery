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
    CameraDiagnostics,
    CameraReconstruction,
    DenseDepthManifest,
    DenseFusionArtifact,
    DenseMVSDiagnostics,
    DenseMVSRequest,
    DenseMVSWorkerManifest,
    DenseSparseModelFile,
    DenseUndistortionManifest,
    DenseWorkspaceManifest,
    IngestManifest,
)
from recon2sim.dense_mvs import (
    ply_counts,
    sha256_file,
)
from recon2sim.genrecon import coordinate_metadata_is_raw_colmap
from recon2sim.ir import StrictModel
from recon2sim.storage import atomic_write_json

DENSE_MVS_WORKER_VERSION = "0.1.0"


def _resolve_python(value: str) -> str | None:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        return str(candidate.absolute()) if candidate.is_file() else None
    return shutil.which(value)


class DenseMVSAdapterConfig(StrictModel):
    execution_mode: Literal["local_worker", "docker", "fake_worker"]
    worker_python: str = "python"
    worker_module: str = "dense_mvs_worker"
    worker_script: str | None = None
    docker_executable: str = "docker"
    docker_image: str = "reconevery/dense-mvs:phase5a"
    executable: str = "colmap"
    official_repository: Literal["https://github.com/colmap/colmap"] = (
        "https://github.com/colmap/colmap"
    )
    official_version: Literal["4.0.4"] = "4.0.4"
    official_commit: Literal["9c23f6942fe69962e06030905e77067c8673382f"] = (
        "9c23f6942fe69962e06030905e77067c8673382f"
    )
    use_gpu: bool = True
    max_image_size: int = Field(default=1600, gt=0)
    patchmatch_cache_size_gb: float = Field(default=16, gt=0)
    geom_consistency: bool = True
    source_view_selection: Literal["auto", "sequential_neighbors", "explicit"] = "auto"
    source_view_ids: dict[str, list[str]] = Field(default_factory=dict)
    sequential_neighbor_count: int = Field(default=10, ge=1)
    min_num_pixels: int = Field(default=5, gt=0)
    max_reproj_error: float = Field(default=2.0, gt=0)
    max_depth_error: float = Field(default=0.01, gt=0)
    max_normal_error: float = Field(default=10.0, gt=0)
    check_num_images: int = Field(default=50, gt=0)
    rgb_remap_tolerance: float = Field(default=3.0, ge=0)
    fake_mode: str = "success"

    @model_validator(mode="after")
    def validate_execution(self) -> DenseMVSAdapterConfig:
        if self.execution_mode == "fake_worker":
            if self.worker_script is None:
                raise ValueError("fake dense MVS execution requires worker_script")
            return self
        if not self.use_gpu:
            raise ValueError(
                "official COLMAP 4.0.4 PatchMatchStereo requires CUDA; "
                "use fake_worker for CPU-only protocol tests"
            )
        if self.execution_mode == "local_worker":
            python = _resolve_python(self.worker_python)
            if python is None:
                raise ValueError(f"configured worker Python {self.worker_python!r} was not found")
            root = Path(python).absolute().parent.parent
            if not (root / "pyvenv.cfg").is_file() and not (root / "conda-meta").is_dir():
                raise ValueError("dense MVS worker_python must be in an isolated environment")
            if root.resolve() == Path(sys.prefix).resolve():
                raise ValueError("dense MVS worker must not use the Reconevery core environment")
            executable = Path(self.executable)
            if executable.is_absolute() and not executable.is_file():
                raise ValueError(f"configured COLMAP executable does not exist: {executable}")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None or visible.strip().lower() in {"", "-1", "none", "void"}:
            raise ValueError("GPU dense MVS requires CUDA_VISIBLE_DEVICES")
        if self.source_view_selection == "explicit" and not self.source_view_ids:
            raise ValueError("explicit source-view selection requires source_view_ids")
        return self


class DenseMVSAdapter:
    name = "dense_mvs"
    version = "0.1.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        diagnostics_path = context.canonical_path("camera", "diagnostics.json")
        diagnostics = CameraDiagnostics.model_validate_json(
            diagnostics_path.read_text(encoding="utf-8")
        )
        if diagnostics.selected_model is None:
            raise ValueError("dense MVS requires a selected sparse COLMAP model")
        camera = CameraReconstruction.model_validate_json(
            context.canonical_path("camera", "reconstruction.json").read_text(encoding="utf-8")
        )
        manifest = IngestManifest.model_validate_json(
            context.canonical_path("inputs", "manifest.json").read_text(encoding="utf-8")
        )
        path_by_id = {frame.frame_id: frame.relative_path for frame in manifest.frames}
        specs = [
            InputSpec("inputs/manifest.json", "ingest_manifest"),
            InputSpec("camera/reconstruction.json", "camera_reconstruction"),
            InputSpec("camera/diagnostics.json", "camera_diagnostics"),
        ]
        model_root = f"camera/colmap/sparse/{diagnostics.selected_model}"
        specs.extend(
            InputSpec(f"{model_root}/{name}", "colmap_raw_model")
            for name in ("cameras.bin", "images.bin", "points3D.bin")
        )
        specs.extend(
            InputSpec(path_by_id[frame_id], "input_frame")
            for frame_id in camera.registered_frame_ids
        )
        return specs

    def prepare(self, context: StageContext) -> None:
        for path in (
            context.path("reconstruction", "dense", "raw", "logs"),
            context.path("reconstruction", "dense", "workspace", "images"),
            context.path("reconstruction", "dense", "workspace", "sparse"),
            context.path("reconstruction", "dense", "workspace", "stereo"),
            context.path("reconstruction", "dense", "previews"),
        ):
            path.mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        root = "reconstruction/dense"
        models: list[tuple[str, str, type[Any]]] = [
            ("request.json", "dense_mvs_request", DenseMVSRequest),
            ("worker_manifest.json", "dense_mvs_worker_manifest", DenseMVSWorkerManifest),
            ("workspace_manifest.json", "dense_workspace_manifest", DenseWorkspaceManifest),
            (
                "undistortion_manifest.json",
                "dense_undistortion_manifest",
                DenseUndistortionManifest,
            ),
            ("depth_manifest.json", "dense_depth_manifest", DenseDepthManifest),
            ("fusion.json", "dense_fusion_artifact", DenseFusionArtifact),
            ("diagnostics.json", "dense_mvs_diagnostics", DenseMVSDiagnostics),
        ]
        outputs = [
            OutputSpec(
                f"{root}/{filename}",
                artifact_type,
                "application/json",
                "colmap_dense",
                validation="json",
                schema_identifier=f"recon2sim/{artifact_type.replace('_', '-')}/0.1.0",
                model=model,
            )
            for filename, artifact_type, model in models
        ]
        outputs.append(
            OutputSpec(
                f"{root}/fused.ply",
                "dense_fused_point_cloud",
                "model/ply",
                "colmap_dense",
            )
        )
        for name in (
            "depth_contact_sheet",
            "normal_contact_sheet",
            "consistency_contact_sheet",
            "fused_point_cloud",
            "camera_dense_coverage",
        ):
            outputs.append(
                OutputSpec(
                    f"{root}/previews/{name}.png",
                    "dense_mvs_preview",
                    "image/png",
                    "colmap_dense",
                    validation="png",
                )
            )
        return outputs

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        if context is None:
            return HealthcheckResult(False, "dense MVS healthcheck requires --config")
        try:
            config = DenseMVSAdapterConfig.model_validate(context.config.adapter.config)
        except ValueError as exc:
            return HealthcheckResult(False, f"invalid dense MVS configuration: {exc}")
        payload = {
            "worker_version": DENSE_MVS_WORKER_VERSION,
            "executable": config.executable,
            "official_version": config.official_version,
            "official_commit": config.official_commit,
            "use_gpu": config.use_gpu,
        }
        with tempfile.TemporaryDirectory(prefix="reconevery-dense-health-") as temp:
            path = Path(temp) / "config.json"
            atomic_write_json(path, payload)
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
                return HealthcheckResult(False, f"dense MVS healthcheck failed: {exc}")
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
            return HealthcheckResult(result.returncode == 0, output or "dense MVS unavailable")

    @staticmethod
    def _docker_healthcheck_command(
        config: DenseMVSAdapterConfig, config_path: Path
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
        gpu = ["--gpus", "all"] if config.use_gpu else []
        return [
            docker,
            "run",
            "--rm",
            *gpu,
            "-v",
            f"{config_path.parent.resolve()}:/health:ro",
            config.docker_image,
            "healthcheck",
            "--config",
            "/health/config.json",
        ]

    def run(self, context: StageContext) -> StageResult:
        config = DenseMVSAdapterConfig.model_validate(context.config.adapter.config)
        manifest_path = context.path("inputs", "manifest.json")
        camera_path = context.path("camera", "reconstruction.json")
        manifest = IngestManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        camera = CameraReconstruction.model_validate_json(camera_path.read_text(encoding="utf-8"))
        diagnostics = CameraDiagnostics.model_validate_json(
            context.path("camera", "diagnostics.json").read_text(encoding="utf-8")
        )
        if manifest.frame_sequence_digest is None:
            raise ValueError("dense MVS requires a frame-sequence digest")
        if camera.frame_sequence_digest != manifest.frame_sequence_digest:
            raise ValueError("dense MVS camera lineage does not match ingest")
        if not coordinate_metadata_is_raw_colmap(camera.coordinate_convention):
            raise ValueError("dense MVS requires raw arbitrary COLMAP camera semantics")
        if diagnostics.selected_model is None:
            raise ValueError("dense MVS requires a selected sparse model")
        registered = set(camera.registered_frame_ids)
        ordered_registered = [
            frame.frame_id for frame in manifest.frames if frame.frame_id in registered
        ]
        model_root = context.path("camera", "colmap", "sparse", str(diagnostics.selected_model))
        request = DenseMVSRequest(
            run_id=context.canonical_run_dir.name,
            manifest_sha256=sha256_file(manifest_path),
            frame_sequence_digest=manifest.frame_sequence_digest,
            master_frame_order=[frame.frame_id for frame in manifest.frames],
            registered_frame_ids=ordered_registered,
            unregistered_frame_ids=[
                frame.frame_id for frame in manifest.frames if frame.frame_id not in registered
            ],
            normalized_frame_paths={
                frame.frame_id: frame.relative_path for frame in manifest.frames
            },
            normalized_frame_hashes={frame.frame_id: frame.sha256 for frame in manifest.frames},
            camera_reconstruction_sha256=sha256_file(camera_path),
            selected_sparse_model_files=[
                DenseSparseModelFile(
                    relative_path=(f"camera/colmap/sparse/{diagnostics.selected_model}/{name}"),
                    sha256=sha256_file(model_root / name),
                )
                for name in ("cameras.bin", "images.bin", "points3D.bin")
            ],
            official_colmap_repository=config.official_repository,
            official_colmap_version=config.official_version,
            official_colmap_commit=config.official_commit,
            executable=config.executable,
            undistortion_configuration={
                "max_image_size": config.max_image_size,
                "rgb_remap_tolerance": config.rgb_remap_tolerance,
            },
            patchmatch_configuration={
                "geom_consistency": config.geom_consistency,
                "use_gpu": config.use_gpu,
                "cache_size_gb": config.patchmatch_cache_size_gb,
                "source_view_selection": config.source_view_selection,
                "source_view_ids": config.source_view_ids,
                "sequential_neighbor_count": config.sequential_neighbor_count,
                "fake_mode": config.fake_mode,
            },
            fusion_configuration={
                "min_num_pixels": config.min_num_pixels,
                "max_reproj_error": config.max_reproj_error,
                "max_depth_error": config.max_depth_error,
                "max_normal_error": config.max_normal_error,
                "check_num_images": config.check_num_images,
            },
            seed=context.seed,
        )
        request_path = context.path("reconstruction", "dense", "request.json")
        atomic_write_json(request_path, request)
        command = self._inference_command(context, config)
        try:
            run_process(
                command,
                context=context,
                name="dense_mvs_worker",
                log_directory="reconstruction/dense/raw/logs",
            )
        except ProcessExecutionError as exc:
            raise self._worker_failure(exc) from exc
        root = context.path("reconstruction", "dense")
        worker = self._model(root / "worker_manifest.json", DenseMVSWorkerManifest)
        workspace = self._model(root / "workspace_manifest.json", DenseWorkspaceManifest)
        undistortion = self._model(root / "undistortion_manifest.json", DenseUndistortionManifest)
        depth = self._model(root / "depth_manifest.json", DenseDepthManifest)
        fusion = self._model(root / "fusion.json", DenseFusionArtifact)
        diagnostics_out = self._model(root / "diagnostics.json", DenseMVSDiagnostics)
        expected_hashes = {
            "request_sha256": sha256_file(request_path),
            "manifest_sha256": request.manifest_sha256,
            "frame_sequence_digest": request.frame_sequence_digest,
            "camera_reconstruction_sha256": request.camera_reconstruction_sha256,
        }
        for key, expected in expected_hashes.items():
            if getattr(worker, key) != expected:
                raise RuntimeError(f"dense MVS worker {key} does not match request")
        if workspace.registered_frame_ids != request.registered_frame_ids:
            raise RuntimeError("dense MVS worker changed registered frame order")
        if len(undistortion.records) != len(request.registered_frame_ids):
            raise RuntimeError("dense MVS worker omitted undistortion records")
        known = set(request.registered_frame_ids)
        if any(record.frame_id not in known for record in depth.records):
            raise RuntimeError("dense MVS depth manifest references an unknown frame")
        fused_path = root / "fused.ply"
        point_count, _ = ply_counts(fused_path)
        if (
            point_count != fusion.point_count
            or sha256_file(fused_path) != fusion.fused_point_cloud_sha256
        ):
            raise RuntimeError("dense fused point cloud does not match its typed artifact")
        if fusion.coordinate_convention != camera.coordinate_convention:
            raise RuntimeError("dense MVS changed camera coordinate semantics")
        if diagnostics_out.successful_depth_map_count != len(depth.records):
            raise RuntimeError("dense MVS diagnostics and depth manifest disagree")
        dynamic = [
            OutputSpec(
                path.relative_to(context.run_dir).as_posix(),
                "dense_mvs_workspace_file",
                "application/octet-stream",
                "colmap_dense",
            )
            for path in sorted((root / "workspace").rglob("*"))
            if path.is_file()
        ]
        return StageResult(
            outputs=dynamic,
            metrics={
                "registered_frames": len(request.registered_frame_ids),
                "depth_maps": len(depth.records),
                "failed_depth_maps": len(depth.failed_frame_ids),
                "fused_points": fusion.point_count,
            },
        )

    @staticmethod
    def _model(path: Path, model: Any) -> Any:
        if not path.is_file():
            raise RuntimeError(f"dense MVS worker omitted {path.name}")
        try:
            return model.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise RuntimeError(f"dense MVS worker produced malformed {path.name}: {exc}") from exc

    def _inference_command(self, context: StageContext, config: DenseMVSAdapterConfig) -> list[str]:
        request = Path("reconstruction/dense/request.json")
        if config.execution_mode != "docker":
            command = self._local_command(config, "infer", request)
            if isinstance(command, str):
                raise RuntimeError(command)
            return [
                *command,
                "--input-root",
                str(context.run_dir.resolve()),
                "--output-dir",
                str(context.path("reconstruction", "dense").resolve()),
            ]
        docker = resolve_executable(config.docker_executable)
        if docker is None:
            raise RuntimeError("Docker executable was not found")
        user = (
            ["--user", f"{os.getuid()}:{os.getgid()}"]
            if hasattr(os, "getuid") and hasattr(os, "getgid")
            else []
        )
        gpu = ["--gpus", "all"] if config.use_gpu else []
        return [
            docker,
            "run",
            "--rm",
            *gpu,
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
            "/workspace/reconstruction/dense",
        ]

    @staticmethod
    def _local_command(config: DenseMVSAdapterConfig, action: str, path: Path) -> list[str] | str:
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
                return f"fake dense MVS worker does not exist: {script}"
            return [python, str(script.resolve()), action, option, str(path)]
        return [python, "-m", config.worker_module, action, option, str(path)]

    @staticmethod
    def _worker_failure(exc: ProcessExecutionError) -> RuntimeError:
        stderr = exc.result.stderr.lower()
        if "out of memory" in stderr or "cuda oom" in stderr:
            return RuntimeError(
                "COLMAP PatchMatch ran out of GPU memory; lower max_image_size or cache size"
            )
        if "cuda" in stderr and "unavailable" in stderr:
            return RuntimeError("COLMAP dense MVS could not access the configured CUDA device")
        return RuntimeError(str(exc))
