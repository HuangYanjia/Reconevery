from __future__ import annotations

import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from recon2sim.adapters.base import HealthcheckResult, OutputSpec, StageContext, StageResult
from recon2sim.adapters.ingest import (
    ProcessExecutionError,
    ProcessResult,
    executable_version,
    resolve_executable,
    run_process,
)
from recon2sim.artifacts import (
    CameraDiagnostics,
    CameraReconstruction,
    ColmapCommandRecord,
    ColmapWorkspaceManifest,
    IngestManifest,
    SparseModelDiagnostics,
)
from recon2sim.colmap import camera_intrinsics, colmap_pose_to_world_from_camera, read_model
from recon2sim.colmap.model import ColmapModel, ColmapModelError
from recon2sim.ir import (
    AlignmentStatus,
    CameraAxes,
    CameraPose,
    ConfidenceRecord,
    CoordinateConvention,
    GeometrySourceType,
    LinearUnits,
    ProvenanceRecord,
    ScaleStatus,
    StrictModel,
    WorldFrame,
)
from recon2sim.storage import atomic_write_json


class MapperConfig(StrictModel):
    multiple_models: bool = True


class SequentialMatcherConfig(StrictModel):
    overlap: int = Field(default=10, gt=0)
    loop_detection: bool = False


class ColmapAdapterConfig(StrictModel):
    execution_mode: Literal["local", "docker"] = "local"
    executable: str = "colmap"
    docker_executable: str = "docker"
    docker_image: str = "reconevery/colmap:phase1"
    docker_user: str | None = None
    matcher: Literal["sequential", "exhaustive"] = "sequential"
    camera_model: str = "OPENCV"
    single_camera: bool = True
    use_gpu: bool = True
    min_registered_frames: int = Field(default=8, gt=0)
    min_registration_ratio: float = Field(default=0.4, gt=0, le=1)
    mapper: MapperConfig = Field(default_factory=MapperConfig)
    sequential_matcher: SequentialMatcherConfig = Field(default_factory=SequentialMatcherConfig)

    @model_validator(mode="after")
    def docker_is_cpu_only(self) -> Self:
        if self.execution_mode == "docker" and self.use_gpu:
            raise ValueError("Docker COLMAP execution currently requires use_gpu=false")
        return self


class ColmapExecutionError(RuntimeError):
    def __init__(self, message: str, *, failed_subcommand: str) -> None:
        super().__init__(message)
        self.details = {"failed_subcommand": failed_subcommand}


class SparseModelThresholdError(ColmapModelError):
    def __init__(
        self,
        message: str,
        diagnostics: list[SparseModelDiagnostics],
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class SparseModelCandidate:
    model_id: str
    path: Path
    model: ColmapModel
    registered_frames: int
    registration_ratio: float
    sparse_points: int
    average_reprojection_error: float | None


def _model_id_key(model_id: str) -> tuple[int, int, str]:
    try:
        return (0, int(model_id), model_id)
    except ValueError:
        return (1, 0, model_id)


def rank_sparse_models(
    candidates: list[SparseModelCandidate],
    *,
    min_registered_frames: int,
    min_registration_ratio: float,
) -> tuple[SparseModelCandidate, list[SparseModelDiagnostics]]:
    diagnostics = [
        SparseModelDiagnostics(
            model_id=candidate.model_id,
            registered_frames=candidate.registered_frames,
            registration_ratio=candidate.registration_ratio,
            sparse_points=candidate.sparse_points,
            average_reprojection_error=candidate.average_reprojection_error,
            rejection_reason=(
                f"registered_frames<{min_registered_frames}"
                if candidate.registered_frames < min_registered_frames
                else (
                    f"registration_ratio<{min_registration_ratio}"
                    if candidate.registration_ratio < min_registration_ratio
                    else None
                )
            ),
        )
        for candidate in candidates
    ]
    eligible = [
        candidate
        for candidate in candidates
        if candidate.registered_frames >= min_registered_frames
        and candidate.registration_ratio >= min_registration_ratio
    ]
    if not eligible:
        summary = ", ".join(
            f"{candidate.model_id}: {candidate.registered_frames} frames "
            f"({candidate.registration_ratio:.3f}), {candidate.sparse_points} points"
            for candidate in candidates
        )
        raise SparseModelThresholdError(
            (
                "no COLMAP sparse model meets registration thresholds "
                f"(min_registered_frames={min_registered_frames}, "
                f"min_registration_ratio={min_registration_ratio}); "
                f"candidates: {summary or 'none'}"
            ),
            diagnostics,
        )
    selected = sorted(
        eligible,
        key=lambda candidate: (
            -candidate.registered_frames,
            -candidate.sparse_points,
            candidate.average_reprojection_error
            if candidate.average_reprojection_error is not None
            else math.inf,
            _model_id_key(candidate.model_id),
        ),
    )[0]
    for item in diagnostics:
        if item.model_id == selected.model_id:
            item.selected = True
            item.rejection_reason = None
    return selected, diagnostics


def camera_confidence(
    *,
    registration_ratio: float,
    registered_frames: int,
    sparse_points: int,
    average_reprojection_error: float | None,
) -> float:
    registration = 0.6 * registration_ratio
    frame_support = 0.15 * min(registered_frames / 20.0, 1.0)
    point_support = 0.15 * min(math.log10(sparse_points + 1) / 4.0, 1.0)
    error_support = (
        0.1 / (1.0 + average_reprojection_error) if average_reprojection_error is not None else 0.0
    )
    return min(1.0, registration + frame_support + point_support + error_support)


class ColmapCameraRecoveryAdapter:
    name = "colmap_camera_recovery"
    version = "0.1.1"

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        config = (
            ColmapAdapterConfig.model_validate(context.config.adapter.config)
            if context is not None
            else ColmapAdapterConfig()
        )
        if config.execution_mode == "docker":
            docker = resolve_executable(config.docker_executable)
            if docker is None:
                return HealthcheckResult(
                    False,
                    f"Docker executable {config.docker_executable!r} was not found; install Docker",
                )
            try:
                docker_version = subprocess.run(
                    [docker, "version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return HealthcheckResult(False, f"docker version failed: {exc}")
            if docker_version.returncode != 0:
                return HealthcheckResult(
                    False,
                    f"docker version failed: {docker_version.stderr.strip()}",
                )
            try:
                inspect = subprocess.run(
                    [
                        docker,
                        "image",
                        "inspect",
                        config.docker_image,
                        "--format",
                        "{{.Id}}",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return HealthcheckResult(False, f"docker image inspect failed: {exc}")
            if inspect.returncode != 0:
                return HealthcheckResult(
                    False,
                    f"Docker image {config.docker_image!r} is unavailable; build or pull it",
                )
            image_identifier = inspect.stdout.strip() or "identifier unavailable"
            try:
                in_container = subprocess.run(
                    [
                        docker,
                        "run",
                        "--rm",
                        *self._docker_user_arguments(config),
                        config.docker_image,
                        "colmap",
                        "-h",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return HealthcheckResult(False, f"in-container colmap -h failed: {exc}")
            if in_container.returncode != 0:
                return HealthcheckResult(
                    False,
                    "in-container colmap -h failed: "
                    f"{(in_container.stderr or in_container.stdout).strip()}",
                )
            first_line = (in_container.stdout or in_container.stderr).splitlines()
            version = first_line[0] if first_line else "COLMAP help available"
            return HealthcheckResult(
                True,
                f"docker={docker}; image={config.docker_image}; id={image_identifier}; {version}",
            )

        executable = resolve_executable(config.executable)
        if executable is None:
            return HealthcheckResult(
                False,
                f"COLMAP executable {config.executable!r} was not found; install COLMAP or set "
                "stages.camera_recovery.adapter.config.executable",
            )
        ok, version = executable_version(executable, "-h")
        if not ok:
            return HealthcheckResult(False, f"COLMAP healthcheck failed: {version}")
        return HealthcheckResult(True, f"colmap={executable} ({version})")

    def prepare(self, context: StageContext) -> None:
        context.path("camera", "colmap", "sparse").mkdir(parents=True, exist_ok=True)
        context.path("camera", "colmap", "logs").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                "camera/reconstruction.json",
                "camera_reconstruction",
                "application/json",
                "colmap",
                validation="json",
                schema_identifier="recon2sim/camera-reconstruction/0.3.0",
                model=CameraReconstruction,
            ),
            OutputSpec(
                "camera/diagnostics.json",
                "camera_diagnostics",
                "application/json",
                "colmap",
                validation="json",
                schema_identifier="recon2sim/camera-diagnostics/0.1.0",
                model=CameraDiagnostics,
            ),
            OutputSpec(
                "camera/colmap/workspace_manifest.json",
                "colmap_workspace_manifest",
                "application/json",
                "colmap",
                validation="json",
                schema_identifier="recon2sim/colmap-workspace-manifest/0.2.0",
                model=ColmapWorkspaceManifest,
            ),
        ]

    def run(self, context: StageContext) -> StageResult:
        config = ColmapAdapterConfig.model_validate(context.config.adapter.config)
        manifest = IngestManifest.model_validate_json(
            context.path("inputs", "manifest.json").read_text(encoding="utf-8")
        )
        self._validate_frames(context, manifest)
        commands: list[ColmapCommandRecord] = []
        executable_or_image, tool_version, image_identifier = self._tool_identity(config)
        workspace_manifest = ColmapWorkspaceManifest(
            execution_mode=config.execution_mode,
            executable_or_image=executable_or_image,
            tool_version=tool_version,
            image_identifier=image_identifier,
            database_path="camera/colmap/database.db",
            image_path="frames",
            sparse_path="camera/colmap/sparse",
        )
        self._write_workspace_manifest(context, workspace_manifest)

        command_steps = self._commands(config, context, tool_version=tool_version)
        for name, command in command_steps:
            try:
                result = run_process(
                    command,
                    context=context,
                    name=name,
                    log_directory="camera/colmap/logs",
                )
            except ProcessExecutionError as exc:
                commands.append(self._command_record(context, name, exc.result))
                workspace_manifest.commands = commands
                workspace_manifest.failed_subcommand = name
                self._write_workspace_manifest(context, workspace_manifest)
                self._write_failure_diagnostics(context, len(manifest.frames), name)
                raise ColmapExecutionError(str(exc), failed_subcommand=name) from exc
            commands.append(self._command_record(context, name, result))
            workspace_manifest.commands = commands
            self._write_workspace_manifest(context, workspace_manifest)
            if (
                name == "feature_extractor"
                and not context.path("camera", "colmap", "database.db").is_file()
            ):
                self._write_failure_diagnostics(context, len(manifest.frames), name)
                raise ColmapExecutionError(
                    "COLMAP feature_extractor returned zero but did not create database.db",
                    failed_subcommand=name,
                )

        try:
            selected, model_diagnostics, warnings = self._select_model(
                context,
                input_frame_count=len(manifest.frames),
                config=config,
            )
            reconstruction = self._reconstruction(manifest, selected, config)
        except Exception as exc:
            workspace_manifest.failed_subcommand = "model_selection"
            self._write_workspace_manifest(context, workspace_manifest)
            self._write_failure_diagnostics(
                context,
                len(manifest.frames),
                "model_selection",
                models=(exc.diagnostics if isinstance(exc, SparseModelThresholdError) else []),
                warnings=[str(exc)],
            )
            raise ColmapExecutionError(
                f"COLMAP model selection/parsing failed: {exc}",
                failed_subcommand="model_selection",
            ) from exc

        diagnostics = CameraDiagnostics(
            input_frame_count=manifest.total_decoded_frames or len(manifest.frames),
            selected_frame_count=len(manifest.frames),
            models=model_diagnostics,
            selected_model=selected.model_id,
            warnings=[
                *warnings,
                "Monocular COLMAP reconstruction has arbitrary scale; coordinates are not metric.",
            ],
        )
        workspace_manifest.selected_model = selected.model_id
        workspace_manifest.failed_subcommand = None
        self._write_workspace_manifest(context, workspace_manifest)
        atomic_write_json(context.path("camera", "diagnostics.json"), diagnostics)
        atomic_write_json(context.path("camera", "reconstruction.json"), reconstruction)

        outputs = self._raw_output_specs(context)
        return StageResult(
            outputs=outputs,
            metrics={
                "registered_frames": selected.registered_frames,
                "registration_ratio": selected.registration_ratio,
                "sparse_points": selected.sparse_points,
                "selected_model": selected.model_id,
            },
        )

    def _commands(
        self,
        config: ColmapAdapterConfig,
        context: StageContext,
        *,
        tool_version: str | None,
    ) -> list[tuple[str, list[str]]]:
        gpu = "1" if config.use_gpu else "0"
        version_match = re.search(r"\bCOLMAP\s+(\d+)", tool_version or "", re.IGNORECASE)
        modern_gpu_options = version_match is not None and int(version_match.group(1)) >= 4
        extraction_gpu_option = (
            "--FeatureExtraction.use_gpu" if modern_gpu_options else "--SiftExtraction.use_gpu"
        )
        matching_gpu_option = (
            "--FeatureMatching.use_gpu" if modern_gpu_options else "--SiftMatching.use_gpu"
        )
        feature = [
            "feature_extractor",
            "--database_path",
            "camera/colmap/database.db",
            "--image_path",
            "frames",
            "--ImageReader.camera_model",
            config.camera_model,
            "--ImageReader.single_camera",
            "1" if config.single_camera else "0",
            extraction_gpu_option,
            gpu,
        ]
        matcher = [
            f"{config.matcher}_matcher",
            "--database_path",
            "camera/colmap/database.db",
            matching_gpu_option,
            gpu,
        ]
        if config.matcher == "sequential":
            matcher.extend(
                [
                    "--SequentialMatching.overlap",
                    str(config.sequential_matcher.overlap),
                    "--SequentialMatching.loop_detection",
                    "1" if config.sequential_matcher.loop_detection else "0",
                ]
            )
        mapper = [
            "mapper",
            "--database_path",
            "camera/colmap/database.db",
            "--image_path",
            "frames",
            "--output_path",
            "camera/colmap/sparse",
            "--Mapper.multiple_models",
            "1" if config.mapper.multiple_models else "0",
        ]
        return [
            ("feature_extractor", self._wrap_command(config, context, feature)),
            (f"{config.matcher}_matcher", self._wrap_command(config, context, matcher)),
            ("mapper", self._wrap_command(config, context, mapper)),
        ]

    def _wrap_command(
        self,
        config: ColmapAdapterConfig,
        context: StageContext,
        arguments: list[str],
    ) -> list[str]:
        if config.execution_mode == "local":
            executable = resolve_executable(config.executable)
            if executable is None:
                raise FileNotFoundError(f"COLMAP executable disappeared: {config.executable}")
            return [executable, *arguments]
        docker = resolve_executable(config.docker_executable)
        if docker is None:
            raise FileNotFoundError(f"Docker executable disappeared: {config.docker_executable}")
        return [
            docker,
            "run",
            "--rm",
            *self._docker_user_arguments(config),
            "--mount",
            f"type=bind,src={context.run_dir.resolve()},dst=/workspace",
            "--workdir",
            "/workspace",
            config.docker_image,
            "colmap",
            *arguments,
        ]

    @staticmethod
    def _docker_user_arguments(config: ColmapAdapterConfig) -> list[str]:
        user = config.docker_user
        if user is None and hasattr(os, "getuid") and hasattr(os, "getgid"):
            user = f"{os.getuid()}:{os.getgid()}"
        return ["--user", user] if user else []

    def _select_model(
        self,
        context: StageContext,
        *,
        input_frame_count: int,
        config: ColmapAdapterConfig,
    ) -> tuple[SparseModelCandidate, list[SparseModelDiagnostics], list[str]]:
        sparse_root = context.path("camera", "colmap", "sparse")
        directories = sorted(
            (path for path in sparse_root.iterdir() if path.is_dir()),
            key=lambda path: _model_id_key(path.name),
        )
        if not directories:
            raise ColmapModelError(
                "COLMAP mapper produced no sparse model directories under camera/colmap/sparse"
            )
        candidates: list[SparseModelCandidate] = []
        warnings: list[str] = []
        for model_dir in directories:
            try:
                model = read_model(model_dir)
            except Exception as exc:
                warnings.append(f"model {model_dir.name} could not be parsed: {exc}")
                continue
            ratio = len(model.images) / input_frame_count if input_frame_count else 0.0
            candidates.append(
                SparseModelCandidate(
                    model_id=model_dir.name,
                    path=model_dir,
                    model=model,
                    registered_frames=len(model.images),
                    registration_ratio=ratio,
                    sparse_points=len(model.points3d),
                    average_reprojection_error=model.average_reprojection_error,
                )
            )
        if not candidates and warnings:
            raise ColmapModelError("; ".join(warnings))
        selected, diagnostics = rank_sparse_models(
            candidates,
            min_registered_frames=config.min_registered_frames,
            min_registration_ratio=config.min_registration_ratio,
        )
        return selected, diagnostics, warnings

    def _reconstruction(
        self,
        manifest: IngestManifest,
        selected: SparseModelCandidate,
        config: ColmapAdapterConfig,
    ) -> CameraReconstruction:
        model = selected.model
        if len(model.cameras) != 1:
            raise ColmapModelError(
                "Phase 1 requires exactly one COLMAP camera; "
                f"selected model {selected.model_id!r} contains {len(model.cameras)}"
            )
        camera = next(iter(model.cameras.values()))
        intrinsics = camera_intrinsics(camera)
        by_name = {Path(frame.relative_path).name: frame for frame in manifest.frames}
        if len(by_name) != len(manifest.frames):
            raise ColmapModelError("ingest manifest contains duplicate frame file names")
        registered: dict[str, Any] = {}
        for image in model.images.values():
            if Path(image.name).name != image.name or image.name not in by_name:
                raise ColmapModelError(
                    f"COLMAP registered inconsistent frame name {image.name!r}; "
                    f"expected one of {sorted(by_name)}"
                )
            if image.camera_id != camera.camera_id:
                raise ColmapModelError(
                    f"COLMAP image {image.name!r} uses incompatible camera {image.camera_id}"
                )
            if image.name in registered:
                raise ColmapModelError(f"COLMAP contains duplicate image name {image.name!r}")
            registered[image.name] = image
        ordered_registered = [
            frame for frame in manifest.frames if Path(frame.relative_path).name in registered
        ]
        score = camera_confidence(
            registration_ratio=selected.registration_ratio,
            registered_frames=selected.registered_frames,
            sparse_points=selected.sparse_points,
            average_reprojection_error=selected.average_reprojection_error,
        )
        confidence = ConfidenceRecord(
            score=score,
            method=(
                "0.60*registration_ratio + 0.15*frame_support + "
                "0.15*point_support + 0.10*reprojection_support"
            ),
        )
        poses = [
            CameraPose(
                frame_id=frame.frame_id,
                transform_world_from_camera=colmap_pose_to_world_from_camera(
                    registered[Path(frame.relative_path).name].qvec_wxyz,
                    registered[Path(frame.relative_path).name].tvec,
                ),
                confidence=confidence,
            )
            for frame in ordered_registered
        ]
        registered_ids = [frame.frame_id for frame in ordered_registered]
        unregistered_ids = [
            frame.frame_id for frame in manifest.frames if frame.frame_id not in set(registered_ids)
        ]
        output_paths = [
            "camera/reconstruction.json",
            "camera/diagnostics.json",
            "camera/colmap/workspace_manifest.json",
        ]
        provenance = ProvenanceRecord(
            adapter_name=self.name,
            adapter_version=self.version,
            configuration=config.model_dump(mode="json"),
            input_artifact_paths=[
                "inputs/manifest.json",
                *[frame.relative_path for frame in manifest.frames],
            ],
            output_artifact_paths=output_paths,
            confidence=confidence,
            source=GeometrySourceType.MEASURED,
        )
        return CameraReconstruction(
            camera_id=f"colmap_camera_{camera.camera_id}",
            model=camera.model_name,
            intrinsics=intrinsics,
            poses=poses,
            registered_frame_ids=registered_ids,
            unregistered_frame_ids=unregistered_ids,
            sparse_point_count=selected.sparse_points,
            average_reprojection_error=selected.average_reprojection_error,
            confidence=confidence,
            coordinate_convention=CoordinateConvention(
                world_frame=WorldFrame.COLMAP_ARBITRARY,
                alignment_status=AlignmentStatus.UNORIENTED,
                camera_axes=CameraAxes.X_RIGHT_Y_DOWN_Z_FORWARD,
                linear_units=LinearUnits.ARBITRARY_UNITS,
                scale_status=ScaleStatus.SCALE_AMBIGUOUS,
            ),
            scale_status=ScaleStatus.SCALE_AMBIGUOUS,
            frame_sequence_digest=manifest.frame_sequence_digest,
            provenance=provenance,
        )

    @staticmethod
    def _validate_frames(context: StageContext, manifest: IngestManifest) -> None:
        for frame in manifest.frames:
            path = context.path(frame.relative_path)
            if path.parent != context.path("frames") or not path.is_file():
                raise ValueError(
                    f"inconsistent frame path {frame.relative_path!r}; COLMAP requires normalized "
                    "frames directly under frames/"
                )

    @staticmethod
    def _tool_identity(
        config: ColmapAdapterConfig,
    ) -> tuple[str, str | None, str | None]:
        if config.execution_mode == "docker":
            docker = resolve_executable(config.docker_executable)
            if docker is None:
                return config.docker_image, None, None
            inspect = subprocess.run(
                [
                    docker,
                    "image",
                    "inspect",
                    config.docker_image,
                    "--format",
                    "{{.Id}}",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            identifier = inspect.stdout.strip() if inspect.returncode == 0 else None
            return config.docker_image, None, identifier or None
        executable = resolve_executable(config.executable)
        if executable is None:
            return config.executable, None, None
        _, version = executable_version(executable, "-h")
        return executable, version, None

    @staticmethod
    def _command_record(
        context: StageContext,
        name: str,
        result: ProcessResult,
    ) -> ColmapCommandRecord:
        prefix = f"camera/colmap/logs/{context.stage_name}.{name}.attempt_{context.attempt}"
        return ColmapCommandRecord(
            name=name,
            command=result.command,
            return_code=result.return_code,
            duration_s=result.duration_s,
            timed_out=result.timed_out,
            stdout_path=f"{prefix}.stdout.log",
            stderr_path=f"{prefix}.stderr.log",
        )

    @staticmethod
    def _write_workspace_manifest(
        context: StageContext,
        manifest: ColmapWorkspaceManifest,
    ) -> None:
        atomic_write_json(
            context.path("camera", "colmap", "workspace_manifest.json"),
            manifest,
        )

    @staticmethod
    def _write_failure_diagnostics(
        context: StageContext,
        input_count: int,
        failed_subcommand: str,
        *,
        models: list[SparseModelDiagnostics] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        atomic_write_json(
            context.path("camera", "diagnostics.json"),
            CameraDiagnostics(
                input_frame_count=input_count,
                selected_frame_count=input_count,
                models=models or [],
                warnings=warnings or [],
                failed_subcommand=failed_subcommand,
            ),
        )

    @staticmethod
    def _raw_output_specs(context: StageContext) -> list[OutputSpec]:
        colmap_root = context.path("camera", "colmap")
        outputs = []
        for path in sorted(colmap_root.rglob("*")):
            if not path.is_file() or path.name == "workspace_manifest.json":
                continue
            relative = path.relative_to(context.run_dir).as_posix()
            if path.suffix == ".log":
                media_type = "text/plain"
                artifact_type = "colmap_log"
            elif path.name == "database.db":
                media_type = "application/vnd.sqlite3"
                artifact_type = "colmap_database"
            else:
                media_type = "application/octet-stream"
                artifact_type = "colmap_raw_model"
            outputs.append(
                OutputSpec(
                    relative,
                    artifact_type,
                    media_type,
                    "colmap",
                )
            )
        return outputs


__all__ = [
    "ColmapAdapterConfig",
    "ColmapCameraRecoveryAdapter",
    "ColmapExecutionError",
    "SparseModelCandidate",
    "SparseModelThresholdError",
    "camera_confidence",
    "rank_sparse_models",
]
