from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel

from recon2sim.adapters.base import HealthcheckResult, OutputSpec, StageContext, StageResult
from recon2sim.adapters.process import ExternalProcessError, ProcessResult, run_external_process
from recon2sim.artifacts import (
    CameraDiagnostics,
    CameraReconstruction,
    ColmapModelDiagnostic,
    ColmapWorkspaceManifest,
    IngestManifest,
    ToolCommandRecord,
)
from recon2sim.colmap import (
    ColmapImage,
    ColmapModel,
    camera_intrinsics,
    colmap_world_to_camera_to_world_from_camera,
    read_colmap_model,
)
from recon2sim.images import png_dimensions
from recon2sim.ir import (
    CameraPose,
    ConfidenceRecord,
    CoordinateConvention,
    GeometrySourceType,
    ProvenanceRecord,
    ScaleStatus,
    WorldFrameStatus,
)
from recon2sim.storage import atomic_write_json


@dataclass(frozen=True)
class SelectedModel:
    model_id: str
    model: ColmapModel
    diagnostic: ColmapModelDiagnostic


class ColmapSubcommandError(RuntimeError):
    def __init__(self, subcommand: str, error: ExternalProcessError) -> None:
        super().__init__(f"COLMAP {subcommand} failed: {error}")
        self.details = {"failed_subcommand": subcommand, **error.details}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_id_key(model_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(model_id))
    except ValueError:
        return (1, model_id)


def rank_sparse_models(
    models: dict[str, ColmapModel],
    *,
    input_frame_count: int,
) -> list[SelectedModel]:
    if input_frame_count <= 0:
        raise ValueError("input_frame_count must be positive")
    candidates: list[SelectedModel] = []
    for model_id, model in models.items():
        registered = len(model.images)
        diagnostic = ColmapModelDiagnostic(
            model_id=model_id,
            registered_frames=registered,
            registration_ratio=min(1.0, registered / input_frame_count),
            sparse_points=len(model.points3d),
            mean_track_length=model.mean_track_length,
            mean_reprojection_error=model.mean_reprojection_error,
        )
        candidates.append(SelectedModel(model_id, model, diagnostic))
    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate.diagnostic.registered_frames,
            -candidate.diagnostic.registration_ratio,
            -candidate.diagnostic.sparse_points,
            -candidate.diagnostic.mean_track_length,
            candidate.diagnostic.mean_reprojection_error
            if candidate.diagnostic.mean_reprojection_error is not None
            else math.inf,
            _model_id_key(candidate.model_id),
        ),
    )


def select_sparse_model(
    models: dict[str, ColmapModel],
    *,
    input_frame_count: int,
    min_registered_frames: int,
    min_registration_ratio: float,
) -> tuple[SelectedModel | None, list[ColmapModelDiagnostic]]:
    ranked = rank_sparse_models(models, input_frame_count=input_frame_count)
    if not ranked:
        return None, []
    winner = ranked[0]
    failures: list[str] = []
    if winner.diagnostic.registered_frames < min_registered_frames:
        failures.append(
            f"registered {winner.diagnostic.registered_frames} frames, below minimum "
            f"{min_registered_frames}"
        )
    if winner.diagnostic.registration_ratio < min_registration_ratio:
        failures.append(
            f"registration ratio {winner.diagnostic.registration_ratio:.3f}, below minimum "
            f"{min_registration_ratio:.3f}"
        )
    selected = winner if not failures else None
    diagnostics: list[ColmapModelDiagnostic] = []
    for candidate in ranked:
        if selected is not None and candidate.model_id == selected.model_id:
            diagnostics.append(candidate.diagnostic.model_copy(update={"selected": True}))
        elif candidate.model_id == winner.model_id and failures:
            diagnostics.append(
                candidate.diagnostic.model_copy(update={"rejection_reason": "; ".join(failures)})
            )
        else:
            diagnostics.append(
                candidate.diagnostic.model_copy(
                    update={"rejection_reason": f"ranked below model {winner.model_id}"}
                )
            )
    return selected, diagnostics


def reconstruction_confidence(diagnostic: ColmapModelDiagnostic) -> float:
    """Return a transparent diagnostic score, not a calibrated probability.

    The score weights registration ratio at 55%, registered-frame support at
    20%, sparse-point support at 15%, and inverse reprojection error at 10%.
    """
    frame_support = min(1.0, diagnostic.registered_frames / 20.0)
    point_support = min(1.0, math.log10(diagnostic.sparse_points + 1) / 4.0)
    reprojection_support = (
        0.0
        if diagnostic.mean_reprojection_error is None
        else 1.0 / (1.0 + diagnostic.mean_reprojection_error)
    )
    return max(
        0.0,
        min(
            1.0,
            0.55 * diagnostic.registration_ratio
            + 0.20 * frame_support
            + 0.15 * point_support
            + 0.10 * reprojection_support,
        ),
    )


def _health_command(arguments: list[str]) -> tuple[bool, str]:
    executable = shutil.which(arguments[0])
    if executable is None:
        return False, f"{arguments[0]!r} was not found on PATH"
    try:
        completed = subprocess.run(
            [executable, *arguments[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"could not execute {executable}: {exc}"
    output = (completed.stdout or completed.stderr).splitlines()
    summary = output[0] if output else "no version text returned"
    return completed.returncode == 0, f"{executable}: {summary}"


def _json_spec(path: str, artifact_type: str, model: type[BaseModel]) -> OutputSpec:
    return OutputSpec(
        path,
        artifact_type,
        "application/json",
        "colmap",
        validation="json",
        schema_identifier=f"recon2sim/{artifact_type.replace('_', '-')}/0.1.0",
        model=model,
    )


def _raw_spec(path: str) -> OutputSpec:
    media_type = (
        "application/vnd.sqlite3"
        if path.endswith(".db")
        else "text/plain"
        if path.endswith(".log")
        else "application/octet-stream"
    )
    return OutputSpec(path, "colmap_raw_workspace", media_type, "colmap")


class ColmapCameraRecoveryAdapter:
    name = "colmap_camera_recovery"
    version = "0.1.0"

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        config = context.config.adapter.config if context is not None else {}
        execution_mode = str(config.get("execution_mode", "local"))
        if execution_mode == "local":
            executable = str(config.get("executable", "colmap"))
            ok, message = _health_command([executable, "-h"])
            if not ok:
                return HealthcheckResult(
                    False,
                    f"local COLMAP unavailable ({message}). Install COLMAP and set "
                    "adapter.config.executable to its path.",
                )
            return HealthcheckResult(True, f"local COLMAP available: {message}")
        if execution_mode != "docker":
            return HealthcheckResult(False, "execution_mode must be either 'local' or 'docker'")
        docker = str(config.get("docker_executable", "docker"))
        image = str(config.get("docker_image", ""))
        if not image:
            return HealthcheckResult(False, "Docker COLMAP requires docker_image")
        version_ok, version_message = _health_command([docker, "version"])
        inspect_ok, inspect_message = _health_command([docker, "image", "inspect", image])
        if not (version_ok and inspect_ok):
            return HealthcheckResult(
                False,
                f"Docker COLMAP unavailable ({version_message}; {inspect_message}). Start "
                f"Docker and build or pull image {image!r}.",
            )
        return HealthcheckResult(
            True, f"Docker engine and image available: {version_message}; image={image}"
        )

    def prepare(self, context: StageContext) -> None:
        context.output_path("camera", "colmap", "logs").mkdir(parents=True, exist_ok=True)
        context.output_path("camera", "colmap", "sparse").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            _json_spec(
                "camera/reconstruction.json",
                "camera_reconstruction",
                CameraReconstruction,
            ),
            _json_spec("camera/diagnostics.json", "camera_diagnostics", CameraDiagnostics),
            _json_spec(
                "camera/colmap/workspace_manifest.json",
                "colmap_workspace_manifest",
                ColmapWorkspaceManifest,
            ),
            _raw_spec("camera/colmap/database.db"),
        ]

    def run(self, context: StageContext) -> StageResult:
        manifest_path = context.path("inputs", "manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"COLMAP camera recovery requires ingest manifest: {manifest_path}"
            )
        manifest = IngestManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        for frame in manifest.frames:
            frame_path = context.path(frame.relative_path)
            if not frame_path.is_file():
                raise FileNotFoundError(
                    f"ingest manifest references missing frame {frame.relative_path!r}"
                )
            if _sha256(frame_path) != frame.sha256:
                raise ValueError(
                    f"ingest frame hash no longer matches manifest: {frame.relative_path}"
                )
            if png_dimensions(frame_path) != (frame.width, frame.height):
                raise ValueError(
                    f"ingest frame dimensions no longer match manifest: {frame.relative_path}"
                )

        config = context.config.adapter.config
        matcher = str(config.get("matcher", "sequential"))
        if matcher not in {"sequential", "exhaustive"}:
            raise ValueError("COLMAP matcher must be 'sequential' or 'exhaustive'")
        min_registered_frames = int(config.get("min_registered_frames", 8))
        min_registration_ratio = float(config.get("min_registration_ratio", 0.4))
        if min_registered_frames <= 0:
            raise ValueError("min_registered_frames must be positive")
        if not 0 <= min_registration_ratio <= 1:
            raise ValueError("min_registration_ratio must be in [0, 1]")

        workspace = context.output_path("camera", "colmap")
        database = workspace / "database.db"
        sparse = workspace / "sparse"
        command_records: list[ToolCommandRecord] = []
        colmap_version = self._run_version(context, command_records)
        execution_mode_value = str(config.get("execution_mode", "local"))
        if execution_mode_value not in {"local", "docker"}:
            raise ValueError("execution_mode must be either 'local' or 'docker'")
        execution_mode = cast(Literal["local", "docker"], execution_mode_value)
        path_set = self._execution_paths(context, execution_mode)
        extraction_gpu_option, matching_gpu_option = self._gpu_option_names(
            context, matcher, command_records
        )

        feature_arguments = [
            "feature_extractor",
            "--database_path",
            path_set["database"],
            "--image_path",
            path_set["images"],
            "--ImageReader.camera_model",
            str(config.get("camera_model", "OPENCV")),
            "--ImageReader.single_camera",
            "1" if bool(config.get("single_camera", True)) else "0",
            extraction_gpu_option,
            "1" if bool(config.get("use_gpu", True)) else "0",
        ]
        self._run_colmap(context, "feature_extractor", feature_arguments, command_records)
        if not database.is_file():
            raise RuntimeError(
                "COLMAP feature_extractor returned success but did not create database.db"
            )
        with database.open("rb") as database_file:
            database_header = database_file.read(16)
        if database_header != b"SQLite format 3\x00":
            raise RuntimeError(
                "COLMAP feature_extractor produced database.db without a valid SQLite header"
            )

        matcher_arguments = [
            f"{matcher}_matcher",
            "--database_path",
            path_set["database"],
            matching_gpu_option,
            "1" if bool(config.get("use_gpu", True)) else "0",
        ]
        if matcher == "sequential":
            sequential = config.get("sequential_matcher", {})
            if not isinstance(sequential, dict):
                raise ValueError("sequential_matcher configuration must be a mapping")
            matcher_arguments.extend(
                [
                    "--SequentialMatching.overlap",
                    str(int(sequential.get("overlap", 10))),
                    "--SequentialMatching.loop_detection",
                    "1" if bool(sequential.get("loop_detection", False)) else "0",
                ]
            )
        self._run_colmap(context, f"{matcher}_matcher", matcher_arguments, command_records)

        mapper_arguments = [
            "mapper",
            "--database_path",
            path_set["database"],
            "--image_path",
            path_set["images"],
            "--output_path",
            path_set["sparse"],
        ]
        mapper_config = config.get("mapper", {})
        if not isinstance(mapper_config, dict):
            raise ValueError("mapper configuration must be a mapping")
        mapper_arguments.extend(
            [
                "--Mapper.multiple_models",
                "1" if bool(mapper_config.get("multiple_models", True)) else "0",
            ]
        )
        self._run_colmap(context, "mapper", mapper_arguments, command_records)

        model_directories = sorted(
            [path for path in sparse.iterdir() if path.is_dir()],
            key=lambda path: _model_id_key(path.name),
        )
        if not model_directories:
            raise RuntimeError(
                "COLMAP mapper produced no sparse model directories; inspect "
                "camera/colmap/logs/mapper.stderr.log"
            )
        models: dict[str, ColmapModel] = {}
        for model_directory in model_directories:
            try:
                models[model_directory.name] = read_colmap_model(model_directory)
            except Exception as exc:
                raise ValueError(
                    f"malformed COLMAP sparse model {model_directory.name!r}: {exc}"
                ) from exc
        selected, model_diagnostics = select_sparse_model(
            models,
            input_frame_count=len(manifest.frames),
            min_registered_frames=min_registered_frames,
            min_registration_ratio=min_registration_ratio,
        )
        decoded_frame_count = manifest.total_decoded_frames or len(manifest.frames)
        if selected is None:
            top = model_diagnostics[0] if model_diagnostics else None
            diagnostics = CameraDiagnostics(
                input_frame_count=decoded_frame_count,
                selected_frame_count=len(manifest.frames),
                registered_frames=top.registered_frames if top else 0,
                registration_ratio=top.registration_ratio if top else 0.0,
                sparse_points=top.sparse_points if top else 0,
                camera_model=None,
                selected_model=None,
                models=model_diagnostics,
                scale_status=ScaleStatus.SCALE_AMBIGUOUS,
                world_frame_status=WorldFrameStatus.COLMAP_UNALIGNED,
                confidence_score=0.0,
                warnings=["no sparse model met configured registration thresholds"],
            )
            atomic_write_json(context.output_path("camera", "diagnostics.json"), diagnostics)
            reason = top.rejection_reason if top else "no model candidates"
            raise RuntimeError(f"no COLMAP sparse model met registration thresholds: {reason}")

        reconstruction, diagnostics = self._normalize_model(
            context,
            manifest,
            selected,
            model_diagnostics,
        )
        atomic_write_json(context.output_path("camera", "reconstruction.json"), reconstruction)
        atomic_write_json(context.output_path("camera", "diagnostics.json"), diagnostics)
        workspace_manifest = ColmapWorkspaceManifest(
            execution_mode=execution_mode,
            colmap_version=colmap_version,
            database_path="camera/colmap/database.db",
            sparse_model_paths=[f"camera/colmap/sparse/{path.name}" for path in model_directories],
            selected_model=selected.model_id,
            input_frame_hashes={frame.frame_id: frame.sha256 for frame in manifest.frames},
            configuration=config,
            commands=command_records,
            provenance=ProvenanceRecord(
                adapter_name=self.name,
                adapter_version=self.version,
                configuration=config,
                input_artifact_paths=[
                    "inputs/manifest.json",
                    *[frame.relative_path for frame in manifest.frames],
                ],
                output_artifact_paths=[
                    "camera/reconstruction.json",
                    "camera/diagnostics.json",
                    "camera/colmap/workspace_manifest.json",
                ],
                confidence=ConfidenceRecord(
                    score=diagnostics.confidence_score,
                    method="colmap_sparse_diagnostics_v1",
                ),
                source=GeometrySourceType.MEASURED,
            ),
        )
        atomic_write_json(
            context.output_path("camera", "colmap", "workspace_manifest.json"),
            workspace_manifest,
        )

        raw_outputs = [
            _raw_spec(path.relative_to(context.attempt_dir).as_posix())
            for path in sorted(workspace.rglob("*"))
            if path.is_file() and path.name != "workspace_manifest.json" and path != database
        ]
        return StageResult(
            outputs=raw_outputs,
            metrics={
                "registered_frames": diagnostics.registered_frames,
                "registration_ratio": diagnostics.registration_ratio,
                "sparse_points": diagnostics.sparse_points,
                "selected_model": selected.model_id,
                "confidence": diagnostics.confidence_score,
            },
        )

    def _run_version(self, context: StageContext, records: list[ToolCommandRecord]) -> str:
        result = self._run_colmap(context, "version", ["-h"], records)
        lines = (result.stdout or result.stderr).splitlines()
        return lines[0] if lines else "COLMAP version unavailable"

    def _gpu_option_names(
        self,
        context: StageContext,
        matcher: str,
        records: list[ToolCommandRecord],
    ) -> tuple[str, str]:
        style = str(context.config.adapter.config.get("gpu_flag_style", "auto"))
        if style == "modern":
            return "--FeatureExtraction.use_gpu", "--FeatureMatching.use_gpu"
        if style == "legacy":
            return "--SiftExtraction.use_gpu", "--SiftMatching.use_gpu"
        if style != "auto":
            raise ValueError("gpu_flag_style must be auto, modern, or legacy")
        extraction_help = self._run_colmap(
            context,
            "feature_extractor_help",
            ["feature_extractor", "-h"],
            records,
        )
        matching_help = self._run_colmap(
            context,
            f"{matcher}_matcher_help",
            [f"{matcher}_matcher", "-h"],
            records,
        )
        extraction_text = extraction_help.stdout + extraction_help.stderr
        matching_text = matching_help.stdout + matching_help.stderr
        if (
            "FeatureExtraction.use_gpu" in extraction_text
            and "FeatureMatching.use_gpu" in matching_text
        ):
            return "--FeatureExtraction.use_gpu", "--FeatureMatching.use_gpu"
        if "SiftExtraction.use_gpu" in extraction_text and "SiftMatching.use_gpu" in matching_text:
            return "--SiftExtraction.use_gpu", "--SiftMatching.use_gpu"
        raise RuntimeError(
            "could not determine COLMAP GPU option names from subcommand help; set "
            "gpu_flag_style to 'modern' or 'legacy' after checking the installed COLMAP help"
        )

    def _run_colmap(
        self,
        context: StageContext,
        name: str,
        colmap_arguments: list[str],
        records: list[ToolCommandRecord],
    ) -> ProcessResult:
        config = context.config.adapter.config
        execution_mode = str(config.get("execution_mode", "local"))
        if execution_mode == "local":
            arguments = [str(config.get("executable", "colmap")), *colmap_arguments]
        elif execution_mode == "docker":
            image = str(config.get("docker_image", ""))
            if not image:
                raise ValueError("Docker COLMAP execution requires docker_image")
            workspace = context.output_path("camera", "colmap").resolve()
            arguments = [
                str(config.get("docker_executable", "docker")),
                "run",
                "--rm",
            ]
            if bool(config.get("use_gpu", True)):
                arguments.extend(["--gpus", "all"])
            arguments.extend(
                [
                    "--volume",
                    f"{context.run_dir.resolve()}:/run:ro",
                    "--volume",
                    f"{workspace}:/workspace",
                    image,
                    str(config.get("executable", "colmap")),
                    *colmap_arguments,
                ]
            )
        else:
            raise ValueError("execution_mode must be either 'local' or 'docker'")
        stdout = context.output_path("camera", "colmap", "logs", f"{name}.stdout.log")
        stderr = context.output_path("camera", "colmap", "logs", f"{name}.stderr.log")
        try:
            result = run_external_process(
                arguments,
                cwd=context.attempt_dir,
                timeout_s=context.config.adapter.timeout_s,
                environment_names=context.config.adapter.env,
                stdout_path=stdout,
                stderr_path=stderr,
                command_name=f"COLMAP {name}",
            )
        except ExternalProcessError as exc:
            raise ColmapSubcommandError(name, exc) from exc
        records.append(
            ToolCommandRecord(
                name=name,
                arguments=arguments,
                return_code=result.return_code,
                duration_s=result.duration_s,
                stdout_path=result.stdout_path.relative_to(context.attempt_dir).as_posix(),
                stderr_path=result.stderr_path.relative_to(context.attempt_dir).as_posix(),
            )
        )
        return result

    @staticmethod
    def _execution_paths(context: StageContext, execution_mode: str) -> dict[str, str]:
        if execution_mode == "local":
            workspace = context.output_path("camera", "colmap").resolve()
            return {
                "database": str(workspace / "database.db"),
                "images": str(context.path("frames").resolve()),
                "sparse": str(workspace / "sparse"),
            }
        if execution_mode == "docker":
            return {
                "database": "/workspace/database.db",
                "images": "/run/frames",
                "sparse": "/workspace/sparse",
            }
        raise ValueError("execution_mode must be either 'local' or 'docker'")

    def _normalize_model(
        self,
        context: StageContext,
        manifest: IngestManifest,
        selected: SelectedModel,
        model_diagnostics: list[ColmapModelDiagnostic],
    ) -> tuple[CameraReconstruction, CameraDiagnostics]:
        model = selected.model
        used_camera_ids = {image.camera_id for image in model.images.values()}
        if len(used_camera_ids) != 1:
            raise ValueError(
                "Phase 1 rejects multi-camera COLMAP reconstructions; found camera IDs "
                f"{sorted(used_camera_ids)}. Run one physical camera per reconstruction."
            )
        camera_id = next(iter(used_camera_ids))
        camera = model.cameras[camera_id]
        intrinsics = camera_intrinsics(camera)
        frame_by_name = {Path(frame.relative_path).name: frame for frame in manifest.frames}
        images_by_name: dict[str, ColmapImage] = {}
        for model_image in model.images.values():
            normalized_name = Path(model_image.name).as_posix()
            if "/" in normalized_name or normalized_name not in frame_by_name:
                raise ValueError(
                    f"COLMAP registered inconsistent frame name {model_image.name!r}; "
                    "expected one of "
                    f"{sorted(frame_by_name)}"
                )
            if normalized_name in images_by_name:
                raise ValueError(f"COLMAP registered duplicate frame name {normalized_name!r}")
            frame = frame_by_name[normalized_name]
            if (frame.width, frame.height) != (camera.width, camera.height):
                raise ValueError(
                    f"COLMAP camera dimensions {(camera.width, camera.height)} do not match "
                    f"manifest frame {frame.frame_id!r} dimensions {(frame.width, frame.height)}"
                )
            images_by_name[normalized_name] = model_image

        confidence_score = reconstruction_confidence(selected.diagnostic)
        confidence = ConfidenceRecord(
            score=confidence_score,
            method="colmap_sparse_diagnostics_v1",
            notes="diagnostic score; not a calibrated probability",
        )
        poses: list[CameraPose] = []
        registered_ids: list[str] = []
        unregistered_ids: list[str] = []
        for frame in manifest.frames:
            pose_image = images_by_name.get(Path(frame.relative_path).name)
            if pose_image is None:
                unregistered_ids.append(frame.frame_id)
                continue
            registered_ids.append(frame.frame_id)
            poses.append(
                CameraPose(
                    frame_id=frame.frame_id,
                    transform_world_from_camera=colmap_world_to_camera_to_world_from_camera(
                        pose_image.qvec_wxyz, pose_image.tvec
                    ),
                    confidence=confidence,
                )
            )

        warnings = [
            "monocular COLMAP reconstruction has unknown metric scale",
            "COLMAP world orientation is not yet aligned to +X forward, +Y left, +Z up",
        ]
        if unregistered_ids:
            warnings.append(f"{len(unregistered_ids)} selected frames were not registered")
        diagnostics = CameraDiagnostics(
            input_frame_count=manifest.total_decoded_frames or len(manifest.frames),
            selected_frame_count=len(manifest.frames),
            registered_frames=len(registered_ids),
            registration_ratio=len(registered_ids) / len(manifest.frames),
            sparse_points=len(model.points3d),
            camera_model=camera.model_name,
            selected_model=selected.model_id,
            models=model_diagnostics,
            scale_status=ScaleStatus.SCALE_AMBIGUOUS,
            world_frame_status=WorldFrameStatus.COLMAP_UNALIGNED,
            confidence_score=confidence_score,
            warnings=warnings,
        )
        reconstruction = CameraReconstruction(
            camera_id=f"colmap_camera_{camera_id}",
            model=camera.model_name,
            intrinsics=intrinsics,
            poses=poses,
            registered_frame_ids=registered_ids,
            unregistered_frame_ids=unregistered_ids,
            confidence=confidence,
            coordinate_convention=CoordinateConvention(
                world_axes="colmap_arbitrary",
                units="arbitrary_scale",
            ),
            scale_status=ScaleStatus.SCALE_AMBIGUOUS,
            world_frame_status=WorldFrameStatus.COLMAP_UNALIGNED,
            provenance=ProvenanceRecord(
                adapter_name=self.name,
                adapter_version=self.version,
                configuration=context.config.adapter.config,
                input_artifact_paths=[
                    "inputs/manifest.json",
                    *[frame.relative_path for frame in manifest.frames],
                ],
                output_artifact_paths=[
                    "camera/reconstruction.json",
                    "camera/diagnostics.json",
                    "camera/colmap/workspace_manifest.json",
                ],
                confidence=confidence,
                source=GeometrySourceType.MEASURED,
            ),
        )
        return reconstruction, diagnostics
