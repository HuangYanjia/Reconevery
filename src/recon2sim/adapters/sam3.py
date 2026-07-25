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
from recon2sim.artifacts import (
    CameraReconstruction,
    FrameQualityReport,
    IngestManifest,
    Sam3InferenceRequest,
    Sam3RawResult,
    Sam3WorkerManifest,
    SegmentationDiagnostics,
    SegmentationPrompt,
    SegmentationPromptManifest,
    SegmentationTrackingArtifact,
)
from recon2sim.ir import ConfidenceRecord, GeometrySourceType, ProvenanceRecord, StrictModel
from recon2sim.segmentation import (
    AnchorStrategy,
    TrackPostprocessingConfig,
    canonicalize_worker_result,
    load_prompt_manifest,
    render_previews,
    select_anchor_frames,
    sha256_file,
    validate_canonical_mask,
    validate_prompt_references,
)
from recon2sim.storage import atomic_write_json

OFFICIAL_REPOSITORY = "https://github.com/facebookresearch/sam3"
OFFICIAL_CODE_COMMIT = "46957e47805eaa273f4aa7bbbd25a88bca9108ce"
DEFAULT_CHECKPOINT_REPOSITORY = "facebook/sam3.1"
DEFAULT_CHECKPOINT_REVISION = "daa63191845a41281374e725f4c9e51c7a824460"
SAM3_CHECKPOINT_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
OFFICIAL_LICENSE = "SAM License (see official checkpoint repository terms)"


def _resolve_worker_python(value: str) -> str | None:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        return str(candidate.absolute()) if candidate.is_file() else None
    return shutil.which(value)


class Sam3AdapterConfig(StrictModel):
    execution_mode: Literal["local_worker", "docker", "fake_worker"]
    prompt_manifest: str
    worker_python: str = "python"
    worker_module: str = "sam3_worker"
    worker_script: str | None = None
    docker_executable: str = "docker"
    docker_image: str = "reconevery/sam3:phase2"
    model_cache_path: str | None = None
    local_checkpoint_path: str | None = None
    official_repository: Literal["https://github.com/facebookresearch/sam3"] = (
        "https://github.com/facebookresearch/sam3"
    )
    official_code_commit: Literal["46957e47805eaa273f4aa7bbbd25a88bca9108ce"] = (
        "46957e47805eaa273f4aa7bbbd25a88bca9108ce"
    )
    checkpoint_repository: Literal["facebook/sam3", "facebook/sam3.1"] = "facebook/sam3.1"
    checkpoint_revision: str = DEFAULT_CHECKPOINT_REVISION
    model_mode: Literal["sam3", "sam3.1"] = "sam3.1"
    device: Literal["cpu", "cuda"] = "cuda"
    precision: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    offline: bool = False
    strategy: Literal["detect_then_track"] = "detect_then_track"
    anchor_strategy: AnchorStrategy = "best_quality_registered_frame"
    anchor_count: int = Field(default=1, gt=0)
    explicit_anchor_frame_ids: list[str] = Field(default_factory=list)
    tracking_direction: Literal["forward", "backward", "forward_backward"] = "forward_backward"
    seed: int = 7
    score_threshold: float = Field(default=0.5, ge=0, le=1)
    mask_threshold: float = Field(default=0.5, ge=0, le=1)
    min_mask_area_pixels: int = Field(default=32, gt=0)
    max_mask_area_ratio: float = Field(default=0.98, gt=0, le=1)
    min_track_observations: int = Field(default=2, gt=0)
    min_track_coverage: float = Field(default=0.1, ge=0, le=1)
    same_prompt_duplicate_iou: float = Field(default=0.9, ge=0, le=1)
    model_box_mask_iou_threshold: float = Field(default=0.05, ge=0, le=1)
    generate_frame_previews: bool = True
    fake_mode: str = "success_multi"

    @model_validator(mode="after")
    def validate_execution(self) -> Sam3AdapterConfig:
        if self.execution_mode == "fake_worker":
            if not self.worker_script:
                raise ValueError("fake_worker execution requires worker_script")
        else:
            if self.device != "cuda":
                raise ValueError(
                    "the pinned official SAM 3 backend requires device=cuda; use fake_worker "
                    "for CPU-only validation"
                )
            if self.precision != "bfloat16":
                raise ValueError("the pinned official SAM 3 backend requires precision=bfloat16")
            if self.anchor_count != 1:
                raise ValueError(
                    "local_worker and docker currently require anchor_count=1; "
                    "multi-anchor inference is not implemented by the pinned official path"
                )
        if self.execution_mode == "local_worker":
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible_devices is None or visible_devices.strip().lower() in {
                "",
                "-1",
                "none",
                "void",
            }:
                raise ValueError(
                    "local_worker requires CUDA_VISIBLE_DEVICES to expose at least one GPU"
                )
            executable = _resolve_worker_python(self.worker_python)
            if executable is None:
                raise ValueError(f"configured worker Python {self.worker_python!r} was not found")
            executable_path = Path(executable)
            environment_roots = {
                executable_path.parent.parent,
                executable_path.resolve().parent.parent,
            }
            isolated_root = next(
                (root for root in environment_roots if (root / "pyvenv.cfg").is_file()),
                None,
            )
            if isolated_root is None:
                raise ValueError(
                    "local_worker worker_python must resolve to an isolated virtual "
                    "environment containing pyvenv.cfg"
                )
            if isolated_root.resolve() == Path(sys.prefix).resolve():
                raise ValueError("local_worker must not use the Reconevery core Python environment")
        for field_name, expected_kind in (
            ("local_checkpoint_path", "file"),
            ("model_cache_path", "directory"),
        ):
            raw_path = getattr(self, field_name)
            if raw_path is None:
                continue
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                raise ValueError(f"{field_name} must be absolute")
            valid = path.is_file() if expected_kind == "file" else path.is_dir()
            if not valid or not os.access(path, os.R_OK):
                raise ValueError(f"{field_name} must reference a readable {expected_kind}: {path}")
        if self.offline and self.local_checkpoint_path is None and self.model_cache_path is None:
            raise ValueError("offline mode requires local_checkpoint_path or model_cache_path")
        expected_checkpoint = {
            "sam3": ("facebook/sam3", SAM3_CHECKPOINT_REVISION),
            "sam3.1": (DEFAULT_CHECKPOINT_REPOSITORY, DEFAULT_CHECKPOINT_REVISION),
        }[self.model_mode]
        if (self.checkpoint_repository, self.checkpoint_revision) != expected_checkpoint:
            repository, revision = expected_checkpoint
            raise ValueError(
                f"model_mode={self.model_mode} requires the pinned official checkpoint "
                f"{repository}@{revision}"
            )
        return self


def _safe_configuration(config: Sam3AdapterConfig) -> dict[str, Any]:
    payload = config.model_dump(mode="json")
    for key in list(payload):
        if "token" in key.lower() or "secret" in key.lower():
            payload[key] = "[REDACTED]"
    return payload


def _redaction_values(context: StageContext) -> tuple[str, ...]:
    return tuple(
        os.environ[name]
        for name in context.config.adapter.env
        if name in os.environ and ("token" in name.lower() or "secret" in name.lower())
    )


def _classify_worker_failure(exc: ProcessExecutionError) -> RuntimeError:
    stderr = exc.result.stderr.lower()
    if "out of memory" in stderr or "cuda oom" in stderr:
        return RuntimeError(
            "SAM worker ran out of GPU memory; reduce frames/objects, disable compilation, "
            "or use a GPU with more memory"
        )
    if any(
        marker in stderr
        for marker in (
            "unauthorized",
            "401",
            "gated repo",
            "terms not accepted",
            "access to model",
        )
    ):
        return RuntimeError(
            "official SAM checkpoint access was denied; accept the official terms and "
            "provide HF_TOKEN through the environment or mount an authorized local checkpoint"
        )
    if "cuda driver" in stderr or "driver version is insufficient" in stderr:
        return RuntimeError(
            "SAM worker reported a CUDA driver/runtime mismatch; install a compatible NVIDIA "
            "driver for the configured CUDA runtime"
        )
    return RuntimeError(str(exc))


class Sam3SegmentationTrackingAdapter:
    name = "sam3_segmentation_tracking"
    version = "0.1.1"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        config = Sam3AdapterConfig.model_validate(context.config.adapter.config)
        manifest_path = context.canonical_path("inputs", "manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "SAM segmentation requires inputs/manifest.json from a successful ingest stage"
            )
        manifest = IngestManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        prompt_source = Path(config.prompt_manifest).expanduser()
        if not prompt_source.is_absolute():
            prompt_source = Path.cwd() / prompt_source
        prompt_source = prompt_source.resolve()
        prompts = load_prompt_manifest(prompt_source)
        specs = [
            InputSpec("inputs/manifest.json", "ingest_manifest"),
            InputSpec("inputs/frame_qa.json", "frame_quality_report"),
            InputSpec("camera/reconstruction.json", "camera_reconstruction"),
            *[
                InputSpec(frame.relative_path, "input_frame", expected_sha256=frame.sha256)
                for frame in manifest.frames
            ],
            InputSpec(
                "observations/prompt_inputs/prompts.yaml",
                "segmentation_prompt_manifest_source",
                expected_sha256=sha256_file(prompt_source),
                source_path=prompt_source,
            ),
        ]
        for prompt in prompts.prompts:
            if prompt.mask_path is None:
                continue
            source = (prompt_source.parent / prompt.mask_path).resolve()
            specs.append(
                InputSpec(
                    f"observations/prompt_inputs/masks/{prompt.prompt_id}.png",
                    "segmentation_seed_mask",
                    expected_sha256=sha256_file(source) if source.is_file() else None,
                    source_path=source,
                )
            )
        return specs

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        if context is None:
            return HealthcheckResult(
                False,
                "SAM healthcheck requires --config so the worker mode, checkpoint, and "
                "device can be checked",
            )
        try:
            config = Sam3AdapterConfig.model_validate(context.config.adapter.config)
        except ValueError as exc:
            return HealthcheckResult(False, f"invalid SAM adapter configuration: {exc}")
        payload = self._worker_configuration(config)
        with tempfile.TemporaryDirectory(prefix="reconevery-sam3-health-") as temp:
            config_path = Path(temp) / "worker_config.json"
            atomic_write_json(config_path, payload)
            if config.execution_mode == "docker":
                return self._docker_healthcheck(context, config, config_path)
            command_or_error = self._local_worker_command(config, "healthcheck", config_path)
            if isinstance(command_or_error, str):
                return HealthcheckResult(False, command_or_error)
            try:
                result = subprocess.run(
                    command_or_error,
                    cwd=Path.cwd(),
                    env=allowed_environment(context),
                    text=True,
                    capture_output=True,
                    timeout=min(context.config.adapter.timeout_s, 60),
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return HealthcheckResult(False, f"SAM worker healthcheck could not run: {exc}")
            output = result.stdout.strip() or result.stderr.strip()
            output = self._redact(output, _redaction_values(context))
            if result.returncode != 0:
                return HealthcheckResult(
                    False,
                    f"SAM worker healthcheck failed (exit {result.returncode}): {output}",
                )
            return HealthcheckResult(True, output or "SAM worker healthcheck succeeded")

    def _docker_healthcheck(
        self,
        context: StageContext,
        config: Sam3AdapterConfig,
        worker_config_path: Path,
    ) -> HealthcheckResult:
        docker = resolve_executable(config.docker_executable)
        if docker is None:
            return HealthcheckResult(
                False,
                f"Docker executable {config.docker_executable!r} was not found",
            )
        try:
            version = subprocess.run(
                [docker, "version", "--format", "{{.Server.Version}}"],
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            inspect = subprocess.run(
                [docker, "image", "inspect", "--format", "{{.Id}}", config.docker_image],
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return HealthcheckResult(False, f"Docker healthcheck failed: {exc}")
        if version.returncode != 0:
            return HealthcheckResult(
                False,
                f"Docker daemon is unavailable: {version.stderr.strip()}",
            )
        if inspect.returncode != 0:
            return HealthcheckResult(
                False,
                f"Docker image {config.docker_image!r} is unavailable: {inspect.stderr.strip()}",
            )
        health_root = worker_config_path.parent
        command = [
            docker,
            "run",
            "--rm",
            "--gpus",
            "all",
            *self._docker_user_arguments(),
            "-v",
            f"{health_root}:/workspace:rw",
            "-w",
            "/workspace",
            *self._docker_cache_arguments(config),
            *self._docker_checkpoint_arguments(config),
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
                timeout=min(context.config.adapter.timeout_s, 120),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return HealthcheckResult(False, f"in-container SAM healthcheck failed: {exc}")
        output = self._redact(
            result.stdout.strip() or result.stderr.strip(),
            _redaction_values(context),
        )
        if result.returncode != 0:
            return HealthcheckResult(
                False,
                f"in-container SAM healthcheck failed (exit {result.returncode}): {output}",
            )
        return HealthcheckResult(
            True,
            f"docker={version.stdout.strip()}; image={config.docker_image} "
            f"({inspect.stdout.strip()}); {output}",
        )

    def prepare(self, context: StageContext) -> None:
        context.path("observations", "raw").mkdir(parents=True, exist_ok=True)
        context.path("observations", "masks").mkdir(parents=True, exist_ok=True)
        context.path("observations", "previews").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                "observations/prompts.json",
                "segmentation_prompt_manifest",
                "application/json",
                "sam3",
                validation="json",
                schema_identifier="recon2sim/segmentation-prompts/0.1.0",
                model=SegmentationPromptManifest,
            ),
            OutputSpec(
                "observations/sam3_request.json",
                "sam3_inference_request",
                "application/json",
                "sam3",
                validation="json",
                schema_identifier="recon2sim/sam3-inference-request/0.1.0",
                model=Sam3InferenceRequest,
            ),
            OutputSpec(
                "observations/worker_manifest.json",
                "sam3_worker_manifest",
                "application/json",
                "sam3",
                validation="json",
                schema_identifier="recon2sim/sam3-worker-manifest/0.1.0",
                model=Sam3WorkerManifest,
            ),
            OutputSpec(
                "observations/object_tracks.json",
                "segmentation_tracking",
                "application/json",
                "sam3",
                validation="json",
                schema_identifier="recon2sim/segmentation-tracking/0.1.0",
                model=SegmentationTrackingArtifact,
            ),
            OutputSpec(
                "observations/diagnostics.json",
                "segmentation_diagnostics",
                "application/json",
                "sam3",
                validation="json",
                schema_identifier="recon2sim/segmentation-diagnostics/0.1.0",
                model=SegmentationDiagnostics,
            ),
            OutputSpec(
                "observations/previews/contact_sheet.png",
                "segmentation_preview",
                "image/png",
                "sam3",
                validation="png",
            ),
            OutputSpec(
                "observations/previews/track_timeline.png",
                "segmentation_preview",
                "image/png",
                "sam3",
                validation="png",
            ),
        ]

    def run(self, context: StageContext) -> StageResult:
        config = Sam3AdapterConfig.model_validate(context.config.adapter.config)
        frame_manifest_path = context.path("inputs", "manifest.json")
        quality_path = context.path("inputs", "frame_qa.json")
        camera_path = context.path("camera", "reconstruction.json")
        frame_manifest = IngestManifest.model_validate_json(
            frame_manifest_path.read_text(encoding="utf-8")
        )
        quality = FrameQualityReport.model_validate_json(quality_path.read_text(encoding="utf-8"))
        camera = CameraReconstruction.model_validate_json(camera_path.read_text(encoding="utf-8"))
        prompts = self._normalized_prompts(context)
        validate_prompt_references(prompts, frame_manifest, prompt_root=context.run_dir)
        prompt_artifact_path = context.path("observations", "prompts.json")
        atomic_write_json(prompt_artifact_path, prompts)
        prompt_manifest_hash = sha256_file(prompt_artifact_path)
        anchors, anchor_diagnostics = select_anchor_frames(
            frame_manifest,
            quality,
            camera,
            strategy=config.anchor_strategy,
            anchor_count=config.anchor_count,
            explicit_frame_ids=config.explicit_anchor_frame_ids,
        )
        request = Sam3InferenceRequest(
            run_id=context.canonical_run_dir.name,
            frame_manifest_path="inputs/manifest.json",
            frame_manifest_sha256=sha256_file(frame_manifest_path),
            frame_order=[frame.frame_id for frame in frame_manifest.frames],
            frame_paths=[frame.relative_path for frame in frame_manifest.frames],
            frame_dimensions={
                frame.frame_id: (frame.width, frame.height) for frame in frame_manifest.frames
            },
            camera_reconstruction_path="camera/reconstruction.json",
            camera_reconstruction_sha256=sha256_file(camera_path),
            registered_frame_ids=camera.registered_frame_ids,
            unregistered_frame_ids=camera.unregistered_frame_ids,
            prompt_manifest=prompts,
            prompt_manifest_sha256=prompt_manifest_hash,
            anchor_frames=anchors,
            strategy=config.strategy,
            tracking_direction=config.tracking_direction,
            model_configuration=self._worker_configuration(config),
            postprocessing_configuration=cast(
                dict[str, object],
                self._postprocessing_configuration(config),
            ),
            output_directory="observations/raw",
            seed=config.seed,
        )
        request_path = context.path("observations", "sam3_request.json")
        atomic_write_json(request_path, request)
        image_identifier = self._docker_image_identifier(config)
        command = self._inference_command(context, config)
        try:
            run_process(
                command,
                context=context,
                name="sam3_worker",
                log_directory="observations/raw/logs",
                redact_values=_redaction_values(context),
            )
        except ProcessExecutionError as exc:
            raise _classify_worker_failure(exc) from exc

        raw_result_path = context.path("observations", "raw", "worker_result.json")
        raw_manifest_path = context.path("observations", "raw", "worker_manifest.json")
        if not raw_result_path.is_file() or not raw_manifest_path.is_file():
            raise RuntimeError(
                "SAM worker completed without worker_result.json and worker_manifest.json"
            )
        try:
            raw_result = Sam3RawResult.model_validate_json(
                raw_result_path.read_text(encoding="utf-8")
            )
            worker_manifest = Sam3WorkerManifest.model_validate_json(
                raw_manifest_path.read_text(encoding="utf-8")
            )
        except ValueError as exc:
            raise RuntimeError(f"SAM worker output is malformed: {exc}") from exc
        self._validate_worker_manifest(config, request, worker_manifest)
        if image_identifier is not None:
            worker_manifest = worker_manifest.model_copy(
                update={"image_identifier": image_identifier}
            )
        atomic_write_json(
            context.path("observations", "worker_manifest.json"),
            worker_manifest,
        )
        postprocessing = TrackPostprocessingConfig(
            score_threshold=config.score_threshold,
            mask_threshold=config.mask_threshold,
            min_mask_area_pixels=config.min_mask_area_pixels,
            max_mask_area_ratio=config.max_mask_area_ratio,
            min_track_observations=config.min_track_observations,
            min_track_coverage=config.min_track_coverage,
            same_prompt_duplicate_iou=config.same_prompt_duplicate_iou,
            model_box_mask_iou_threshold=config.model_box_mask_iou_threshold,
        )
        safe_config = _safe_configuration(config)
        tracks, dropped = canonicalize_worker_result(
            raw_result,
            frame_manifest,
            camera,
            prompts,
            root=context.run_dir,
            config=postprocessing,
            adapter_name=self.name,
            adapter_version=self.version,
            provenance_configuration=safe_config,
            provenance_timestamp=frame_manifest.provenance.timestamp,
        )
        for track in tracks:
            for observation in track.observations:
                frame = next(
                    frame
                    for frame in frame_manifest.frames
                    if frame.frame_id == observation.frame_id
                )
                validate_canonical_mask(
                    context.path(observation.mask_path),
                    expected_size=(frame.width, frame.height),
                    expected_area=observation.mask_area_pixels,
                    expected_bbox=observation.bbox_xywh,
                )
        confidence = (
            sum(track.confidence.score for track in tracks) / len(tracks) if tracks else 1.0
        )
        provenance = ProvenanceRecord(
            adapter_name=self.name,
            adapter_version=self.version,
            configuration=safe_config,
            input_artifact_paths=[
                "inputs/manifest.json",
                "inputs/frame_qa.json",
                "camera/reconstruction.json",
                *sorted(prompts.input_hashes),
            ],
            output_artifact_paths=[
                "observations/object_tracks.json",
                "observations/diagnostics.json",
                "observations/worker_manifest.json",
            ],
            timestamp=frame_manifest.provenance.timestamp,
            confidence=ConfidenceRecord(
                score=confidence,
                method="mean_canonical_track_confidence_or_valid_empty_result",
            ),
            source=GeometrySourceType.GENERATED,
        )
        artifact = SegmentationTrackingArtifact(
            frame_count=len(frame_manifest.frames),
            tracks=tracks,
            prompt_manifest_path="observations/prompts.json",
            worker_manifest_path="observations/worker_manifest.json",
            diagnostics_path="observations/diagnostics.json",
            provenance=provenance,
        )
        seen_prompt_ids = {track.prompt_id for track in tracks}
        diagnostics = SegmentationDiagnostics(
            backend_mode=config.execution_mode,
            input_frame_count=len(frame_manifest.frames),
            registered_frame_count=len(camera.registered_frame_ids),
            unregistered_frame_count=len(camera.unregistered_frame_ids),
            prompt_count=len([prompt for prompt in prompts.prompts if prompt.enabled]),
            anchor_frames=anchor_diagnostics,
            raw_track_count=len(raw_result.tracks),
            kept_track_count=len(tracks),
            dropped_tracks=dropped,
            mask_count=sum(track.observation_count for track in tracks),
            mean_coverage=(
                sum(track.coverage_ratio for track in tracks) / len(tracks) if tracks else 0.0
            ),
            mean_confidence=(
                sum(track.confidence.score for track in tracks) / len(tracks) if tracks else 0.0
            ),
            runtime_seconds=worker_manifest.runtime_seconds,
            peak_gpu_memory_bytes=worker_manifest.peak_gpu_memory_bytes,
            thresholds=self._postprocessing_configuration(config),
            no_matching_prompt_ids=sorted(
                prompt.prompt_id
                for prompt in prompts.prompts
                if prompt.enabled and prompt.prompt_id not in seen_prompt_ids
            ),
            warnings=[*worker_manifest.warnings, *raw_result.warnings],
        )
        atomic_write_json(context.path("observations", "object_tracks.json"), artifact)
        atomic_write_json(context.path("observations", "diagnostics.json"), diagnostics)
        preview_paths = render_previews(
            context.run_dir,
            frame_manifest,
            artifact,
            camera,
            include_frame_previews=config.generate_frame_previews,
        )

        dynamic_outputs: list[OutputSpec] = []
        declared_paths = {spec.relative_path for spec in self.expected_outputs(context)}
        for path in sorted(context.path("observations", "prompt_inputs").rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(context.run_dir).as_posix()
            is_mask = path.suffix.lower() == ".png"
            dynamic_outputs.append(
                OutputSpec(
                    relative,
                    "segmentation_seed_mask" if is_mask else "segmentation_prompt_manifest_source",
                    "image/png" if is_mask else "application/yaml",
                    "configured_input",
                )
            )
        for path in sorted(context.path("observations", "raw").rglob("*")):
            if path.is_file():
                relative = path.relative_to(context.run_dir).as_posix()
                dynamic_outputs.append(
                    OutputSpec(
                        relative,
                        "sam3_raw_output",
                        "application/octet-stream",
                        "sam3",
                    )
                )
        for track in tracks:
            for observation in track.observations:
                dynamic_outputs.append(
                    OutputSpec(
                        observation.mask_path,
                        "canonical_object_mask",
                        "image/png",
                        "sam3",
                    )
                )
        for preview_path in preview_paths:
            if preview_path not in declared_paths:
                dynamic_outputs.append(
                    OutputSpec(
                        preview_path,
                        "segmentation_preview",
                        "image/png",
                        "sam3",
                        validation="png",
                    )
                )
        return StageResult(
            outputs=dynamic_outputs,
            metrics={
                "prompt_count": diagnostics.prompt_count,
                "raw_track_count": diagnostics.raw_track_count,
                "track_count": diagnostics.kept_track_count,
                "dropped_track_count": len(dropped),
                "mask_count": diagnostics.mask_count,
            },
        )

    def _normalized_prompts(
        self,
        context: StageContext,
    ) -> SegmentationPromptManifest:
        source_path = context.path("observations", "prompt_inputs", "prompts.yaml")
        source_hash = sha256_file(source_path)
        manifest = load_prompt_manifest(source_path)
        prompts: list[SegmentationPrompt] = []
        input_hashes = {"observations/prompt_inputs/prompts.yaml": source_hash}
        for prompt in manifest.prompts:
            if prompt.mask_path is None:
                prompts.append(prompt)
                continue
            normalized_path = f"observations/prompt_inputs/masks/{prompt.prompt_id}.png"
            input_hashes[normalized_path] = sha256_file(context.path(normalized_path))
            prompts.append(
                prompt.model_copy(
                    update={
                        "mask_path": normalized_path,
                    }
                )
            )
        return SegmentationPromptManifest(
            prompts=prompts,
            source_path="observations/prompt_inputs/prompts.yaml",
            source_sha256=source_hash,
            input_hashes=input_hashes,
        )

    @staticmethod
    def _postprocessing_configuration(
        config: Sam3AdapterConfig,
    ) -> dict[str, float | int]:
        return {
            "score_threshold": config.score_threshold,
            "mask_threshold": config.mask_threshold,
            "min_mask_area_pixels": config.min_mask_area_pixels,
            "max_mask_area_ratio": config.max_mask_area_ratio,
            "min_track_observations": config.min_track_observations,
            "min_track_coverage": config.min_track_coverage,
            "same_prompt_duplicate_iou": config.same_prompt_duplicate_iou,
            "model_box_mask_iou_threshold": config.model_box_mask_iou_threshold,
        }

    @staticmethod
    def _worker_configuration(config: Sam3AdapterConfig) -> dict[str, Any]:
        access_mode: str
        if config.execution_mode == "fake_worker":
            access_mode = "fake"
        elif config.local_checkpoint_path is not None:
            access_mode = "local_path"
        elif config.offline:
            access_mode = "offline_cache"
        else:
            access_mode = "authenticated_remote"
        local_checkpoint_path = config.local_checkpoint_path
        model_cache_path = config.model_cache_path
        if config.execution_mode == "docker":
            if local_checkpoint_path is not None:
                local_checkpoint_path = f"/checkpoints/{Path(local_checkpoint_path).name}"
            if model_cache_path is not None:
                model_cache_path = "/model-cache"
        return {
            "official_repository": OFFICIAL_REPOSITORY,
            "official_code_commit": OFFICIAL_CODE_COMMIT,
            "checkpoint_repository": config.checkpoint_repository,
            "checkpoint_revision": config.checkpoint_revision,
            "checkpoint_access_mode": access_mode,
            "local_checkpoint_path": local_checkpoint_path,
            "model_cache_path": model_cache_path,
            "offline": config.offline,
            "model_mode": config.model_mode,
            "device": config.device,
            "precision": config.precision,
            "seed": config.seed,
            "fake_mode": config.fake_mode,
        }

    def _inference_command(
        self,
        context: StageContext,
        config: Sam3AdapterConfig,
    ) -> list[str]:
        if config.execution_mode != "docker":
            command_or_error = self._local_worker_command(
                config,
                "infer",
                Path("observations/sam3_request.json"),
            )
            if isinstance(command_or_error, str):
                raise RuntimeError(command_or_error)
            return command_or_error
        docker = resolve_executable(config.docker_executable)
        if docker is None:
            raise RuntimeError(f"Docker executable {config.docker_executable!r} was not found")
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
            *self._docker_cache_arguments(config),
            *self._docker_checkpoint_arguments(config),
            *self._docker_environment_arguments(context),
            "--entrypoint",
            "python",
            config.docker_image,
            "-m",
            config.worker_module,
            "infer",
            "--request",
            "/workspace/observations/sam3_request.json",
            "--output-dir",
            "/workspace/observations/raw",
        ]

    @staticmethod
    def _local_worker_command(
        config: Sam3AdapterConfig,
        action: str,
        path: Path,
    ) -> list[str] | str:
        python = _resolve_worker_python(config.worker_python)
        if python is None:
            return f"configured worker Python {config.worker_python!r} was not found"
        option = "--config" if action == "healthcheck" else "--request"
        if config.execution_mode == "fake_worker":
            assert config.worker_script is not None
            script = Path(config.worker_script).expanduser()
            if not script.is_absolute():
                script = Path.cwd() / script
            script = script.resolve()
            if not script.is_file():
                return f"configured fake worker script does not exist: {script}"
            command = [python, str(script), action, option, str(path)]
        else:
            command = [
                python,
                "-m",
                config.worker_module,
                action,
                option,
                str(path),
            ]
        if action == "infer":
            command.extend(["--output-dir", "observations/raw"])
        return command

    @staticmethod
    def _docker_user_arguments() -> list[str]:
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            return ["--user", f"{os.getuid()}:{os.getgid()}"]
        return []

    @staticmethod
    def _docker_environment_arguments(context: StageContext) -> list[str]:
        arguments: list[str] = []
        for name in context.config.adapter.env:
            if name in os.environ:
                arguments.extend(["-e", name])
        return arguments

    @staticmethod
    def _docker_cache_arguments(config: Sam3AdapterConfig) -> list[str]:
        if config.model_cache_path is None:
            return []
        return [
            "-v",
            f"{Path(config.model_cache_path).expanduser().resolve()}:/model-cache:rw",
        ]

    @staticmethod
    def _docker_checkpoint_arguments(config: Sam3AdapterConfig) -> list[str]:
        if config.local_checkpoint_path is None:
            return []
        checkpoint = Path(config.local_checkpoint_path).expanduser().resolve()
        return ["-v", f"{checkpoint}:/checkpoints/{checkpoint.name}:ro"]

    @staticmethod
    def _docker_image_identifier(config: Sam3AdapterConfig) -> str | None:
        if config.execution_mode != "docker":
            return None
        docker = resolve_executable(config.docker_executable)
        if docker is None:
            raise RuntimeError(f"Docker executable {config.docker_executable!r} was not found")
        try:
            result = subprocess.run(
                [
                    docker,
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    config.docker_image,
                ],
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Docker image inspection failed: {exc}") from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"Docker image {config.docker_image!r} could not be inspected: "
                f"{result.stderr.strip()}"
            )
        identifier = result.stdout.strip()
        return identifier or None

    @staticmethod
    def _redact(text: str, values: tuple[str, ...]) -> str:
        for value in values:
            if value:
                text = text.replace(value, "[REDACTED]")
        return text

    @staticmethod
    def _validate_worker_manifest(
        config: Sam3AdapterConfig,
        request: Sam3InferenceRequest,
        manifest: Sam3WorkerManifest,
    ) -> None:
        expected_access = request.model_configuration["checkpoint_access_mode"]
        mismatches = {
            "official_repository": (
                manifest.official_repository,
                OFFICIAL_REPOSITORY,
            ),
            "official_code_commit": (
                manifest.official_code_commit,
                OFFICIAL_CODE_COMMIT,
            ),
            "checkpoint_repository": (
                manifest.checkpoint_repository,
                config.checkpoint_repository,
            ),
            "checkpoint_revision": (
                manifest.checkpoint_revision,
                config.checkpoint_revision,
            ),
            "checkpoint_access_mode": (
                manifest.checkpoint_access_mode,
                expected_access,
            ),
            "prompt_manifest_hash": (
                manifest.prompt_manifest_hash,
                request.prompt_manifest_sha256,
            ),
            "frame_manifest_hash": (
                manifest.frame_manifest_hash,
                request.frame_manifest_sha256,
            ),
            "strategy": (manifest.strategy, config.strategy),
            "model_mode": (manifest.model_mode, config.model_mode),
            "device": (manifest.device, config.device),
            "precision": (manifest.precision, config.precision),
        }
        invalid = [
            f"{name}: worker={actual!r}, expected={expected!r}"
            for name, (actual, expected) in mismatches.items()
            if actual != expected
        ]
        if invalid:
            raise RuntimeError(
                "SAM worker manifest is inconsistent with the request: " + "; ".join(invalid)
            )
