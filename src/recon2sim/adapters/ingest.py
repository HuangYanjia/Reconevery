from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError
from pydantic import Field, model_validator

from recon2sim.adapters.base import HealthcheckResult, OutputSpec, StageContext, StageResult
from recon2sim.adapters.process import terminate_process_group
from recon2sim.artifacts import (
    FrameManifestEntry,
    FrameQualityEntry,
    FrameQualityReport,
    FrameSelectionStatus,
    IngestManifest,
    InputSourceType,
)
from recon2sim.frame_qa import measure_frame, normalized_signature_difference
from recon2sim.ir import ConfidenceRecord, GeometrySourceType, ProvenanceRecord, StrictModel
from recon2sim.lineage import frame_sequence_digest
from recon2sim.storage import atomic_write_json

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


class FFmpegIngestConfig(StrictModel):
    input_mode: Literal["auto", "video", "image_directory", "mock"] = "auto"
    executable: str = "ffmpeg"
    ffprobe_executable: str = "ffprobe"
    target_fps: float = Field(default=3.0, gt=0)
    max_frames: int = Field(default=300, gt=0)
    image_format: Literal["png"] = "png"
    resize_max_edge: int | None = Field(default=1920, gt=0)
    scene_change_threshold: float | None = Field(default=None, ge=0, le=1)
    blur_threshold: float = Field(default=0.0, ge=0)
    duplicate_threshold: float = Field(default=0.0, ge=0, le=1)
    min_brightness: float = Field(default=5.0, ge=0, le=255)
    max_brightness: float = Field(default=250.0, ge=0, le=255)
    keep_rejected_frames: bool = True

    @model_validator(mode="after")
    def valid_brightness_range(self) -> FFmpegIngestConfig:
        if self.min_brightness > self.max_brightness:
            raise ValueError("min_brightness must not exceed max_brightness")
        return self


@dataclass(frozen=True)
class DetectedInput:
    source_type: InputSourceType
    video: Path | None
    images: tuple[Path, ...]


@dataclass(frozen=True)
class ProcessResult:
    command: list[str]
    return_code: int
    duration_s: float
    timed_out: bool
    stdout: str
    stderr: str


class ProcessExecutionError(RuntimeError):
    def __init__(self, message: str, result: ProcessResult) -> None:
        super().__init__(message)
        self.result = result


def deterministic_frame_name(index: int) -> str:
    if index < 0:
        raise ValueError("frame index must be non-negative")
    return f"frame_{index:06d}.png"


def detect_input(path: Path, mode: str = "auto") -> DetectedInput:
    if mode == "mock":
        raise ValueError("input_mode=mock must use the existing mock_ingest adapter")
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in VIDEO_EXTENSIONS and mode in {"auto", "video"}:
            return DetectedInput(InputSourceType.VIDEO, path, ())
        if suffix in IMAGE_EXTENSIONS and mode in {"auto", "image_directory"}:
            return DetectedInput(InputSourceType.IMAGE_DIRECTORY, None, (path,))
        raise ValueError(
            f"unsupported input file {path}; video extensions: {sorted(VIDEO_EXTENSIONS)}, "
            f"image extensions: {sorted(IMAGE_EXTENSIONS)}"
        )
    if not path.is_dir():
        raise FileNotFoundError(f"input path does not exist: {path}")

    videos = tuple(
        candidate
        for candidate in sorted(path.rglob("*"))
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS
    )
    image_root = path / "images" if (path / "images").is_dir() else path
    images = tuple(
        candidate
        for candidate in sorted(image_root.rglob("*"))
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
    )
    if mode == "video":
        images = ()
    elif mode == "image_directory":
        videos = ()
    if videos and images:
        raise ValueError(f"input {path} contains both videos and images; set input_mode explicitly")
    if len(videos) > 1:
        raise ValueError(f"input {path} contains multiple videos; keep exactly one: {videos}")
    if videos:
        return DetectedInput(InputSourceType.VIDEO, videos[0], ())
    if images:
        return DetectedInput(InputSourceType.IMAGE_DIRECTORY, None, images)
    raise FileNotFoundError(
        f"no supported video or JPEG/PNG images found under {path}; "
        f"supported video extensions: {sorted(VIDEO_EXTENSIONS)}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collection_hash(root: Path, paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def resolve_executable(value: str) -> str | None:
    candidate = Path(value)
    if candidate.parent != Path("."):
        return str(candidate.resolve()) if candidate.is_file() else None
    return shutil.which(value)


def executable_version(executable: str, flag: str = "-version") -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [executable, flag],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (result.stdout or result.stderr).splitlines()
    first_line = output[0].strip() if output else "version output was empty"
    return result.returncode == 0, first_line


def allowed_environment(context: StageContext) -> dict[str, str]:
    environment = {
        name: os.environ[name] for name in context.config.adapter.env if name in os.environ
    }
    environment["LC_ALL"] = "C"
    return environment


def run_process(
    command: list[str],
    *,
    context: StageContext,
    name: str,
    log_directory: str = "logs",
    redact_values: tuple[str, ...] = (),
) -> ProcessResult:
    start = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=context.run_dir,
        env=allowed_environment(context),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    interruption: KeyboardInterrupt | SystemExit | None = None
    try:
        stdout, stderr = process.communicate(timeout=context.config.adapter.timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        stdout, stderr = terminate_process_group(process)
    except (KeyboardInterrupt, SystemExit) as exc:
        interruption = exc
        stdout, stderr = terminate_process_group(process)
    duration = time.monotonic() - start
    for value in redact_values:
        if value:
            stdout = stdout.replace(value, "[REDACTED]")
            stderr = stderr.replace(value, "[REDACTED]")
    log_root = context.path(log_directory)
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{context.stage_name}.{name}.attempt_{context.attempt}.stdout.log"
    stderr_path = log_root / f"{context.stage_name}.{name}.attempt_{context.attempt}.stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    result = ProcessResult(command, process.returncode, duration, timed_out, stdout, stderr)
    if interruption is not None:
        raise interruption
    if timed_out:
        raise ProcessExecutionError(
            (
                f"{name} timed out after {context.config.adapter.timeout_s} seconds; "
                f"see {stderr_path.relative_to(context.run_dir)}"
            ),
            result,
        )
    if process.returncode != 0:
        raise ProcessExecutionError(
            (
                f"{name} failed with return code {process.returncode}; "
                f"see {stderr_path.relative_to(context.run_dir)}"
            ),
            result,
        )
    return result


def _normalize_image(source: Path, destination: Path, resize_max_edge: int | None) -> None:
    try:
        with Image.open(source) as image:
            image.load()
            normalized = ImageOps.exif_transpose(image)
            if resize_max_edge is not None and max(normalized.size) > resize_max_edge:
                normalized.thumbnail(
                    (resize_max_edge, resize_max_edge),
                    Image.Resampling.LANCZOS,
                )
            normalized = normalized.convert("RGB")
            destination.parent.mkdir(parents=True, exist_ok=True)
            normalized.save(destination, format="PNG", compress_level=6, optimize=False)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"unreadable or unsupported image {source}: {exc}") from exc


def _exif_datetime(path: Path) -> datetime | None:
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            raw = exif.get(ExifTags.Base.DateTimeOriginal) or exif.get(ExifTags.Base.DateTime)
    except (OSError, UnidentifiedImageError):
        return None
    if not isinstance(raw, str):
        return None
    try:
        return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def _provenance(
    config: FFmpegIngestConfig,
    outputs: list[str],
    source: GeometrySourceType,
) -> ProvenanceRecord:
    return ProvenanceRecord(
        adapter_name=FFmpegIngestAdapter.name,
        adapter_version=FFmpegIngestAdapter.version,
        configuration=config.model_dump(mode="json"),
        input_artifact_paths=[],
        output_artifact_paths=outputs,
        confidence=ConfidenceRecord(score=1.0, method="deterministic_ingest"),
        source=source,
    )


class FFmpegIngestAdapter:
    name = "ffmpeg_ingest"
    version = "0.1.2"

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        config = (
            FFmpegIngestConfig.model_validate(context.config.adapter.config)
            if context is not None
            else FFmpegIngestConfig()
        )
        if config.input_mode == "image_directory":
            return HealthcheckResult(True, "Pillow image ingest is available")
        ffmpeg = resolve_executable(config.executable)
        ffprobe = resolve_executable(config.ffprobe_executable)
        if ffmpeg is None:
            return HealthcheckResult(
                False,
                f"FFmpeg executable {config.executable!r} was not found; install FFmpeg or set "
                "stages.ingest.adapter.config.executable",
            )
        if ffprobe is None:
            return HealthcheckResult(
                False,
                f"FFprobe executable {config.ffprobe_executable!r} was not found; install FFmpeg "
                "or set stages.ingest.adapter.config.ffprobe_executable",
            )
        ffmpeg_ok, ffmpeg_version = executable_version(ffmpeg)
        ffprobe_ok, ffprobe_version = executable_version(ffprobe)
        if not ffmpeg_ok or not ffprobe_ok:
            return HealthcheckResult(
                False,
                f"FFmpeg check: {ffmpeg_version}; FFprobe check: {ffprobe_version}",
            )
        return HealthcheckResult(
            True,
            f"ffmpeg={ffmpeg} ({ffmpeg_version}); ffprobe={ffprobe} ({ffprobe_version})",
        )

    def prepare(self, context: StageContext) -> None:
        context.run_dir.mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                "inputs/manifest.json",
                "ingest_manifest",
                "application/json",
                "ingest",
                validation="json",
                schema_identifier="recon2sim/ingest-manifest/0.2.0",
                model=IngestManifest,
            ),
            OutputSpec(
                "inputs/frame_qa.json",
                "frame_quality_report",
                "application/json",
                "ingest",
                validation="json",
                schema_identifier="recon2sim/frame-quality-report/0.1.0",
                model=FrameQualityReport,
            ),
            OutputSpec(
                "inputs/commands.json",
                "ingest_commands",
                "application/json",
                "ingest",
                validation="json",
            ),
        ]

    def run(self, context: StageContext) -> StageResult:
        config = FFmpegIngestConfig.model_validate(context.config.adapter.config)
        detected = detect_input(context.input_dir, config.input_mode)
        command_records: list[dict[str, Any]] = []
        ffmpeg_version: str | None = None
        ffprobe_version: str | None = None
        total_decoded_frames: int | None = None
        source_hash: str
        candidates: list[tuple[Path, str, int, int | None, float]]

        if detected.source_type is InputSourceType.VIDEO:
            assert detected.video is not None
            (
                candidates,
                total_decoded_frames,
                ffmpeg_version,
                ffprobe_version,
                video_commands,
            ) = self._extract_video(context, config, detected.video)
            command_records.extend(video_commands)
            source_hash = _sha256(detected.video)
        else:
            candidates = self._normalize_images(context, config, detected.images)
            source_hash = _collection_hash(context.input_dir, detected.images)

        manifest_frames, quality_entries, output_specs = self._quality_select(
            context,
            config,
            detected.source_type,
            candidates,
        )
        if not manifest_frames:
            raise ValueError(
                "frame QA rejected every extracted frame; inspect the retained attempt workspace "
                "and tune blur/brightness/duplicate thresholds for this dataset"
            )

        output_paths = [
            "inputs/manifest.json",
            "inputs/frame_qa.json",
            "inputs/commands.json",
            *[frame.relative_path for frame in manifest_frames],
        ]
        provenance = _provenance(
            config,
            output_paths,
            GeometrySourceType.MEASURED,
        )
        report = FrameQualityReport(
            entries=quality_entries,
            configuration={
                "blur_threshold": config.blur_threshold,
                "duplicate_threshold": config.duplicate_threshold,
                "min_brightness": config.min_brightness,
                "max_brightness": config.max_brightness,
                "keep_rejected_frames": config.keep_rejected_frames,
            },
            provenance=provenance,
        )
        atomic_write_json(context.path("inputs", "frame_qa.json"), report)

        source_path = detected.video or context.input_dir
        output_hashes = {frame.relative_path: frame.sha256 for frame in manifest_frames}
        manifest = IngestManifest(
            source_type=detected.source_type,
            frames=manifest_frames,
            source_input_path=str(source_path),
            source_hash=source_hash,
            ffmpeg_version=ffmpeg_version,
            ffprobe_version=ffprobe_version,
            extraction_config=config.model_dump(mode="json"),
            total_decoded_frames=total_decoded_frames or len(candidates),
            selected_frames=len(manifest_frames),
            dropped_frames=len(candidates) - len(manifest_frames),
            output_hashes=output_hashes,
            frame_qa_path="inputs/frame_qa.json",
            frame_sequence_digest=frame_sequence_digest(manifest_frames),
            provenance=provenance,
        )
        atomic_write_json(context.path("inputs", "manifest.json"), manifest)
        atomic_write_json(context.path("inputs", "commands.json"), {"commands": command_records})

        for path in sorted(context.path("logs").glob("*")) if context.path("logs").is_dir() else []:
            if path.is_file():
                output_specs.append(
                    OutputSpec(
                        path.relative_to(context.run_dir).as_posix(),
                        "ingest_tool_log",
                        "text/plain",
                        "ingest",
                    )
                )
        return StageResult(
            outputs=output_specs,
            metrics={
                "decoded_frames": len(candidates),
                "selected_frames": len(manifest_frames),
                "dropped_frames": len(candidates) - len(manifest_frames),
            },
        )

    def _extract_video(
        self,
        context: StageContext,
        config: FFmpegIngestConfig,
        video: Path,
    ) -> tuple[
        list[tuple[Path, str, int, int | None, float]],
        int | None,
        str,
        str,
        list[dict[str, Any]],
    ]:
        ffmpeg = resolve_executable(config.executable)
        ffprobe = resolve_executable(config.ffprobe_executable)
        if ffmpeg is None or ffprobe is None:
            raise FileNotFoundError("FFmpeg or FFprobe became unavailable after healthcheck")
        _, ffmpeg_version = executable_version(ffmpeg)
        _, ffprobe_version = executable_version(ffprobe)
        probe_command = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,avg_frame_rate:format=duration",
            "-of",
            "json",
            str(video.resolve()),
        ]
        probe = run_process(probe_command, context=context, name="ffprobe")
        try:
            metadata = json.loads(probe.stdout)
            streams = metadata["streams"]
            if not streams:
                raise ValueError("no video stream")
            stream = streams[0]
        except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"FFprobe returned malformed metadata for {video}: {exc}") from exc
        raw_total = stream.get("nb_frames")
        total = int(raw_total) if isinstance(raw_total, str) and raw_total.isdigit() else None
        raw_frame_rate = stream.get("avg_frame_rate")
        source_fps: float | None = None
        if isinstance(raw_frame_rate, str) and "/" in raw_frame_rate:
            numerator_text, denominator_text = raw_frame_rate.split("/", maxsplit=1)
            try:
                denominator = float(denominator_text)
                source_fps = float(numerator_text) / denominator if denominator else None
            except ValueError:
                source_fps = None

        extracted_root = context.path("inputs", "extracted")
        extracted_root.mkdir(parents=True, exist_ok=False)
        filters = [f"fps={config.target_fps:.12g}"]
        if config.scene_change_threshold is not None:
            threshold = f"{config.scene_change_threshold:.12g}"
            filters.extend(
                [
                    f"select='eq(n\\,0)+gt(scene\\,{threshold})'",
                    f"setpts=N/({config.target_fps:.12g}*TB)",
                ]
            )
        if config.resize_max_edge is not None:
            edge = config.resize_max_edge
            filters.append(
                f"scale='if(gte(iw,ih),min(iw,{edge}),-2)':'if(lt(iw,ih),min(ih,{edge}),-2)'"
            )
        output_pattern = extracted_root / "frame_%06d.png"
        ffmpeg_command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(video.resolve()),
            "-vf",
            ",".join(filters),
            "-frames:v",
            str(config.max_frames),
            "-start_number",
            "0",
            str(output_pattern.resolve()),
        ]
        extraction = run_process(ffmpeg_command, context=context, name="ffmpeg")
        extracted = sorted(extracted_root.glob("frame_*.png"))
        if not extracted:
            raise ValueError(
                f"FFmpeg decoded no frames from {video}; inspect logs for codec/input errors"
            )
        candidates = [
            (
                path,
                video.name,
                index,
                (
                    round(index / config.target_fps * source_fps)
                    if source_fps is not None and config.scene_change_threshold is None
                    else None
                ),
                index / config.target_fps,
            )
            for index, path in enumerate(extracted)
        ]
        records = [
            self._process_record(context, "ffprobe", probe),
            self._process_record(context, "ffmpeg", extraction),
        ]
        return candidates, total, ffmpeg_version, ffprobe_version, records

    def _normalize_images(
        self,
        context: StageContext,
        config: FFmpegIngestConfig,
        images: tuple[Path, ...],
    ) -> list[tuple[Path, str, int, int | None, float]]:
        normalized_root = context.path("inputs", "normalized")
        normalized_root.mkdir(parents=True, exist_ok=False)
        exif_times = [_exif_datetime(path) for path in images]
        known_times = [value for value in exif_times if value is not None]
        baseline = min(known_times) if known_times else None
        candidates: list[tuple[Path, str, int, int | None, float]] = []
        for index, source in enumerate(images[: config.max_frames]):
            destination = normalized_root / deterministic_frame_name(index)
            _normalize_image(source, destination, config.resize_max_edge)
            source_reference = (
                source.name
                if context.input_dir.is_file()
                else source.relative_to(context.input_dir).as_posix()
            )
            exif_time = exif_times[index]
            timestamp = (
                (exif_time - baseline).total_seconds()
                if exif_time is not None and baseline is not None
                else index / config.target_fps
            )
            candidates.append((destination, source_reference, index, index, timestamp))
        return candidates

    def _quality_select(
        self,
        context: StageContext,
        config: FFmpegIngestConfig,
        source_type: InputSourceType,
        candidates: list[tuple[Path, str, int, int | None, float]],
    ) -> tuple[list[FrameManifestEntry], list[FrameQualityEntry], list[OutputSpec]]:
        frames: list[FrameManifestEntry] = []
        quality_entries: list[FrameQualityEntry] = []
        outputs: list[OutputSpec] = []
        previous_signature: tuple[int, ...] | None = None
        previous_frame_id: str | None = None
        for candidate, source_file, output_index, original_index, timestamp in candidates:
            frame_name = deterministic_frame_name(output_index)
            frame_id = frame_name.removesuffix(".png")
            metrics = measure_frame(candidate)
            duplicate_difference = (
                normalized_signature_difference(previous_signature, metrics.signature)
                if previous_signature is not None
                else None
            )
            near_duplicate = (
                duplicate_difference is not None
                and duplicate_difference <= config.duplicate_threshold
            )
            reason: str | None = None
            if metrics.mean_brightness < config.min_brightness:
                reason = "brightness_below_minimum"
            elif metrics.mean_brightness > config.max_brightness:
                reason = "brightness_above_maximum"
            elif metrics.blur_score < config.blur_threshold:
                reason = "blur_below_threshold"
            elif near_duplicate:
                reason = "near_duplicate"

            if reason is None:
                relative_path = f"frames/{frame_name}"
                destination = context.path(relative_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(candidate, destination)
                digest = _sha256(destination)
                with Image.open(destination) as selected_image:
                    width, height = selected_image.size
                frames.append(
                    FrameManifestEntry(
                        frame_id=frame_id,
                        relative_path=relative_path,
                        sha256=digest,
                        width=width,
                        height=height,
                        timestamp_s=timestamp,
                        source_type=source_type,
                        source_file=source_file,
                        original_frame_index=original_index,
                    )
                )
                outputs.append(
                    OutputSpec(
                        relative_path,
                        "input_frame",
                        "image/png",
                        "ingest",
                        validation="png",
                    )
                )
                status = FrameSelectionStatus.SELECTED
                previous_signature = metrics.signature
                previous_frame_id = frame_id
            else:
                relative_path = f"inputs/rejected_frames/{frame_name}"
                if config.keep_rejected_frames:
                    destination = context.path(relative_path)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(candidate, destination)
                    outputs.append(
                        OutputSpec(
                            relative_path,
                            "rejected_input_frame",
                            "image/png",
                            "ingest",
                            validation="png",
                        )
                    )
                else:
                    candidate.unlink()
                status = FrameSelectionStatus.REJECTED
            quality_entries.append(
                FrameQualityEntry(
                    frame_id=frame_id,
                    relative_path=relative_path,
                    blur_score=metrics.blur_score,
                    mean_brightness=metrics.mean_brightness,
                    grayscale_variance=metrics.grayscale_variance,
                    near_duplicate=near_duplicate,
                    duplicate_of_frame_id=previous_frame_id if near_duplicate else None,
                    status=status,
                    rejection_reason=reason,
                )
            )
        return frames, quality_entries, outputs

    @staticmethod
    def _process_record(
        context: StageContext,
        name: str,
        result: ProcessResult,
    ) -> dict[str, Any]:
        prefix = f"logs/{context.stage_name}.{name}.attempt_{context.attempt}"
        return {
            "name": name,
            "command": result.command,
            "return_code": result.return_code,
            "duration_s": result.duration_s,
            "timed_out": result.timed_out,
            "stdout_path": f"{prefix}.stdout.log",
            "stderr_path": f"{prefix}.stderr.log",
        }


__all__ = [
    "DetectedInput",
    "FFmpegIngestAdapter",
    "FFmpegIngestConfig",
    "ProcessExecutionError",
    "ProcessResult",
    "allowed_environment",
    "detect_input",
    "deterministic_frame_name",
    "executable_version",
    "resolve_executable",
    "run_process",
]
