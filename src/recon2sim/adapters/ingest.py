from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from PIL import Image, ImageOps, UnidentifiedImageError

from recon2sim.adapters.base import HealthcheckResult, OutputSpec, StageContext, StageResult
from recon2sim.adapters.process import ProcessResult, run_external_process
from recon2sim.artifacts import (
    FrameManifestEntry,
    FrameQualityEntry,
    FrameQualityReport,
    IngestManifest,
    InputSourceType,
)
from recon2sim.ir import ConfidenceRecord, GeometrySourceType, ProvenanceRecord
from recon2sim.storage import atomic_write_json

InputMode = Literal["video", "image_directory", "mock"]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
UNSUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".gif", ".heic", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class InputDetection:
    mode: InputMode
    sources: list[Path]


@dataclass(frozen=True)
class FrameCandidate:
    raw_path: Path
    source_reference: str
    original_index: int
    timestamp_s: float


@dataclass(frozen=True)
class FrameMetrics:
    blur_score: float
    mean_brightness: float
    intensity_variance: float
    signature: tuple[float, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_digest(root: Path, sources: list[Path]) -> str:
    if root.is_file():
        return _sha256(root)
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(source)))
    return digest.hexdigest()


def _relative_source(root: Path, source: Path) -> str:
    return source.name if root.is_file() else source.relative_to(root).as_posix()


def deterministic_frame_name(index: int) -> str:
    if index < 0:
        raise ValueError("frame indices must be non-negative")
    return f"frame_{index:06d}.png"


def detect_input_mode(input_path: Path, requested: str = "auto") -> InputDetection:
    if requested not in {"auto", "video", "image_directory", "mock"}:
        raise ValueError(
            f"unsupported input_mode {requested!r}; choose auto, video, image_directory, or mock"
        )
    if not input_path.exists():
        raise FileNotFoundError(f"input path does not exist: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(
                f"input file {input_path} is not a supported video; supported extensions are "
                f"{sorted(VIDEO_EXTENSIONS)}"
            )
        if requested in {"image_directory", "mock"}:
            raise ValueError(f"input_mode={requested} requires a directory of JPEG or PNG images")
        return InputDetection("video", [input_path])

    files = sorted(path for path in input_path.rglob("*") if path.is_file())
    videos = [path for path in files if path.suffix.lower() in VIDEO_EXTENSIONS]
    images = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
    unsupported_images = [
        path for path in files if path.suffix.lower() in UNSUPPORTED_IMAGE_EXTENSIONS
    ]
    if unsupported_images:
        references = [path.relative_to(input_path).as_posix() for path in unsupported_images[:5]]
        raise ValueError(
            f"unsupported image formats were found; Phase 1 accepts JPEG and PNG only: {references}"
        )
    if requested == "video":
        if len(videos) != 1:
            raise ValueError(
                f"input_mode=video requires exactly one supported video in {input_path}; "
                f"found {len(videos)}"
            )
        return InputDetection("video", videos)
    if requested in {"image_directory", "mock"}:
        if not images:
            raise ValueError(f"input_mode={requested} found no JPEG or PNG images in {input_path}")
        return InputDetection(cast(InputMode, requested), images)
    if videos and images:
        raise ValueError(
            f"input auto-detection found both videos and images in {input_path}; set input_mode "
            "explicitly"
        )
    if len(videos) > 1:
        raise ValueError(
            f"input auto-detection found {len(videos)} videos in {input_path}; keep one video or "
            "set up separate runs"
        )
    if videos:
        return InputDetection("video", videos)
    if images:
        return InputDetection("image_directory", images)
    raise ValueError(
        f"no supported video, JPEG, or PNG input was found in {input_path}; "
        f"video extensions: {sorted(VIDEO_EXTENSIONS)}, image extensions: "
        f"{sorted(IMAGE_EXTENSIONS)}"
    )


def _tool_version(executable: str, arguments: list[str]) -> tuple[bool, str]:
    resolved = shutil.which(executable)
    if resolved is None:
        return False, f"{executable!r} was not found on PATH"
    try:
        completed = subprocess.run(
            [resolved, *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"could not execute {resolved}: {exc}"
    output = completed.stdout or completed.stderr
    first_line = output.splitlines()[0] if output.splitlines() else "version unavailable"
    return completed.returncode == 0, f"{resolved}: {first_line}"


def _version_line(result: ProcessResult) -> str:
    lines = (result.stdout or result.stderr).splitlines()
    return lines[0] if lines else "unknown"


def _resize(image: Image.Image, max_edge: int | None) -> Image.Image:
    converted = ImageOps.exif_transpose(image).convert("RGB")
    if max_edge is None or max(converted.size) <= max_edge:
        return converted
    converted.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return converted


def _save_normalized(source: Path, destination: Path, max_edge: int | None) -> None:
    try:
        with Image.open(source) as image:
            normalized = _resize(image, max_edge)
            normalized.load()
            destination.parent.mkdir(parents=True, exist_ok=True)
            normalized.save(destination, format="PNG", compress_level=6, optimize=False)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"could not decode supported image {source}: {exc}") from exc


def _frame_metrics(path: Path) -> FrameMetrics:
    try:
        with Image.open(path) as image:
            gray = image.convert("L")
            gray.thumbnail((256, 256), Image.Resampling.BILINEAR)
            width, height = gray.size
            flattened = cast(tuple[int, ...], gray.get_flattened_data())
            pixels = [float(value) for value in flattened]
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"could not decode extracted frame {path}: {exc}") from exc
    if not pixels:
        raise ValueError(f"extracted frame has no pixels: {path}")
    mean = sum(pixels) / len(pixels)
    variance = sum((value - mean) ** 2 for value in pixels) / len(pixels)
    laplacians: list[float] = []
    for y in range(1, height - 1):
        offset = y * width
        for x in range(1, width - 1):
            index = offset + x
            laplacians.append(
                pixels[index - width]
                + pixels[index + width]
                + pixels[index - 1]
                + pixels[index + 1]
                - 4 * pixels[index]
            )
    if laplacians:
        laplacian_mean = sum(laplacians) / len(laplacians)
        blur_score = sum((value - laplacian_mean) ** 2 for value in laplacians) / len(laplacians)
    else:
        blur_score = 0.0
    with Image.open(path) as image:
        signature_image = image.convert("L").resize((16, 16), Image.Resampling.BILINEAR)
        flattened_signature = cast(tuple[int, ...], signature_image.get_flattened_data())
        signature = tuple(float(value) for value in flattened_signature)
    return FrameMetrics(blur_score, mean, variance, signature)


def _duplicate_score(current: tuple[float, ...], previous: tuple[float, ...]) -> float:
    if len(current) != len(previous) or not current:
        raise ValueError("frame QA signatures must have equal non-zero lengths")
    mean_absolute_difference = sum(
        abs(current_value - previous_value)
        for current_value, previous_value in zip(current, previous, strict=True)
    ) / len(current)
    return max(0.0, min(1.0, 1.0 - mean_absolute_difference / 255.0))


def _read_exif_time(path: Path) -> datetime | None:
    try:
        with Image.open(path) as image:
            value = image.getexif().get(36867) or image.getexif().get(306)
    except (OSError, UnidentifiedImageError):
        return None
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y:%m:%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def _json_spec(
    path: str, artifact_type: str, model: type[IngestManifest] | type[FrameQualityReport]
) -> OutputSpec:
    return OutputSpec(
        relative_path=path,
        artifact_type=artifact_type,
        media_type="application/json",
        source_type="measured",
        validation="json",
        schema_identifier=f"recon2sim/{artifact_type.replace('_', '-')}/0.1.0",
        model=model,
    )


def _plain_output(path: str, artifact_type: str, media_type: str) -> OutputSpec:
    return OutputSpec(path, artifact_type, media_type, "measured")


class FfmpegIngestAdapter:
    name = "ffmpeg_ingest"
    version = "0.1.0"

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        if context is not None:
            requested = str(context.config.adapter.config.get("input_mode", "auto"))
            detection = detect_input_mode(context.input_dir, requested)
            if detection.mode != "video":
                return HealthcheckResult(
                    True,
                    f"Pillow image ingest ready for {len(detection.sources)} source images",
                )
            ffmpeg = str(context.config.adapter.config.get("ffmpeg_executable", "ffmpeg"))
            ffprobe = str(context.config.adapter.config.get("ffprobe_executable", "ffprobe"))
        else:
            ffmpeg = "ffmpeg"
            ffprobe = "ffprobe"
        ffmpeg_ok, ffmpeg_message = _tool_version(ffmpeg, ["-version"])
        ffprobe_ok, ffprobe_message = _tool_version(ffprobe, ["-version"])
        if not (ffmpeg_ok and ffprobe_ok):
            return HealthcheckResult(
                False,
                f"FFmpeg video ingest unavailable ({ffmpeg_message}; {ffprobe_message}). "
                "Install FFmpeg and ensure both ffmpeg and ffprobe are on PATH; image-directory "
                "ingest remains available.",
            )
        return HealthcheckResult(True, f"{ffmpeg_message}; {ffprobe_message}")

    def prepare(self, context: StageContext) -> None:
        context.attempt_dir.mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            _json_spec("inputs/manifest.json", "ingest_manifest", IngestManifest),
            _json_spec("inputs/frame_qa.json", "frame_quality_report", FrameQualityReport),
        ]

    def run(self, context: StageContext) -> StageResult:
        config = context.config.adapter.config
        requested = str(config.get("input_mode", "auto"))
        detection = detect_input_mode(context.input_dir, requested)
        max_frames = int(config.get("max_frames", 300))
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        target_fps = float(config.get("target_fps", 3.0))
        if not math.isfinite(target_fps) or target_fps <= 0:
            raise ValueError("target_fps must be a positive finite number")
        resize_value = config.get("resize_max_edge", 1920)
        resize_max_edge = None if resize_value is None else int(resize_value)
        if resize_max_edge is not None and resize_max_edge <= 0:
            raise ValueError("resize_max_edge must be positive or null")
        image_format = str(config.get("image_format", "png")).lower()
        if image_format != "png":
            raise ValueError("Phase 1 normalized ingest supports image_format=png only")
        scene_change_value = config.get("scene_change_threshold")
        scene_change_threshold = None if scene_change_value is None else float(scene_change_value)
        if scene_change_threshold is not None and not 0 <= scene_change_threshold <= 1:
            raise ValueError("scene_change_threshold must be in [0, 1] or null")

        command_results: list[ProcessResult] = []
        ffmpeg_version: str | None = None
        ffprobe_version: str | None = None
        extraction_configuration = dict(config)
        if detection.mode == "video":
            candidates, command_results, video_metadata = self._extract_video(
                context,
                detection.sources[0],
                target_fps=target_fps,
                max_frames=max_frames,
                resize_max_edge=resize_max_edge,
                scene_change_threshold=scene_change_threshold,
            )
            ffmpeg_version = cast(str, video_metadata["ffmpeg_version"])
            ffprobe_version = cast(str, video_metadata["ffprobe_version"])
            extraction_configuration.update(video_metadata)
            source_type = InputSourceType.VIDEO
        else:
            candidates = self._normalize_images(
                context,
                detection.sources[:max_frames],
                target_fps=target_fps,
                resize_max_edge=resize_max_edge,
            )
            source_type = (
                InputSourceType.MOCK
                if detection.mode == "mock"
                else InputSourceType.IMAGE_DIRECTORY
            )
        if not candidates:
            raise RuntimeError("ingest produced no decodable frame candidates")

        frame_entries, qa_report, frame_outputs = self._run_qa(
            context,
            candidates,
            source_type,
        )
        qa_path = "inputs/frame_qa.json"
        atomic_write_json(context.output_path(qa_path), qa_report)
        if not frame_entries:
            raise RuntimeError(
                "frame QA rejected every extracted frame; inspect inputs/frame_qa.json in the "
                "attempt workspace and tune conservative dataset-specific thresholds"
            )

        manifest_path = "inputs/manifest.json"
        source_root = context.input_dir
        source_reference = (
            _relative_source(source_root, detection.sources[0])
            if detection.mode == "video"
            else "."
        )
        source_hash = _input_digest(source_root, detection.sources)
        output_hashes = {entry.relative_path: entry.sha256 for entry in frame_entries}
        output_paths = [manifest_path, qa_path, *output_hashes]
        provenance = ProvenanceRecord(
            adapter_name=self.name,
            adapter_version=self.version,
            configuration=config,
            input_artifact_paths=[],
            output_artifact_paths=output_paths,
            confidence=ConfidenceRecord(score=1.0, method="deterministic_decode_and_hash"),
            source=GeometrySourceType.MEASURED,
        )
        manifest = IngestManifest(
            source_type=source_type,
            frames=frame_entries,
            source_input_reference=source_reference,
            source_sha256=source_hash,
            ffmpeg_version=ffmpeg_version,
            ffprobe_version=ffprobe_version,
            extraction_configuration=extraction_configuration,
            total_decoded_frames=len(candidates),
            selected_frames=len(frame_entries),
            dropped_frames=len(candidates) - len(frame_entries),
            output_hashes=output_hashes,
            frame_qa_path=qa_path,
            provenance=provenance,
        )
        atomic_write_json(context.output_path(manifest_path), manifest)

        command_outputs: list[OutputSpec] = []
        for result in command_results:
            for path, kind in (
                (result.stdout_path, "tool_stdout"),
                (result.stderr_path, "tool_stderr"),
            ):
                relative = path.relative_to(context.attempt_dir).as_posix()
                command_outputs.append(_plain_output(relative, kind, "text/plain"))
        return StageResult(
            outputs=[*frame_outputs, *command_outputs],
            metrics={
                "decoded_frames": len(candidates),
                "selected_frames": len(frame_entries),
                "dropped_frames": len(candidates) - len(frame_entries),
                "input_mode": detection.mode,
            },
        )

    def _extract_video(
        self,
        context: StageContext,
        source: Path,
        *,
        target_fps: float,
        max_frames: int,
        resize_max_edge: int | None,
        scene_change_threshold: float | None,
    ) -> tuple[list[FrameCandidate], list[ProcessResult], dict[str, Any]]:
        config = context.config.adapter.config
        ffmpeg = str(config.get("ffmpeg_executable", "ffmpeg"))
        ffprobe = str(config.get("ffprobe_executable", "ffprobe"))
        env_names = context.config.adapter.env
        timeout = context.config.adapter.timeout_s
        logs = context.output_path("inputs", "logs")

        ffprobe_version_result = run_external_process(
            [ffprobe, "-version"],
            cwd=context.attempt_dir,
            timeout_s=min(timeout, 30),
            environment_names=env_names,
            stdout_path=logs / "ffprobe_version.stdout.log",
            stderr_path=logs / "ffprobe_version.stderr.log",
            command_name="ffprobe version check",
        )
        ffmpeg_version_result = run_external_process(
            [ffmpeg, "-version"],
            cwd=context.attempt_dir,
            timeout_s=min(timeout, 30),
            environment_names=env_names,
            stdout_path=logs / "ffmpeg_version.stdout.log",
            stderr_path=logs / "ffmpeg_version.stderr.log",
            command_name="ffmpeg version check",
        )
        probe_arguments = [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(source.resolve()),
        ]
        probe_result = run_external_process(
            probe_arguments,
            cwd=context.attempt_dir,
            timeout_s=timeout,
            environment_names=env_names,
            stdout_path=logs / "ffprobe_metadata.stdout.log",
            stderr_path=logs / "ffprobe_metadata.stderr.log",
            command_name="ffprobe metadata inspection",
        )
        try:
            probe_payload = json.loads(probe_result.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"ffprobe returned malformed JSON for {source}; see {probe_result.stdout_path}"
            ) from exc
        streams = probe_payload.get("streams") if isinstance(probe_payload, dict) else None
        if not isinstance(streams, list) or not any(
            isinstance(stream, dict) and stream.get("codec_type") == "video" for stream in streams
        ):
            raise ValueError(f"ffprobe found no readable video stream in {source}")

        raw_root = context.output_path("raw_frames")
        raw_root.mkdir(parents=True, exist_ok=False)
        filters = [f"fps={target_fps:g}"]
        if scene_change_threshold is not None:
            filters.append(f"select=eq(n\\,0)+gt(scene\\,{scene_change_threshold:g})")
        if resize_max_edge is not None:
            filters.append(
                f"scale=w=min({resize_max_edge}\\,iw):h=min({resize_max_edge}\\,ih):"
                "force_original_aspect_ratio=decrease"
            )
        output_pattern = raw_root / "frame_%06d.png"
        ffmpeg_arguments = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "info",
            "-n",
            "-i",
            str(source.resolve()),
            "-vf",
            ",".join(filters),
            "-frames:v",
            str(max_frames),
            "-start_number",
            "0",
            "-fps_mode",
            "vfr",
            str(output_pattern),
        ]
        extraction_result = run_external_process(
            ffmpeg_arguments,
            cwd=context.attempt_dir,
            timeout_s=timeout,
            environment_names=env_names,
            stdout_path=logs / "ffmpeg_extract.stdout.log",
            stderr_path=logs / "ffmpeg_extract.stderr.log",
            command_name="FFmpeg frame extraction",
        )
        raw_frames = sorted(raw_root.glob("frame_*.png"))
        if not raw_frames:
            raise RuntimeError(
                f"FFmpeg completed but extracted no frames from {source}; inspect "
                f"{extraction_result.stderr_path}"
            )
        source_reference = _relative_source(context.input_dir, source)
        candidates = [
            FrameCandidate(
                path,
                source_reference,
                index,
                index / target_fps,
            )
            for index, path in enumerate(raw_frames)
        ]
        results = [
            ffprobe_version_result,
            ffmpeg_version_result,
            probe_result,
            extraction_result,
        ]
        metadata = {
            "ffmpeg_version": _version_line(ffmpeg_version_result),
            "ffprobe_version": _version_line(ffprobe_version_result),
            "ffprobe_command": probe_arguments,
            "ffmpeg_command": ffmpeg_arguments,
            "probed_metadata": probe_payload,
        }
        return candidates, results, metadata

    def _normalize_images(
        self,
        context: StageContext,
        sources: list[Path],
        *,
        target_fps: float,
        resize_max_edge: int | None,
    ) -> list[FrameCandidate]:
        raw_root = context.output_path("raw_frames")
        raw_root.mkdir(parents=True, exist_ok=False)
        exif_times = [_read_exif_time(source) for source in sources]
        known_times = [value for value in exif_times if value is not None]
        origin = min(known_times) if known_times else None
        candidates: list[FrameCandidate] = []
        for index, source in enumerate(sources):
            destination = raw_root / deterministic_frame_name(index)
            _save_normalized(source, destination, resize_max_edge)
            exif_time = exif_times[index]
            timestamp = (
                max(0.0, (exif_time - origin).total_seconds())
                if exif_time is not None and origin is not None
                else index / target_fps
            )
            candidates.append(
                FrameCandidate(
                    destination,
                    _relative_source(context.input_dir, source),
                    index,
                    timestamp,
                )
            )
        return candidates

    def _run_qa(
        self,
        context: StageContext,
        candidates: list[FrameCandidate],
        source_type: InputSourceType,
    ) -> tuple[list[FrameManifestEntry], FrameQualityReport, list[OutputSpec]]:
        config = context.config.adapter.config
        blur_threshold = float(config.get("blur_threshold", 0.0))
        duplicate_threshold_value = config.get("duplicate_threshold", 0.995)
        duplicate_threshold = (
            None if duplicate_threshold_value is None else float(duplicate_threshold_value)
        )
        min_brightness = float(config.get("min_brightness", 5.0))
        max_brightness = float(config.get("max_brightness", 250.0))
        keep_rejected = bool(config.get("keep_rejected_frames", True))
        if blur_threshold < 0:
            raise ValueError("blur_threshold must be non-negative")
        if duplicate_threshold is not None and not 0 <= duplicate_threshold <= 1:
            raise ValueError("duplicate_threshold must be in [0, 1] or null")
        if not 0 <= min_brightness <= max_brightness <= 255:
            raise ValueError("brightness thresholds must satisfy 0 <= min <= max <= 255")

        entries: list[FrameManifestEntry] = []
        qa_entries: list[FrameQualityEntry] = []
        outputs: list[OutputSpec] = []
        previous_selected_signature: tuple[float, ...] | None = None
        for candidate in candidates:
            metrics = _frame_metrics(candidate.raw_path)
            duplicate_score = (
                _duplicate_score(metrics.signature, previous_selected_signature)
                if previous_selected_signature is not None
                else None
            )
            rejection_reason: str | None = None
            if metrics.mean_brightness < min_brightness:
                rejection_reason = "brightness_below_minimum"
            elif metrics.mean_brightness > max_brightness:
                rejection_reason = "brightness_above_maximum"
            elif metrics.blur_score < blur_threshold:
                rejection_reason = "blur_score_below_threshold"
            elif (
                duplicate_threshold is not None
                and duplicate_score is not None
                and duplicate_score >= duplicate_threshold
            ):
                rejection_reason = "near_duplicate"
            selected = rejection_reason is None
            frame_name = deterministic_frame_name(candidate.original_index)
            normalized_path: str | None = None
            rejected_path: str | None = None
            if selected:
                normalized_path = f"frames/{frame_name}"
                destination = context.output_path(normalized_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(candidate.raw_path, destination)
                with Image.open(destination) as image:
                    width, height = image.size
                entries.append(
                    FrameManifestEntry(
                        frame_id=Path(frame_name).stem,
                        relative_path=normalized_path,
                        sha256=_sha256(destination),
                        width=width,
                        height=height,
                        timestamp_s=candidate.timestamp_s,
                        source_type=source_type,
                        source_file_reference=candidate.source_reference,
                        original_frame_index=candidate.original_index,
                    )
                )
                outputs.append(
                    OutputSpec(
                        normalized_path,
                        "normalized_input_frame",
                        "image/png",
                        "measured",
                        validation="png",
                    )
                )
                previous_selected_signature = metrics.signature
            elif keep_rejected:
                rejected_path = f"diagnostics/rejected_frames/{frame_name}"
                destination = context.output_path(rejected_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(candidate.raw_path, destination)
                outputs.append(
                    OutputSpec(
                        rejected_path,
                        "rejected_input_frame",
                        "image/png",
                        "measured",
                        validation="png",
                    )
                )
            qa_entries.append(
                FrameQualityEntry(
                    frame_id=Path(frame_name).stem,
                    source_file_reference=candidate.source_reference,
                    normalized_path=normalized_path,
                    rejected_path=rejected_path,
                    original_frame_index=candidate.original_index,
                    blur_score=metrics.blur_score,
                    mean_brightness=metrics.mean_brightness,
                    intensity_variance=metrics.intensity_variance,
                    duplicate_score=duplicate_score,
                    is_duplicate=rejection_reason == "near_duplicate",
                    selected=selected,
                    rejection_reason=rejection_reason,
                )
            )

        provenance = ProvenanceRecord(
            adapter_name=self.name,
            adapter_version=self.version,
            configuration={
                "blur_threshold": blur_threshold,
                "duplicate_threshold": duplicate_threshold,
                "min_brightness": min_brightness,
                "max_brightness": max_brightness,
                "keep_rejected_frames": keep_rejected,
            },
            input_artifact_paths=[],
            output_artifact_paths=["inputs/frame_qa.json"],
            confidence=ConfidenceRecord(score=1.0, method="deterministic_cpu_statistics"),
            source=GeometrySourceType.MEASURED,
        )
        report = FrameQualityReport(
            thresholds={
                "blur_threshold": blur_threshold,
                "duplicate_threshold": (
                    duplicate_threshold if duplicate_threshold is not None else False
                ),
                "min_brightness": min_brightness,
                "max_brightness": max_brightness,
                "keep_rejected_frames": keep_rejected,
            },
            entries=qa_entries,
            selected_count=len(entries),
            dropped_count=len(candidates) - len(entries),
            provenance=provenance,
        )
        return entries, report, outputs
