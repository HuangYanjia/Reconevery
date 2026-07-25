from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from PIL import Image, ImageDraw, ImageFilter

from recon2sim.artifacts import (
    AnchorFrameDiagnostic,
    CameraReconstruction,
    DroppedTrackDiagnostic,
    FrameQualityEntry,
    FrameQualityReport,
    IngestManifest,
    Sam3AnchorFrame,
    Sam3RawObservation,
    Sam3RawResult,
    SegmentationObservation,
    SegmentationPrompt,
    SegmentationPromptManifest,
    SegmentationTrack,
    SegmentationTrackingArtifact,
)
from recon2sim.ir import ConfidenceRecord, GeometrySourceType, ProvenanceRecord
from recon2sim.storage import atomic_write_json

AnchorStrategy = Literal[
    "first_frame",
    "first_registered_frame",
    "best_quality_frame",
    "best_quality_registered_frame",
    "explicit",
]


@dataclass(frozen=True)
class TrackPostprocessingConfig:
    score_threshold: float = 0.5
    mask_threshold: float = 0.5
    min_mask_area_pixels: int = 32
    max_mask_area_ratio: float = 0.98
    min_track_observations: int = 2
    min_track_coverage: float = 0.1
    same_prompt_duplicate_iou: float = 0.9
    model_box_mask_iou_threshold: float = 0.05


@dataclass
class _CandidateObservation:
    raw: Sam3RawObservation
    frame_index: int
    binary_mask: Image.Image
    bbox_xywh: tuple[int, int, int, int]
    area: int
    area_ratio: float
    centroid: tuple[float, float]
    camera_pose_available: bool


@dataclass
class _CandidateTrack:
    raw_model_object_id: str
    prompt: SegmentationPrompt
    semantic_label: str
    normalized_label: str
    observations: list[_CandidateObservation]
    mean_score: float
    minimum_score: float
    coverage: float

    @property
    def sort_key(self) -> tuple[str, str, int, float, float, int, str]:
        first = self.observations[0]
        return (
            self.normalized_label,
            self.prompt.prompt_id,
            first.frame_index,
            first.centroid[0],
            first.centroid[1],
            first.area,
            self.raw_model_object_id,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompt_manifest(path: Path) -> SegmentationPromptManifest:
    if not path.is_file():
        raise FileNotFoundError(f"prompt manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"prompt manifest must contain a mapping: {path}")
    return SegmentationPromptManifest.model_validate(payload)


def validate_prompt_references(
    prompt_manifest: SegmentationPromptManifest,
    frame_manifest: IngestManifest,
    *,
    prompt_root: Path,
) -> None:
    frames = {frame.frame_id: frame for frame in frame_manifest.frames}
    for prompt in prompt_manifest.prompts:
        if not prompt.enabled:
            continue
        if prompt.frame_id is not None and prompt.frame_id not in frames:
            raise ValueError(
                f"prompt {prompt.prompt_id!r} references unknown frame {prompt.frame_id!r}"
            )
        frame = frames.get(prompt.frame_id or "")
        if prompt.box_xyxy is not None and frame is not None:
            x0, y0, x1, y1 = prompt.box_xyxy
            if x0 < 0 or y0 < 0 or x1 > frame.width or y1 > frame.height:
                raise ValueError(
                    f"prompt {prompt.prompt_id!r} box is outside "
                    f"{frame.frame_id} ({frame.width}x{frame.height})"
                )
        if prompt.points is not None and frame is not None:
            for point in prompt.points:
                if point.x >= frame.width or point.y >= frame.height:
                    raise ValueError(
                        f"prompt {prompt.prompt_id!r} point ({point.x}, {point.y}) is outside "
                        f"{frame.frame_id} ({frame.width}x{frame.height})"
                    )
        if prompt.mask_path is not None and frame is not None:
            mask_path = prompt_root / prompt.mask_path
            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"prompt {prompt.prompt_id!r} seed mask does not exist: {mask_path}"
                )
            with Image.open(mask_path) as image:
                image.load()
                if image.size != (frame.width, frame.height):
                    raise ValueError(
                        f"prompt {prompt.prompt_id!r} seed mask dimensions {image.size} do not "
                        f"match {frame.frame_id} ({frame.width}, {frame.height})"
                    )
                values = set(image.convert("L").tobytes())
            if not values <= {0, 255}:
                raise ValueError(
                    f"prompt {prompt.prompt_id!r} seed mask must contain only 0 and 255"
                )


def frame_quality_score(entry: FrameQualityEntry, camera_pose_available: bool) -> float:
    blur = entry.blur_score / (entry.blur_score + 100.0)
    variance = entry.grayscale_variance / (entry.grayscale_variance + 500.0)
    brightness = max(0.0, 1.0 - abs(entry.mean_brightness - 127.5) / 127.5)
    registration = 1.0 if camera_pose_available else 0.0
    return 0.4 * blur + 0.25 * variance + 0.2 * brightness + 0.15 * registration


def select_anchor_frames(
    frame_manifest: IngestManifest,
    quality_report: FrameQualityReport,
    camera: CameraReconstruction,
    *,
    strategy: AnchorStrategy,
    anchor_count: int = 1,
    explicit_frame_ids: list[str] | None = None,
) -> tuple[list[Sam3AnchorFrame], list[AnchorFrameDiagnostic]]:
    if anchor_count <= 0:
        raise ValueError("anchor_count must be positive")
    frames = frame_manifest.frames
    frame_ids = [frame.frame_id for frame in frames]
    frame_index = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    registered = set(camera.registered_frame_ids)
    quality = {entry.frame_id: entry for entry in quality_report.entries}
    missing_quality = set(frame_ids) - set(quality)
    if missing_quality:
        raise ValueError(f"frame QA is missing selected frames: {sorted(missing_quality)}")

    if strategy == "explicit":
        selected_ids = explicit_frame_ids or []
        if not selected_ids:
            raise ValueError("explicit anchor strategy requires explicit_frame_ids")
        unknown = set(selected_ids) - set(frame_ids)
        if unknown:
            raise ValueError(f"explicit anchors reference unknown frames: {sorted(unknown)}")
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("explicit anchor frame IDs must be unique")
    elif strategy == "first_frame":
        selected_ids = frame_ids[:anchor_count]
    elif strategy == "first_registered_frame":
        selected_ids = [frame_id for frame_id in frame_ids if frame_id in registered][:anchor_count]
    else:
        candidates = frame_ids
        if strategy == "best_quality_registered_frame":
            candidates = [frame_id for frame_id in frame_ids if frame_id in registered]
        selected_ids = sorted(
            candidates,
            key=lambda frame_id: (
                -frame_quality_score(quality[frame_id], frame_id in registered),
                frame_index[frame_id],
            ),
        )[:anchor_count]
    if not selected_ids:
        raise ValueError(
            f"anchor strategy {strategy!r} produced no frames; camera registration may be empty"
        )

    anchors: list[Sam3AnchorFrame] = []
    diagnostics: list[AnchorFrameDiagnostic] = []
    for frame_id in selected_ids:
        score = frame_quality_score(quality[frame_id], frame_id in registered)
        reason = (
            "explicitly configured"
            if strategy == "explicit"
            else f"selected by {strategy} using documented frame-quality score"
        )
        anchors.append(
            Sam3AnchorFrame(
                frame_id=frame_id,
                score=score,
                camera_pose_available=frame_id in registered,
                selection_reason=reason,
            )
        )
        diagnostics.append(
            AnchorFrameDiagnostic(
                frame_id=frame_id,
                strategy=strategy,
                selection_score=score,
                selection_reason=reason,
                camera_pose_available=frame_id in registered,
            )
        )
    return anchors, diagnostics


def normalize_semantic_label(label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", label.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "object"


def _bbox_and_centroid(
    mask: Image.Image,
) -> tuple[tuple[int, int, int, int], int, tuple[float, float]]:
    pixels = mask.load()
    if pixels is None:
        raise ValueError("mask pixels could not be loaded")
    width, height = mask.size
    min_x, min_y = width, height
    max_x = max_y = -1
    area = 0
    sum_x = 0
    sum_y = 0
    for y in range(height):
        for x in range(width):
            if pixels[x, y] == 0:
                continue
            area += 1
            sum_x += x
            sum_y += y
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    if area == 0:
        raise ValueError("mask is empty")
    return (
        (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1),
        area,
        (sum_x / area, sum_y / area),
    )


def _box_iou_xyxy(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _mask_iou(first: Image.Image, second: Image.Image) -> float:
    if first.size != second.size:
        return 0.0
    first_data = first.tobytes()
    second_data = second.tobytes()
    intersection = 0
    union = 0
    for first_pixel, second_pixel in zip(first_data, second_data, strict=True):
        first_active = first_pixel != 0
        second_active = second_pixel != 0
        intersection += int(first_active and second_active)
        union += int(first_active or second_active)
    return intersection / union if union else 0.0


def _drop(
    raw_model_object_id: str,
    semantic_label: str,
    prompt_id: str,
    reason_code: str,
    explanation: str,
) -> DroppedTrackDiagnostic:
    return DroppedTrackDiagnostic(
        raw_model_object_id=raw_model_object_id,
        semantic_label=semantic_label,
        prompt_id=prompt_id,
        reason_code=reason_code,
        explanation=explanation,
    )


def _load_raw_mask(
    root: Path,
    observation: Sam3RawObservation,
    expected_size: tuple[int, int],
    mask_threshold: float,
) -> Image.Image:
    path = root / observation.mask_path
    if not path.is_file():
        raise FileNotFoundError(f"worker mask does not exist: {observation.mask_path}")
    with Image.open(path) as image:
        image.load()
        if image.size != expected_size:
            raise ValueError(f"mask dimensions {image.size} do not match expected {expected_size}")
        grayscale = image.convert("L")
        values = set(grayscale.tobytes())
        if observation.mask_encoding == "binary_png":
            if not values <= {0, 255}:
                raise ValueError("binary worker mask contains values other than 0 and 255")
            return grayscale.copy()
        threshold = round(mask_threshold * 255)
        return grayscale.point(lambda value: 255 if value >= threshold else 0, mode="L")


def _candidate_from_raw_track(
    raw_track_index: int,
    raw_result: Sam3RawResult,
    prompt_by_id: dict[str, SegmentationPrompt],
    frame_manifest: IngestManifest,
    camera: CameraReconstruction,
    root: Path,
    config: TrackPostprocessingConfig,
) -> tuple[_CandidateTrack | None, DroppedTrackDiagnostic | None]:
    raw_track = raw_result.tracks[raw_track_index]
    prompt = prompt_by_id.get(raw_track.prompt_id)
    if prompt is None:
        return None, _drop(
            raw_track.raw_model_object_id,
            raw_track.semantic_label,
            raw_track.prompt_id,
            "missing_prompt",
            "worker track references a prompt that is not in the validated manifest",
        )
    if raw_track.semantic_label != prompt.label:
        return None, _drop(
            raw_track.raw_model_object_id,
            raw_track.semantic_label,
            raw_track.prompt_id,
            "prompt_label_mismatch",
            "worker semantic label does not match the validated prompt label",
        )
    if not raw_track.observations:
        return None, _drop(
            raw_track.raw_model_object_id,
            raw_track.semantic_label,
            raw_track.prompt_id,
            "no_observations",
            "worker object contains no observations",
        )
    frames = {frame.frame_id: frame for frame in frame_manifest.frames}
    frame_index = {frame.frame_id: index for index, frame in enumerate(frame_manifest.frames)}
    registered = set(camera.registered_frame_ids)
    seen_frames: set[str] = set()
    observations: list[_CandidateObservation] = []
    threshold = (
        prompt.confidence_threshold
        if prompt.confidence_threshold is not None
        else config.score_threshold
    )
    try:
        for raw_observation in raw_track.observations:
            if raw_observation.raw_model_object_id != raw_track.raw_model_object_id:
                raise ValueError("observation raw object ID does not match its track")
            if raw_observation.prompt_id != raw_track.prompt_id:
                raise ValueError("observation prompt ID does not match its track")
            if raw_observation.semantic_label != raw_track.semantic_label:
                raise ValueError("observation semantic label does not match its track")
            if raw_observation.frame_id not in frames:
                raise ValueError(
                    f"observation references unknown frame {raw_observation.frame_id!r}"
                )
            if raw_observation.frame_id in seen_frames:
                raise ValueError(f"duplicate observation for frame {raw_observation.frame_id!r}")
            seen_frames.add(raw_observation.frame_id)
            if not math.isfinite(raw_observation.score):
                raise ValueError("observation score is non-finite")
            if not 0 <= raw_observation.score <= 1:
                raise ValueError("observation score is outside [0, 1]")
            if raw_observation.score < threshold:
                continue
            frame = frames[raw_observation.frame_id]
            mask = _load_raw_mask(
                root,
                raw_observation,
                (frame.width, frame.height),
                config.mask_threshold,
            )
            bbox, area, centroid = _bbox_and_centroid(mask)
            area_ratio = area / (frame.width * frame.height)
            if area < config.min_mask_area_pixels:
                raise ValueError(
                    f"mask area {area} is below min_mask_area_pixels={config.min_mask_area_pixels}"
                )
            if area_ratio > config.max_mask_area_ratio:
                raise ValueError(
                    f"mask area ratio {area_ratio:.6f} exceeds max_mask_area_ratio="
                    f"{config.max_mask_area_ratio}"
                )
            if raw_observation.model_box_xyxy is not None:
                model_box = raw_observation.model_box_xyxy
                if (
                    any(not math.isfinite(value) for value in model_box)
                    or model_box[0] < 0
                    or model_box[1] < 0
                    or model_box[2] > frame.width
                    or model_box[3] > frame.height
                    or model_box[2] <= model_box[0]
                    or model_box[3] <= model_box[1]
                ):
                    raise ValueError("model box is invalid or outside the frame")
                x, y, width, height = bbox
                mask_box = (float(x), float(y), float(x + width), float(y + height))
                if _box_iou_xyxy(model_box, mask_box) < config.model_box_mask_iou_threshold:
                    raise ValueError("model box is inconsistent with its mask")
            observations.append(
                _CandidateObservation(
                    raw=raw_observation,
                    frame_index=frame_index[raw_observation.frame_id],
                    binary_mask=mask,
                    bbox_xywh=bbox,
                    area=area,
                    area_ratio=area_ratio,
                    centroid=centroid,
                    camera_pose_available=raw_observation.frame_id in registered,
                )
            )
    except (OSError, ValueError, FileNotFoundError) as exc:
        return None, _drop(
            raw_track.raw_model_object_id,
            raw_track.semantic_label,
            raw_track.prompt_id,
            "invalid_observation",
            str(exc),
        )

    observations.sort(key=lambda item: item.frame_index)
    if len(observations) < config.min_track_observations:
        return None, _drop(
            raw_track.raw_model_object_id,
            raw_track.semantic_label,
            raw_track.prompt_id,
            "short_track",
            f"{len(observations)} observations is below "
            f"min_track_observations={config.min_track_observations}",
        )
    coverage = len(observations) / len(frame_manifest.frames)
    if coverage < config.min_track_coverage:
        return None, _drop(
            raw_track.raw_model_object_id,
            raw_track.semantic_label,
            raw_track.prompt_id,
            "insufficient_coverage",
            f"coverage {coverage:.6f} is below min_track_coverage={config.min_track_coverage}",
        )
    scores = [observation.raw.score for observation in observations]
    return (
        _CandidateTrack(
            raw_model_object_id=raw_track.raw_model_object_id,
            prompt=prompt,
            semantic_label=prompt.label,
            normalized_label=normalize_semantic_label(prompt.label),
            observations=observations,
            mean_score=sum(scores) / len(scores),
            minimum_score=min(scores),
            coverage=coverage,
        ),
        None,
    )


def _same_deduplication_group(first: _CandidateTrack, second: _CandidateTrack) -> bool:
    if first.prompt.prompt_id == second.prompt.prompt_id:
        return True
    return bool(
        first.prompt.synonym_group and first.prompt.synonym_group == second.prompt.synonym_group
    )


def _tracks_duplicate(
    first: _CandidateTrack,
    second: _CandidateTrack,
    threshold: float,
) -> bool:
    if not _same_deduplication_group(first, second):
        return False
    second_by_frame = {observation.raw.frame_id: observation for observation in second.observations}
    common = [
        (observation, second_by_frame[observation.raw.frame_id])
        for observation in first.observations
        if observation.raw.frame_id in second_by_frame
    ]
    return bool(common) and all(
        _mask_iou(first_observation.binary_mask, second_observation.binary_mask) >= threshold
        for first_observation, second_observation in common
    )


def canonicalize_worker_result(
    raw_result: Sam3RawResult,
    frame_manifest: IngestManifest,
    camera: CameraReconstruction,
    prompt_manifest: SegmentationPromptManifest,
    *,
    root: Path,
    config: TrackPostprocessingConfig,
    adapter_name: str,
    adapter_version: str,
    provenance_configuration: dict[str, Any],
    provenance_timestamp: datetime,
) -> tuple[list[SegmentationTrack], list[DroppedTrackDiagnostic]]:
    prompt_by_id = {
        prompt.prompt_id: prompt for prompt in prompt_manifest.prompts if prompt.enabled
    }
    candidates: list[_CandidateTrack] = []
    dropped: list[DroppedTrackDiagnostic] = []
    for index in range(len(raw_result.tracks)):
        candidate, diagnostic = _candidate_from_raw_track(
            index,
            raw_result,
            prompt_by_id,
            frame_manifest,
            camera,
            root,
            config,
        )
        if diagnostic is not None:
            dropped.append(diagnostic)
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(key=lambda item: (-item.mean_score, item.sort_key))
    kept: list[_CandidateTrack] = []
    for candidate in candidates:
        duplicate = next(
            (
                existing
                for existing in kept
                if _tracks_duplicate(
                    candidate,
                    existing,
                    config.same_prompt_duplicate_iou,
                )
            ),
            None,
        )
        if duplicate is None:
            kept.append(candidate)
            continue
        dropped.append(
            _drop(
                candidate.raw_model_object_id,
                candidate.semantic_label,
                candidate.prompt.prompt_id,
                "duplicate_track",
                f"near-identical masks duplicate raw object "
                f"{duplicate.raw_model_object_id!r} in the same prompt or synonym group",
            )
        )

    limited: list[_CandidateTrack] = []
    prompt_counts: dict[str, int] = {}
    for candidate in kept:
        count = prompt_counts.get(candidate.prompt.prompt_id, 0)
        limit = candidate.prompt.instance_limit
        if limit is not None and count >= limit:
            dropped.append(
                _drop(
                    candidate.raw_model_object_id,
                    candidate.semantic_label,
                    candidate.prompt.prompt_id,
                    "instance_limit",
                    f"prompt instance_limit={limit} was already reached by higher-ranked tracks",
                )
            )
            continue
        prompt_counts[candidate.prompt.prompt_id] = count + 1
        limited.append(candidate)
    kept = limited
    kept.sort(key=lambda item: item.sort_key)
    label_counts: dict[str, int] = {}
    tracks: list[SegmentationTrack] = []
    for candidate in kept:
        label_counts[candidate.normalized_label] = (
            label_counts.get(candidate.normalized_label, 0) + 1
        )
        object_id = f"{candidate.normalized_label}_{label_counts[candidate.normalized_label]:04d}"
        canonical_observations: list[SegmentationObservation] = []
        for observation in candidate.observations:
            mask_path = f"observations/masks/{object_id}/{observation.raw.frame_id}.png"
            destination = root / mask_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            observation.binary_mask.save(
                destination,
                format="PNG",
                compress_level=6,
                optimize=False,
            )
            canonical_observations.append(
                SegmentationObservation(
                    frame_id=observation.raw.frame_id,
                    mask_path=mask_path,
                    bbox_xywh=observation.bbox_xywh,
                    model_box_xyxy=observation.raw.model_box_xyxy,
                    frame_score=observation.raw.score,
                    mask_area_pixels=observation.area,
                    mask_area_ratio=observation.area_ratio,
                    camera_pose_available=observation.camera_pose_available,
                    occluded=observation.raw.occluded,
                    raw_model_object_id=candidate.raw_model_object_id,
                    prompt_id=candidate.prompt.prompt_id,
                )
            )
        confidence_score = 0.8 * candidate.mean_score + 0.2 * candidate.coverage
        output_paths = [observation.mask_path for observation in canonical_observations]
        provenance = ProvenanceRecord(
            adapter_name=adapter_name,
            adapter_version=adapter_version,
            configuration=provenance_configuration,
            input_artifact_paths=[
                "inputs/manifest.json",
                "inputs/frame_qa.json",
                "camera/reconstruction.json",
                "observations/prompts.json",
                "observations/raw/worker_result.json",
            ],
            output_artifact_paths=output_paths,
            timestamp=provenance_timestamp,
            confidence=ConfidenceRecord(
                score=confidence_score,
                method="0.8_mean_frame_score_plus_0.2_track_coverage",
            ),
            source=GeometrySourceType.GENERATED,
        )
        tracks.append(
            SegmentationTrack(
                object_id=object_id,
                semantic_label=candidate.semantic_label,
                normalized_semantic_label=candidate.normalized_label,
                prompt_id=candidate.prompt.prompt_id,
                raw_model_object_id=candidate.raw_model_object_id,
                asset_type_hint=candidate.prompt.asset_type_hint,
                asset_type_hint_source=(
                    "configured_semantic_hint"
                    if candidate.prompt.asset_type_hint is not None
                    else None
                ),
                first_frame_id=canonical_observations[0].frame_id,
                last_frame_id=canonical_observations[-1].frame_id,
                observation_count=len(canonical_observations),
                coverage_ratio=candidate.coverage,
                mean_score=candidate.mean_score,
                minimum_score=candidate.minimum_score,
                observations=canonical_observations,
                provenance=provenance,
                confidence=ConfidenceRecord(
                    score=confidence_score,
                    method="0.8_mean_frame_score_plus_0.2_track_coverage",
                ),
            )
        )
    dropped.sort(
        key=lambda item: (
            item.semantic_label,
            item.prompt_id,
            item.raw_model_object_id,
            item.reason_code,
        )
    )
    return tracks, dropped


def validate_canonical_mask(
    path: Path,
    *,
    expected_size: tuple[int, int],
    expected_area: int | None = None,
    expected_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[int, tuple[int, int, int, int]]:
    with Image.open(path) as image:
        image.load()
        if image.mode != "L":
            raise ValueError(f"canonical mask must use grayscale mode L, found {image.mode}")
        if image.size != expected_size:
            raise ValueError(f"canonical mask dimensions {image.size} do not match {expected_size}")
        if not set(image.tobytes()) <= {0, 255}:
            raise ValueError("canonical mask must contain only 0 and 255")
        bbox, area, _ = _bbox_and_centroid(image)
    if expected_area is not None and area != expected_area:
        raise ValueError(f"canonical mask area {area} does not match reported {expected_area}")
    if expected_bbox is not None and bbox != expected_bbox:
        raise ValueError(f"canonical mask box {bbox} does not match reported {expected_bbox}")
    return area, bbox


def _object_color(object_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(object_id.encode("utf-8")).digest()
    return (
        64 + digest[0] % 160,
        64 + digest[1] % 160,
        64 + digest[2] % 160,
    )


def _render_frame(
    root: Path,
    frame_path: str,
    frame_id: str,
    tracks: list[SegmentationTrack],
) -> Image.Image:
    with Image.open(root / frame_path) as source:
        rendered = source.convert("RGB")
    draw = ImageDraw.Draw(rendered)
    label_y = 4.0
    for track in tracks:
        observation = next(
            (item for item in track.observations if item.frame_id == frame_id),
            None,
        )
        if observation is None:
            continue
        color = _object_color(track.object_id)
        with Image.open(root / observation.mask_path) as mask_file:
            mask = mask_file.convert("L")
            outline = mask.filter(ImageFilter.FIND_EDGES).point(
                lambda value: 255 if value else 0,
                mode="L",
            )
        color_layer = Image.new("RGB", rendered.size, color)
        rendered.paste(color_layer, mask=outline)
        x, y, width, height = observation.bbox_xywh
        draw.rectangle((x, y, x + width - 1, y + height - 1), outline=color, width=2)
        label = f"{track.object_id} {track.semantic_label} {observation.frame_score:.2f}"
        text_bbox = draw.textbbox((4, label_y), label)
        draw.rectangle(text_bbox, fill=(0, 0, 0))
        draw.text((4, label_y), label, fill=color)
        label_y = text_bbox[3] + 3
    frame_label = f"{frame_id}"
    text_bbox = draw.textbbox((rendered.width - 4, rendered.height - 4), frame_label, anchor="rb")
    draw.rectangle(text_bbox, fill=(0, 0, 0))
    draw.text(
        (rendered.width - 4, rendered.height - 4),
        frame_label,
        fill=(255, 255, 255),
        anchor="rb",
    )
    return rendered


def render_previews(
    root: Path,
    frame_manifest: IngestManifest,
    artifact: SegmentationTrackingArtifact,
    camera: CameraReconstruction,
    *,
    include_frame_previews: bool = True,
) -> list[str]:
    preview_root = root / "observations" / "previews"
    preview_root.mkdir(parents=True, exist_ok=True)
    frame_ids = [frame.frame_id for frame in frame_manifest.frames]
    representative_indexes = sorted(
        {
            round(index * (len(frame_ids) - 1) / max(1, min(12, len(frame_ids)) - 1))
            for index in range(min(12, len(frame_ids)))
        }
    )
    rendered: dict[str, Image.Image] = {}
    output_paths: list[str] = []
    for frame in frame_manifest.frames:
        if include_frame_previews or frame_ids.index(frame.frame_id) in representative_indexes:
            rendered[frame.frame_id] = _render_frame(
                root,
                frame.relative_path,
                frame.frame_id,
                artifact.tracks,
            )
        if include_frame_previews:
            output_path = f"observations/previews/frames/{frame.frame_id}.png"
            destination = root / output_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            rendered[frame.frame_id].save(
                destination, format="PNG", compress_level=6, optimize=False
            )
            output_paths.append(output_path)

    tiles = [rendered[frame_ids[index]] for index in representative_indexes]
    tile_width = 240
    resized_tiles: list[Image.Image] = []
    for tile in tiles:
        resized = tile.copy()
        resized.thumbnail((tile_width, 180), Image.Resampling.LANCZOS)
        resized_tiles.append(resized)
    columns = min(4, max(1, len(resized_tiles)))
    rows = math.ceil(len(resized_tiles) / columns)
    cell_height = max((tile.height for tile in resized_tiles), default=180)
    contact_sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * cell_height),
        (24, 24, 24),
    )
    for index, tile in enumerate(resized_tiles):
        contact_sheet.paste(
            tile,
            (index % columns * tile_width, index // columns * cell_height),
        )
    contact_path = "observations/previews/contact_sheet.png"
    contact_sheet.save(root / contact_path, format="PNG", compress_level=6, optimize=False)
    output_paths.append(contact_path)

    cell_width = max(3, min(12, 1200 // max(1, len(frame_ids))))
    label_width = 180
    row_height = 18
    timeline = Image.new(
        "RGB",
        (label_width + cell_width * len(frame_ids), row_height * (len(artifact.tracks) + 2)),
        (250, 250, 250),
    )
    draw = ImageDraw.Draw(timeline)
    registered = set(camera.registered_frame_ids)
    draw.text((4, 2), "camera pose", fill=(0, 0, 0))
    for frame_index, frame_id in enumerate(frame_ids):
        x = label_width + frame_index * cell_width
        draw.rectangle(
            (x, 0, x + cell_width - 1, row_height - 1),
            fill=(45, 140, 80) if frame_id in registered else (170, 170, 170),
        )
    for row_index, track in enumerate(artifact.tracks, start=1):
        y = row_index * row_height
        color = _object_color(track.object_id)
        draw.text((4, y + 2), track.object_id, fill=(0, 0, 0))
        visible = {observation.frame_id for observation in track.observations}
        for frame_index, frame_id in enumerate(frame_ids):
            x = label_width + frame_index * cell_width
            draw.rectangle(
                (x, y, x + cell_width - 1, y + row_height - 1),
                fill=color if frame_id in visible else (225, 225, 225),
            )
    timeline_path = "observations/previews/track_timeline.png"
    timeline.save(root / timeline_path, format="PNG", compress_level=6, optimize=False)
    output_paths.append(timeline_path)
    return output_paths


def _uncompressed_rle(mask: Image.Image) -> list[int]:
    pixels = mask.load()
    if pixels is None:
        raise ValueError("mask pixels could not be loaded")
    width, height = mask.size
    counts: list[int] = []
    current = 0
    run = 0
    for x in range(width):
        for y in range(height):
            value = 1 if pixels[x, y] else 0
            if value == current:
                run += 1
            else:
                counts.append(run)
                run = 1
                current = value
    counts.append(run)
    return counts


def export_coco(
    root: Path,
    frame_manifest: IngestManifest,
    artifact: SegmentationTrackingArtifact,
    output: Path,
) -> None:
    frame_index = {frame.frame_id: index + 1 for index, frame in enumerate(frame_manifest.frames)}
    labels = sorted({track.normalized_semantic_label for track in artifact.tracks})
    category_index = {label: index + 1 for index, label in enumerate(labels)}
    images = [
        {
            "id": index + 1,
            "file_name": frame.relative_path,
            "width": frame.width,
            "height": frame.height,
        }
        for index, frame in enumerate(frame_manifest.frames)
    ]
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for track in sorted(artifact.tracks, key=lambda item: item.object_id):
        for observation in sorted(
            track.observations,
            key=lambda item: frame_index[item.frame_id],
        ):
            with Image.open(root / observation.mask_path) as mask_file:
                mask = mask_file.convert("L")
                counts = _uncompressed_rle(mask)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": frame_index[observation.frame_id],
                    "category_id": category_index[track.normalized_semantic_label],
                    "track_id": track.object_id,
                    "bbox": list(observation.bbox_xywh),
                    "area": observation.mask_area_pixels,
                    "iscrowd": 0,
                    "segmentation": {
                        "size": [mask.height, mask.width],
                        "counts": counts,
                    },
                }
            )
            annotation_id += 1
    atomic_write_json(
        output,
        {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": category_index[label], "name": label} for label in labels],
        },
    )


def parse_tracking_artifact(path: Path) -> SegmentationTrackingArtifact:
    return SegmentationTrackingArtifact.model_validate_json(path.read_text(encoding="utf-8"))
