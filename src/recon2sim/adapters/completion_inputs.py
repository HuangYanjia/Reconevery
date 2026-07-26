from __future__ import annotations

import math
from pathlib import Path

from PIL import Image
from pydantic import Field

from recon2sim.adapters.base import (
    HealthcheckResult,
    InputSpec,
    OutputSpec,
    StageContext,
    StageResult,
)
from recon2sim.adapters.completion_common import (
    CompletionWorkerConfig,
    completion_healthcheck,
    worker_command,
)
from recon2sim.adapters.ingest import ProcessExecutionError, run_process
from recon2sim.artifacts import (
    CameraReconstruction,
    CompletionAnchorRecord,
    CompletionCropManifest,
    CompletionEligibilityArtifact,
    CompletionEligibilityRecord,
    CompletionEligibilityStatus,
    CompletionEvidencePackage,
    CompletionEvidencePreparationRequest,
    CompletionEvidenceSplit,
    CompletionWorkerManifest,
    DenseDepthManifest,
    FrameQualityReport,
    IngestManifest,
    MeasuredObjectGeometryArtifact,
    SegmentationTrackingArtifact,
)
from recon2sim.completion import (
    completion_eligibility,
    select_diverse_anchors,
    sha256_file,
    split_object_evidence,
)
from recon2sim.storage import atomic_write_json


class CompletionInputsAdapterConfig(CompletionWorkerConfig):
    worker_module: str = "completion_evaluation_worker"
    docker_image: str = "reconevery/completion-evaluation:phase5b"
    allow_unclassified: bool = False
    object_eligibility_overrides: dict[str, CompletionEligibilityStatus] = Field(
        default_factory=dict
    )
    maximum_anchors_per_object: int = Field(default=2, ge=1, le=4)
    minimum_anchor_angle_degrees: float = Field(default=12.0, ge=0, le=180)
    minimum_heldout_frames: int = Field(default=2, ge=1)
    fitting_fraction: float = Field(default=0.6, gt=0, le=1)
    crop_margin_ratio: float = Field(default=0.15, ge=0, le=1)
    crop_output_size: int = Field(default=1024, ge=64, le=4096)


def _camera_direction(camera: CameraReconstruction, frame_id: str) -> tuple[float, float, float]:
    pose = next(item for item in camera.poses if item.frame_id == frame_id)
    x, y, z, w = pose.transform_world_from_camera.rotation_xyzw
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return (
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x * x + y * y),
    )


def _crop_rgba(
    frame_path: Path,
    mask_path: Path,
    output_path: Path,
    *,
    bbox_xywh: tuple[int, int, int, int],
    margin_ratio: float,
    output_size: int,
) -> tuple[
    tuple[float, float, float, float, float, float, float, float, float],
    tuple[float, float, float, float, float, float, float, float, float],
]:
    image = Image.open(frame_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if image.size != mask.size:
        raise ValueError("completion crop source frame and canonical mask dimensions differ")
    x, y, width, height = bbox_xywh
    margin = math.ceil(margin_ratio * max(width, height))
    left = x - margin
    top = y - margin
    right = x + width + margin
    bottom = y + height + margin
    side = max(right - left, bottom - top)
    square_left = left - (side - (right - left)) // 2
    square_top = top - (side - (bottom - top)) // 2
    scale = output_size / side
    source_to_crop = (
        scale,
        0.0,
        -square_left * scale,
        0.0,
        scale,
        -square_top * scale,
        0.0,
        0.0,
        1.0,
    )
    crop_to_source = (
        1 / scale,
        0.0,
        square_left,
        0.0,
        1 / scale,
        square_top,
        0.0,
        0.0,
        1.0,
    )
    square_box = (
        square_left,
        square_top,
        square_left + side,
        square_top + side,
    )
    # Pillow deterministically pads crop regions outside the source with zero.
    canvas_rgb = image.crop(square_box)
    canvas_alpha = mask.crop(square_box)
    rgb = canvas_rgb.resize((output_size, output_size), Image.Resampling.LANCZOS)
    alpha = canvas_alpha.resize((output_size, output_size), Image.Resampling.NEAREST)
    rgba = Image.merge("RGBA", (*rgb.split(), alpha))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(output_path, format="PNG", optimize=False, compress_level=9)
    return crop_to_source, source_to_crop


class CompletionEvidencePackageAdapter:
    name = "completion_evidence_package"
    version = "0.1.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        config = CompletionInputsAdapterConfig.model_validate(context.config.adapter.config)
        manifest = IngestManifest.model_validate_json(
            context.canonical_path("inputs", "manifest.json").read_text(encoding="utf-8")
        )
        frame_qa = FrameQualityReport.model_validate_json(
            context.canonical_path("inputs", "frame_qa.json").read_text(encoding="utf-8")
        )
        camera = CameraReconstruction.model_validate_json(
            context.canonical_path("camera", "reconstruction.json").read_text(encoding="utf-8")
        )
        tracks = SegmentationTrackingArtifact.model_validate_json(
            context.canonical_path("observations", "object_tracks.json").read_text(encoding="utf-8")
        )
        depth = DenseDepthManifest.model_validate_json(
            context.canonical_path("reconstruction", "dense", "depth_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        measured = MeasuredObjectGeometryArtifact.model_validate_json(
            context.canonical_path(
                "reconstruction", "measured_objects", "geometry_manifest.json"
            ).read_text(encoding="utf-8")
        )
        qa_by_id = {entry.frame_id: entry for entry in frame_qa.entries}
        depth_by_id = {record.frame_id: record for record in depth.records}
        measured_by_id = {item.object_id: item for item in measured.hypotheses}
        registered = set(camera.registered_frame_ids)
        frame_ids_to_materialize: set[str] = set()
        mask_paths_to_materialize: set[str] = set()
        depth_frame_ids_to_materialize: set[str] = set()
        for track in tracks.tracks:
            status, _, _ = completion_eligibility(
                track.semantic_label,
                track.asset_type_hint,
                allow_unclassified=config.allow_unclassified,
                override=config.object_eligibility_overrides.get(track.object_id),
            )
            if status not in {
                CompletionEligibilityStatus.ELIGIBLE_RIGID,
                CompletionEligibilityStatus.ELIGIBLE_STATIC,
            }:
                continue
            measured_item = measured_by_id.get(track.object_id)
            sample_count = measured_item.validated_sample_count if measured_item is not None else 0
            observations = {item.frame_id: item for item in track.observations}
            scored: list[tuple[str, float, tuple[float, float, float]]] = []
            for frame_id, observation in observations.items():
                if frame_id not in registered or frame_id not in depth_by_id:
                    continue
                qa = qa_by_id[frame_id]
                dense_ratio = depth_by_id[frame_id].valid_depth_ratio
                blur_quality = qa.blur_score / (qa.blur_score + 100.0)
                score = (
                    0.35 * observation.frame_score
                    + 0.20 * min(1.0, observation.mask_area_ratio * 8)
                    + 0.15 * blur_quality
                    + 0.20 * dense_ratio
                    + 0.10 * min(1.0, sample_count / 10000)
                )
                scored.append((frame_id, score, _camera_direction(camera, frame_id)))
            anchors = select_diverse_anchors(
                scored,
                maximum_count=config.maximum_anchors_per_object,
                minimum_angle_degrees=config.minimum_anchor_angle_degrees,
            )
            split = split_object_evidence(
                track.object_id,
                [item.frame_id for item in track.observations if item.frame_id in registered],
                anchors,
                minimum_heldout_frames=config.minimum_heldout_frames,
                fitting_fraction=config.fitting_fraction,
            )
            training_frames = {
                *split.generation_anchor_frames,
                *split.registration_fitting_frames,
            }
            frame_ids_to_materialize.update(split.generation_anchor_frames)
            depth_frame_ids_to_materialize.update(training_frames)
            mask_paths_to_materialize.update(
                observations[frame_id].mask_path for frame_id in training_frames
            )
        specs = [
            InputSpec("inputs/manifest.json", "ingest_manifest"),
            InputSpec("inputs/frame_qa.json", "frame_quality_report"),
            InputSpec("camera/reconstruction.json", "camera_reconstruction"),
            InputSpec("observations/object_tracks.json", "segmentation_tracking"),
            InputSpec("reconstruction/dense/depth_manifest.json", "dense_depth_manifest"),
            InputSpec(
                "reconstruction/dense/undistortion_manifest.json",
                "dense_undistortion_manifest",
            ),
            InputSpec(
                "reconstruction/measured_objects/geometry_manifest.json",
                "measured_object_geometry",
            ),
        ]
        specs.extend(
            InputSpec(frame.relative_path, "input_frame")
            for frame in manifest.frames
            if frame.frame_id in frame_ids_to_materialize
        )
        specs.extend(
            InputSpec(path, "canonical_object_mask") for path in sorted(mask_paths_to_materialize)
        )
        for record in depth.records:
            if record.frame_id not in depth_frame_ids_to_materialize:
                continue
            specs.extend(
                [
                    InputSpec(record.depth_path, "dense_mvs_workspace_file"),
                    InputSpec(record.normal_path, "dense_mvs_workspace_file"),
                    InputSpec(record.consistency_graph_path, "dense_mvs_workspace_file"),
                ]
            )
        unique: dict[str, InputSpec] = {}
        for spec in specs:
            unique[spec.relative_path] = spec
        return list(unique.values())

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return completion_healthcheck(
            context,
            CompletionInputsAdapterConfig,
            worker_name="completion evidence worker",
        )

    def prepare(self, context: StageContext) -> None:
        for path in (
            context.path("reconstruction", "completion", "inputs"),
            context.path("reconstruction", "completion", "evidence"),
            context.path("reconstruction", "completion", "raw", "evidence_logs"),
        ):
            path.mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        root = "reconstruction/completion"
        return [
            OutputSpec(
                f"{root}/eligibility.json",
                "completion_eligibility",
                "application/json",
                "completion_inputs",
                validation="json",
                model=CompletionEligibilityArtifact,
            ),
            OutputSpec(
                f"{root}/evidence_split.json",
                "completion_evidence_split",
                "application/json",
                "completion_inputs",
                validation="json",
                model=CompletionEvidenceSplit,
            ),
            OutputSpec(
                f"{root}/crop_manifest.json",
                "completion_crop_manifest",
                "application/json",
                "completion_inputs",
                validation="json",
                model=CompletionCropManifest,
            ),
            OutputSpec(
                f"{root}/evidence_request.json",
                "completion_evidence_request",
                "application/json",
                "completion_inputs",
                validation="json",
                model=CompletionEvidencePreparationRequest,
            ),
            OutputSpec(
                f"{root}/evidence/worker_manifest.json",
                "completion_worker_manifest",
                "application/json",
                "completion_inputs",
                validation="json",
                model=CompletionWorkerManifest,
            ),
            OutputSpec(
                f"{root}/evidence/evidence_package.json",
                "completion_evidence_package",
                "application/json",
                "completion_inputs",
                validation="json",
                model=CompletionEvidencePackage,
            ),
        ]

    def run(self, context: StageContext) -> StageResult:
        config = CompletionInputsAdapterConfig.model_validate(context.config.adapter.config)
        manifest_path = context.path("inputs", "manifest.json")
        camera_path = context.path("camera", "reconstruction.json")
        tracks_path = context.path("observations", "object_tracks.json")
        depth_path = context.path("reconstruction", "dense", "depth_manifest.json")
        undistortion_path = context.path("reconstruction", "dense", "undistortion_manifest.json")
        measured_path = context.path("reconstruction", "measured_objects", "geometry_manifest.json")
        manifest = IngestManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        frame_qa = FrameQualityReport.model_validate_json(
            context.path("inputs", "frame_qa.json").read_text(encoding="utf-8")
        )
        camera = CameraReconstruction.model_validate_json(camera_path.read_text(encoding="utf-8"))
        tracks = SegmentationTrackingArtifact.model_validate_json(
            tracks_path.read_text(encoding="utf-8")
        )
        depth = DenseDepthManifest.model_validate_json(depth_path.read_text(encoding="utf-8"))
        measured = MeasuredObjectGeometryArtifact.model_validate_json(
            measured_path.read_text(encoding="utf-8")
        )
        if manifest.frame_sequence_digest is None:
            raise ValueError("completion requires a frame-sequence digest")
        if any(
            digest != manifest.frame_sequence_digest
            for digest in (
                camera.frame_sequence_digest,
                tracks.frame_sequence_digest,
                measured.frame_sequence_digest,
            )
        ):
            raise ValueError("completion inputs do not share a frame lineage")
        eligibility = CompletionEligibilityArtifact(
            manifest_sha256=sha256_file(manifest_path),
            frame_sequence_digest=manifest.frame_sequence_digest,
            segmentation_tracking_sha256=sha256_file(tracks_path),
            measured_geometry_sha256=sha256_file(measured_path),
            records=[
                CompletionEligibilityRecord(
                    object_id=track.object_id,
                    semantic_label=track.semantic_label,
                    asset_type_hint=track.asset_type_hint,
                    status=result[0],
                    reason=result[1],
                    explicitly_overridden=result[2],
                )
                for track in tracks.tracks
                for result in (
                    completion_eligibility(
                        track.semantic_label,
                        track.asset_type_hint,
                        allow_unclassified=config.allow_unclassified,
                        override=config.object_eligibility_overrides.get(track.object_id),
                    ),
                )
            ],
        )
        root = context.path("reconstruction", "completion")
        atomic_write_json(root / "eligibility.json", eligibility)
        eligible_ids = {
            record.object_id
            for record in eligibility.records
            if record.status
            in {
                CompletionEligibilityStatus.ELIGIBLE_RIGID,
                CompletionEligibilityStatus.ELIGIBLE_STATIC,
            }
        }
        frame_by_id = {frame.frame_id: frame for frame in manifest.frames}
        qa_by_id = {entry.frame_id: entry for entry in frame_qa.entries}
        depth_by_id = {record.frame_id: record for record in depth.records}
        measured_by_id = {item.object_id: item for item in measured.hypotheses}
        pose_ids = set(camera.registered_frame_ids)
        anchor_records: list[CompletionAnchorRecord] = []
        object_anchors: dict[str, list[str]] = {}
        for track in tracks.tracks:
            if track.object_id not in eligible_ids:
                continue
            scored: list[tuple[str, float, tuple[float, float, float]]] = []
            observations = {item.frame_id: item for item in track.observations}
            measured_item = measured_by_id.get(track.object_id)
            sample_count = measured_item.validated_sample_count if measured_item is not None else 0
            for frame_id, observation in observations.items():
                if frame_id not in pose_ids or frame_id not in depth_by_id:
                    continue
                qa = qa_by_id[frame_id]
                dense_ratio = depth_by_id[frame_id].valid_depth_ratio
                blur_quality = qa.blur_score / (qa.blur_score + 100.0)
                score = (
                    0.35 * observation.frame_score
                    + 0.20 * min(1.0, observation.mask_area_ratio * 8)
                    + 0.15 * blur_quality
                    + 0.20 * dense_ratio
                    + 0.10 * min(1.0, sample_count / 10000)
                )
                scored.append((frame_id, score, _camera_direction(camera, frame_id)))
            anchors = select_diverse_anchors(
                scored,
                maximum_count=config.maximum_anchors_per_object,
                minimum_angle_degrees=config.minimum_anchor_angle_degrees,
            )
            object_anchors[track.object_id] = anchors
            scored_by_id = {item[0]: item for item in scored}
            for rank, frame_id in enumerate(anchors, 1):
                observation = observations[frame_id]
                frame = frame_by_id[frame_id]
                crop_relative = f"reconstruction/completion/inputs/{track.object_id}/{frame_id}.png"
                metadata_relative = (
                    f"reconstruction/completion/inputs/{track.object_id}/{frame_id}.json"
                )
                crop_to_source, source_to_crop = _crop_rgba(
                    context.path(*Path(frame.relative_path).parts),
                    context.path(*Path(observation.mask_path).parts),
                    context.path(*Path(crop_relative).parts),
                    bbox_xywh=observation.bbox_xywh,
                    margin_ratio=config.crop_margin_ratio,
                    output_size=config.crop_output_size,
                )
                record = CompletionAnchorRecord(
                    object_id=track.object_id,
                    frame_id=frame_id,
                    rank=rank,
                    selection_score=scored_by_id[frame_id][1],
                    camera_direction=scored_by_id[frame_id][2],
                    mask_bbox_xywh=observation.bbox_xywh,
                    mask_area_ratio=observation.mask_area_ratio,
                    dense_valid_ratio=depth_by_id[frame_id].valid_depth_ratio,
                    measured_sample_count=sample_count,
                    selection_reason=(
                        "deterministic weighted SAM, mask, QA, dense-depth, measured-support "
                        "score with view-direction diversity"
                    ),
                    crop_path=crop_relative,
                    crop_metadata_path=metadata_relative,
                    crop_sha256=sha256_file(context.path(*Path(crop_relative).parts)),
                    source_frame_sha256=frame.sha256,
                    source_mask_sha256=sha256_file(
                        context.path(*Path(observation.mask_path).parts)
                    ),
                    crop_to_source_transform=crop_to_source,
                    source_to_crop_transform=source_to_crop,
                )
                atomic_write_json(context.path(*Path(metadata_relative).parts), record)
                anchor_records.append(record)
        crop_manifest = CompletionCropManifest(
            output_size=config.crop_output_size,
            margin_ratio=config.crop_margin_ratio,
            anchors=anchor_records,
        )
        atomic_write_json(root / "crop_manifest.json", crop_manifest)
        splits = CompletionEvidenceSplit(
            frame_sequence_digest=manifest.frame_sequence_digest,
            objects=[
                split_object_evidence(
                    track.object_id,
                    [
                        observation.frame_id
                        for observation in track.observations
                        if observation.frame_id in pose_ids
                    ],
                    object_anchors.get(track.object_id, []),
                    minimum_heldout_frames=config.minimum_heldout_frames,
                    fitting_fraction=config.fitting_fraction,
                )
                for track in tracks.tracks
                if track.object_id in eligible_ids
            ],
            seed=context.seed,
        )
        atomic_write_json(root / "evidence_split.json", splits)
        tracks_by_id = {track.object_id: track for track in tracks.tracks}
        object_inputs: dict[str, dict[str, object]] = {}
        for split in splits.objects:
            track = tracks_by_id[split.object_id]
            observation_by_id = {item.frame_id: item for item in track.observations}
            training_frames = [
                *split.generation_anchor_frames,
                *split.registration_fitting_frames,
            ]
            object_inputs[split.object_id] = {
                "semantic_label": track.semantic_label,
                "training_frame_ids": training_frames,
                "heldout_frame_ids": split.heldout_validation_frames,
                "training_masks": {
                    frame_id: observation_by_id[frame_id].mask_path for frame_id in training_frames
                },
                "training_dense_maps": {
                    frame_id: {
                        "depth_path": depth_by_id[frame_id].depth_path,
                        "normal_path": depth_by_id[frame_id].normal_path,
                        "consistency_graph_path": (depth_by_id[frame_id].consistency_graph_path),
                    }
                    for frame_id in training_frames
                },
                "measured_point_cloud_path": None,
                "training_geometry_source": "dense_depth_backprojection_only",
            }
        request = CompletionEvidencePreparationRequest(
            manifest_sha256=sha256_file(manifest_path),
            frame_sequence_digest=manifest.frame_sequence_digest,
            camera_reconstruction_path="camera/reconstruction.json",
            camera_reconstruction_sha256=sha256_file(camera_path),
            segmentation_tracking_path="observations/object_tracks.json",
            segmentation_tracking_sha256=sha256_file(tracks_path),
            dense_depth_manifest_path="reconstruction/dense/depth_manifest.json",
            dense_depth_manifest_sha256=sha256_file(depth_path),
            dense_undistortion_manifest_path=("reconstruction/dense/undistortion_manifest.json"),
            dense_undistortion_manifest_sha256=sha256_file(undistortion_path),
            measured_geometry_path="reconstruction/measured_objects/geometry_manifest.json",
            measured_geometry_sha256=sha256_file(measured_path),
            evidence_split_path="reconstruction/completion/evidence_split.json",
            evidence_split_sha256=sha256_file(root / "evidence_split.json"),
            crop_manifest_path="reconstruction/completion/crop_manifest.json",
            crop_manifest_sha256=sha256_file(root / "crop_manifest.json"),
            object_inputs=object_inputs,
            coordinate_convention=camera.coordinate_convention,
            output_directory="reconstruction/completion/evidence",
            seed=context.seed,
        )
        atomic_write_json(root / "evidence_request.json", request)
        try:
            run_process(
                worker_command(
                    context,
                    config,
                    "prepare-evidence",
                    "reconstruction/completion/evidence_request.json",
                    "reconstruction/completion/evidence",
                ),
                context=context,
                name="completion_evidence_worker",
                log_directory="reconstruction/completion/raw/evidence_logs",
            )
        except ProcessExecutionError as exc:
            raise RuntimeError(str(exc)) from exc
        worker = CompletionWorkerManifest.model_validate_json(
            (root / "evidence" / "worker_manifest.json").read_text(encoding="utf-8")
        )
        package = CompletionEvidencePackage.model_validate_json(
            (root / "evidence" / "evidence_package.json").read_text(encoding="utf-8")
        )
        if worker.request_sha256 != sha256_file(root / "evidence_request.json"):
            raise RuntimeError("completion evidence worker request hash mismatch")
        expected = {
            "manifest_sha256": request.manifest_sha256,
            "frame_sequence_digest": request.frame_sequence_digest,
            "camera_reconstruction_sha256": request.camera_reconstruction_sha256,
            "segmentation_tracking_sha256": request.segmentation_tracking_sha256,
            "dense_depth_manifest_sha256": request.dense_depth_manifest_sha256,
            "measured_geometry_sha256": request.measured_geometry_sha256,
            "evidence_split_sha256": request.evidence_split_sha256,
            "crop_manifest_sha256": request.crop_manifest_sha256,
        }
        if any(getattr(package, key) != value for key, value in expected.items()):
            raise RuntimeError("completion evidence package lineage does not match request")
        dynamic = [
            OutputSpec(
                path.relative_to(context.run_dir).as_posix(),
                "completion_evidence_file",
                "image/png" if path.suffix == ".png" else "application/octet-stream",
                "completion_inputs",
                validation="exists",
            )
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.relative_to(root).as_posix()
            not in {
                "eligibility.json",
                "evidence_split.json",
                "crop_manifest.json",
                "evidence_request.json",
                "evidence/worker_manifest.json",
                "evidence/evidence_package.json",
            }
            and not path.relative_to(root).as_posix().startswith("raw/")
        ]
        return StageResult(
            outputs=dynamic,
            metrics={
                "eligible_objects": len(splits.objects),
                "anchor_crops": len(anchor_records),
                "heldout_frames": sum(
                    len(item.heldout_validation_frames) for item in splits.objects
                ),
            },
        )
