from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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
from recon2sim.adapters.completion_registration import GENERATION_MANIFESTS
from recon2sim.adapters.ingest import ProcessExecutionError, run_process
from recon2sim.artifacts import (
    CandidateEvaluationManifest,
    CandidateEvaluationRequest,
    CandidateGenerationManifest,
    CandidateRegistrationManifest,
    CompletionEvidencePackage,
    CompletionEvidenceSplit,
    CompletionWorkerManifest,
    DenseDepthManifest,
    DenseWorkspaceManifest,
    SegmentationTrackingArtifact,
)
from recon2sim.completion import sha256_file
from recon2sim.storage import atomic_write_json


class CompletionEvaluationAdapterConfig(CompletionWorkerConfig):
    worker_module: str = "completion_evaluation_worker"
    docker_image: str = "reconevery/completion-evaluation:phase5b"
    minimum_validation_views: int = Field(default=2, ge=1)
    minimum_mask_iou: float = Field(default=0.25, ge=0, le=1)
    minimum_mask_precision: float = Field(default=0.60, ge=0, le=1)
    maximum_median_relative_depth_residual: float = Field(default=0.08, ge=0)
    minimum_depth_inlier_fraction: float = Field(default=0.50, ge=0, le=1)
    maximum_negative_space_violation_ratio: float = Field(default=0.10, ge=0, le=1)
    maximum_front_of_scene_violation_ratio: float = Field(default=0.05, ge=0, le=1)
    minimum_recall_gain_over_measured_baseline: float = Field(default=0.05, ge=-1, le=1)
    maximum_precision_drop_from_measured_baseline: float = Field(default=0.15, ge=0, le=1)


class CompletionCandidateEvaluationAdapter:
    name = "completion_candidate_evaluation"
    version = "0.1.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        split = CompletionEvidenceSplit.model_validate_json(
            context.canonical_path("reconstruction", "completion", "evidence_split.json").read_text(
                encoding="utf-8"
            )
        )
        tracks = SegmentationTrackingArtifact.model_validate_json(
            context.canonical_path("observations", "object_tracks.json").read_text(encoding="utf-8")
        )
        depth = DenseDepthManifest.model_validate_json(
            context.canonical_path("reconstruction", "dense", "depth_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        workspace = DenseWorkspaceManifest.model_validate_json(
            context.canonical_path("reconstruction", "dense", "workspace_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        heldout = {
            frame_id for item in split.objects for frame_id in item.heldout_validation_frames
        }
        specs = [
            InputSpec("camera/reconstruction.json", "camera_reconstruction"),
            InputSpec("observations/object_tracks.json", "segmentation_tracking"),
            InputSpec("reconstruction/dense/depth_manifest.json", "dense_depth_manifest"),
            InputSpec(
                "reconstruction/dense/undistortion_manifest.json",
                "dense_undistortion_manifest",
            ),
            InputSpec(
                "reconstruction/dense/workspace_manifest.json",
                "dense_workspace_manifest",
            ),
            InputSpec(
                "reconstruction/completion/evidence/evidence_package.json",
                "completion_evidence_package",
            ),
            InputSpec(
                "reconstruction/completion/evidence_split.json",
                "completion_evidence_split",
            ),
            InputSpec(
                "reconstruction/completion/registration_manifest.json",
                "candidate_registration_manifest",
            ),
        ]
        for path in GENERATION_MANIFESTS.values():
            specs.append(InputSpec(path, "candidate_generation_manifest"))
            generation = CandidateGenerationManifest.model_validate_json(
                context.canonical_path(*Path(path).parts).read_text(encoding="utf-8")
            )
            for candidate in generation.candidates:
                specs.extend(
                    InputSpec(
                        asset.relative_path,
                        (
                            "measured_object_geometry_file"
                            if candidate.backend.value == "measured_partial_baseline"
                            else "completion_candidate_file"
                        ),
                        materialization_mode="reflink_or_copy",
                    )
                    for asset in candidate.native_assets
                )
        specs.extend(
            InputSpec(observation.mask_path, "canonical_object_mask")
            for track in tracks.tracks
            for observation in track.observations
            if observation.frame_id in heldout
        )
        for record in depth.records:
            if record.frame_id in heldout:
                specs.extend(
                    [
                        InputSpec(record.depth_path, "dense_mvs_workspace_file"),
                        InputSpec(record.normal_path, "dense_mvs_workspace_file"),
                        InputSpec(
                            record.consistency_graph_path,
                            "dense_mvs_workspace_file",
                        ),
                    ]
                )
        specs.extend(
            InputSpec(
                frame.workspace_filename,
                "dense_mvs_workspace_file",
            )
            for frame in workspace.frames
            if frame.frame_id in heldout
        )
        return [replace(spec, include_producer_signature=False) for spec in specs]

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return completion_healthcheck(
            context,
            CompletionEvaluationAdapterConfig,
            worker_name="completion held-out evaluation worker",
        )

    def prepare(self, context: StageContext) -> None:
        context.path("reconstruction", "completion", "raw", "evaluation_logs").mkdir(
            parents=True, exist_ok=True
        )

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        root = "reconstruction/completion"
        outputs = [
            OutputSpec(
                f"{root}/evaluation_request.json",
                "candidate_evaluation_request",
                "application/json",
                "completion_evaluation",
                validation="json",
                model=CandidateEvaluationRequest,
            ),
            OutputSpec(
                f"{root}/evaluation_worker_manifest.json",
                "completion_worker_manifest",
                "application/json",
                "completion_evaluation",
                validation="json",
                model=CompletionWorkerManifest,
            ),
            OutputSpec(
                f"{root}/evaluation_manifest.json",
                "candidate_evaluation_manifest",
                "application/json",
                "completion_evaluation",
                validation="json",
                model=CandidateEvaluationManifest,
            ),
        ]
        registration_path = context.canonical_path(
            "reconstruction", "completion", "registration_manifest.json"
        )
        if registration_path.is_file():
            registration = CandidateRegistrationManifest.model_validate_json(
                registration_path.read_text(encoding="utf-8")
            )
            for item in registration.registrations:
                if item.frozen_transform is None:
                    continue
                for frame_id in item.heldout_frame_ids:
                    outputs.append(
                        OutputSpec(
                            f"{root}/renders/{item.candidate_id}/{frame_id}.png",
                            "candidate_heldout_render",
                            "image/png",
                            "completion_evaluation",
                            validation="png",
                        )
                    )
        return outputs

    def run(self, context: StageContext) -> StageResult:
        config = CompletionEvaluationAdapterConfig.model_validate(context.config.adapter.config)
        root = context.path("reconstruction", "completion")
        registration_path = root / "registration_manifest.json"
        package_path = root / "evidence" / "evidence_package.json"
        split_path = root / "evidence_split.json"
        tracks_path = context.path("observations", "object_tracks.json")
        camera_path = context.path("camera", "reconstruction.json")
        depth_path = context.path("reconstruction", "dense", "depth_manifest.json")
        undistortion_path = context.path("reconstruction", "dense", "undistortion_manifest.json")
        workspace_path = context.path("reconstruction", "dense", "workspace_manifest.json")
        registration = CandidateRegistrationManifest.model_validate_json(
            registration_path.read_text(encoding="utf-8")
        )
        CompletionEvidencePackage.model_validate_json(package_path.read_text(encoding="utf-8"))
        split = CompletionEvidenceSplit.model_validate_json(split_path.read_text(encoding="utf-8"))
        tracks = SegmentationTrackingArtifact.model_validate_json(
            tracks_path.read_text(encoding="utf-8")
        )
        depth = DenseDepthManifest.model_validate_json(depth_path.read_text(encoding="utf-8"))
        workspace = DenseWorkspaceManifest.model_validate_json(
            workspace_path.read_text(encoding="utf-8")
        )
        dense_frames = {item.frame_id: item for item in workspace.frames}
        observations = {
            track.object_id: {item.frame_id: item for item in track.observations}
            for track in tracks.tracks
        }
        depth_by_id = {item.frame_id: item for item in depth.records}
        heldout_inputs: dict[str, dict[str, object]] = {}
        for item in split.objects:
            heldout_inputs[item.object_id] = {
                "frame_ids": item.heldout_validation_frames,
                "mask_paths": {
                    frame_id: observations[item.object_id][frame_id].mask_path
                    for frame_id in item.heldout_validation_frames
                },
                "depth_paths": {
                    frame_id: depth_by_id[frame_id].depth_path
                    for frame_id in item.heldout_validation_frames
                },
                "normal_paths": {
                    frame_id: depth_by_id[frame_id].normal_path
                    for frame_id in item.heldout_validation_frames
                },
                "dense_depth_hashes": {
                    frame_id: depth_by_id[frame_id].depth_sha256
                    for frame_id in item.heldout_validation_frames
                },
                "dense_image_paths": {
                    frame_id: dense_frames[frame_id].workspace_filename
                    for frame_id in item.heldout_validation_frames
                },
            }
        manifest_paths = dict(GENERATION_MANIFESTS)
        manifest_hashes = {
            name: sha256_file(context.path(*Path(path).parts))
            for name, path in manifest_paths.items()
        }
        request = CandidateEvaluationRequest(
            registration_manifest_path="reconstruction/completion/registration_manifest.json",
            registration_manifest_sha256=sha256_file(registration_path),
            evidence_package_path=("reconstruction/completion/evidence/evidence_package.json"),
            evidence_package_sha256=sha256_file(package_path),
            evidence_split_path="reconstruction/completion/evidence_split.json",
            evidence_split_sha256=sha256_file(split_path),
            generation_manifest_paths=manifest_paths,
            generation_manifest_hashes=manifest_hashes,
            segmentation_tracking_path="observations/object_tracks.json",
            segmentation_tracking_sha256=sha256_file(tracks_path),
            camera_reconstruction_path="camera/reconstruction.json",
            camera_reconstruction_sha256=sha256_file(camera_path),
            dense_depth_manifest_path="reconstruction/dense/depth_manifest.json",
            dense_depth_manifest_sha256=sha256_file(depth_path),
            dense_undistortion_manifest_path=("reconstruction/dense/undistortion_manifest.json"),
            dense_undistortion_manifest_sha256=sha256_file(undistortion_path),
            heldout_inputs=heldout_inputs,
            evaluation_configuration={
                "occlusion_policy": "dense_depth_visibility_v1",
                "minimum_validation_views": config.minimum_validation_views,
                "minimum_mask_iou": config.minimum_mask_iou,
                "minimum_mask_precision": config.minimum_mask_precision,
                "maximum_median_relative_depth_residual": (
                    config.maximum_median_relative_depth_residual
                ),
                "minimum_depth_inlier_fraction": config.minimum_depth_inlier_fraction,
                "maximum_negative_space_violation_ratio": (
                    config.maximum_negative_space_violation_ratio
                ),
                "maximum_front_of_scene_violation_ratio": (
                    config.maximum_front_of_scene_violation_ratio
                ),
                "minimum_recall_gain_over_measured_baseline": (
                    config.minimum_recall_gain_over_measured_baseline
                ),
                "maximum_precision_drop_from_measured_baseline": (
                    config.maximum_precision_drop_from_measured_baseline
                ),
                "heldout_only": True,
                "transforms_frozen": True,
                "fake_mode": config.fake_mode,
            },
            output_directory="reconstruction/completion",
            seed=context.seed,
        )
        request_path = root / "evaluation_request.json"
        atomic_write_json(request_path, request)
        try:
            run_process(
                worker_command(
                    context,
                    config,
                    "evaluate",
                    "reconstruction/completion/evaluation_request.json",
                    "reconstruction/completion",
                ),
                context=context,
                name="completion_evaluation_worker",
                log_directory="reconstruction/completion/raw/evaluation_logs",
            )
        except ProcessExecutionError as exc:
            if "out of memory" in exc.result.stderr.lower():
                raise RuntimeError("completion evaluation worker ran out of memory") from exc
            raise RuntimeError(str(exc)) from exc
        worker = CompletionWorkerManifest.model_validate_json(
            (root / "evaluation_worker_manifest.json").read_text(encoding="utf-8")
        )
        evaluation = CandidateEvaluationManifest.model_validate_json(
            (root / "evaluation_manifest.json").read_text(encoding="utf-8")
        )
        if worker.request_sha256 != sha256_file(request_path):
            raise RuntimeError("evaluation worker request hash mismatch")
        if evaluation.registration_manifest_sha256 != request.registration_manifest_sha256:
            raise RuntimeError("evaluation worker changed the registration lineage")
        registered = {
            item.candidate_id: item
            for item in registration.registrations
            if item.frozen_transform is not None
        }
        split_by_object = {item.object_id: item for item in split.objects}
        for evaluation_item in evaluation.evaluations:
            if evaluation_item.candidate_id not in registered:
                raise RuntimeError("evaluation references an unregistered candidate")
            if (
                evaluation_item.heldout_frame_ids
                != split_by_object[evaluation_item.object_id].heldout_validation_frames
            ):
                raise RuntimeError("evaluation did not use the declared held-out frames")
            if evaluation_item.metrics.validation_view_count != len(
                evaluation_item.heldout_frame_ids
            ):
                raise RuntimeError("evaluation view count does not match held-out evidence")
            if set(evaluation_item.render_paths) != set(evaluation_item.heldout_frame_ids):
                raise RuntimeError("evaluation renders do not cover every held-out frame")
            for render_path in evaluation_item.render_paths.values():
                if not context.path(*Path(render_path).parts).is_file():
                    raise RuntimeError(f"evaluation render is missing: {render_path}")
        return StageResult(
            metrics={
                "evaluated_candidates": len(evaluation.evaluations),
                "passing_candidates": sum(
                    item.passed_hard_gates for item in evaluation.evaluations
                ),
            }
        )
