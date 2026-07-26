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
from recon2sim.adapters.ingest import ProcessExecutionError, run_process
from recon2sim.artifacts import (
    CandidateGenerationManifest,
    CandidateRegistrationManifest,
    CandidateRegistrationRequest,
    CompletionEvidencePackage,
    CompletionEvidenceSplit,
    CompletionWorkerManifest,
    DenseDepthManifest,
    SegmentationTrackingArtifact,
)
from recon2sim.completion import positive_scale_sim3, sha256_file
from recon2sim.storage import atomic_write_json

GENERATION_MANIFESTS = {
    "sam3d_objects": "reconstruction/completion/sam3d_generation_manifest.json",
    "trellis2": "reconstruction/completion/trellis2_generation_manifest.json",
    "measured_partial_baseline": ("reconstruction/completion/measured_generation_manifest.json"),
}


class CompletionRegistrationAdapterConfig(CompletionWorkerConfig):
    worker_module: str = "completion_evaluation_worker"
    docker_image: str = "reconevery/completion-evaluation:phase5b"
    maximum_surface_samples: int = Field(default=100_000, ge=1_000)
    trimmed_fraction: float = Field(default=0.8, gt=0, le=1)
    robust_loss: str = "huber"
    maximum_iterations: int = Field(default=100, ge=1)
    fitting_refinement_iterations: int = Field(default=30, ge=0, le=200)
    fitting_refinement_maximum_points: int = Field(default=10_000, ge=1_000)
    fitting_refinement_maximum_rotation_degrees: float = Field(default=20.0, ge=0, le=90)
    fitting_refinement_maximum_scale_ratio: float = Field(default=1.5, ge=1.0, le=4.0)
    fitting_refinement_translation_extent_ratio: float = Field(default=0.25, ge=0, le=2)
    minimum_scale: float = Field(default=0.01, gt=0)
    maximum_scale: float = Field(default=100.0, gt=0)


class CompletionCandidateRegistrationAdapter:
    name = "completion_candidate_registration"
    version = "0.1.2"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        package = CompletionEvidencePackage.model_validate_json(
            context.canonical_path(
                "reconstruction", "completion", "evidence", "evidence_package.json"
            ).read_text(encoding="utf-8")
        )
        specs = [
            InputSpec("camera/reconstruction.json", "camera_reconstruction"),
            InputSpec("observations/object_tracks.json", "segmentation_tracking"),
            InputSpec(
                "reconstruction/dense/depth_manifest.json",
                "dense_depth_manifest",
            ),
            InputSpec(
                "reconstruction/dense/undistortion_manifest.json",
                "dense_undistortion_manifest",
            ),
            InputSpec(
                "reconstruction/completion/evidence/evidence_package.json",
                "completion_evidence_package",
            ),
            InputSpec(
                "reconstruction/completion/evidence_split.json",
                "completion_evidence_split",
            ),
        ]
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
        fitting = {
            frame_id for item in split.objects for frame_id in item.registration_fitting_frames
        }
        specs.extend(
            InputSpec(observation.mask_path, "canonical_object_mask")
            for track in tracks.tracks
            for observation in track.observations
            if observation.frame_id in fitting
        )
        specs.extend(
            InputSpec(record.depth_path, "dense_mvs_workspace_file")
            for record in depth.records
            if record.frame_id in fitting
        )
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
                            "completion_evidence_file"
                            if asset.relative_path.startswith("reconstruction/completion/evidence/")
                            else (
                                "measured_object_geometry_file"
                                if candidate.backend.value == "measured_partial_baseline"
                                else "completion_candidate_file"
                            )
                        ),
                        materialization_mode="reflink_or_copy",
                    )
                    for asset in candidate.native_assets
                )
        for item in package.objects:
            if item.training_points_path is not None:
                specs.append(
                    InputSpec(
                        item.training_points_path,
                        "completion_evidence_file",
                        materialization_mode="reflink_or_copy",
                    )
                )
        return [replace(spec, include_producer_signature=False) for spec in specs]

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return completion_healthcheck(
            context,
            CompletionRegistrationAdapterConfig,
            worker_name="completion registration worker",
        )

    def prepare(self, context: StageContext) -> None:
        context.path("reconstruction", "completion", "raw", "registration_logs").mkdir(
            parents=True, exist_ok=True
        )

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        root = "reconstruction/completion"
        return [
            OutputSpec(
                f"{root}/registration_request.json",
                "candidate_registration_request",
                "application/json",
                "completion_registration",
                validation="json",
                model=CandidateRegistrationRequest,
            ),
            OutputSpec(
                f"{root}/registration_worker_manifest.json",
                "completion_worker_manifest",
                "application/json",
                "completion_registration",
                validation="json",
                model=CompletionWorkerManifest,
            ),
            OutputSpec(
                f"{root}/registration_manifest.json",
                "candidate_registration_manifest",
                "application/json",
                "completion_registration",
                validation="json",
                model=CandidateRegistrationManifest,
            ),
        ]

    def run(self, context: StageContext) -> StageResult:
        config = CompletionRegistrationAdapterConfig.model_validate(context.config.adapter.config)
        root = context.path("reconstruction", "completion")
        package_path = root / "evidence" / "evidence_package.json"
        package = CompletionEvidencePackage.model_validate_json(
            package_path.read_text(encoding="utf-8")
        )
        split = CompletionEvidenceSplit.model_validate_json(
            (root / "evidence_split.json").read_text(encoding="utf-8")
        )
        tracks = SegmentationTrackingArtifact.model_validate_json(
            context.path("observations", "object_tracks.json").read_text(encoding="utf-8")
        )
        depth = DenseDepthManifest.model_validate_json(
            context.path("reconstruction", "dense", "depth_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        observations = {
            track.object_id: {item.frame_id: item for item in track.observations}
            for track in tracks.tracks
        }
        depth_by_id = {item.frame_id: item for item in depth.records}
        fitting_inputs: dict[str, dict[str, object]] = {}
        for item in split.objects:
            fitting_inputs[item.object_id] = {
                "frame_ids": item.registration_fitting_frames,
                "mask_paths": {
                    frame_id: observations[item.object_id][frame_id].mask_path
                    for frame_id in item.registration_fitting_frames
                },
                "depth_paths": {
                    frame_id: depth_by_id[frame_id].depth_path
                    for frame_id in item.registration_fitting_frames
                },
            }
        candidate_ids: list[str] = []
        candidates = {}
        generation_hashes: dict[str, str] = {}
        for backend, path in GENERATION_MANIFESTS.items():
            full_path = context.path(*Path(path).parts)
            generation = CandidateGenerationManifest.model_validate_json(
                full_path.read_text(encoding="utf-8")
            )
            generation_hashes[backend] = sha256_file(full_path)
            candidate_ids.extend(item.candidate_id for item in generation.candidates)
            candidates.update({item.candidate_id: item for item in generation.candidates})
        request = CandidateRegistrationRequest(
            evidence_package_sha256=sha256_file(package_path),
            generation_manifest_hashes=generation_hashes,
            candidate_ids=sorted(candidate_ids),
            camera_reconstruction_path="camera/reconstruction.json",
            camera_reconstruction_sha256=sha256_file(context.path("camera", "reconstruction.json")),
            dense_undistortion_manifest_path=("reconstruction/dense/undistortion_manifest.json"),
            dense_undistortion_manifest_sha256=sha256_file(
                context.path("reconstruction", "dense", "undistortion_manifest.json")
            ),
            fitting_inputs=fitting_inputs,
            registration_configuration={
                "method": "asymmetric_measured_to_candidate_sim3_with_fitting_views_v2",
                "maximum_surface_samples": config.maximum_surface_samples,
                "trimmed_fraction": config.trimmed_fraction,
                "robust_loss": config.robust_loss,
                "maximum_iterations": config.maximum_iterations,
                "minimum_scale": config.minimum_scale,
                "maximum_scale": config.maximum_scale,
                "fitting_refinement_iterations": config.fitting_refinement_iterations,
                "fitting_refinement_maximum_points": (config.fitting_refinement_maximum_points),
                "fitting_refinement_maximum_rotation_degrees": (
                    config.fitting_refinement_maximum_rotation_degrees
                ),
                "fitting_refinement_maximum_scale_ratio": (
                    config.fitting_refinement_maximum_scale_ratio
                ),
                "fitting_refinement_translation_extent_ratio": (
                    config.fitting_refinement_translation_extent_ratio
                ),
                "heldout_evidence_used": False,
                "fake_mode": config.fake_mode,
            },
            output_directory="reconstruction/completion",
            seed=context.seed,
        )
        request_path = root / "registration_request.json"
        atomic_write_json(request_path, request)
        try:
            run_process(
                worker_command(
                    context,
                    config,
                    "register",
                    "reconstruction/completion/registration_request.json",
                    "reconstruction/completion",
                ),
                context=context,
                name="completion_registration_worker",
                log_directory="reconstruction/completion/raw/registration_logs",
            )
        except ProcessExecutionError as exc:
            if "out of memory" in exc.result.stderr.lower():
                raise RuntimeError("completion registration worker ran out of memory") from exc
            raise RuntimeError(str(exc)) from exc
        worker = CompletionWorkerManifest.model_validate_json(
            (root / "registration_worker_manifest.json").read_text(encoding="utf-8")
        )
        registration = CandidateRegistrationManifest.model_validate_json(
            (root / "registration_manifest.json").read_text(encoding="utf-8")
        )
        if worker.request_sha256 != sha256_file(request_path):
            raise RuntimeError("registration worker request hash mismatch")
        if registration.request_sha256 != sha256_file(request_path):
            raise RuntimeError("registration manifest request hash mismatch")
        if {item.candidate_id for item in registration.registrations} != set(request.candidate_ids):
            raise RuntimeError("registration manifest does not cover all candidate IDs")
        splits = {object_split.object_id: object_split for object_split in split.objects}
        evidence = {record.object_id: record for record in package.objects}
        for registration_item in registration.registrations:
            candidate = candidates[registration_item.candidate_id]
            if (
                registration_item.registration_asset_id != candidate.registration_asset_id
                or registration_item.registration_asset_path != candidate.registration_asset_path
            ):
                raise RuntimeError("registration worker changed the candidate representation")
            object_split = splits[registration_item.object_id]
            if registration_item.fitting_frame_ids != object_split.registration_fitting_frames:
                raise RuntimeError("registration worker changed fitting evidence")
            if registration_item.heldout_frame_ids != object_split.heldout_validation_frames:
                raise RuntimeError("registration worker changed held-out evidence")
            if (
                registration_item.training_points_sha256
                != evidence[registration_item.object_id].training_points_sha256
            ):
                raise RuntimeError("registration worker used the wrong measured evidence")
            if registration_item.frozen_transform is not None and not positive_scale_sim3(
                registration_item.frozen_transform.matrix_world_from_candidate
            ):
                raise RuntimeError("registration worker emitted an improper Sim(3)")
        return StageResult(
            metrics={
                "registered_candidates": sum(
                    item.status != "registration_failed" for item in registration.registrations
                ),
                "failed_candidates": sum(
                    item.status == "registration_failed" for item in registration.registrations
                ),
            }
        )
