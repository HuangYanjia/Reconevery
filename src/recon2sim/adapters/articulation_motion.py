from __future__ import annotations

from dataclasses import replace

from recon2sim.adapters.articulation_common import (
    ArticulationWorkerConfig,
    articulation_healthcheck,
    run_articulation_worker,
)
from recon2sim.adapters.base import (
    HealthcheckResult,
    InputSpec,
    OutputSpec,
    StageContext,
    StageResult,
)
from recon2sim.articulation import (
    effective_evidence_level,
    ordered_motion_state_ids,
    sha256_file,
)
from recon2sim.artifacts import (
    ArticulatedPartStateGeometryManifest,
    ArticulationCaptureManifest,
    ArticulationEvidenceSplit,
    ArticulationPartPromptManifest,
    ArticulationStateAlignmentArtifact,
    MeasuredPartMotionArtifact,
)
from recon2sim.storage import atomic_write_json


class ArticulationMotionConfig(ArticulationWorkerConfig):
    worker_module: str = "articulation_alignment_worker"
    docker_image: str = "reconevery/articulation-alignment:phase5c"
    maximum_fixed_translation_part_diagonals: float = 0.02
    maximum_fixed_rotation_degrees: float = 3.0
    maximum_prismatic_rotation_degrees: float = 5.0
    maximum_prismatic_orthogonal_residual: float = 0.05
    maximum_revolute_axis_error_degrees: float = 15.0
    maximum_revolute_pivot_residual_part_diagonals: float = 0.10


class ArticulationMotionAdapter:
    name = "articulation_motion"
    version = "0.1.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        geometry = ArticulatedPartStateGeometryManifest.model_validate_json(
            context.canonical_path(
                "reconstruction",
                "articulation",
                "measured_states",
                "fitting_manifest.json",
            ).read_text(encoding="utf-8")
        )
        specs = [
            InputSpec(
                "reconstruction/articulation/capture_manifest.json",
                "articulation_capture_manifest",
            ),
            InputSpec(
                "reconstruction/articulation/part_prompt_manifest.json",
                "articulation_part_prompt_manifest",
            ),
            InputSpec(
                "reconstruction/articulation/measured_states/fitting_manifest.json",
                "articulated_fitting_part_state_geometry_manifest",
            ),
            InputSpec(
                "reconstruction/articulation/state_alignment.json",
                "articulation_state_alignment",
            ),
            InputSpec(
                "reconstruction/articulation/evidence_split.json",
                "articulation_evidence_split",
            ),
        ]
        specs.extend(
            InputSpec(
                item.measured_point_cloud_path,
                "measured_articulated_part_point_cloud",
                materialization_mode="reflink_or_copy",
            )
            for item in geometry.geometries
        )
        return [replace(spec, include_producer_signature=False) for spec in specs]

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return articulation_healthcheck(
            context,
            ArticulationMotionConfig,
            worker_name=self.name,
        )

    def prepare(self, context: StageContext) -> None:
        context.path("reconstruction", "articulation", "raw", "logs").mkdir(
            parents=True, exist_ok=True
        )
        context.path("reconstruction", "articulation", "previews").mkdir(
            parents=True, exist_ok=True
        )

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                "reconstruction/articulation/measured_motion.json",
                "measured_part_motion",
                "application/json",
                self.name,
                validation="json",
                model=MeasuredPartMotionArtifact,
            ),
            *[
                OutputSpec(
                    f"reconstruction/articulation/previews/{name}.png",
                    "articulation_preview",
                    "image/png",
                    self.name,
                    validation="png",
                )
                for name in ("measured_part_motion", "joint_axis_and_pivot")
            ],
        ]

    def run(self, context: StageContext) -> StageResult:
        config = ArticulationMotionConfig.model_validate(context.config.adapter.config)
        root = context.path("reconstruction", "articulation")
        capture = ArticulationCaptureManifest.model_validate_json(
            (root / "capture_manifest.json").read_text(encoding="utf-8")
        )
        prompt = ArticulationPartPromptManifest.model_validate_json(
            (root / "part_prompt_manifest.json").read_text(encoding="utf-8")
        )
        alignment = ArticulationStateAlignmentArtifact.model_validate_json(
            (root / "state_alignment.json").read_text(encoding="utf-8")
        )
        split = ArticulationEvidenceSplit.model_validate_json(
            (root / "evidence_split.json").read_text(encoding="utf-8")
        )
        request_path = root / "raw" / "measured_motion_request.json"
        accepted_nonheldout = {
            item.state_id
            for item in alignment.transforms
            if item.accepted and item.state_id not in set(split.heldout_validation_states)
        }
        try:
            accepted_state_ids = ordered_motion_state_ids(
                capture.reference_state_id,
                [state.state_id for state in capture.states],
                accepted_nonheldout,
            )
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        provisional_motion = len(accepted_state_ids) >= 2
        atomic_write_json(
            request_path,
            {
                "schema_version": "0.1.0",
                "capture_manifest_path": "reconstruction/articulation/capture_manifest.json",
                "capture_manifest_sha256": sha256_file(root / "capture_manifest.json"),
                "part_prompt_manifest_path": (
                    "reconstruction/articulation/part_prompt_manifest.json"
                ),
                "part_prompt_manifest_sha256": sha256_file(root / "part_prompt_manifest.json"),
                "measured_states_manifest_path": (
                    "reconstruction/articulation/measured_states/fitting_manifest.json"
                ),
                "measured_states_manifest_sha256": sha256_file(
                    root / "measured_states/fitting_manifest.json"
                ),
                "state_alignment_path": "reconstruction/articulation/state_alignment.json",
                "state_alignment_sha256": sha256_file(root / "state_alignment.json"),
                "articulated_object_id": capture.articulated_object_id,
                "reference_state_id": capture.reference_state_id,
                "capture_state_count": capture.capture_state_count,
                "accepted_alignment_state_ids": alignment.accepted_alignment_state_ids,
                "effective_motion_evidence_level": effective_evidence_level(
                    len(alignment.accepted_alignment_state_ids),
                    valid_measured_motion=provisional_motion,
                ),
                "base_part_id": prompt.objects[0].base.part_id,
                "movable_parts": [
                    part.model_dump(mode="json")
                    for part in prompt.objects[0].movable_parts
                    if part.include
                ],
                "accepted_state_ids": accepted_state_ids,
                "motion_configuration": {
                    key: value
                    for key, value in config.model_dump(mode="json").items()
                    if key.startswith("maximum_")
                },
                "output_directory": "reconstruction/articulation",
                "seed": context.seed,
                "fake_mode": config.fake_mode,
            },
        )
        run_articulation_worker(
            context,
            config,
            action="estimate-motion",
            request_path=request_path.relative_to(context.run_dir).as_posix(),
            output_directory="reconstruction/articulation",
            log_name="measured_motion",
        )
        result = MeasuredPartMotionArtifact.model_validate_json(
            (root / "measured_motion.json").read_text(encoding="utf-8")
        )
        if result.state_alignment_sha256 != sha256_file(root / "state_alignment.json"):
            raise RuntimeError("measured motion state-alignment hash mismatch")
        return StageResult(
            metrics={
                "joint_hypotheses": len(result.joint_hypotheses),
                "part_state_geometries": len(result.part_geometries),
            }
        )


__all__ = ["ArticulationMotionAdapter", "ArticulationMotionConfig"]
