from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pydantic import Field

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
from recon2sim.articulation import sha256_file
from recon2sim.artifacts import (
    ArticulatedCandidateManifest,
    ArticulatedEvaluationManifest,
    ArticulatedPartStateGeometry,
    ArticulatedPartStateGeometryManifest,
    ArticulatedSourceFamily,
    ArticulationCaptureManifest,
    ArticulationEvidenceSplit,
    ArticulationFittingManifest,
    ArticulationStateAlignmentArtifact,
    DenseDepthManifest,
)
from recon2sim.storage import atomic_write_json


class ArticulationEvaluationConfig(ArticulationWorkerConfig):
    worker_module: str = "articulation_evaluation_worker"
    docker_image: str = "reconevery/articulation-evaluation:phase5c"
    minimum_valid_states: int = 3
    minimum_heldout_states: int = 1
    minimum_usable_heldout_views: int = Field(default=1, ge=1)
    minimum_base_mask_iou: float = 0.45
    minimum_movable_part_mask_iou: float = 0.40
    minimum_whole_object_mask_iou: float = 0.45
    minimum_depth_inlier_fraction: float = 0.50
    maximum_negative_space_violation_ratio: float = 0.15
    maximum_front_of_scene_violation_ratio: float = 0.10
    maximum_base_motion_scene_diagonals: float = 0.02
    maximum_prismatic_orthogonal_residual: float = 0.05
    maximum_prismatic_rotation_degrees: float = 5.0
    maximum_revolute_axis_error_degrees: float = 15.0
    maximum_revolute_pivot_residual_part_diagonals: float = 0.10


class ArticulationEvaluationAdapter:
    name = "articulation_evaluation"
    version = "0.1.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        geometry = ArticulatedPartStateGeometryManifest.model_validate_json(
            context.canonical_path(
                "reconstruction", "articulation", "measured_states", "manifest.json"
            ).read_text(encoding="utf-8")
        )
        candidates = ArticulatedCandidateManifest.model_validate_json(
            context.canonical_path(
                "reconstruction", "articulation", "candidate_manifest.json"
            ).read_text(encoding="utf-8")
        )
        capture = ArticulationCaptureManifest.model_validate_json(
            context.canonical_path(
                "reconstruction", "articulation", "capture_manifest.json"
            ).read_text(encoding="utf-8")
        )
        split = ArticulationEvidenceSplit.model_validate_json(
            context.canonical_path(
                "reconstruction", "articulation", "evidence_split.json"
            ).read_text(encoding="utf-8")
        )
        alignment = ArticulationStateAlignmentArtifact.model_validate_json(
            context.canonical_path(
                "reconstruction", "articulation", "state_alignment.json"
            ).read_text(encoding="utf-8")
        )
        accepted_states = {item.state_id for item in alignment.transforms if item.accepted}
        heldout_states = set(split.heldout_validation_states) & accepted_states
        specs = [
            InputSpec(
                "reconstruction/articulation/fitting_manifest.json",
                "articulation_fitting_manifest",
            ),
            InputSpec(
                "reconstruction/articulation/link_assignments.json",
                "articulated_link_assignment_manifest",
            ),
            InputSpec(
                "reconstruction/articulation/candidate_manifest.json",
                "articulated_candidate_manifest",
            ),
            InputSpec(
                "reconstruction/articulation/evidence_split.json",
                "articulation_evidence_split",
            ),
            InputSpec(
                "reconstruction/articulation/measured_states/manifest.json",
                "articulated_part_state_geometry_manifest",
            ),
            InputSpec(
                "reconstruction/articulation/measured_motion.json",
                "measured_part_motion",
            ),
            InputSpec(
                "reconstruction/articulation/state_alignment.json",
                "articulation_state_alignment",
            ),
            InputSpec(
                "reconstruction/articulation/capture_manifest.json",
                "articulation_capture_manifest",
            ),
        ]
        for state in capture.states:
            if state.state_id not in heldout_states:
                continue
            specs.extend(
                [
                    InputSpec(state.camera_evidence_path, "camera_reconstruction"),
                    InputSpec(state.segmentation_evidence_path, "segmentation_tracking"),
                    InputSpec(
                        state.undistortion_evidence_path,
                        "dense_undistortion_manifest",
                    ),
                    InputSpec(state.depth_evidence_path, "dense_depth_manifest"),
                ]
            )
            depth = DenseDepthManifest.model_validate_json(
                context.canonical_path(*Path(state.depth_evidence_path).parts).read_text(
                    encoding="utf-8"
                )
            )
            specs.extend(
                InputSpec(
                    record.depth_path,
                    "articulation_dense_depth_map",
                    materialization_mode="reflink_or_copy",
                )
                for record in depth.records
            )
        specs.extend(
            InputSpec(path, "articulated_part_mask")
            for item in geometry.geometries
            if item.state_id in heldout_states
            for path in item.mask_paths
        )
        specs.extend(
            InputSpec(
                item.measured_point_cloud_path,
                "measured_articulated_part_point_cloud",
                materialization_mode="reflink_or_copy",
            )
            for item in geometry.geometries
            if item.state_id in heldout_states
        )
        specs.extend(
            InputSpec(
                path,
                (
                    "measured_articulated_part_point_cloud"
                    if candidate.source_family is ArticulatedSourceFamily.MEASURED_MOTION
                    else "articulated_candidate_visual_link"
                ),
                materialization_mode="reflink_or_copy",
            )
            for candidate in candidates.candidates
            for link in candidate.links
            for path in link.visual_asset_paths
        )
        unique = {spec.relative_path: spec for spec in specs}
        return [replace(spec, include_producer_signature=False) for spec in unique.values()]

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return articulation_healthcheck(
            context,
            ArticulationEvaluationConfig,
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
        root = "reconstruction/articulation"
        return [
            OutputSpec(
                f"{root}/evaluation_manifest.json",
                "articulated_evaluation_manifest",
                "application/json",
                self.name,
                validation="json",
                model=ArticulatedEvaluationManifest,
            ),
            *[
                OutputSpec(
                    f"{root}/previews/{name}.png",
                    "articulation_preview",
                    "image/png",
                    self.name,
                    validation="png",
                )
                for name in (
                    "link_assignment",
                    "fitting_states",
                    "heldout_state_evaluation",
                )
            ],
        ]

    def run(self, context: StageContext) -> StageResult:
        config = ArticulationEvaluationConfig.model_validate(context.config.adapter.config)
        root = context.path("reconstruction", "articulation")
        split = ArticulationEvidenceSplit.model_validate_json(
            (root / "evidence_split.json").read_text(encoding="utf-8")
        )
        fitting = ArticulationFittingManifest.model_validate_json(
            (root / "fitting_manifest.json").read_text(encoding="utf-8")
        )
        alignment = ArticulationStateAlignmentArtifact.model_validate_json(
            (root / "state_alignment.json").read_text(encoding="utf-8")
        )
        accepted_states = {item.state_id for item in alignment.transforms if item.accepted}
        heldout_state_ids = [
            state_id for state_id in split.heldout_validation_states if state_id in accepted_states
        ]
        capture = ArticulationCaptureManifest.model_validate_json(
            (root / "capture_manifest.json").read_text(encoding="utf-8")
        )
        geometry = ArticulatedPartStateGeometryManifest.model_validate_json(
            (root / "measured_states/manifest.json").read_text(encoding="utf-8")
        )
        geometry_by_state: dict[str, list[ArticulatedPartStateGeometry]] = {}
        for item in geometry.geometries:
            geometry_by_state.setdefault(item.state_id, []).append(item)
        state_evidence = []
        for state in capture.states:
            if state.state_id not in set(heldout_state_ids):
                continue
            part_masks: dict[str, dict[str, str]] = {}
            for item in geometry_by_state.get(state.state_id, []):
                part_masks[item.part_id] = {Path(path).stem: path for path in item.mask_paths}
            state_evidence.append(
                {
                    "state_id": state.state_id,
                    "camera_reconstruction_path": state.camera_evidence_path,
                    "camera_reconstruction_sha256": state.camera_reconstruction_sha256,
                    "undistortion_manifest_path": state.undistortion_evidence_path,
                    "depth_manifest_path": state.depth_evidence_path,
                    "part_mask_paths": part_masks,
                }
            )
        request_path = root / "raw" / "articulation_evaluation_request.json"
        gates = {
            key: value
            for key, value in config.model_dump(mode="json").items()
            if key.startswith(("minimum_", "maximum_"))
        }
        atomic_write_json(
            request_path,
            {
                "schema_version": "0.1.0",
                "fitting_manifest_path": ("reconstruction/articulation/fitting_manifest.json"),
                "fitting_manifest_sha256": sha256_file(root / "fitting_manifest.json"),
                "link_assignments_path": ("reconstruction/articulation/link_assignments.json"),
                "link_assignments_sha256": sha256_file(root / "link_assignments.json"),
                "candidate_manifest_path": ("reconstruction/articulation/candidate_manifest.json"),
                "candidate_manifest_sha256": sha256_file(root / "candidate_manifest.json"),
                "evidence_split_path": "reconstruction/articulation/evidence_split.json",
                "evidence_split_sha256": sha256_file(root / "evidence_split.json"),
                "measured_states_manifest_path": (
                    "reconstruction/articulation/measured_states/manifest.json"
                ),
                "measured_states_manifest_sha256": sha256_file(
                    root / "measured_states/manifest.json"
                ),
                "state_alignment_path": ("reconstruction/articulation/state_alignment.json"),
                "state_alignment_sha256": sha256_file(root / "state_alignment.json"),
                "measured_motion_path": "reconstruction/articulation/measured_motion.json",
                "measured_motion_sha256": sha256_file(root / "measured_motion.json"),
                "heldout_state_ids": heldout_state_ids,
                "reference_state_id": capture.reference_state_id,
                "capture_state_count": capture.capture_state_count,
                "accepted_alignment_state_ids": alignment.accepted_alignment_state_ids,
                "generation_state_ids": split.candidate_generation_states,
                "fitting_state_ids": split.kinematic_fitting_states,
                "heldout_views_by_state": {
                    state_id: split.heldout_views_by_state.get(state_id, [])
                    for state_id in heldout_state_ids
                },
                "state_evidence": state_evidence,
                "frozen_candidate_ids": [
                    item.candidate_id for item in fitting.fittings if item.status != "failed"
                ],
                "acceptance_gates": gates,
                "heldout_q_policy": "measured_geometry",
                "output_directory": "reconstruction/articulation",
                "seed": context.seed,
                "fake_mode": config.fake_mode,
            },
        )
        run_articulation_worker(
            context,
            config,
            action="evaluate",
            request_path=request_path.relative_to(context.run_dir).as_posix(),
            output_directory="reconstruction/articulation",
            log_name="articulation_evaluation",
        )
        result = ArticulatedEvaluationManifest.model_validate_json(
            (root / "evaluation_manifest.json").read_text(encoding="utf-8")
        )
        if result.fitting_manifest_sha256 != sha256_file(root / "fitting_manifest.json"):
            raise RuntimeError("articulation evaluation fitting hash mismatch")
        if heldout_state_ids and not any(
            state.heldout
            for evaluation in result.evaluations
            for state in evaluation.state_evaluations
        ):
            raise RuntimeError("articulation evaluation omitted held-out states")
        render_outputs = [
            OutputSpec(
                path,
                "articulation_heldout_render",
                "image/png",
                self.name,
                validation="png",
            )
            for evaluation in result.evaluations
            for state in evaluation.state_evaluations
            for path in state.render_paths.values()
        ]
        return StageResult(
            outputs=render_outputs,
            metrics={
                "evaluated_candidates": len(result.evaluations),
                "passing_candidates": sum(item.passed_hard_gates for item in result.evaluations),
            },
        )


__all__ = ["ArticulationEvaluationAdapter", "ArticulationEvaluationConfig"]
