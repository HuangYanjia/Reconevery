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
    ArticulatedLinkAssignmentManifest,
    ArticulatedPartStateGeometry,
    ArticulatedPartStateGeometryManifest,
    ArticulatedSourceFamily,
    ArticulationCaptureManifest,
    ArticulationEvidenceSplit,
    ArticulationFittingManifest,
    ArticulationStateAlignmentArtifact,
    DenseDepthManifest,
    MeasuredPartMotionArtifact,
)
from recon2sim.storage import atomic_write_json


class ArticulationFittingConfig(ArticulationWorkerConfig):
    worker_module: str = "articulation_evaluation_worker"
    docker_image: str = "reconevery/articulation-evaluation:phase5c"
    maximum_axis_refinement_degrees: float = 10.0
    maximum_pivot_refinement_part_diagonals: float = 0.05
    maximum_fitting_views_per_state: int = Field(default=6, ge=1, le=64)
    allow_nonuniform_scale: bool = False
    allow_per_state_arbitrary_link_transforms: bool = False


class ArticulationFittingAdapter:
    name = "articulation_fitting"
    version = "0.1.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        candidates = ArticulatedCandidateManifest.model_validate_json(
            context.canonical_path(
                "reconstruction", "articulation", "candidate_manifest.json"
            ).read_text(encoding="utf-8")
        )
        geometries = ArticulatedPartStateGeometryManifest.model_validate_json(
            context.canonical_path(
                "reconstruction",
                "articulation",
                "measured_states",
                "fitting_manifest.json",
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
        fitting_states = set(split.kinematic_fitting_states) & accepted_states
        specs = [
            InputSpec(
                "reconstruction/articulation/candidate_manifest.json",
                "articulated_candidate_manifest",
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
                "reconstruction/articulation/evidence_split.json",
                "articulation_evidence_split",
            ),
            InputSpec(
                "reconstruction/articulation/capture_manifest.json",
                "articulation_capture_manifest",
            ),
            InputSpec(
                "reconstruction/articulation/measured_states/fitting_manifest.json",
                "articulated_fitting_part_state_geometry_manifest",
            ),
        ]
        for state in capture.states:
            if state.state_id not in fitting_states:
                continue
            specs.extend(
                [
                    InputSpec(state.camera_evidence_path, "camera_reconstruction"),
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
        specs.extend(
            InputSpec(path, "articulated_part_mask")
            for item in geometries.geometries
            for path in item.mask_paths
        )
        specs.extend(
            InputSpec(
                item.measured_point_cloud_path,
                "measured_articulated_part_point_cloud",
                materialization_mode="reflink_or_copy",
            )
            for item in geometries.geometries
        )
        unique = {spec.relative_path: spec for spec in specs}
        return [replace(spec, include_producer_signature=False) for spec in unique.values()]

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return articulation_healthcheck(
            context,
            ArticulationFittingConfig,
            worker_name=self.name,
        )

    def prepare(self, context: StageContext) -> None:
        context.path("reconstruction", "articulation", "raw", "logs").mkdir(
            parents=True, exist_ok=True
        )

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        root = "reconstruction/articulation"
        return [
            OutputSpec(
                f"{root}/link_assignments.json",
                "articulated_link_assignment_manifest",
                "application/json",
                self.name,
                validation="json",
                model=ArticulatedLinkAssignmentManifest,
            ),
            OutputSpec(
                f"{root}/fitting_manifest.json",
                "articulation_fitting_manifest",
                "application/json",
                self.name,
                validation="json",
                model=ArticulationFittingManifest,
            ),
        ]

    def run(self, context: StageContext) -> StageResult:
        config = ArticulationFittingConfig.model_validate(context.config.adapter.config)
        if config.allow_nonuniform_scale:
            raise ValueError("Phase 5C forbids non-uniform candidate scale")
        if config.allow_per_state_arbitrary_link_transforms:
            raise ValueError("Phase 5C forbids arbitrary per-state link transforms")
        root = context.path("reconstruction", "articulation")
        candidates = ArticulatedCandidateManifest.model_validate_json(
            (root / "candidate_manifest.json").read_text(encoding="utf-8")
        )
        measured = MeasuredPartMotionArtifact.model_validate_json(
            (root / "measured_motion.json").read_text(encoding="utf-8")
        )
        split = ArticulationEvidenceSplit.model_validate_json(
            (root / "evidence_split.json").read_text(encoding="utf-8")
        )
        capture = ArticulationCaptureManifest.model_validate_json(
            (root / "capture_manifest.json").read_text(encoding="utf-8")
        )
        alignment = ArticulationStateAlignmentArtifact.model_validate_json(
            (root / "state_alignment.json").read_text(encoding="utf-8")
        )
        accepted_states = {item.state_id for item in alignment.transforms if item.accepted}
        fitting_state_ids = [
            state_id for state_id in split.kinematic_fitting_states if state_id in accepted_states
        ]
        geometries = ArticulatedPartStateGeometryManifest.model_validate_json(
            (root / "measured_states/fitting_manifest.json").read_text(encoding="utf-8")
        )
        geometry_by_state: dict[str, list[ArticulatedPartStateGeometry]] = {}
        for geometry in geometries.geometries:
            geometry_by_state.setdefault(geometry.state_id, []).append(geometry)
        state_evidence = []
        for state in capture.states:
            if state.state_id not in set(fitting_state_ids):
                continue
            state_evidence.append(
                {
                    "state_id": state.state_id,
                    "camera_reconstruction_path": state.camera_evidence_path,
                    "undistortion_manifest_path": state.undistortion_evidence_path,
                    "depth_manifest_path": state.depth_evidence_path,
                    "registered_frame_ids": state.registered_frame_ids[
                        : config.maximum_fitting_views_per_state
                    ],
                    "part_mask_paths": {
                        geometry.part_id: {Path(path).stem: path for path in geometry.mask_paths}
                        for geometry in geometry_by_state.get(state.state_id, [])
                    },
                }
            )
        request_path = root / "raw" / "articulation_fitting_request.json"
        atomic_write_json(
            request_path,
            {
                "schema_version": "0.1.0",
                "candidate_manifest_path": ("reconstruction/articulation/candidate_manifest.json"),
                "candidate_manifest_sha256": sha256_file(root / "candidate_manifest.json"),
                "measured_motion_path": "reconstruction/articulation/measured_motion.json",
                "measured_motion_sha256": sha256_file(root / "measured_motion.json"),
                "state_alignment_path": ("reconstruction/articulation/state_alignment.json"),
                "state_alignment_sha256": sha256_file(root / "state_alignment.json"),
                "measured_states_manifest_path": (
                    "reconstruction/articulation/measured_states/fitting_manifest.json"
                ),
                "measured_states_manifest_sha256": sha256_file(
                    root / "measured_states/fitting_manifest.json"
                ),
                "evidence_split_path": "reconstruction/articulation/evidence_split.json",
                "evidence_split_sha256": sha256_file(root / "evidence_split.json"),
                "state_evidence": state_evidence,
                "candidate_ids": [item.candidate_id for item in candidates.candidates],
                "fitting_state_ids": fitting_state_ids,
                "heldout_state_ids": split.heldout_validation_states,
                "joint_hypotheses": [
                    item.model_dump(mode="json") for item in measured.joint_hypotheses
                ],
                "fitting_configuration": {
                    "maximum_axis_refinement_degrees": (config.maximum_axis_refinement_degrees),
                    "maximum_pivot_refinement_part_diagonals": (
                        config.maximum_pivot_refinement_part_diagonals
                    ),
                    "maximum_fitting_views_per_state": (config.maximum_fitting_views_per_state),
                    "allow_nonuniform_scale": False,
                    "allow_per_state_arbitrary_link_transforms": False,
                },
                "output_directory": "reconstruction/articulation",
                "seed": context.seed,
                "fake_mode": config.fake_mode,
            },
        )
        run_articulation_worker(
            context,
            config,
            action="fit",
            request_path=request_path.relative_to(context.run_dir).as_posix(),
            output_directory="reconstruction/articulation",
            log_name="articulation_fitting",
        )
        assignments = ArticulatedLinkAssignmentManifest.model_validate_json(
            (root / "link_assignments.json").read_text(encoding="utf-8")
        )
        fitting = ArticulationFittingManifest.model_validate_json(
            (root / "fitting_manifest.json").read_text(encoding="utf-8")
        )
        candidate_hash = sha256_file(root / "candidate_manifest.json")
        if assignments.candidate_manifest_sha256 != candidate_hash:
            raise RuntimeError("link assignment candidate-manifest hash mismatch")
        if fitting.candidate_manifest_sha256 != candidate_hash:
            raise RuntimeError("articulation fitting candidate-manifest hash mismatch")
        if fitting.evidence_split_sha256 != sha256_file(root / "evidence_split.json"):
            raise RuntimeError("articulation fitting evidence-split hash mismatch")
        return StageResult(
            metrics={
                "candidate_count": len(candidates.candidates),
                "fitted_candidates": sum(item.status != "failed" for item in fitting.fittings),
            }
        )


__all__ = ["ArticulationFittingAdapter", "ArticulationFittingConfig"]
