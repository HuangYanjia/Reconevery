from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree

from PIL import Image, ImageDraw

from recon2sim.adapters.base import (
    HealthcheckResult,
    InputSpec,
    OutputSpec,
    StageContext,
    StageResult,
)
from recon2sim.articulation import (
    invert_sim3,
    proper_positive_sim3,
    select_articulated_candidate,
    sha256_file,
    stable_digest,
)
from recon2sim.artifacts import (
    ArticulatedCandidateManifest,
    ArticulatedCandidateSelection,
    ArticulatedCandidateStatus,
    ArticulatedEligibilityArtifact,
    ArticulatedEvaluationManifest,
    ArticulatedJointType,
    ArticulatedLicenseMode,
    ArticulatedObjectSelection,
    ArticulatedPartStateGeometryManifest,
    ArticulatedSourceFamily,
    ArticulationCaptureManifest,
    ArticulationDiagnostics,
    ArticulationEvidenceLevel,
    ArticulationEvidenceSplit,
    ArticulationFittingManifest,
    ArticulationPartPromptManifest,
    ArticulationPreviewManifest,
    EndToEndConsistencyCheck,
    MeasuredPartMotionArtifact,
    Phase5CConsistencyReport,
)
from recon2sim.ir import (
    Articulation,
    AssetType,
    ConfidenceRecord,
    GeometryAsset,
    GeometrySourceType,
    Joint,
    Link,
    ObjectInstance,
    ProvenanceRecord,
    ScaleStatus,
    SceneIR,
)
from recon2sim.storage import atomic_write_json


def _preview(path: Path, title: str, lines: list[str]) -> None:
    image = Image.new("RGB", (1280, 720), (246, 247, 249))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1280, 74), fill=(25, 34, 44))
    draw.text((28, 24), title, fill=(255, 255, 255))
    for index, line in enumerate(lines):
        draw.text((42, 112 + index * 42), line, fill=(30, 38, 46))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)


class ArticulationSelectionAdapter:
    name = "articulation_selection"
    version = "0.1.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        candidates = ArticulatedCandidateManifest.model_validate_json(
            context.canonical_path(
                "reconstruction", "articulation", "candidate_manifest.json"
            ).read_text(encoding="utf-8")
        )
        geometry = ArticulatedPartStateGeometryManifest.model_validate_json(
            context.canonical_path(
                "reconstruction", "articulation", "measured_states", "manifest.json"
            ).read_text(encoding="utf-8")
        )
        specs = [
            InputSpec(
                "reconstruction/articulation/reference_phase5a_scene.json",
                "scene_ir",
            ),
            InputSpec(
                "reconstruction/articulation/capture_manifest.json",
                "articulation_capture_manifest",
            ),
            InputSpec(
                "reconstruction/articulation/measured_motion.json",
                "measured_part_motion",
            ),
            InputSpec(
                "reconstruction/articulation/measured_states/manifest.json",
                "articulated_part_state_geometry_manifest",
            ),
            InputSpec(
                "reconstruction/articulation/candidate_manifest.json",
                "articulated_candidate_manifest",
            ),
            InputSpec(
                "reconstruction/articulation/fitting_manifest.json",
                "articulation_fitting_manifest",
            ),
            InputSpec(
                "reconstruction/articulation/evaluation_manifest.json",
                "articulated_evaluation_manifest",
            ),
        ]
        specs.extend(
            InputSpec(item.measured_point_cloud_path, "measured_articulated_part_point_cloud")
            for item in geometry.geometries
        )
        specs.extend(
            InputSpec(path, "articulated_candidate_visual_link")
            for candidate in candidates.candidates
            if candidate.source_family is not ArticulatedSourceFamily.MEASURED_MOTION
            for link in candidate.links
            for path in link.visual_asset_paths
        )
        return specs

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return HealthcheckResult(
            True,
            "deterministic license-aware articulated selection available",
        )

    def prepare(self, context: StageContext) -> None:
        context.path("reconstruction", "articulation", "selected").mkdir(
            parents=True, exist_ok=True
        )
        context.path("reconstruction", "articulation", "previews").mkdir(
            parents=True, exist_ok=True
        )
        context.path("scene_ir").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        root = "reconstruction/articulation"
        return [
            OutputSpec(
                f"{root}/selection.json",
                "articulated_candidate_selection",
                "application/json",
                self.name,
                validation="json",
                model=ArticulatedCandidateSelection,
            ),
            OutputSpec(
                f"{root}/diagnostics.json",
                "articulation_diagnostics",
                "application/json",
                self.name,
                validation="json",
                model=ArticulationDiagnostics,
            ),
            OutputSpec(
                f"{root}/preview_manifest.json",
                "articulation_preview_manifest",
                "application/json",
                self.name,
                validation="json",
                model=ArticulationPreviewManifest,
            ),
            OutputSpec(
                "scene_ir/phase5c_scene.json",
                "scene_ir",
                "application/json",
                self.name,
                validation="scene_ir",
                model=SceneIR,
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
                    "candidate_grid",
                    "selected_articulation",
                )
            ],
        ]

    def run(self, context: StageContext) -> StageResult:
        mode = ArticulatedLicenseMode(
            context.config.adapter.config.get("license_mode", "research_evaluation")
        )
        root = context.path("reconstruction", "articulation")
        capture = ArticulationCaptureManifest.model_validate_json(
            (root / "capture_manifest.json").read_text(encoding="utf-8")
        )
        measured = MeasuredPartMotionArtifact.model_validate_json(
            (root / "measured_motion.json").read_text(encoding="utf-8")
        )
        geometry = ArticulatedPartStateGeometryManifest.model_validate_json(
            (root / "measured_states/manifest.json").read_text(encoding="utf-8")
        )
        candidates = ArticulatedCandidateManifest.model_validate_json(
            (root / "candidate_manifest.json").read_text(encoding="utf-8")
        )
        fitting = ArticulationFittingManifest.model_validate_json(
            (root / "fitting_manifest.json").read_text(encoding="utf-8")
        )
        evaluation = ArticulatedEvaluationManifest.model_validate_json(
            (root / "evaluation_manifest.json").read_text(encoding="utf-8")
        )
        candidate_by_id = {item.candidate_id: item for item in candidates.candidates}
        production = {
            item.candidate_id: item.production_selectable for item in candidates.candidates
        }
        research_id, production_id, selected_id = select_articulated_candidate(
            evaluation.evaluations,
            production_selectable=production,
            mode=mode,
        )
        selected_candidate = candidate_by_id.get(selected_id) if selected_id else None
        if (
            selected_candidate is not None
            and selected_candidate.source_family is ArticulatedSourceFamily.MEASURED_MOTION
        ):
            status = {
                ArticulationEvidenceLevel.SINGLE_STATE_PRIOR_ONLY: (
                    ArticulatedCandidateStatus.PRIOR_ONLY
                ),
                ArticulationEvidenceLevel.TWO_STATE_MOTION_SUPPORTED: (
                    ArticulatedCandidateStatus.TWO_STATE
                ),
                ArticulationEvidenceLevel.MULTI_STATE_HELDOUT_VALIDATED: (
                    ArticulatedCandidateStatus.MULTI_STATE
                ),
            }[capture.evidence_level]
            rationale = [
                "measured-motion analytic baseline passed its available evidence gates",
                "no hidden visual geometry is implied by the measured baseline",
            ]
        elif selected_id is not None:
            status = ArticulatedCandidateStatus.ACCEPTED
            rationale = [
                "candidate passed frozen-structure held-out state gates",
                f"selection policy={mode.value}",
            ]
        elif research_id is not None and mode is ArticulatedLicenseMode.PRODUCTION_CANDIDATE:
            status = ArticulatedCandidateStatus.LICENSE_BLOCKED
            rationale = [
                "a research candidate passed but no production-selectable candidate exists"
            ]
        elif evaluation.evaluations:
            status = ArticulatedCandidateStatus.REJECTED_HELDOUT
            rationale = ["no candidate passed all preconfigured held-out state gates"]
        else:
            status = ArticulatedCandidateStatus.UNRESOLVED
            rationale = ["no articulated candidate was available"]
        selected = ArticulatedObjectSelection(
            articulated_object_id=capture.articulated_object_id,
            status=status,
            evidence_level=capture.evidence_level,
            best_research_articulated_candidate=research_id,
            best_production_eligible_articulated_candidate=production_id,
            selected_candidate_id=selected_id,
            selection_rationale=rationale,
            geometry_status=(
                (
                    "partial_measured"
                    if selected_candidate is not None
                    and selected_candidate.source_family is ArticulatedSourceFamily.MEASURED_MOTION
                    else "articulated_visual_candidate"
                )
                if selected_id
                else None
            ),
            completion_status=(
                "selected_by_multi_state_validation"
                if selected_id
                and selected_candidate is not None
                and selected_candidate.source_family is not ArticulatedSourceFamily.MEASURED_MOTION
                and capture.evidence_level.value == "multi_state_heldout_validated"
                else None
            ),
        )
        selection = ArticulatedCandidateSelection(
            license_mode=mode,
            ranking_policy="hard_gates_heldout_pareto_deterministic_v1",
            objects=[selected],
            deterministic_selection_digest=stable_digest(
                {
                    "mode": mode,
                    "research": research_id,
                    "production": production_id,
                    "selected": selected_id,
                    "evaluations": [
                        item.model_dump(mode="json") for item in evaluation.evaluations
                    ],
                }
            ),
        )
        atomic_write_json(root / "selection.json", selection)
        scene = SceneIR.model_validate_json(
            (root / "reference_phase5a_scene.json").read_text(encoding="utf-8")
        )
        scene = self._integrate_scene(
            scene,
            capture,
            measured,
            geometry,
            selected_candidate,
        )
        atomic_write_json(context.path("scene_ir", "phase5c_scene.json"), scene)
        diagnostics = ArticulationDiagnostics(
            state_count=len(capture.states),
            aligned_state_count=len(capture.states),
            measured_part_count=len({item.part_id for item in geometry.geometries}),
            joint_hypothesis_count=len(measured.joint_hypotheses),
            candidate_count_by_family={
                family: sum(
                    candidate.source_family.value == family for candidate in candidates.candidates
                )
                for family in sorted(
                    {candidate.source_family.value for candidate in candidates.candidates}
                )
            },
            fitted_candidate_count=sum(item.status != "failed" for item in fitting.fittings),
            evaluated_candidate_count=len(evaluation.evaluations),
            passing_candidate_count=sum(item.passed_hard_gates for item in evaluation.evaluations),
            total_runtime_seconds=(
                candidates.runtime_seconds
                + fitting.runtime_seconds
                + evaluation.runtime_seconds
                + measured.runtime_seconds
            ),
            peak_gpu_memory_bytes=max(
                (
                    value
                    for value in (
                        fitting.peak_gpu_memory_bytes,
                        evaluation.peak_gpu_memory_bytes,
                    )
                    if value is not None
                ),
                default=None,
            ),
            peak_host_memory_bytes=max(
                (
                    value
                    for value in (
                        fitting.peak_host_memory_bytes,
                        evaluation.peak_host_memory_bytes,
                    )
                    if value is not None
                ),
                default=None,
            ),
            warnings=[
                "articulated result is visual-only and remains in arbitrary unoriented units"
            ],
        )
        atomic_write_json(root / "diagnostics.json", diagnostics)
        _preview(
            root / "previews/candidate_grid.png",
            "Phase 5C articulated candidates",
            [
                f"object: {capture.articulated_object_id}",
                f"states: {len(capture.states)}",
                f"candidates: {len(candidates.candidates)}",
                f"passing: {diagnostics.passing_candidate_count}",
            ],
        )
        _preview(
            root / "previews/selected_articulation.png",
            "Selected articulated visual hypothesis",
            [
                f"status: {status.value}",
                f"research: {research_id or 'none'}",
                f"production: {production_id or 'none'}",
                f"selected: {selected_id or 'none'}",
                "collision=false | dynamics=false | sim_ready=false",
            ],
        )
        previews = ArticulationPreviewManifest(
            preview_paths={
                "state_alignment": "reconstruction/articulation/previews/state_alignment.png",
                "measured_part_motion": (
                    "reconstruction/articulation/previews/measured_part_motion.png"
                ),
                "joint_axis_and_pivot": (
                    "reconstruction/articulation/previews/joint_axis_and_pivot.png"
                ),
                "candidate_grid": ("reconstruction/articulation/previews/candidate_grid.png"),
                "link_assignment": ("reconstruction/articulation/previews/link_assignment.png"),
                "fitting_states": ("reconstruction/articulation/previews/fitting_states.png"),
                "heldout_state_evaluation": (
                    "reconstruction/articulation/previews/heldout_state_evaluation.png"
                ),
                "selected_articulation": (
                    "reconstruction/articulation/previews/selected_articulation.png"
                ),
            }
        )
        atomic_write_json(root / "preview_manifest.json", previews)
        dynamic_outputs: list[OutputSpec] = []
        if selected_candidate := candidate_by_id.get(selected_id) if selected_id else None:
            selected_root = root / "selected" / capture.articulated_object_id
            selected_root.mkdir(parents=True, exist_ok=True)
            bundle_path = selected_root / "kinematic_bundle.json"
            atomic_write_json(
                bundle_path,
                {
                    "schema_version": "0.1.0",
                    "articulated_object_id": capture.articulated_object_id,
                    "candidate": selected_candidate.model_dump(mode="json"),
                    "measured_joint_hypotheses": [
                        item.model_dump(mode="json") for item in measured.joint_hypotheses
                    ],
                    "evidence_level": capture.evidence_level,
                    "coordinate_convention": scene.metadata.coordinate_convention,
                    "scale_status": "scale_ambiguous",
                    "physical_validation": "not_implemented",
                    "collision_ready": False,
                    "sim_ready": False,
                },
            )
            urdf_path = selected_root / "preview_only.urdf"
            self._write_preview_urdf(
                urdf_path,
                capture.articulated_object_id,
                selected_candidate,
            )
            relative_root = f"reconstruction/articulation/selected/{capture.articulated_object_id}"
            dynamic_outputs.extend(
                [
                    OutputSpec(
                        f"{relative_root}/kinematic_bundle.json",
                        "articulated_kinematic_bundle",
                        "application/json",
                        self.name,
                        validation="json",
                    ),
                    OutputSpec(
                        f"{relative_root}/preview_only.urdf",
                        "visual_only_articulation_preview",
                        "application/xml",
                        self.name,
                        validation="exists",
                    ),
                ]
            )
        return StageResult(
            outputs=dynamic_outputs,
            metrics={
                "candidate_count": len(candidates.candidates),
                "selected": selected_id is not None,
            },
        )

    @staticmethod
    def _write_preview_urdf(
        path: Path,
        object_id: str,
        candidate: object,
    ) -> None:
        from recon2sim.artifacts import ArticulatedCandidate

        if not isinstance(candidate, ArticulatedCandidate):
            raise TypeError("preview URDF requires an articulated candidate")
        robot = ElementTree.Element(
            "robot",
            {
                "name": object_id,
                "simulation_ready": "false",
                "linear_units": "arbitrary_units",
            },
        )
        robot.append(
            ElementTree.Comment(
                "Visual diagnostic only: no collision, inertial, dynamics, or metric claims."
            )
        )
        for link in candidate.links:
            link_node = ElementTree.SubElement(robot, "link", {"name": link.link_id})
            for visual_path in link.visual_asset_paths:
                visual = ElementTree.SubElement(link_node, "visual")
                geometry = ElementTree.SubElement(visual, "geometry")
                ElementTree.SubElement(geometry, "mesh", {"filename": visual_path})
        for joint in candidate.joints:
            joint_type = (
                "continuous"
                if joint.joint_type is ArticulatedJointType.CONTINUOUS_CANDIDATE
                else joint.joint_type.value
            )
            node = ElementTree.SubElement(
                robot,
                "joint",
                {"name": joint.joint_id, "type": joint_type},
            )
            ElementTree.SubElement(node, "parent", {"link": joint.parent_link_id})
            ElementTree.SubElement(node, "child", {"link": joint.child_link_id})
            ElementTree.SubElement(
                node,
                "axis",
                {"xyz": " ".join(f"{value:.12g}" for value in joint.axis)},
            )
            if joint.pivot is not None:
                ElementTree.SubElement(
                    node,
                    "origin",
                    {"xyz": " ".join(f"{value:.12g}" for value in joint.pivot)},
                )
            if (
                joint.candidate_limit_lower is not None
                and joint.candidate_limit_upper is not None
                and joint_type != "continuous"
            ):
                ElementTree.SubElement(
                    node,
                    "limit",
                    {
                        "lower": f"{joint.candidate_limit_lower:.12g}",
                        "upper": f"{joint.candidate_limit_upper:.12g}",
                        "effort": "0",
                        "velocity": "0",
                    },
                )
        ElementTree.indent(robot, space="  ")
        path.write_bytes(
            ElementTree.tostring(
                robot,
                encoding="utf-8",
                xml_declaration=True,
            )
        )

    @staticmethod
    def render_previews(run_dir: Path) -> None:
        root = run_dir / "reconstruction/articulation"
        capture = ArticulationCaptureManifest.model_validate_json(
            (root / "capture_manifest.json").read_text(encoding="utf-8")
        )
        candidates = ArticulatedCandidateManifest.model_validate_json(
            (root / "candidate_manifest.json").read_text(encoding="utf-8")
        )
        evaluation = ArticulatedEvaluationManifest.model_validate_json(
            (root / "evaluation_manifest.json").read_text(encoding="utf-8")
        )
        selection = ArticulatedCandidateSelection.model_validate_json(
            (root / "selection.json").read_text(encoding="utf-8")
        )
        selected = selection.objects[0]
        _preview(
            root / "previews/candidate_grid.png",
            "Phase 5C articulated candidates",
            [
                f"object: {capture.articulated_object_id}",
                f"states: {len(capture.states)}",
                f"candidates: {len(candidates.candidates)}",
                f"passing: {sum(item.passed_hard_gates for item in evaluation.evaluations)}",
            ],
        )
        _preview(
            root / "previews/selected_articulation.png",
            "Selected articulated visual hypothesis",
            [
                f"status: {selected.status.value}",
                (f"research: {selected.best_research_articulated_candidate or 'none'}"),
                (
                    "production: "
                    f"{selected.best_production_eligible_articulated_candidate or 'none'}"
                ),
                f"selected: {selected.selected_candidate_id or 'none'}",
                "collision=false | dynamics=false | sim_ready=false",
            ],
        )

    def _integrate_scene(
        self,
        scene: SceneIR,
        capture: ArticulationCaptureManifest,
        measured: MeasuredPartMotionArtifact,
        geometry: ArticulatedPartStateGeometryManifest,
        candidate: object,
    ) -> SceneIR:
        from recon2sim.artifacts import ArticulatedCandidate

        selected_candidate = candidate if isinstance(candidate, ArticulatedCandidate) else None
        reference_geometries = [
            item for item in geometry.geometries if item.state_id == capture.reference_state_id
        ]
        measured_assets = [
            GeometryAsset(
                asset_id=f"{item.part_id}.measured.phase5c",
                asset_type=AssetType.ARTICULATED,
                uri=item.measured_point_cloud_path,
                format="ply",
                source=GeometrySourceType.MEASURED,
                coordinate_convention=item.coordinate_convention,
                scale_status=ScaleStatus.SCALE_AMBIGUOUS,
                geometry_status="partial_measured",
                completion_status="not_completed",
                asset_role="measured_anchor",
                observation_grounded=True,
                physical_validation="not_implemented",
                collision_ready=False,
                sim_ready=False,
                provenance=ProvenanceRecord(
                    adapter_name=self.name,
                    adapter_version=self.version,
                    input_artifact_paths=[item.measured_point_cloud_path],
                    output_artifact_paths=["scene_ir/phase5c_scene.json"],
                    timestamp=datetime.now(UTC),
                    confidence=ConfidenceRecord(
                        score=0.8,
                        method="phase5a_measured_part_geometry",
                    ),
                    source=GeometrySourceType.MEASURED,
                ),
            )
            for item in reference_geometries
        ]
        selected_assets: list[GeometryAsset] = []
        if (
            selected_candidate is not None
            and selected_candidate.source_family is not ArticulatedSourceFamily.MEASURED_MOTION
        ):
            for link in selected_candidate.links:
                for index, path in enumerate(link.visual_asset_paths):
                    selected_assets.append(
                        GeometryAsset(
                            asset_id=f"{selected_candidate.candidate_id}.{link.link_id}.{index}",
                            asset_type=AssetType.ARTICULATED,
                            uri=path,
                            format="ply",
                            source=GeometrySourceType.GENERATED,
                            coordinate_convention=scene.metadata.coordinate_convention,
                            scale_status=ScaleStatus.SCALE_AMBIGUOUS,
                            geometry_status="articulated_visual_candidate",
                            completion_status="selected_by_multi_state_validation",
                            asset_role="articulated_visual_link",
                            observation_grounded=True,
                            physical_validation="not_implemented",
                            collision_ready=False,
                            usage_policy="research_evaluation",
                            production_selectable=selected_candidate.production_selectable,
                            sim_ready=False,
                            provenance=selected_candidate.provenance,
                        )
                    )
        preserved_assets = [
            asset
            for asset in scene.geometry_assets
            if asset.asset_id
            not in {
                *(item.asset_id for item in measured_assets),
                *(item.asset_id for item in selected_assets),
            }
        ]
        objects = [
            item for item in scene.objects if item.object_id != capture.articulated_object_id
        ]
        measured_ids_by_part = {
            item.part_id: f"{item.part_id}.measured.phase5c" for item in reference_geometries
        }
        if selected_candidate is not None:
            visual_ids_by_link = {
                link.link_id: [
                    f"{selected_candidate.candidate_id}.{link.link_id}.{index}"
                    for index in range(len(link.visual_asset_paths))
                ]
                for link in selected_candidate.links
            }
            if selected_candidate.source_family is ArticulatedSourceFamily.MEASURED_MOTION:
                visual_ids_by_link = {}
            links = [
                Link(
                    link_id=link.link_id,
                    name=link.name,
                    geometry_asset_ids=[
                        *(
                            [measured_ids_by_part[link.link_id]]
                            if link.link_id in measured_ids_by_part
                            else []
                        ),
                        *visual_ids_by_link.get(link.link_id, []),
                    ],
                )
                for link in selected_candidate.links
            ]
            measured_by_joint = {item.joint_id: item for item in measured.joint_hypotheses}
            joints = []
            for item in selected_candidate.joints:
                scene_joint_type: Literal["fixed", "prismatic", "revolute"]
                if item.joint_type is ArticulatedJointType.FIXED:
                    scene_joint_type = "fixed"
                elif item.joint_type is ArticulatedJointType.PRISMATIC:
                    scene_joint_type = "prismatic"
                elif item.joint_type is ArticulatedJointType.REVOLUTE:
                    scene_joint_type = "revolute"
                else:
                    continue
                measured_joint = measured_by_joint.get(item.joint_id)
                observed_range = (
                    (
                        measured_joint.observed_position_min,
                        measured_joint.observed_position_max,
                    )
                    if measured_joint is not None
                    and measured_joint.observed_position_min is not None
                    and measured_joint.observed_position_max is not None
                    else None
                )
                joints.append(
                    Joint(
                        joint_id=item.joint_id,
                        parent_link_id=item.parent_link_id,
                        child_link_id=item.child_link_id,
                        joint_type=scene_joint_type,
                        axis_xyz=item.axis,
                        origin_xyz=item.pivot,
                        limits=(
                            (item.candidate_limit_lower, item.candidate_limit_upper)
                            if item.candidate_limit_lower is not None
                            and item.candidate_limit_upper is not None
                            else None
                        ),
                        observed_position_range=observed_range,
                        observed_state_positions=(
                            {state.state_id: state.position for state in measured_joint.states}
                            if measured_joint is not None
                            else {}
                        ),
                        limit_source=item.limit_source,
                    )
                )
        else:
            links = [
                Link(
                    link_id=item.part_id,
                    name=item.semantic_label,
                    geometry_asset_ids=[measured_ids_by_part[item.part_id]],
                )
                for item in reference_geometries
            ]
            joints = []
            for measured_hypothesis in measured.joint_hypotheses:
                measured_scene_joint_type: Literal["fixed", "prismatic", "revolute"]
                if measured_hypothesis.joint_type is ArticulatedJointType.FIXED:
                    measured_scene_joint_type = "fixed"
                elif measured_hypothesis.joint_type is ArticulatedJointType.PRISMATIC:
                    measured_scene_joint_type = "prismatic"
                elif measured_hypothesis.joint_type is ArticulatedJointType.REVOLUTE:
                    measured_scene_joint_type = "revolute"
                else:
                    continue
                joints.append(
                    Joint(
                        joint_id=measured_hypothesis.joint_id,
                        parent_link_id=measured_hypothesis.parent_part_id,
                        child_link_id=measured_hypothesis.child_part_id,
                        joint_type=measured_scene_joint_type,
                        axis_xyz=measured_hypothesis.axis or (1.0, 0.0, 0.0),
                        origin_xyz=measured_hypothesis.pivot,
                        limits=(
                            (
                                measured_hypothesis.candidate_limit_lower,
                                measured_hypothesis.candidate_limit_upper,
                            )
                            if measured_hypothesis.candidate_limit_lower is not None
                            and measured_hypothesis.candidate_limit_upper is not None
                            else None
                        ),
                        observed_position_range=(
                            (
                                measured_hypothesis.observed_position_min,
                                measured_hypothesis.observed_position_max,
                            )
                            if measured_hypothesis.observed_position_min is not None
                            and measured_hypothesis.observed_position_max is not None
                            else None
                        ),
                        observed_state_positions={
                            state.state_id: state.position for state in measured_hypothesis.states
                        },
                        limit_source=measured_hypothesis.limit_source,
                    )
                )
        objects.append(
            ObjectInstance(
                object_id=capture.articulated_object_id,
                name=capture.articulated_object_id,
                asset_type=AssetType.ARTICULATED,
                geometry_asset_ids=list(measured_ids_by_part.values()),
                articulation=Articulation(
                    articulation_id=f"{capture.articulated_object_id}.articulation",
                    links=links,
                    joints=joints,
                    evidence_level=capture.evidence_level.value,
                    validation_artifact_path=("validation/phase5c_articulated_reconstruction.json"),
                    physical_validation="not_implemented",
                    collision_ready=False,
                    sim_ready=False,
                ),
                geometry_status=(
                    "articulated_visual_candidate"
                    if selected_candidate is not None
                    and selected_candidate.source_family
                    is not ArticulatedSourceFamily.MEASURED_MOTION
                    else "partial_measured"
                ),
                completion_status=(
                    "selected_by_multi_state_validation"
                    if selected_candidate is not None
                    and selected_candidate.source_family
                    is not ArticulatedSourceFamily.MEASURED_MOTION
                    else "not_completed"
                ),
                observation_grounded=True,
                physical_validation="not_implemented",
                sim_ready=False,
                provenance=[
                    ProvenanceRecord(
                        adapter_name=self.name,
                        adapter_version=self.version,
                        input_artifact_paths=[
                            "reconstruction/articulation/measured_motion.json",
                            "reconstruction/articulation/evaluation_manifest.json",
                        ],
                        output_artifact_paths=["scene_ir/phase5c_scene.json"],
                        timestamp=datetime.now(UTC),
                        confidence=ConfidenceRecord(
                            score=(
                                max(
                                    (item.confidence for item in measured.joint_hypotheses),
                                    default=0.0,
                                )
                            ),
                            method="multi_state_articulation_validation",
                        ),
                        source=GeometrySourceType.FUSED,
                    )
                ],
                confidence=ConfidenceRecord(
                    score=max(
                        (item.confidence for item in measured.joint_hypotheses),
                        default=0.0,
                    ),
                    method="multi_state_articulation_validation",
                ),
            )
        )
        return scene.model_copy(
            update={
                "schema_version": "0.1.6",
                "geometry_assets": [*preserved_assets, *measured_assets, *selected_assets],
                "objects": objects,
                "collision_assets": scene.collision_assets,
            }
        )


class Phase5CConsistencyValidationAdapter:
    name = "phase5c_consistency_validation"
    version = "0.1.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        geometry = ArticulatedPartStateGeometryManifest.model_validate_json(
            context.canonical_path(
                "reconstruction", "articulation", "measured_states", "manifest.json"
            ).read_text(encoding="utf-8")
        )
        specs = [
            InputSpec(
                "reconstruction/articulation/eligibility.json",
                "articulated_eligibility",
            ),
            InputSpec(
                "reconstruction/articulation/part_prompt_manifest.json",
                "articulation_part_prompt_manifest",
            ),
            InputSpec(
                "reconstruction/articulation/capture_manifest.json",
                "articulation_capture_manifest",
            ),
            InputSpec(
                "reconstruction/articulation/state_alignment.json",
                "articulation_state_alignment",
            ),
            InputSpec(
                "reconstruction/articulation/measured_motion.json",
                "measured_part_motion",
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
                "reconstruction/articulation/candidate_manifest.json",
                "articulated_candidate_manifest",
            ),
            InputSpec(
                "reconstruction/articulation/link_assignments.json",
                "articulated_link_assignment_manifest",
            ),
            InputSpec(
                "reconstruction/articulation/fitting_manifest.json",
                "articulation_fitting_manifest",
            ),
            InputSpec(
                "reconstruction/articulation/evaluation_manifest.json",
                "articulated_evaluation_manifest",
            ),
            InputSpec(
                "reconstruction/articulation/selection.json",
                "articulated_candidate_selection",
            ),
            InputSpec("scene_ir/phase5c_scene.json", "scene_ir"),
        ]
        specs.extend(
            InputSpec(
                item.measured_point_cloud_path,
                "measured_articulated_part_point_cloud",
                materialization_mode="reflink_or_copy",
            )
            for item in geometry.geometries
        )
        return specs

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return HealthcheckResult(True, "Phase 5C consistency validation available")

    def prepare(self, context: StageContext) -> None:
        context.path("validation").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                "validation/phase5c_articulated_reconstruction.json",
                "phase5c_consistency_report",
                "application/json",
                self.name,
                validation="json",
                model=Phase5CConsistencyReport,
            )
        ]

    def run(self, context: StageContext) -> StageResult:
        root = context.path("reconstruction", "articulation")
        capture = ArticulationCaptureManifest.model_validate_json(
            (root / "capture_manifest.json").read_text(encoding="utf-8")
        )
        eligibility = ArticulatedEligibilityArtifact.model_validate_json(
            (root / "eligibility.json").read_text(encoding="utf-8")
        )
        prompts = ArticulationPartPromptManifest.model_validate_json(
            (root / "part_prompt_manifest.json").read_text(encoding="utf-8")
        )
        split = ArticulationEvidenceSplit.model_validate_json(
            (root / "evidence_split.json").read_text(encoding="utf-8")
        )
        geometry = ArticulatedPartStateGeometryManifest.model_validate_json(
            (root / "measured_states/manifest.json").read_text(encoding="utf-8")
        )
        candidates = ArticulatedCandidateManifest.model_validate_json(
            (root / "candidate_manifest.json").read_text(encoding="utf-8")
        )
        from recon2sim.artifacts import ArticulationStateAlignmentArtifact

        alignment = ArticulationStateAlignmentArtifact.model_validate_json(
            (root / "state_alignment.json").read_text(encoding="utf-8")
        )
        measured = MeasuredPartMotionArtifact.model_validate_json(
            (root / "measured_motion.json").read_text(encoding="utf-8")
        )
        fitting = ArticulationFittingManifest.model_validate_json(
            (root / "fitting_manifest.json").read_text(encoding="utf-8")
        )
        evaluation = ArticulatedEvaluationManifest.model_validate_json(
            (root / "evaluation_manifest.json").read_text(encoding="utf-8")
        )
        selection = ArticulatedCandidateSelection.model_validate_json(
            (root / "selection.json").read_text(encoding="utf-8")
        )
        scene = SceneIR.model_validate_json(
            context.path("scene_ir", "phase5c_scene.json").read_text(encoding="utf-8")
        )
        checks: list[EndToEndConsistencyCheck] = []

        def check(check_id: str, passed: bool, message: str) -> None:
            checks.append(
                EndToEndConsistencyCheck(
                    check_id=check_id,
                    passed=passed,
                    message=message,
                )
            )

        check(
            "phase5b_merge_lineage_present",
            eligibility.phase5b_selection_sha256 != "0" * 64,
            "articulated routing records a Phase 5B selection or explicit override digest",
        )
        check(
            "all_states_pass_phase5a",
            all(item.phase5a_consistency_passed for item in capture.states),
            "every state independently passed Phase 5A",
        )
        check(
            "source_state_hashes_recorded",
            all(
                all(
                    len(value) == 64
                    for value in (
                        state.ingest_manifest_sha256,
                        state.camera_reconstruction_sha256,
                        state.segmentation_tracking_sha256,
                        state.dense_depth_manifest_sha256,
                        state.measured_geometry_sha256,
                    )
                )
                for state in capture.states
            ),
            "state manifests retain immutable upstream artifact hashes",
        )
        prompt_object = next(
            item
            for item in prompts.objects
            if item.articulated_object_id == capture.articulated_object_id
        )
        movable_ids = {item.part_id for item in prompt_object.movable_parts if item.include}
        check(
            "state_alignment_excludes_movable_regions",
            alignment.static_evidence_only,
            "state alignment is explicitly static-evidence-only",
        )
        check(
            "state_alignment_records_exclusions",
            all(
                movable_ids.issubset(set(item.excluded_movable_part_ids))
                for item in alignment.transforms
            ),
            "every state transform records all movable parts as excluded",
        )
        check(
            "state_transforms_proper",
            all(
                proper_positive_sim3(
                    item.matrix_reference_from_state,
                    item.inverse_matrix,
                )
                for item in alignment.transforms
            ),
            "state transforms are finite invertible positive-scale Sim(3)",
        )
        accepted_alignment_states = {
            item.state_id for item in alignment.transforms if item.accepted
        }
        used_alignment_states = (
            {state.state_id for joint in measured.joint_hypotheses for state in joint.states}
            | {state_id for item in fitting.fittings for state_id in item.fitting_state_ids}
            | {
                state.state_id
                for item in evaluation.evaluations
                for state in item.state_evaluations
            }
        )
        check(
            "state_alignment_gates",
            bool(used_alignment_states)
            and used_alignment_states.issubset(accepted_alignment_states),
            "every state actually used by Phase 5C passed static-alignment gates",
        )
        expected_parts = {prompt_object.base.prompt_id, *movable_ids}
        geometry_parts_by_state = {
            state.state_id: {
                item.part_id for item in geometry.geometries if item.state_id == state.state_id
            }
            for state in capture.states
        }
        check(
            "stable_part_ids",
            all(parts == expected_parts for parts in geometry_parts_by_state.values()),
            "configured part IDs are stable and complete across states",
        )
        check(
            "measured_part_geometry_lineage",
            all(
                item.state_id in geometry_parts_by_state
                and item.measured_point_cloud_sha256
                == sha256_file(context.path(*Path(item.measured_point_cloud_path).parts))
                for item in geometry.geometries
            ),
            "measured part clouds match their declared hashes and states",
        )
        check(
            "measured_joint_uses_aligned_states",
            all(
                {state.state_id for state in joint.states}.issubset(accepted_alignment_states)
                for joint in measured.joint_hypotheses
            ),
            "measured joint hypotheses use only accepted aligned states",
        )
        split_sets = (
            set(split.candidate_generation_states),
            set(split.kinematic_fitting_states),
            set(split.heldout_validation_states),
        )
        check(
            "heldout_not_used_for_generation",
            not (split_sets[0] & split_sets[2]),
            "held-out states are disjoint from candidate generation",
        )
        heldout_states = {
            state.state_id
            for item in evaluation.evaluations
            for state in item.state_evaluations
            if state.heldout
        }
        fitting_states = {
            state_id for item in fitting.fittings for state_id in item.fitting_state_ids
        }
        check(
            "heldout_state_leakage",
            not (heldout_states & fitting_states) and not (split_sets[1] & split_sets[2]),
            "held-out states are disjoint from fitting states",
        )
        assignment_by_candidate = {item.candidate_id: item for item in fitting.link_assignments}
        check(
            "explicit_link_assignments",
            set(assignment_by_candidate) == {item.candidate_id for item in candidates.candidates},
            "every normalized candidate has an explicit link assignment",
        )
        check(
            "candidate_joints_finite",
            all(
                all(
                    math.isfinite(value)
                    for value in (
                        *joint.axis,
                        *(joint.pivot or ()),
                    )
                )
                for candidate in candidates.candidates
                for joint in candidate.joints
            ),
            "candidate joint axes and pivots contain only finite values",
        )
        check(
            "prismatic_axes_normalized",
            all(
                abs(math.sqrt(sum(value * value for value in joint.axis)) - 1.0) <= 1e-6
                for candidate in candidates.candidates
                for joint in candidate.joints
                if joint.joint_type is ArticulatedJointType.PRISMATIC
            ),
            "all prismatic axes are normalized",
        )
        check(
            "revolute_axes_normalized",
            all(
                abs(math.sqrt(sum(value * value for value in joint.axis)) - 1.0) <= 1e-6
                for candidate in candidates.candidates
                for joint in candidate.joints
                if joint.joint_type
                in {
                    ArticulatedJointType.REVOLUTE,
                    ArticulatedJointType.CONTINUOUS_CANDIDATE,
                }
            ),
            "all revolute/continuous candidate axes are normalized",
        )

        def acyclic(candidate: object) -> bool:
            from recon2sim.artifacts import ArticulatedCandidate

            if not isinstance(candidate, ArticulatedCandidate):
                return False
            children: dict[str, list[str]] = {}
            for joint in candidate.joints:
                children.setdefault(joint.parent_link_id, []).append(joint.child_link_id)
            visiting: set[str] = set()
            visited: set[str] = set()

            def visit(link_id: str) -> bool:
                if link_id in visiting:
                    return False
                if link_id in visited:
                    return True
                visiting.add(link_id)
                if not all(visit(child) for child in children.get(link_id, [])):
                    return False
                visiting.remove(link_id)
                visited.add(link_id)
                return True

            return all(visit(link.link_id) for link in candidate.links)

        check(
            "acyclic_joint_graphs",
            all(acyclic(candidate) for candidate in candidates.candidates),
            "candidate joint graphs contain no unsupported cycles",
        )
        check(
            "candidate_base_transforms",
            all(
                item.matrix_reference_world_from_candidate_base is None
                or (
                    (inverse := invert_sim3(item.matrix_reference_world_from_candidate_base))
                    is not None
                    and proper_positive_sim3(
                        item.matrix_reference_world_from_candidate_base,
                        inverse,
                    )
                )
                for item in fitting.fittings
            ),
            "successful candidate base transforms are finite positive-scale Sim(3)",
        )
        check(
            "frozen_joint_model",
            all(item.structure_frozen_before_heldout for item in fitting.fittings),
            "candidate structure is frozen before held-out evaluation",
        )
        check(
            "heldout_only_fits_q",
            evaluation.candidate_structures_frozen_before_heldout
            and all(
                state.joint_position_source
                in {"measured_geometry", "interpolated", "discrete_state"}
                for item in evaluation.evaluations
                for state in item.state_evaluations
                if state.heldout
            ),
            "held-out evaluation estimates only allowed joint positions",
        )
        check(
            "license_policy",
            all(
                item.selected_candidate_id is None
                or item.best_research_articulated_candidate == item.selected_candidate_id
                or selection.license_mode is ArticulatedLicenseMode.PRODUCTION_CANDIDATE
                for item in selection.objects
            ),
            "selection follows the configured research/production license policy",
        )
        evaluation_by_id = {item.candidate_id: item for item in evaluation.evaluations}
        check(
            "selected_candidate_was_evaluated",
            all(
                item.selected_candidate_id is None
                or (
                    item.selected_candidate_id in evaluation_by_id
                    and evaluation_by_id[item.selected_candidate_id].passed_hard_gates
                )
                for item in selection.objects
            ),
            "every selected Scene IR candidate passed its held-out evaluation",
        )
        check(
            "measured_geometry_retained",
            all(item.measured_geometry_retained for item in selection.objects),
            "measured geometry remains present beside articulated candidates",
        )
        scene_object = next(
            (item for item in scene.objects if item.object_id == capture.articulated_object_id),
            None,
        )
        check(
            "scene_ir_selection_matches",
            scene_object is not None
            and scene_object.articulation is not None
            and all(
                selected.selected_candidate_id is None
                or selected.selected_candidate_id
                in {asset.asset_id.split(".", 1)[0] for asset in scene.geometry_assets}
                or next(
                    (
                        candidate.source_family is ArticulatedSourceFamily.MEASURED_MOTION
                        for candidate in candidates.candidates
                        if candidate.candidate_id == selected.selected_candidate_id
                    ),
                    False,
                )
                for selected in selection.objects
            ),
            "Scene IR articulation matches the deterministic selection artifact",
        )
        check(
            "no_collision_assets",
            not scene.collision_assets
            and all(not item.collision_asset_ids for item in scene.objects),
            "Phase 5C produced no collision assets",
        )
        check(
            "no_dynamics_claims",
            all(
                item.physical_validation == "not_implemented" and not item.sim_ready
                for item in scene.objects
            ),
            "Phase 5C makes no dynamics or simulation-ready claim",
        )
        check(
            "coordinate_semantics",
            scene.metadata.coordinate_convention.world_frame.value == "colmap_arbitrary"
            and scene.metadata.coordinate_convention.alignment_status.value == "unoriented"
            and scene.metadata.coordinate_convention.scale_status.value == "scale_ambiguous",
            "Scene IR remains arbitrary, unoriented, and scale-ambiguous",
        )
        prohibited = (
            "camera/colmap/database.db",
            "camera/colmap/logs",
            "observations/raw",
            "reconstruction/global/raw",
            "reconstruction/completion/raw",
        )
        check(
            "selective_input_materialization",
            not any(context.path(*Path(path).parts).exists() for path in prohibited),
            "validator attempt contains no undeclared upstream model workspaces",
        )
        check(
            "upstream_artifacts_immutable",
            all(
                item.measured_point_cloud_sha256
                == sha256_file(context.path(*Path(item.measured_point_cloud_path).parts))
                for item in geometry.geometries
            ),
            "materialized upstream measured geometry remained byte-identical",
        )
        report = Phase5CConsistencyReport(
            passed=all(item.passed for item in checks),
            checks=checks,
            multi_state_evidence_available=len(capture.states) >= 2,
            measured_joint_motion_available=bool(measured.joint_hypotheses),
            heldout_state_validation_used=bool(heldout_states),
            warnings=["articulated hypotheses are visual and kinematic only; dynamics are unknown"],
        )
        atomic_write_json(
            context.path("validation", "phase5c_articulated_reconstruction.json"),
            report,
        )
        return StageResult(
            metrics={
                "passed": report.passed,
                "check_count": len(checks),
                "heldout_state_count": len(heldout_states),
            }
        )


__all__ = [
    "ArticulationSelectionAdapter",
    "Phase5CConsistencyValidationAdapter",
]
