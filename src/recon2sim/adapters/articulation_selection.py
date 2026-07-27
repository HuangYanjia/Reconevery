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
    effective_evidence_level,
    invert_sim3,
    proper_positive_sim3,
    select_articulated_candidate,
    sha256_file,
    stable_digest,
)
from recon2sim.artifacts import (
    ArticulatedAssetSpace,
    ArticulatedCandidate,
    ArticulatedCandidateEvaluation,
    ArticulatedCandidateManifest,
    ArticulatedCandidateSelection,
    ArticulatedCandidateStatus,
    ArticulatedEligibilityArtifact,
    ArticulatedEvaluationManifest,
    ArticulatedJointType,
    ArticulatedKinematicBundle,
    ArticulatedLicenseMode,
    ArticulatedLinkAssignment,
    ArticulatedObjectSelection,
    ArticulatedPartStateGeometryManifest,
    ArticulatedSelectedIdentityManifest,
    ArticulatedSourceFamily,
    ArticulationCaptureManifest,
    ArticulationDiagnostics,
    ArticulationEvidenceLevel,
    ArticulationEvidenceSplit,
    ArticulationFittingManifest,
    ArticulationPartPromptManifest,
    ArticulationPreviewManifest,
    ArticulationStateAlignmentArtifact,
    EndToEndConsistencyCheck,
    FittedArticulatedJoint,
    FittedArticulatedKinematicModel,
    MeasuredPartMotionArtifact,
    Phase5CConsistencyReport,
    SelectedArtifactReference,
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
    Transform,
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


_EVIDENCE_ORDER = {
    ArticulationEvidenceLevel.SINGLE_STATE_PRIOR_ONLY: 0,
    ArticulationEvidenceLevel.TWO_STATE_MOTION_SUPPORTED: 1,
    ArticulationEvidenceLevel.MULTI_STATE_HELDOUT_AVAILABLE: 2,
    ArticulationEvidenceLevel.MULTI_STATE_HELDOUT_VALIDATED: 3,
}


def _matrix_to_transform(values: tuple[float, ...]) -> Transform:
    if len(values) != 16:
        raise ValueError("candidate base transform must contain 16 values")
    matrix = [list(values[row * 4 : row * 4 + 4]) for row in range(4)]
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if not math.isfinite(determinant) or determinant <= 0:
        raise ValueError("candidate base transform is not a positive-scale Sim(3)")
    scale = determinant ** (1.0 / 3.0)
    rotation = [[matrix[row][column] / scale for column in range(3)] for row in range(3)]
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0:
        factor = math.sqrt(trace + 1.0) * 2
        quaternion = (
            (rotation[2][1] - rotation[1][2]) / factor,
            (rotation[0][2] - rotation[2][0]) / factor,
            (rotation[1][0] - rotation[0][1]) / factor,
            0.25 * factor,
        )
    elif rotation[0][0] > rotation[1][1] and rotation[0][0] > rotation[2][2]:
        factor = math.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]) * 2
        quaternion = (
            0.25 * factor,
            (rotation[0][1] + rotation[1][0]) / factor,
            (rotation[0][2] + rotation[2][0]) / factor,
            (rotation[2][1] - rotation[1][2]) / factor,
        )
    elif rotation[1][1] > rotation[2][2]:
        factor = math.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]) * 2
        quaternion = (
            (rotation[0][1] + rotation[1][0]) / factor,
            0.25 * factor,
            (rotation[1][2] + rotation[2][1]) / factor,
            (rotation[0][2] - rotation[2][0]) / factor,
        )
    else:
        factor = math.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]) * 2
        quaternion = (
            (rotation[0][2] + rotation[2][0]) / factor,
            (rotation[1][2] + rotation[2][1]) / factor,
            0.25 * factor,
            (rotation[1][0] - rotation[0][1]) / factor,
        )
    norm = math.sqrt(sum(value * value for value in quaternion))
    normalized_quaternion = tuple(float(value / norm) for value in quaternion)
    return Transform(
        translation=(matrix[0][3], matrix[1][3], matrix[2][3]),
        rotation_xyzw=(
            normalized_quaternion[0],
            normalized_quaternion[1],
            normalized_quaternion[2],
            normalized_quaternion[3],
        ),
        scale=(scale, scale, scale),
    )


def _matrix_to_urdf_origin(
    values: tuple[float, ...],
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    if len(values) != 16:
        raise ValueError("articulated visual transform must contain 16 values")
    scale = math.sqrt(values[0] ** 2 + values[4] ** 2 + values[8] ** 2)
    if scale <= 0:
        raise ValueError("articulated visual transform scale must be positive")
    rotation = (
        (values[0] / scale, values[1] / scale, values[2] / scale),
        (values[4] / scale, values[5] / scale, values[6] / scale),
        (values[8] / scale, values[9] / scale, values[10] / scale),
    )
    pitch = math.asin(max(-1.0, min(1.0, -rotation[2][0])))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(rotation[2][1], rotation[2][2])
        yaw = math.atan2(rotation[1][0], rotation[0][0])
    else:
        roll = math.atan2(-rotation[1][2], rotation[1][1])
        yaw = 0.0
    return (
        (values[3], values[7], values[11]),
        (roll, pitch, yaw),
        scale,
    )


def _identity_matrix(values: tuple[float, ...], tolerance: float = 1e-8) -> bool:
    identity = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    return len(values) == 16 and all(
        abs(actual - expected) <= tolerance
        for actual, expected in zip(values, identity, strict=True)
    )


def _geometry_format(path: str) -> Literal["obj", "glb", "ply"]:
    suffix = Path(path).suffix.lower().removeprefix(".")
    if suffix not in {"obj", "glb", "ply"}:
        raise ValueError(f"unsupported articulated visual asset format: {path}")
    return suffix  # type: ignore[return-value]


def _selected_artifact_reference(path: Path, run_root: Path) -> SelectedArtifactReference:
    return SelectedArtifactReference(
        path=path.relative_to(run_root).as_posix(),
        sha256=sha256_file(path),
    )


def _optional_selected_artifact_reference(
    path: str | None,
    digest: str | None,
) -> SelectedArtifactReference | None:
    if path is None or digest is None:
        return None
    return SelectedArtifactReference(path=path, sha256=digest)


def _float_vector(value: str | None, count: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        result = tuple(float(item) for item in value.split())
    except ValueError:
        return None
    return result if len(result) == count else None


def _preview_urdf_asset_transforms_match(
    path: Path,
    candidate: ArticulatedCandidate,
    fitted_model: FittedArticulatedKinematicModel,
) -> bool:
    try:
        root = ElementTree.parse(path).getroot()
    except (ElementTree.ParseError, OSError):
        return False
    metadata = root.find("reconevery_metadata")
    if metadata is None:
        return False
    recorded_base = _float_vector(metadata.get("matrix_reference_world_from_candidate_base"), 16)
    if recorded_base is None or any(
        abs(actual - expected) > 1e-8
        for actual, expected in zip(
            recorded_base,
            fitted_model.matrix_reference_world_from_candidate_base,
            strict=True,
        )
    ):
        return False
    xml_links = {item.get("name"): item for item in root.findall("link")}
    for link in candidate.links:
        xml_link = xml_links.get(link.link_id)
        if xml_link is None:
            return False
        visuals_by_path: dict[str, ElementTree.Element] = {}
        for visual in xml_link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is not None and mesh.get("filename") is not None:
                visuals_by_path[mesh.get("filename", "")] = visual
        if set(visuals_by_path) != set(link.visual_asset_paths):
            return False
        for visual_path in link.visual_asset_paths:
            visual = visuals_by_path[visual_path]
            if visual.get("reconevery_asset_space") != link.visual_asset_spaces[visual_path].value:
                return False
            origin = visual.find("origin")
            mesh = visual.find("geometry/mesh")
            if origin is None or mesh is None:
                return False
            actual_xyz = _float_vector(origin.get("xyz"), 3)
            actual_rpy = _float_vector(origin.get("rpy"), 3)
            actual_scale = _float_vector(mesh.get("scale"), 3)
            expected_xyz, expected_rpy, expected_scale = _matrix_to_urdf_origin(
                link.visual_asset_transforms_candidate_base[visual_path]
            )
            if actual_xyz is None or actual_rpy is None or actual_scale is None:
                return False
            if any(
                abs(actual - expected) > 1e-8
                for actual, expected in zip(actual_xyz, expected_xyz, strict=True)
            ) or any(
                abs(actual - expected) > 1e-8
                for actual, expected in zip(actual_rpy, expected_rpy, strict=True)
            ):
                return False
            if any(abs(value - expected_scale) > 1e-8 for value in actual_scale):
                return False
    return True


class ArticulationSelectionAdapter:
    name = "articulation_selection"
    version = "0.3.0"

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
                "reconstruction/articulation/state_alignment.json",
                "articulation_state_alignment",
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
                "reconstruction/articulation/fitting_manifest.json",
                "articulation_fitting_manifest",
            ),
            InputSpec(
                "reconstruction/articulation/link_assignments.json",
                "articulated_link_assignment_manifest",
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
        alignment = ArticulationStateAlignmentArtifact.model_validate_json(
            (root / "state_alignment.json").read_text(encoding="utf-8")
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
        fitting_by_id = {item.candidate_id: item for item in fitting.fittings}
        assignment_by_id = {item.candidate_id: item for item in fitting.link_assignments}
        evaluation_by_id = {item.candidate_id: item for item in evaluation.evaluations}
        expected_evaluation_manifest_hashes = {
            "fitting_manifest_sha256": sha256_file(root / "fitting_manifest.json"),
            "link_assignments_sha256": sha256_file(root / "link_assignments.json"),
            "candidate_manifest_sha256": sha256_file(root / "candidate_manifest.json"),
            "evidence_split_sha256": sha256_file(root / "evidence_split.json"),
            "measured_states_manifest_sha256": sha256_file(root / "measured_states/manifest.json"),
            "state_alignment_sha256": sha256_file(root / "state_alignment.json"),
            "measured_motion_sha256": sha256_file(root / "measured_motion.json"),
        }
        for field, expected_hash in expected_evaluation_manifest_hashes.items():
            if getattr(evaluation, field) != expected_hash:
                raise ValueError(f"evaluation manifest {field} does not match canonical input")
        for item in evaluation.evaluations:
            candidate = candidate_by_id[item.candidate_id]
            fitted = fitting_by_id[item.candidate_id]
            assignment = assignment_by_id[item.candidate_id]
            if item.candidate_sha256 != stable_digest(candidate.model_dump(mode="json")):
                raise ValueError("evaluation candidate identity does not match selection input")
            if item.fitted_model_sha256 != (
                fitted.fitted_model_sha256
                if fitted.fitted_model_sha256 is not None
                else stable_digest(None)
            ):
                raise ValueError("evaluation fitted-model identity does not match selection input")
            if item.link_assignment_sha256 != stable_digest(assignment.model_dump(mode="json")):
                raise ValueError(
                    "evaluation link-assignment identity does not match selection input"
                )
        production = {
            item.candidate_id: item.production_selectable for item in candidates.candidates
        }
        research_id, production_id, selected_id = select_articulated_candidate(
            evaluation.evaluations,
            production_selectable=production,
            mode=mode,
        )
        selected_candidate = candidate_by_id.get(selected_id) if selected_id else None
        selected_fitting = fitting_by_id.get(selected_id) if selected_id else None
        selected_assignment = assignment_by_id.get(selected_id) if selected_id else None
        selected_evaluation = evaluation_by_id.get(selected_id) if selected_id else None
        if selected_id is not None and (
            selected_candidate is None
            or selected_fitting is None
            or selected_fitting.fitted_model is None
            or selected_assignment is None
            or selected_evaluation is None
            or not selected_evaluation.passed_hard_gates
        ):
            raise ValueError("selected candidate lacks a passing fitted/evaluated model")
        selected_validation_level = (
            selected_evaluation.selected_candidate_validation_level
            if selected_evaluation is not None
            else measured.effective_motion_evidence_level
        )
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
                ArticulationEvidenceLevel.MULTI_STATE_HELDOUT_AVAILABLE: (
                    ArticulatedCandidateStatus.TWO_STATE
                ),
                ArticulationEvidenceLevel.MULTI_STATE_HELDOUT_VALIDATED: (
                    ArticulatedCandidateStatus.MULTI_STATE
                ),
            }[selected_validation_level]
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
        scene = SceneIR.model_validate_json(
            (root / "reference_phase5a_scene.json").read_text(encoding="utf-8")
        )
        selected_candidate_reference: SelectedArtifactReference | None = None
        fitted_model_reference: SelectedArtifactReference | None = None
        link_assignment_reference: SelectedArtifactReference | None = None
        evaluation_reference: SelectedArtifactReference | None = None
        identity_reference: SelectedArtifactReference | None = None
        bundle_reference: SelectedArtifactReference | None = None
        dynamic_outputs: list[OutputSpec] = []
        if (
            selected_candidate is not None
            and selected_fitting is not None
            and selected_fitting.fitted_model is not None
            and selected_assignment is not None
            and selected_evaluation is not None
        ):
            selected_root = root / "selected" / capture.articulated_object_id
            selected_root.mkdir(parents=True, exist_ok=True)
            candidate_path = selected_root / "selected_candidate.json"
            fitted_model_path = selected_root / "fitted_kinematic_model.json"
            link_assignment_path = selected_root / "selected_link_assignment.json"
            evaluation_path = selected_root / "selected_evaluation.json"
            identity_path = selected_root / "selected_identity_manifest.json"
            bundle_path = selected_root / "kinematic_bundle.json"
            urdf_path = selected_root / "preview_only.urdf"

            atomic_write_json(candidate_path, selected_candidate)
            atomic_write_json(fitted_model_path, selected_fitting.fitted_model)
            atomic_write_json(link_assignment_path, selected_assignment)
            atomic_write_json(evaluation_path, selected_evaluation)
            selected_candidate_reference = _selected_artifact_reference(
                candidate_path, context.run_dir
            )
            fitted_model_reference = _selected_artifact_reference(
                fitted_model_path, context.run_dir
            )
            link_assignment_reference = _selected_artifact_reference(
                link_assignment_path, context.run_dir
            )
            evaluation_reference = _selected_artifact_reference(evaluation_path, context.run_dir)
            identity = ArticulatedSelectedIdentityManifest(
                articulated_object_id=capture.articulated_object_id,
                candidate_id=selected_candidate.candidate_id,
                selected_candidate=selected_candidate_reference,
                fitted_kinematic_model=fitted_model_reference,
                selected_link_assignment=link_assignment_reference,
                selected_evaluation=evaluation_reference,
            )
            atomic_write_json(identity_path, identity)
            identity_reference = _selected_artifact_reference(identity_path, context.run_dir)
            bundle = ArticulatedKinematicBundle(
                articulated_object_id=capture.articulated_object_id,
                candidate_id=selected_candidate.candidate_id,
                selected_identity_manifest=identity_reference,
                selected_candidate=selected_candidate_reference,
                fitted_kinematic_model=fitted_model_reference,
                selected_link_assignment=link_assignment_reference,
                selected_evaluation=evaluation_reference,
                base_sim3=(
                    selected_fitting.fitted_model.matrix_reference_world_from_candidate_base
                ),
                fitting_state_q=selected_fitting.fitted_joint_positions,
                heldout_inferred_q={
                    state.state_id: state.inferred_joint_positions
                    for state in selected_evaluation.state_evaluations
                },
                license_record=selected_candidate.license_record,
                measured_joint_hypotheses=measured.joint_hypotheses,
                evidence_level=selected_validation_level,
                coordinate_convention=scene.metadata.coordinate_convention,
            )
            atomic_write_json(bundle_path, bundle)
            bundle_reference = _selected_artifact_reference(bundle_path, context.run_dir)
            self._write_preview_urdf(
                urdf_path,
                capture.articulated_object_id,
                selected_candidate,
                selected_fitting,
            )
            relative_root = f"reconstruction/articulation/selected/{capture.articulated_object_id}"
            dynamic_outputs.extend(
                [
                    OutputSpec(
                        f"{relative_root}/selected_candidate.json",
                        "selected_articulated_candidate",
                        "application/json",
                        self.name,
                        validation="json",
                        model=ArticulatedCandidate,
                    ),
                    OutputSpec(
                        f"{relative_root}/fitted_kinematic_model.json",
                        "selected_fitted_kinematic_model",
                        "application/json",
                        self.name,
                        validation="json",
                        model=FittedArticulatedKinematicModel,
                    ),
                    OutputSpec(
                        f"{relative_root}/selected_link_assignment.json",
                        "selected_articulated_link_assignment",
                        "application/json",
                        self.name,
                        validation="json",
                        model=ArticulatedLinkAssignment,
                    ),
                    OutputSpec(
                        f"{relative_root}/selected_evaluation.json",
                        "selected_articulated_evaluation",
                        "application/json",
                        self.name,
                        validation="json",
                        model=ArticulatedCandidateEvaluation,
                    ),
                    OutputSpec(
                        f"{relative_root}/selected_identity_manifest.json",
                        "selected_articulated_identity_manifest",
                        "application/json",
                        self.name,
                        validation="json",
                        model=ArticulatedSelectedIdentityManifest,
                    ),
                    OutputSpec(
                        f"{relative_root}/kinematic_bundle.json",
                        "articulated_kinematic_bundle",
                        "application/json",
                        self.name,
                        validation="json",
                        model=ArticulatedKinematicBundle,
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
        selected = ArticulatedObjectSelection(
            articulated_object_id=capture.articulated_object_id,
            status=status,
            capture_state_count=capture.capture_state_count,
            capture_evidence_tier=capture.capture_evidence_tier,
            accepted_alignment_state_ids=alignment.accepted_alignment_state_ids,
            effective_motion_evidence_level=measured.effective_motion_evidence_level,
            selected_candidate_validation_level=selected_validation_level,
            best_research_articulated_candidate=research_id,
            best_production_eligible_articulated_candidate=production_id,
            selected_candidate_id=selected_id,
            candidate_manifest_sha256=sha256_file(root / "candidate_manifest.json"),
            selected_candidate_path=(
                selected_candidate_reference.path
                if selected_candidate_reference is not None
                else None
            ),
            selected_candidate_sha256=(
                selected_candidate_reference.sha256
                if selected_candidate_reference is not None
                else None
            ),
            fitted_model_path=(
                fitted_model_reference.path if fitted_model_reference is not None else None
            ),
            fitted_model_sha256=(
                fitted_model_reference.sha256 if fitted_model_reference is not None else None
            ),
            link_assignment_path=(
                link_assignment_reference.path if link_assignment_reference is not None else None
            ),
            link_assignment_sha256=(
                link_assignment_reference.sha256 if link_assignment_reference is not None else None
            ),
            evaluation_path=(
                evaluation_reference.path if evaluation_reference is not None else None
            ),
            evaluation_sha256=(
                evaluation_reference.sha256 if evaluation_reference is not None else None
            ),
            selected_identity_manifest_path=(
                identity_reference.path if identity_reference is not None else None
            ),
            selected_identity_manifest_sha256=(
                identity_reference.sha256 if identity_reference is not None else None
            ),
            kinematic_bundle_path=(bundle_reference.path if bundle_reference is not None else None),
            kinematic_bundle_sha256=(
                bundle_reference.sha256 if bundle_reference is not None else None
            ),
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
                and selected_validation_level
                is ArticulationEvidenceLevel.MULTI_STATE_HELDOUT_VALIDATED
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
        scene = self._integrate_scene(
            scene,
            capture,
            measured,
            geometry,
            selected_candidate,
            selected_fitting,
            selected_evaluation,
            selected,
        )
        atomic_write_json(context.path("scene_ir", "phase5c_scene.json"), scene)
        diagnostics = ArticulationDiagnostics(
            capture_state_count=capture.capture_state_count,
            capture_evidence_tier=capture.capture_evidence_tier,
            accepted_alignment_state_ids=alignment.accepted_alignment_state_ids,
            effective_motion_evidence_level=measured.effective_motion_evidence_level,
            selected_candidate_validation_level=selected_validation_level,
            aligned_state_count=alignment.aligned_state_count,
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
        fitting: object,
    ) -> None:
        from recon2sim.artifacts import ArticulatedCandidate, ArticulationFittingArtifact

        if not isinstance(candidate, ArticulatedCandidate):
            raise TypeError("preview URDF requires an articulated candidate")
        fitted_model = (
            fitting.fitted_model if isinstance(fitting, ArticulationFittingArtifact) else None
        )
        fitted_by_joint = (
            {item.candidate_joint_id: item for item in fitted_model.fitted_joints}
            if fitted_model is not None
            else {}
        )
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
        if fitted_model is not None:
            ElementTree.SubElement(
                robot,
                "reconevery_metadata",
                {
                    "candidate_id": fitted_model.candidate_id,
                    "matrix_reference_world_from_candidate_base": " ".join(
                        f"{value:.12g}"
                        for value in fitted_model.matrix_reference_world_from_candidate_base
                    ),
                    "simulation_ready": "false",
                },
            )
        for link in candidate.links:
            link_node = ElementTree.SubElement(robot, "link", {"name": link.link_id})
            for visual_path in link.visual_asset_paths:
                space = link.visual_asset_spaces[visual_path]
                transform = link.visual_asset_transforms_candidate_base[visual_path]
                if (
                    space is ArticulatedAssetSpace.REFERENCE_WORLD
                    and fitted_model is not None
                    and not _identity_matrix(
                        fitted_model.matrix_reference_world_from_candidate_base
                    )
                ):
                    raise ValueError(
                        "reference-world measured visual cannot enter a transformed URDF link"
                    )
                xyz, rpy, scale = _matrix_to_urdf_origin(transform)
                visual = ElementTree.SubElement(
                    link_node,
                    "visual",
                    {"reconevery_asset_space": space.value},
                )
                ElementTree.SubElement(
                    visual,
                    "origin",
                    {
                        "xyz": " ".join(f"{value:.12g}" for value in xyz),
                        "rpy": " ".join(f"{value:.12g}" for value in rpy),
                    },
                )
                geometry = ElementTree.SubElement(visual, "geometry")
                ElementTree.SubElement(
                    geometry,
                    "mesh",
                    {
                        "filename": visual_path,
                        "scale": " ".join(f"{scale:.12g}" for _ in range(3)),
                    },
                )
        for joint in candidate.joints:
            fitted_joint = fitted_by_joint.get(joint.joint_id)
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
                {
                    "xyz": " ".join(
                        f"{value:.12g}"
                        for value in (
                            fitted_joint.fitted_axis if fitted_joint is not None else joint.axis
                        )
                    )
                },
            )
            pivot = fitted_joint.fitted_pivot if fitted_joint is not None else joint.pivot
            if pivot is not None:
                ElementTree.SubElement(
                    node,
                    "origin",
                    {"xyz": " ".join(f"{value:.12g}" for value in pivot)},
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
        fitting: object,
        evaluation: object,
        selection: ArticulatedObjectSelection,
    ) -> SceneIR:
        from recon2sim.artifacts import (
            ArticulatedCandidate,
            ArticulatedCandidateEvaluation,
            ArticulationFittingArtifact,
        )

        selected_candidate = candidate if isinstance(candidate, ArticulatedCandidate) else None
        selected_fitting = fitting if isinstance(fitting, ArticulationFittingArtifact) else None
        selected_evaluation = (
            evaluation if isinstance(evaluation, ArticulatedCandidateEvaluation) else None
        )
        fitted_model = selected_fitting.fitted_model if selected_fitting is not None else None
        reference_geometries = [
            item for item in geometry.geometries if item.state_id == capture.reference_state_id
        ]
        measured_assets = [
            GeometryAsset(
                asset_id=f"{item.part_id}.measured.phase5c",
                asset_type=AssetType.ARTICULATED,
                uri=item.measured_point_cloud_path,
                format=_geometry_format(item.measured_point_cloud_path),
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
                articulated_asset_space="reference_world",
                content_sha256=item.measured_point_cloud_sha256,
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
                            format=_geometry_format(path),
                            source=GeometrySourceType.GENERATED,
                            coordinate_convention=scene.metadata.coordinate_convention,
                            scale_status=ScaleStatus.SCALE_AMBIGUOUS,
                            geometry_status="articulated_visual_candidate",
                            completion_status="selected_by_multi_state_validation",
                            asset_role="articulated_visual_link",
                            observation_grounded=True,
                            physical_validation="not_implemented",
                            collision_ready=False,
                            articulated_asset_space=link.visual_asset_spaces[path].value,
                            asset_to_candidate_base_transform=_matrix_to_transform(
                                link.visual_asset_transforms_candidate_base[path]
                            ),
                            content_sha256=link.visual_asset_hashes[path],
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
            links = []
            for link in selected_candidate.links:
                links.append(
                    Link(
                        link_id=link.link_id,
                        name=link.name,
                        geometry_asset_ids=visual_ids_by_link.get(link.link_id, []),
                    )
                )
            measured_by_joint = {item.joint_id: item for item in measured.joint_hypotheses}
            fitted_by_candidate_joint = (
                {item.candidate_joint_id: item for item in fitted_model.fitted_joints}
                if fitted_model is not None
                else {}
            )
            heldout_positions_by_joint: dict[str, dict[str, float]] = {}
            if selected_evaluation is not None:
                for state in selected_evaluation.state_evaluations:
                    for joint_id, position in state.inferred_joint_positions.items():
                        heldout_positions_by_joint.setdefault(joint_id, {})[state.state_id] = (
                            position
                        )
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
                fitted_joint = fitted_by_candidate_joint.get(item.joint_id)
                if fitted_model is not None and fitted_joint is None:
                    continue
                measured_joint = (
                    measured_by_joint.get(fitted_joint.measured_joint_id)
                    if fitted_joint is not None
                    else measured_by_joint.get(item.joint_id)
                )
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
                        axis_xyz=(
                            fitted_joint.fitted_axis if fitted_joint is not None else item.axis
                        ),
                        origin_xyz=(
                            fitted_joint.fitted_pivot if fitted_joint is not None else item.pivot
                        ),
                        limits=(
                            (item.candidate_limit_lower, item.candidate_limit_upper)
                            if item.candidate_limit_lower is not None
                            and item.candidate_limit_upper is not None
                            else None
                        ),
                        observed_position_range=observed_range,
                        observed_state_positions=(
                            {
                                **(
                                    fitted_joint.fitting_state_q
                                    if fitted_joint is not None
                                    else (
                                        {
                                            state.state_id: state.position
                                            for state in measured_joint.states
                                        }
                                        if measured_joint is not None
                                        else {}
                                    )
                                ),
                                **heldout_positions_by_joint.get(item.joint_id, {}),
                            }
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
                transform=(
                    _matrix_to_transform(fitted_model.matrix_reference_world_from_candidate_base)
                    if fitted_model is not None
                    else Transform()
                ),
                geometry_asset_ids=list(measured_ids_by_part.values()),
                articulation=Articulation(
                    articulation_id=f"{capture.articulated_object_id}.articulation",
                    links=links,
                    joints=joints,
                    evidence_level=(
                        selected_evaluation.selected_candidate_validation_level.value
                        if selected_evaluation is not None
                        else measured.effective_motion_evidence_level.value
                    ),
                    validation_artifact_path=("validation/phase5c_articulated_reconstruction.json"),
                    selected_candidate_artifact_path=selection.selected_candidate_path,
                    selected_candidate_artifact_sha256=(selection.selected_candidate_sha256),
                    fitting_artifact_path=selection.fitted_model_path,
                    fitting_artifact_sha256=selection.fitted_model_sha256,
                    link_assignment_artifact_path=selection.link_assignment_path,
                    link_assignment_artifact_sha256=selection.link_assignment_sha256,
                    evaluation_artifact_path=selection.evaluation_path,
                    evaluation_artifact_sha256=selection.evaluation_sha256,
                    selected_identity_manifest_path=(selection.selected_identity_manifest_path),
                    selected_identity_manifest_sha256=(selection.selected_identity_manifest_sha256),
                    kinematic_bundle_path=selection.kinematic_bundle_path,
                    kinematic_bundle_sha256=selection.kinematic_bundle_sha256,
                    selected_candidate_id=(
                        selected_candidate.candidate_id if selected_candidate is not None else None
                    ),
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
    version = "0.3.0"

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
        selection = ArticulatedCandidateSelection.model_validate_json(
            context.canonical_path("reconstruction", "articulation", "selection.json").read_text(
                encoding="utf-8"
            )
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
        specs.extend(
            InputSpec(
                path,
                "articulated_candidate_visual_link",
                materialization_mode="reflink_or_copy",
            )
            for candidate in candidates.candidates
            if candidate.source_family is not ArticulatedSourceFamily.MEASURED_MOTION
            for link in candidate.links
            for path in link.visual_asset_paths
        )
        for selected in selection.objects:
            for path, kind in (
                (selected.selected_candidate_path, "selected_articulated_candidate"),
                (selected.fitted_model_path, "selected_fitted_kinematic_model"),
                (
                    selected.link_assignment_path,
                    "selected_articulated_link_assignment",
                ),
                (selected.evaluation_path, "selected_articulated_evaluation"),
                (
                    selected.selected_identity_manifest_path,
                    "selected_articulated_identity_manifest",
                ),
                (selected.kinematic_bundle_path, "articulated_kinematic_bundle"),
            ):
                if path is not None:
                    specs.append(InputSpec(path, kind))
            if selected.selected_candidate_id is not None:
                specs.append(
                    InputSpec(
                        (
                            "reconstruction/articulation/selected/"
                            f"{selected.articulated_object_id}/preview_only.urdf"
                        ),
                        "visual_only_articulation_preview",
                    )
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
        candidate_by_id = {item.candidate_id: item for item in candidates.candidates}
        fitted_by_id = {item.candidate_id: item for item in fitting.fittings}
        assignment_by_candidate = {item.candidate_id: item for item in fitting.link_assignments}
        evaluation_by_id = {item.candidate_id: item for item in evaluation.evaluations}
        selected_record = selection.objects[0] if selection.objects else None
        selected_candidate_file: ArticulatedCandidate | None = None
        selected_fitted_file: FittedArticulatedKinematicModel | None = None
        selected_assignment_file: ArticulatedLinkAssignment | None = None
        selected_evaluation_file: ArticulatedCandidateEvaluation | None = None
        selected_identity_file: ArticulatedSelectedIdentityManifest | None = None
        selected_bundle_file: ArticulatedKinematicBundle | None = None
        if selected_record is not None and selected_record.selected_candidate_id is not None:
            try:
                selected_candidate_file = ArticulatedCandidate.model_validate_json(
                    context.path(
                        *Path(selected_record.selected_candidate_path or "").parts
                    ).read_text(encoding="utf-8")
                )
                selected_fitted_file = FittedArticulatedKinematicModel.model_validate_json(
                    context.path(*Path(selected_record.fitted_model_path or "").parts).read_text(
                        encoding="utf-8"
                    )
                )
                selected_assignment_file = ArticulatedLinkAssignment.model_validate_json(
                    context.path(*Path(selected_record.link_assignment_path or "").parts).read_text(
                        encoding="utf-8"
                    )
                )
                selected_evaluation_file = ArticulatedCandidateEvaluation.model_validate_json(
                    context.path(*Path(selected_record.evaluation_path or "").parts).read_text(
                        encoding="utf-8"
                    )
                )
                selected_identity_file = ArticulatedSelectedIdentityManifest.model_validate_json(
                    context.path(
                        *Path(selected_record.selected_identity_manifest_path or "").parts
                    ).read_text(encoding="utf-8")
                )
                selected_bundle_file = ArticulatedKinematicBundle.model_validate_json(
                    context.path(
                        *Path(selected_record.kinematic_bundle_path or "").parts
                    ).read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                pass
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
        valid_measured_motion = any(
            item.joint_type
            in {
                ArticulatedJointType.FIXED,
                ArticulatedJointType.PRISMATIC,
                ArticulatedJointType.REVOLUTE,
            }
            for item in measured.joint_hypotheses
        )
        expected_effective_level = effective_evidence_level(
            len(accepted_alignment_states),
            valid_measured_motion=valid_measured_motion,
        )
        check(
            "accepted_alignment_state_count",
            alignment.accepted_alignment_state_ids
            == [item.state_id for item in alignment.transforms if item.accepted]
            and alignment.aligned_state_count == len(accepted_alignment_states),
            "aligned_state_count and accepted IDs derive from accepted transforms",
        )
        check(
            "effective_evidence_from_accepted_states",
            measured.capture_state_count == capture.capture_state_count
            and set(measured.accepted_alignment_state_ids) == accepted_alignment_states
            and measured.effective_motion_evidence_level == expected_effective_level,
            "effective motion evidence derives from accepted alignments and a valid motion model",
        )
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
        expected_parts = {prompt_object.base.part_id, *movable_ids}
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
        geometry_by_state_part = {
            (item.state_id, item.part_id): item for item in geometry.geometries
        }
        check(
            "state_local_track_mapping",
            all(
                set(state.part_track_ids) == expected_parts
                and len(set(state.part_track_ids.values())) == len(expected_parts)
                and all(
                    (geometry_item := geometry_by_state_part.get((state.state_id, part_id)))
                    is not None
                    and geometry_item.source_track_id == track_id
                    for part_id, track_id in state.part_track_ids.items()
                )
                for state in capture.states
            ),
            "stable part IDs map explicitly to unique state-local SAM track IDs",
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
        check(
            "declared_reference_state_used",
            measured.reference_state_id == capture.reference_state_id
            and capture.reference_state_id in accepted_alignment_states
            and all(
                any(state.state_id == capture.reference_state_id for state in joint.states)
                for joint in measured.joint_hypotheses
            ),
            "measured motion uses the declared accepted reference state",
        )

        def normalized_residual_valid(
            raw: float | None,
            normalized: float | None,
            diagonal: float | None,
        ) -> bool:
            if raw is None:
                return normalized is None
            return (
                normalized is not None
                and diagonal is not None
                and diagonal > 0
                and math.isclose(
                    normalized,
                    raw / diagonal,
                    rel_tol=1e-6,
                    abs_tol=1e-9,
                )
            )

        normalized_thresholds_valid = all(
            normalized_residual_valid(
                joint.fixed_translation_residual_arbitrary_units,
                joint.fixed_translation_residual_part_diagonals,
                joint.normalization_part_diagonal,
            )
            and normalized_residual_valid(
                joint.pivot_residual_arbitrary_units,
                joint.pivot_residual_part_diagonals,
                joint.normalization_part_diagonal,
            )
            for joint in measured.joint_hypotheses
        )
        check(
            "normalized_part_threshold_semantics",
            normalized_thresholds_valid,
            "part-diagonal thresholds record consistent raw and normalized residuals",
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
            "particulate_working_frame_audit",
            all(
                candidate.source_family is not ArticulatedSourceFamily.PARTICULATE
                or (
                    candidate.working_frame_hypothesis is not None
                    and candidate.working_frame_hypothesis
                    in candidate.working_frame_hypotheses_evaluated
                    and candidate.working_frame_selection_evidence is not None
                    and candidate.working_transform_source_to_particulate is not None
                    and candidate.working_transform_particulate_to_source is not None
                    and proper_positive_sim3(
                        candidate.working_transform_source_to_particulate,
                        candidate.working_transform_particulate_to_source,
                    )
                )
                for candidate in candidates.candidates
            ),
            "Particulate candidates record an explicit reversible working-frame prior",
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
            "explicit_articulated_asset_spaces",
            all(
                set(link.visual_asset_paths)
                == set(link.visual_asset_hashes)
                == set(link.visual_asset_spaces)
                == set(link.visual_asset_transforms_candidate_base)
                and all(
                    link.visual_asset_hashes[path] == sha256_file(context.path(*Path(path).parts))
                    for path in link.visual_asset_paths
                )
                and all(
                    (
                        candidate.source_family is ArticulatedSourceFamily.MEASURED_MOTION
                        and link.visual_asset_spaces[path] is ArticulatedAssetSpace.REFERENCE_WORLD
                    )
                    or (
                        candidate.source_family is not ArticulatedSourceFamily.MEASURED_MOTION
                        and link.visual_asset_spaces[path]
                        in {
                            ArticulatedAssetSpace.CANDIDATE_BASE,
                            ArticulatedAssetSpace.LINK_LOCAL,
                        }
                    )
                    for path in link.visual_asset_paths
                )
                for candidate in candidates.candidates
                for link in candidate.links
            ),
            "measured anchors remain reference-world while candidate visuals use candidate spaces",
        )
        check(
            "original_measured_geometry_immutable",
            all(
                item.measured_point_cloud_sha256
                == sha256_file(context.path(*Path(item.measured_point_cloud_path).parts))
                for item in geometry.geometries
            ),
            "original Phase 5A measured point clouds remain byte-identical",
        )
        check(
            "fitted_model_hashes",
            all(
                (
                    item.fitted_model is None
                    and item.fitted_model_sha256 is None
                    and item.status == "failed"
                )
                or (
                    item.fitted_model is not None
                    and item.fitted_model_sha256
                    == stable_digest(item.fitted_model.model_dump(mode="json"))
                    and item.matrix_reference_world_from_candidate_base
                    == item.fitted_model.matrix_reference_world_from_candidate_base
                )
                for item in fitting.fittings
            ),
            "every successful fit has an exact typed fitted-model hash",
        )
        check(
            "canonical_axis_q_sign_convention",
            all(
                all(
                    joint.axis_sign_role == "native_axis_flip_provenance_only"
                    and (
                        math.isclose(joint.q_scale, 1.0 / item.fitted_model.scale)
                        if joint.joint_type is ArticulatedJointType.PRISMATIC
                        else math.isclose(joint.q_scale, 1.0)
                        if joint.joint_type
                        in {
                            ArticulatedJointType.REVOLUTE,
                            ArticulatedJointType.CONTINUOUS_CANDIDATE,
                        }
                        else math.isclose(joint.q_scale, 0.0)
                    )
                    for joint in item.fitted_model.fitted_joints
                )
                for item in fitting.fittings
                if item.fitted_model is not None
            ),
            "axis sign is provenance-only and q scale follows the canonical convention",
        )
        check(
            "ambiguous_assignments_not_accepted",
            all(
                not any(record.ambiguous for record in assignment.assignments)
                or next(
                    (
                        fit.status == "ambiguous"
                        for fit in fitting.fittings
                        if fit.candidate_id == assignment.candidate_id
                    ),
                    False,
                )
                for assignment in fitting.link_assignments
            ),
            "ambiguous link assignments cannot become accepted fits",
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
            "heldout_render_coverage",
            all(
                state.requested_heldout_view_count > 0
                and state.usable_heldout_view_count > 0
                and state.rendered_heldout_view_count > 0
                and state.views_with_target_masks > 0
                and state.views_with_valid_depth > 0
                and all(
                    not view.missing_link_ids
                    and set(view.required_link_ids) == set(view.rendered_link_ids)
                    and view.target_masks_complete
                    and view.valid_depth
                    and view.visible_candidate_pixel_count > 0
                    and view.render_sha256 is not None
                    for view in state.view_evaluations
                    if view.usable
                )
                for item in evaluation.evaluations
                if item.passed_hard_gates
                for state in item.state_evaluations
                if state.heldout
            ),
            "passing candidates have rendered target masks and dense depth on held-out views",
        )
        fitted_by_id = {item.candidate_id: item for item in fitting.fittings}

        def fitted_joints_for(candidate_id: str) -> list[FittedArticulatedJoint]:
            fitted = fitted_by_id[candidate_id].fitted_model
            return list(fitted.fitted_joints) if fitted is not None else []

        check(
            "joint_metrics_are_measured",
            all(
                state.base_point_residual_arbitrary_units is not None
                and state.base_motion_scene_diagonals is not None
                and state.joint_constraint_residual is not None
                and all(
                    (
                        joint.joint_type is not ArticulatedJointType.PRISMATIC
                        or state.prismatic_orthogonal_residual is not None
                        and state.prismatic_rotation_leakage_degrees is not None
                    )
                    and (
                        joint.joint_type
                        not in {
                            ArticulatedJointType.REVOLUTE,
                            ArticulatedJointType.CONTINUOUS_CANDIDATE,
                        }
                        or state.axis_error_degrees is not None
                        and state.pivot_residual_part_diagonals is not None
                    )
                    for joint in fitted_joints_for(item.candidate_id)
                )
                for item in evaluation.evaluations
                if item.passed_hard_gates
                for state in item.state_evaluations
                if state.heldout
            ),
            "passing held-out evaluations contain non-placeholder joint-specific metrics",
        )
        check(
            "exact_three_state_accounting",
            all(
                item.selected_candidate_validation_level
                is not ArticulationEvidenceLevel.MULTI_STATE_HELDOUT_VALIDATED
                or (
                    (model := fitted_by_id[item.candidate_id].fitted_model) is not None
                    and len(
                        (
                            set(model.generation_state_ids)
                            | set(model.fitting_state_ids)
                            | {state.state_id for state in item.state_evaluations if state.heldout}
                        )
                        & accepted_alignment_states
                    )
                    >= 3
                )
                for item in evaluation.evaluations
            ),
            "three-state validity counts generation, fitting, and evaluated held-out states",
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
        candidate_by_id = {item.candidate_id: item for item in candidates.candidates}
        check(
            "heldout_artifact_identity",
            evaluation.fitting_manifest_sha256 == sha256_file(root / "fitting_manifest.json")
            and evaluation.link_assignments_sha256 == sha256_file(root / "link_assignments.json")
            and evaluation.candidate_manifest_sha256
            == sha256_file(root / "candidate_manifest.json")
            and evaluation.evidence_split_sha256 == sha256_file(root / "evidence_split.json")
            and evaluation.measured_states_manifest_sha256
            == sha256_file(root / "measured_states/manifest.json")
            and evaluation.state_alignment_sha256 == sha256_file(root / "state_alignment.json")
            and evaluation.measured_motion_sha256 == sha256_file(root / "measured_motion.json")
            and all(
                item.candidate_sha256
                == stable_digest(candidate_by_id[item.candidate_id].model_dump(mode="json"))
                and item.fitted_model_sha256
                == (fitted_by_id[item.candidate_id].fitted_model_sha256 or stable_digest(None))
                and item.link_assignment_sha256
                == stable_digest(assignment_by_candidate[item.candidate_id].model_dump(mode="json"))
                for item in evaluation.evaluations
            ),
            "held-out evaluations bind exact upstream, candidate, fit, and assignment identities",
        )
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
        selected_path_hash_pairs = (
            (
                (
                    selected_record.selected_candidate_path,
                    selected_record.selected_candidate_sha256,
                ),
                (selected_record.fitted_model_path, selected_record.fitted_model_sha256),
                (
                    selected_record.link_assignment_path,
                    selected_record.link_assignment_sha256,
                ),
                (selected_record.evaluation_path, selected_record.evaluation_sha256),
                (
                    selected_record.selected_identity_manifest_path,
                    selected_record.selected_identity_manifest_sha256,
                ),
                (
                    selected_record.kinematic_bundle_path,
                    selected_record.kinematic_bundle_sha256,
                ),
            )
            if selected_record is not None
            else ()
        )
        check(
            "selected_file_hashes_exact",
            selected_record is not None
            and (
                selected_record.selected_candidate_id is None
                or all(
                    path is not None
                    and digest is not None
                    and sha256_file(context.path(*Path(path).parts)) == digest
                    for path, digest in selected_path_hash_pairs
                )
            ),
            "every selected path/SHA pair hashes the exact declared file bytes",
        )
        selected_id = selected_record.selected_candidate_id if selected_record is not None else None
        check(
            "dedicated_selected_records_match_canonical",
            selected_id is None
            or (
                selected_candidate_file == candidate_by_id.get(selected_id)
                and selected_fitted_file
                == (fitted_by_id[selected_id].fitted_model if selected_id in fitted_by_id else None)
                and selected_assignment_file == assignment_by_candidate.get(selected_id)
                and selected_evaluation_file == evaluation_by_id.get(selected_id)
            ),
            "dedicated selected records contain exactly one canonical selected object",
        )
        expected_candidate_reference = _optional_selected_artifact_reference(
            selected_record.selected_candidate_path if selected_record is not None else None,
            selected_record.selected_candidate_sha256 if selected_record is not None else None,
        )
        expected_fitted_reference = _optional_selected_artifact_reference(
            selected_record.fitted_model_path if selected_record is not None else None,
            selected_record.fitted_model_sha256 if selected_record is not None else None,
        )
        expected_assignment_reference = _optional_selected_artifact_reference(
            selected_record.link_assignment_path if selected_record is not None else None,
            selected_record.link_assignment_sha256 if selected_record is not None else None,
        )
        expected_evaluation_reference = _optional_selected_artifact_reference(
            selected_record.evaluation_path if selected_record is not None else None,
            selected_record.evaluation_sha256 if selected_record is not None else None,
        )
        expected_identity_reference = _optional_selected_artifact_reference(
            (
                selected_record.selected_identity_manifest_path
                if selected_record is not None
                else None
            ),
            (
                selected_record.selected_identity_manifest_sha256
                if selected_record is not None
                else None
            ),
        )
        check(
            "selected_identity_manifest_exact",
            selected_id is None
            or (
                selected_identity_file is not None
                and selected_identity_file.articulated_object_id == capture.articulated_object_id
                and selected_identity_file.candidate_id == selected_id
                and selected_identity_file.selected_candidate == expected_candidate_reference
                and selected_identity_file.fitted_kinematic_model == expected_fitted_reference
                and selected_identity_file.selected_link_assignment == expected_assignment_reference
                and selected_identity_file.selected_evaluation == expected_evaluation_reference
            ),
            "selected identity manifest binds candidate, fit, assignment, and evaluation files",
        )
        check(
            "kinematic_bundle_selected_identity",
            selected_id is None
            or (
                selected_bundle_file is not None
                and selected_bundle_file.candidate_id == selected_id
                and selected_bundle_file.selected_candidate == expected_candidate_reference
                and selected_bundle_file.fitted_kinematic_model == expected_fitted_reference
                and selected_bundle_file.selected_link_assignment == expected_assignment_reference
                and selected_bundle_file.selected_evaluation == expected_evaluation_reference
                and selected_bundle_file.selected_identity_manifest == expected_identity_reference
            ),
            "kinematic bundle references every dedicated selected record by exact file hash",
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
        selected_record = selection.objects[0] if selection.objects else None
        selected_fit = (
            fitted_by_id.get(selected_record.selected_candidate_id)
            if selected_record is not None and selected_record.selected_candidate_id is not None
            else None
        )
        expected_scene_transform = (
            _matrix_to_transform(
                selected_fit.fitted_model.matrix_reference_world_from_candidate_base
            )
            if selected_fit is not None and selected_fit.fitted_model is not None
            else Transform()
        )
        check(
            "scene_ir_fitted_base_transform",
            scene_object is not None and scene_object.transform == expected_scene_transform,
            "Scene IR object transform equals the selected fitted base Sim(3)",
        )
        expected_fitted_joints = (
            {item.candidate_joint_id: item for item in selected_fit.fitted_model.fitted_joints}
            if selected_fit is not None and selected_fit.fitted_model is not None
            else {}
        )
        scene_joints = (
            {item.joint_id: item for item in scene_object.articulation.joints}
            if scene_object is not None and scene_object.articulation is not None
            else {}
        )
        check(
            "scene_ir_fitted_joints",
            all(
                joint_id in scene_joints
                and scene_joints[joint_id].axis_xyz == fitted.fitted_axis
                and scene_joints[joint_id].origin_xyz == fitted.fitted_pivot
                for joint_id, fitted in expected_fitted_joints.items()
            ),
            "Scene IR joints use selected fitted/refined axes and pivots",
        )
        check(
            "scene_ir_visual_asset_formats",
            all(asset.format == _geometry_format(asset.uri) for asset in scene.geometry_assets),
            "Scene IR visual formats match their actual file extensions",
        )
        check(
            "scene_ir_visual_asset_spaces",
            all(
                (
                    asset.articulated_asset_space in {"candidate_base", "link_local"}
                    and asset.asset_to_candidate_base_transform is not None
                    and asset.content_sha256 == sha256_file(context.path(*Path(asset.uri).parts))
                )
                for asset in scene.geometry_assets
                if asset.asset_role == "articulated_visual_link"
            ),
            "Scene IR candidate visuals preserve explicit spaces and exact content hashes",
        )
        measured_asset_ids = {
            f"{item.part_id}.measured.phase5c": item
            for item in geometry.geometries
            if item.state_id == capture.reference_state_id
        }
        scene_assets_by_id = {asset.asset_id: asset for asset in scene.geometry_assets}
        check(
            "scene_ir_measured_assets_reference_world",
            all(
                asset_id in scene_assets_by_id
                and scene_assets_by_id[asset_id].articulated_asset_space == "reference_world"
                and scene_assets_by_id[asset_id].asset_to_candidate_base_transform is None
                and scene_assets_by_id[asset_id].content_sha256
                == geometry_item.measured_point_cloud_sha256
                and sha256_file(context.path(*Path(scene_assets_by_id[asset_id].uri).parts))
                == geometry_item.measured_point_cloud_sha256
                for asset_id, geometry_item in measured_asset_ids.items()
            ),
            "original measured anchors remain immutable reference-world evidence",
        )
        linked_reference_world_assets = (
            {
                asset_id
                for link in scene_object.articulation.links
                for asset_id in link.geometry_asset_ids
                if asset_id in scene_assets_by_id
                and scene_assets_by_id[asset_id].articulated_asset_space == "reference_world"
            }
            if scene_object is not None and scene_object.articulation is not None
            else set()
        )
        selected_source_family = (
            candidate_by_id[selected_id].source_family
            if selected_id is not None and selected_id in candidate_by_id
            else None
        )
        check(
            "no_reference_world_double_transform",
            not linked_reference_world_assets
            or (
                selected_source_family in {None, ArticulatedSourceFamily.MEASURED_MOTION}
                and scene_object is not None
                and scene_object.transform == Transform()
            ),
            "reference-world measured evidence is not attached below a candidate transform",
        )
        scene_articulation = (
            scene_object.articulation
            if scene_object is not None and scene_object.articulation is not None
            else None
        )
        check(
            "scene_ir_selected_artifact_paths",
            selected_id is None
            or (
                scene_articulation is not None
                and selected_record is not None
                and scene_articulation.selected_candidate_artifact_path
                == selected_record.selected_candidate_path
                and scene_articulation.selected_candidate_artifact_sha256
                == selected_record.selected_candidate_sha256
                and scene_articulation.fitting_artifact_path == selected_record.fitted_model_path
                and scene_articulation.fitting_artifact_sha256
                == selected_record.fitted_model_sha256
                and scene_articulation.link_assignment_artifact_path
                == selected_record.link_assignment_path
                and scene_articulation.link_assignment_artifact_sha256
                == selected_record.link_assignment_sha256
                and scene_articulation.evaluation_artifact_path == selected_record.evaluation_path
                and scene_articulation.evaluation_artifact_sha256
                == selected_record.evaluation_sha256
                and scene_articulation.selected_identity_manifest_path
                == selected_record.selected_identity_manifest_path
                and scene_articulation.selected_identity_manifest_sha256
                == selected_record.selected_identity_manifest_sha256
                and scene_articulation.kinematic_bundle_path
                == selected_record.kinematic_bundle_path
                and scene_articulation.kinematic_bundle_sha256
                == selected_record.kinematic_bundle_sha256
            ),
            "Scene IR references exact dedicated selected artifacts",
        )
        urdf_path = root / "selected" / capture.articulated_object_id / "preview_only.urdf"
        check(
            "preview_urdf_asset_transforms",
            selected_id is None
            or (
                selected_candidate_file is not None
                and selected_fitted_file is not None
                and _preview_urdf_asset_transforms_match(
                    urdf_path,
                    selected_candidate_file,
                    selected_fitted_file,
                )
            ),
            "preview URDF preserves every visual asset transform and fitted-base metadata",
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
            multi_state_evidence_available=(
                alignment.aligned_state_count >= 3 and bool(split.heldout_validation_states)
            ),
            measured_joint_motion_available=bool(measured.joint_hypotheses),
            heldout_state_validation_used=bool(heldout_states),
            capture_state_count=capture.capture_state_count,
            capture_evidence_tier=capture.capture_evidence_tier,
            accepted_alignment_state_ids=alignment.accepted_alignment_state_ids,
            effective_motion_evidence_level=measured.effective_motion_evidence_level,
            selected_candidate_validation_level=(
                selection.objects[0].selected_candidate_validation_level
                if selection.objects
                else measured.effective_motion_evidence_level
            ),
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
