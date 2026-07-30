from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from recon2sim.artifacts import (
    ArticulatedAssetSpace,
    ArticulatedCandidate,
    ArticulatedCandidateEvaluation,
    ArticulatedCandidateManifest,
    ArticulatedCandidateSelection,
    ArticulatedEvaluationManifest,
    ArticulatedKinematicBundle,
    ArticulatedLicenseRecord,
    ArticulatedLinkAssignment,
    ArticulatedLinkAssignmentManifest,
    ArticulatedObjectSelection,
    ArticulatedPartStateGeometryManifest,
    ArticulatedSelectedIdentityManifest,
    ArticulatedSourceFamily,
    ArticulationCaptureManifest,
    ArticulationFittingManifest,
    ArticulationStateAlignmentArtifact,
    CameraReconstruction,
    CandidateEvaluationManifest,
    CandidateGenerationManifest,
    CandidateRegistrationManifest,
    CandidateRepresentationParityArtifact,
    CandidateSelectionArtifact,
    CanonicalSceneWrapper,
    CompletionLicenseRecord,
    FittedArticulatedKinematicModel,
    GenReconWorkerManifest,
    GlobalContextSourceAsset,
    GlobalContextSourceManifest,
    GlobalSceneReconstructionArtifact,
    MeasuredObjectGeometryArtifact,
    SceneAssemblyAssetRole,
    SceneAssemblyAssetSpace,
    SceneAssemblyInputManifest,
    SceneAssemblyLicenseRecord,
    SceneAssemblySourceArtifactType,
    SceneAssemblySourceReference,
    WorldCalibrationArtifact,
    WorldCalibrationStatus,
)
from recon2sim.assembly import IDENTITY_MATRIX4
from recon2sim.calibration import sha256_file, stable_digest
from recon2sim.ir import GeometrySourceType, SceneIR
from recon2sim.storage import atomic_write_json


def _reference(value: object, expected: SceneAssemblySourceArtifactType) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"assembly requires a typed {expected.value} source reference")
    actual = value.get("artifact_type")
    if actual != expected.value:
        raise ValueError(
            f"assembly source type mismatch: expected {expected.value!r}, got {actual!r}"
        )
    return value


def _load[ModelT: BaseModel](
    root: Path,
    value: object,
    expected: SceneAssemblySourceArtifactType,
    model: type[ModelT],
) -> ModelT:
    raw = _reference(value, expected)
    reference = SceneAssemblySourceReference.model_validate(raw)
    path = root / reference.path
    if not path.is_file() or sha256_file(path) != reference.sha256:
        raise ValueError(f"assembly source reference hash mismatch: {reference.path}")
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _assert_or_set(raw: dict[str, Any], key: str, value: object, *, label: str) -> None:
    declared = raw.get(key)
    if declared is not None and declared != value:
        raise ValueError(f"{label} disagrees with upstream typed artifact")
    raw[key] = value


def _license_status(
    value: str,
) -> Literal["approved", "not_reviewed", "research_only", "blocked"]:
    if value == "approved_by_project_policy":
        return "approved"
    if value == "research_only":
        return "research_only"
    return "not_reviewed"


def _completion_license(
    value: CompletionLicenseRecord,
    reference: dict[str, object],
) -> dict[str, object]:
    return SceneAssemblyLicenseRecord(
        license_id=f"{value.backend.value}:{value.code_license}:{value.checkpoint_license}",
        license_name=value.asset_license,
        research_evaluation_allowed=value.research_evaluation_allowed,
        production_selectable=value.production_selectable,
        commercial_review_status=_license_status(value.commercial_use_review_status),
        restrictions=[
            *value.access_conditions,
            *(
                ["commercial use requires project-policy approval"]
                if value.commercial_use_review_status != "approved_by_project_policy"
                else []
            ),
        ],
        source_record=SceneAssemblySourceReference.model_validate(reference),
    ).model_dump(mode="json")


def _articulated_license(
    value: ArticulatedLicenseRecord,
    reference: dict[str, object],
) -> dict[str, object]:
    return SceneAssemblyLicenseRecord(
        license_id=f"{value.source_family.value}:{value.code_license}:{value.checkpoint_license}",
        license_name=value.asset_license,
        research_evaluation_allowed=value.research_evaluation_allowed,
        production_selectable=value.production_selectable,
        commercial_review_status=_license_status(value.commercial_review_status),
        restrictions=[
            *value.training_data_notes,
            *(
                ["commercial use requires project-policy approval"]
                if value.commercial_review_status != "approved_by_project_policy"
                else []
            ),
        ],
        source_record=SceneAssemblySourceReference.model_validate(reference),
    ).model_dump(mode="json")


def _normalize_lineages(raw: dict[str, Any], root: Path) -> None:
    lineages = raw.get("lineages")
    if not isinstance(lineages, list):
        raise ValueError("assembly lineages must be a list")
    lineages_by_id = {
        str(value["lineage_id"]): value
        for value in lineages
        if isinstance(value, dict) and value.get("lineage_id") is not None
    }
    for value in lineages:
        if not isinstance(value, dict):
            raise ValueError("assembly lineage record must be a mapping")
        camera = _load(
            root,
            value.get("camera_reconstruction"),
            SceneAssemblySourceArtifactType.CAMERA_RECONSTRUCTION,
            CameraReconstruction,
        )
        scene = _load(
            root,
            value.get("source_scene_ir"),
            SceneAssemblySourceArtifactType.SOURCE_SCENE_IR,
            SceneIR,
        )
        if camera.frame_sequence_digest is None:
            raise ValueError("camera reconstruction lacks a frame-sequence digest")
        _assert_or_set(
            value,
            "frame_sequence_digest",
            camera.frame_sequence_digest,
            label="lineage frame-sequence digest",
        )
        _assert_or_set(
            value,
            "world_frame",
            scene.metadata.coordinate_convention.world_frame.value,
            label="lineage world frame",
        )
        if value.get("accepted_alignment") is not None:
            alignment = _load(
                root,
                value["accepted_alignment"],
                SceneAssemblySourceArtifactType.STATE_ALIGNMENT,
                ArticulationStateAlignmentArtifact,
            )
            capture_ref = _reference(
                value.get("alignment_capture_manifest"),
                SceneAssemblySourceArtifactType.ARTICULATION_CAPTURE_MANIFEST,
            )
            capture = _load(
                root,
                capture_ref,
                SceneAssemblySourceArtifactType.ARTICULATION_CAPTURE_MANIFEST,
                ArticulationCaptureManifest,
            )
            capture_reference = SceneAssemblySourceReference.model_validate(capture_ref)
            if alignment.capture_manifest_sha256 != capture_reference.sha256:
                raise ValueError("state alignment is not bound to the referenced capture manifest")
            state_id = value.get("alignment_state_id")
            transform = next(
                (item for item in alignment.transforms if item.state_id == state_id),
                None,
            )
            if transform is None or not transform.accepted:
                raise ValueError("lineage connection does not reference an accepted alignment")
            child_state = next(
                (item for item in capture.states if item.state_id == state_id),
                None,
            )
            reference_state = next(
                (item for item in capture.states if item.state_id == alignment.reference_state_id),
                None,
            )
            if child_state is None or reference_state is None:
                raise ValueError("state alignment IDs are absent from its capture manifest")
            camera_reference = SceneAssemblySourceReference.model_validate(
                value["camera_reconstruction"]
            )
            if (
                camera_reference.sha256 != child_state.camera_reconstruction_sha256
                or camera.frame_sequence_digest != child_state.frame_sequence_digest
            ):
                raise ValueError(
                    "child lineage camera hash/digest does not match the capture state"
                )
            _assert_or_set(
                value,
                "source_state_id",
                transform.state_id,
                label="lineage alignment source state",
            )
            parent = lineages_by_id.get(str(value.get("connected_to_lineage_id")))
            if parent is None:
                raise ValueError("lineage alignment target lineage is not declared")
            parent_camera = _load(
                root,
                parent.get("camera_reconstruction"),
                SceneAssemblySourceArtifactType.CAMERA_RECONSTRUCTION,
                CameraReconstruction,
            )
            parent_camera_reference = SceneAssemblySourceReference.model_validate(
                parent["camera_reconstruction"]
            )
            if (
                parent_camera_reference.sha256 != reference_state.camera_reconstruction_sha256
                or parent_camera.frame_sequence_digest != reference_state.frame_sequence_digest
            ):
                raise ValueError(
                    "reference lineage camera hash/digest does not match the capture state"
                )
            _assert_or_set(
                parent,
                "source_state_id",
                alignment.reference_state_id,
                label="lineage alignment reference state",
            )
            _assert_or_set(
                value,
                "transform_connected_from_lineage",
                list(transform.matrix_reference_from_state),
                label="lineage alignment transform",
            )


def _normalize_measured_assets(
    raw: dict[str, Any],
    root: Path,
    assets_by_id: dict[str, dict[str, Any]],
) -> None:
    for asset in assets_by_id.values():
        if asset.get("role") != SceneAssemblyAssetRole.MEASURED_ANCHOR.value:
            continue
        reference_value = asset.get("measured_geometry")
        reference = _reference(
            reference_value,
            SceneAssemblySourceArtifactType.MEASURED_GEOMETRY,
        )
        source_reference = SceneAssemblySourceReference.model_validate(reference)
        source_path = root / source_reference.path
        if not source_path.is_file() or sha256_file(source_path) != source_reference.sha256:
            raise ValueError("measured-geometry source hash mismatch")
        payload = source_path.read_text(encoding="utf-8")
        object_id = asset.get("object_id")
        try:
            artifact = MeasuredObjectGeometryArtifact.model_validate_json(payload)
        except ValueError:
            articulated = ArticulatedPartStateGeometryManifest.model_validate_json(payload)
            part_id = asset.get("part_id")
            geometry = next(
                (
                    item
                    for item in articulated.geometries
                    if item.articulated_object_id == object_id and item.part_id == part_id
                ),
                None,
            )
            if geometry is None:
                raise ValueError(
                    f"measured anchor has no articulated part geometry for "
                    f"{object_id!r}/{part_id!r}"
                ) from None
            expected_path = geometry.measured_point_cloud_path
            expected_sha = geometry.measured_point_cloud_sha256
        else:
            hypothesis = next(
                (item for item in artifact.hypotheses if item.object_id == object_id),
                None,
            )
            if hypothesis is None or hypothesis.point_cloud is None:
                raise ValueError(f"measured anchor has no upstream geometry for {object_id!r}")
            if hypothesis.geometry_source != "measured":
                raise ValueError("measured anchor source artifact is not measured geometry")
            expected_path = hypothesis.point_cloud.relative_path
            expected_sha = hypothesis.point_cloud.sha256
        if (
            asset.get("asset_sha256") != expected_sha
            or asset.get("source_native_asset_path", asset.get("asset_path")) != expected_path
        ):
            raise ValueError("measured anchor path/hash does not match its object hypothesis")
        asset["source"] = GeometrySourceType.MEASURED.value
        asset["asset_native_space"] = SceneAssemblyAssetSpace.REFERENCE_WORLD.value
        asset["selected_upstream"] = False
        asset["observation_validation_passed"] = True
        expected_rights = {
            "license_id": "user_measured_evidence",
            "license_name": "User-owned measured reconstruction evidence",
            "research_evaluation_allowed": True,
            "production_selectable": True,
            "commercial_review_status": "approved",
            "restrictions": [],
            "source_record": None,
        }
        declared_license = asset.get("license")
        if declared_license is not None and (
            SceneAssemblyLicenseRecord.model_validate(declared_license).model_dump(mode="json")
            != expected_rights
        ):
            raise ValueError(
                "measured-asset project-owned rights policy disagrees with upstream typed artifact"
            )
        asset["license"] = expected_rights


def _normalize_global_context_assets(
    raw: dict[str, Any],
    root: Path,
    assets_by_id: dict[str, dict[str, Any]],
) -> None:
    lineages = {
        str(value["lineage_id"]): value for value in raw["lineages"] if isinstance(value, dict)
    }
    for asset in assets_by_id.values():
        if asset.get("role") != SceneAssemblyAssetRole.GLOBAL_CONTEXT.value:
            continue
        reconstruction_ref = _reference(
            asset.get("global_scene_reconstruction"),
            SceneAssemblySourceArtifactType.PHASE3_GLOBAL_RECONSTRUCTION,
        )
        reconstruction = _load(
            root,
            reconstruction_ref,
            SceneAssemblySourceArtifactType.PHASE3_GLOBAL_RECONSTRUCTION,
            GlobalSceneReconstructionArtifact,
        )
        worker_ref = _reference(
            asset.get("license_source_record"),
            SceneAssemblySourceArtifactType.GLOBAL_CONTEXT_MANIFEST,
        )
        worker = _load(
            root,
            worker_ref,
            SceneAssemblySourceArtifactType.GLOBAL_CONTEXT_MANIFEST,
            GenReconWorkerManifest,
        )
        lineage = lineages.get(str(asset.get("lineage_id")))
        if lineage is None:
            raise ValueError("global-context asset has an undeclared lineage")
        camera_ref = SceneAssemblySourceReference.model_validate(lineage["camera_reconstruction"])
        scene_ref = SceneAssemblySourceReference.model_validate(lineage["source_scene_ir"])
        scene = _load(
            root,
            lineage["source_scene_ir"],
            SceneAssemblySourceArtifactType.SOURCE_SCENE_IR,
            SceneIR,
        )
        if (
            reconstruction.frame_sequence_digest != lineage["frame_sequence_digest"]
            or reconstruction.camera_reconstruction_sha256 != camera_ref.sha256
            or reconstruction.coordinate_convention != scene.metadata.coordinate_convention
            or worker.frame_sequence_digest != reconstruction.frame_sequence_digest
            or worker.official_repository != reconstruction.official_repository
            or worker.official_code_commit != reconstruction.official_code_commit
            or worker.runtime_model_revision != reconstruction.runtime_model_revision
            or worker.runtime_repository_revisions != reconstruction.runtime_repository_revisions
        ):
            raise ValueError("global-context Phase 3 lineage/backend identity mismatch")
        raw_asset_format = asset.get("format")
        if raw_asset_format not in {"glb", "ply"}:
            raise ValueError("global-context format is absent from the Phase 3 artifact")
        asset_format: Literal["glb", "ply"] = raw_asset_format
        expected_native_path = (
            reconstruction.scene_asset_path
            if asset_format == "glb"
            else reconstruction.mesh_asset_path
        )
        _assert_or_set(
            asset,
            "source_native_asset_path",
            expected_native_path,
            label="global-context native representation",
        )
        _assert_or_set(
            asset,
            "asset_path",
            expected_native_path,
            label="global-context promoted Phase 3 geometry path",
        )
        source_geometry = next(
            (
                value
                for value in scene.geometry_assets
                if value.uri == expected_native_path
                and value.format == asset_format
                and value.source is GeometrySourceType.GENERATED
            ),
            None,
        )
        if (
            source_geometry is None
            or source_geometry.coordinate_convention != reconstruction.coordinate_convention
        ):
            raise ValueError("global-context geometry is absent from the exact Phase 3 Scene IR")
        staged_path = root / str(asset["asset_path"])
        staged_sha256 = sha256_file(staged_path)
        if staged_sha256 != asset.get("asset_sha256"):
            raise ValueError("global-context geometry hash differs from staged asset bytes")
        source_manifest = GlobalContextSourceManifest(
            lineage_id=str(asset["lineage_id"]),
            frame_sequence_digest=reconstruction.frame_sequence_digest,
            camera_reconstruction_sha256=reconstruction.camera_reconstruction_sha256,
            coordinate_convention=reconstruction.coordinate_convention,
            phase3_reconstruction=SceneAssemblySourceReference.model_validate(reconstruction_ref),
            genrecon_worker_manifest=SceneAssemblySourceReference.model_validate(worker_ref),
            source_scene_ir=scene_ref,
            assets=[
                GlobalContextSourceAsset(
                    assembly_asset_id=str(asset["asset_id"]),
                    source_geometry_asset_id=source_geometry.asset_id,
                    source_native_asset_path=expected_native_path,
                    sha256=staged_sha256,
                    format=asset_format,
                )
            ],
        )
        manifest_path = (
            "assembly/source/global_context_sources/"
            f"{stable_digest(source_manifest.model_dump(mode='json'))}.json"
        )
        atomic_write_json(root / manifest_path, source_manifest)
        source_reference = SceneAssemblySourceReference(
            path=manifest_path,
            sha256=sha256_file(root / manifest_path),
            artifact_type=SceneAssemblySourceArtifactType.GLOBAL_CONTEXT_SOURCE,
        )
        _assert_or_set(
            asset,
            "global_context_source",
            source_reference.model_dump(mode="json"),
            label="global-context source manifest",
        )
        derived = SceneAssemblyLicenseRecord(
            license_id=(f"genrecon:{worker.official_code_commit}:{worker.official_license}"),
            license_name=(
                f"GenRecon {worker.official_license} code; generated output review pending"
            ),
            research_evaluation_allowed=True,
            production_selectable=False,
            commercial_review_status="not_reviewed",
            restrictions=["model-output deployment review required"],
            source_record=SceneAssemblySourceReference.model_validate(worker_ref),
        ).model_dump(mode="json")
        _assert_or_set(
            asset,
            "license",
            derived,
            label="global-context license",
        )


def _normalize_rigid_object(
    item: dict[str, Any],
    root: Path,
    assets_by_id: dict[str, dict[str, Any]],
) -> None:
    selection_ref = _reference(
        item.get("rigid_selection_artifact"),
        SceneAssemblySourceArtifactType.RIGID_SELECTION,
    )
    evaluation_ref = _reference(
        item.get("rigid_evaluation_artifact"),
        SceneAssemblySourceArtifactType.RIGID_EVALUATION,
    )
    registration_ref = _reference(
        item.get("rigid_registration_artifact"),
        SceneAssemblySourceArtifactType.RIGID_REGISTRATION,
    )
    selection = _load(
        root,
        selection_ref,
        SceneAssemblySourceArtifactType.RIGID_SELECTION,
        CandidateSelectionArtifact,
    )
    evaluation = _load(
        root,
        evaluation_ref,
        SceneAssemblySourceArtifactType.RIGID_EVALUATION,
        CandidateEvaluationManifest,
    )
    registration = _load(
        root,
        registration_ref,
        SceneAssemblySourceArtifactType.RIGID_REGISTRATION,
        CandidateRegistrationManifest,
    )
    evaluation_reference = SceneAssemblySourceReference.model_validate(evaluation_ref)
    registration_reference = SceneAssemblySourceReference.model_validate(registration_ref)
    if selection.evaluation_manifest_sha256 != evaluation_reference.sha256:
        raise ValueError("rigid selection is not bound to the referenced evaluation manifest")
    if evaluation.registration_manifest_sha256 != registration_reference.sha256:
        raise ValueError("rigid evaluation is not bound to the referenced registration manifest")
    generation_refs = item.get("rigid_generation_artifacts")
    if not isinstance(generation_refs, list) or not generation_refs:
        raise ValueError("rigid assembly object requires generation-manifest references")
    generations = [
        (
            _reference(value, SceneAssemblySourceArtifactType.RIGID_GENERATION),
            _load(
                root,
                value,
                SceneAssemblySourceArtifactType.RIGID_GENERATION,
                CandidateGenerationManifest,
            ),
        )
        for value in generation_refs
    ]
    parity_refs = item.get("representation_parity_artifacts", [])
    if not isinstance(parity_refs, list):
        raise ValueError("rigid representation-parity references must be a list")
    parity_by_path: dict[
        str,
        tuple[CandidateRepresentationParityArtifact, dict[str, object]],
    ] = {}
    for value in parity_refs:
        reference = _reference(value, SceneAssemblySourceArtifactType.REPRESENTATION_PARITY)
        normalized_reference = SceneAssemblySourceReference.model_validate(reference)
        parity_by_path[normalized_reference.path] = (
            _load(
                root,
                reference,
                SceneAssemblySourceArtifactType.REPRESENTATION_PARITY,
                CandidateRepresentationParityArtifact,
            ),
            reference,
        )
    object_id = str(item["object_id"])
    selected = next((value for value in selection.objects if value.object_id == object_id), None)
    if selected is None:
        raise ValueError(f"rigid selection has no object {object_id!r}")
    _assert_or_set(item, "upstream_status", selected.status, label="rigid selection status")
    _assert_or_set(
        item,
        "preferred_research_candidate_id",
        selected.best_research_candidate,
        label="rigid research selection",
    )
    _assert_or_set(
        item,
        "preferred_deployment_candidate_id",
        selected.best_production_eligible_candidate,
        label="rigid deployment selection",
    )
    evaluations = {value.candidate_id: value for value in evaluation.evaluations}
    registrations = {value.candidate_id: value for value in registration.registrations}
    candidates = {
        candidate.candidate_id: (candidate, reference)
        for reference, generation in generations
        for candidate in generation.candidates
    }
    preferred = {
        selected.best_research_candidate,
        selected.best_production_eligible_candidate,
    } - {None}
    if any(
        candidate_id not in evaluations or not evaluations[candidate_id].passed_hard_gates
        for candidate_id in preferred
    ):
        raise ValueError("rigid selection names a candidate that did not pass hard gates")
    normalized_candidate_ids: set[str] = set()
    for asset_id in item.get("candidate_asset_ids", []):
        asset = assets_by_id[str(asset_id)]
        candidate_id = asset.get("candidate_id")
        if candidate_id not in candidates or candidate_id not in evaluations:
            raise ValueError("rigid candidate asset lacks generation/evaluation identity")
        candidate, generation_ref = candidates[str(candidate_id)]
        evaluated = evaluations[str(candidate_id)]
        registered = registrations.get(str(candidate_id))
        if (
            candidate.object_id != object_id
            or evaluated.object_id != object_id
            or registered is None
            or registered.object_id != object_id
        ):
            raise ValueError("rigid candidate/evaluation object identity mismatch")
        if (
            candidate.backend is not evaluated.backend
            or candidate.license_record.backend is not candidate.backend
            or candidate.license_record != evaluated.license_record
        ):
            raise ValueError("rigid candidate and evaluation license identities mismatch")
        representation_pairs_used = {
            (evaluated.registration_asset_id, evaluated.registration_asset_path),
            (evaluated.evaluation_asset_id, evaluated.evaluation_asset_path),
            (evaluated.selection_asset_id, evaluated.selection_asset_path),
        }
        transfer_required = len(representation_pairs_used) > 1
        parity_record = (
            parity_by_path.get(evaluated.representation_parity_path)
            if evaluated.representation_parity_path is not None
            else None
        )
        if transfer_required:
            if parity_record is None:
                raise ValueError("rigid representation transfer lacks its parity artifact")
            parity, _ = parity_record
            representation_pairs = {
                (parity.gaussian_asset_id, parity.gaussian_asset_path),
                (parity.glb_asset_id, parity.glb_asset_path),
            }
            if (
                parity.candidate_id != candidate.candidate_id
                or parity.accepted != evaluated.representation_parity_accepted
                or not parity.accepted
                or not representation_pairs_used <= representation_pairs
            ):
                raise ValueError("rigid representation parity identity is inconsistent")
        elif evaluated.representation_parity_accepted and parity_record is None:
            raise ValueError("rigid evaluation claims parity without an artifact")
        elif parity_record is not None:
            parity, _ = parity_record
            if (
                parity.candidate_id != candidate.candidate_id
                or parity.accepted != evaluated.representation_parity_accepted
            ):
                raise ValueError("rigid diagnostic representation parity is stale")
        native = next(
            (
                value
                for value in candidate.native_assets
                if value.asset_id == evaluated.selection_asset_id
            ),
            None,
        )
        if native is None:
            raise ValueError("evaluated rigid representation is absent from native assets")
        if (
            evaluated.selection_asset_path != native.relative_path
            or asset.get("asset_sha256") != native.sha256
        ):
            raise ValueError("rigid candidate selected asset path/hash mismatch")
        if registered.frozen_transform is None:
            raise ValueError("selected rigid candidate lacks a frozen registration")
        if candidate_id == selected.selected_candidate and (
            selected.selected_native_asset_path != native.relative_path
            or selected.selected_asset_id != evaluated.selection_asset_id
            or selected.evaluated_asset_id != evaluated.evaluation_asset_id
            or selected.representation_parity_path != evaluated.representation_parity_path
        ):
            raise ValueError("rigid selected representation identity is stale")
        _assert_or_set(
            asset,
            "representation_id",
            evaluated.selection_asset_id,
            label="rigid representation",
        )
        derived = {
            "source_native_asset_path": native.relative_path,
            "selected_upstream": candidate_id in preferred,
            "observation_validation_passed": evaluated.passed_hard_gates,
            "candidate_selection": selection_ref,
            "candidate_evaluation": evaluation_ref,
            "candidate_generation": generation_ref,
            "object_to_source_world": list(registered.frozen_transform.matrix_world_from_candidate),
            "asset_to_object": list(IDENTITY_MATRIX4),
            "license": _completion_license(evaluated.license_record, evaluation_ref),
            "license_source_record": evaluation_ref,
            "source": GeometrySourceType.GENERATED.value,
            "asset_native_space": SceneAssemblyAssetSpace.CANDIDATE_BASE.value,
        }
        for key, value in derived.items():
            _assert_or_set(
                asset,
                key,
                value,
                label=f"rigid candidate {key.replace('_', ' ')}",
            )
        normalized_candidate_ids.add(candidate.candidate_id)
    if not preferred <= normalized_candidate_ids:
        raise ValueError(
            "assembly manifest omits an upstream preferred rigid candidate representation"
        )


def _load_selected_file[ModelT: BaseModel](
    root: Path,
    path: str | None,
    sha256: str | None,
    model: type[ModelT],
    *,
    label: str,
) -> ModelT:
    if path is None or sha256 is None:
        raise ValueError(f"selected articulation lacks {label} identity")
    artifact_path = root / path
    if not artifact_path.is_file() or sha256_file(artifact_path) != sha256:
        raise ValueError(f"selected articulation {label} path/hash mismatch")
    return model.model_validate_json(artifact_path.read_text(encoding="utf-8"))


def _validate_selected_articulated_identity(
    root: Path,
    item: dict[str, Any],
    selected: ArticulatedObjectSelection,
    candidate: ArticulatedCandidate,
    evaluation: ArticulatedCandidateEvaluation,
    fitted_model: FittedArticulatedKinematicModel,
    assignment: ArticulatedLinkAssignment,
) -> SceneAssemblySourceReference:
    candidate_id = candidate.candidate_id
    if selected.selected_candidate_id != candidate_id:
        raise ValueError("selected articulation candidate identity mismatch")
    selected_candidate = _load_selected_file(
        root,
        selected.selected_candidate_path,
        selected.selected_candidate_sha256,
        ArticulatedCandidate,
        label="candidate",
    )
    selected_model = _load_selected_file(
        root,
        selected.fitted_model_path,
        selected.fitted_model_sha256,
        FittedArticulatedKinematicModel,
        label="fitted model",
    )
    selected_assignment = _load_selected_file(
        root,
        selected.link_assignment_path,
        selected.link_assignment_sha256,
        ArticulatedLinkAssignment,
        label="link assignment",
    )
    selected_evaluation = _load_selected_file(
        root,
        selected.evaluation_path,
        selected.evaluation_sha256,
        ArticulatedCandidateEvaluation,
        label="evaluation",
    )
    identity = _load_selected_file(
        root,
        selected.selected_identity_manifest_path,
        selected.selected_identity_manifest_sha256,
        ArticulatedSelectedIdentityManifest,
        label="identity manifest",
    )
    bundle = _load_selected_file(
        root,
        selected.kinematic_bundle_path,
        selected.kinematic_bundle_sha256,
        ArticulatedKinematicBundle,
        label="kinematic bundle",
    )
    if (
        selected_candidate != candidate
        or selected_model != fitted_model
        or selected_assignment != assignment
        or selected_evaluation != evaluation
    ):
        raise ValueError("dedicated selected articulation records differ from source manifests")
    if (
        identity.articulated_object_id != candidate.articulated_object_id
        or identity.candidate_id != candidate_id
        or bundle.articulated_object_id != candidate.articulated_object_id
        or bundle.candidate_id != candidate_id
        or bundle.license_record != candidate.license_record
        or tuple(bundle.base_sim3) != tuple(fitted_model.matrix_reference_world_from_candidate_base)
    ):
        raise ValueError("selected articulation identity manifest or bundle is stale")
    selected_pairs = (
        (identity.selected_candidate, bundle.selected_candidate),
        (identity.fitted_kinematic_model, bundle.fitted_kinematic_model),
        (identity.selected_link_assignment, bundle.selected_link_assignment),
        (identity.selected_evaluation, bundle.selected_evaluation),
    )
    if any(left != right for left, right in selected_pairs):
        raise ValueError("selected articulation bundle references differ from identity manifest")
    selection_pairs = (
        (
            identity.selected_candidate.path,
            identity.selected_candidate.sha256,
            selected.selected_candidate_path,
            selected.selected_candidate_sha256,
        ),
        (
            identity.fitted_kinematic_model.path,
            identity.fitted_kinematic_model.sha256,
            selected.fitted_model_path,
            selected.fitted_model_sha256,
        ),
        (
            identity.selected_link_assignment.path,
            identity.selected_link_assignment.sha256,
            selected.link_assignment_path,
            selected.link_assignment_sha256,
        ),
        (
            identity.selected_evaluation.path,
            identity.selected_evaluation.sha256,
            selected.evaluation_path,
            selected.evaluation_sha256,
        ),
    )
    if any(
        (path, sha256) != (selected_path, selected_sha256)
        for path, sha256, selected_path, selected_sha256 in selection_pairs
    ):
        raise ValueError("selected identity manifest disagrees with selection path/hash pairs")
    identity_reference = SceneAssemblySourceReference.model_validate(
        _reference(
            item.get("selected_identity_manifest"),
            SceneAssemblySourceArtifactType.SELECTED_IDENTITY_MANIFEST,
        )
    )
    if (
        identity_reference.path != selected.selected_identity_manifest_path
        or identity_reference.sha256 != selected.selected_identity_manifest_sha256
        or bundle.selected_identity_manifest.path != identity_reference.path
        or bundle.selected_identity_manifest.sha256 != identity_reference.sha256
    ):
        raise ValueError("selected articulation identity reference does not match selection")
    bundle_reference = SceneAssemblySourceReference.model_validate(
        _reference(
            item.get("kinematic_bundle"),
            SceneAssemblySourceArtifactType.KINEMATIC_BUNDLE,
        )
    )
    if (
        bundle_reference.path != selected.kinematic_bundle_path
        or bundle_reference.sha256 != selected.kinematic_bundle_sha256
    ):
        raise ValueError("selected articulation kinematic-bundle reference is stale")
    return bundle_reference


def _normalize_articulated_object(
    item: dict[str, Any],
    root: Path,
    assets_by_id: dict[str, dict[str, Any]],
) -> None:
    selection_ref = _reference(
        item.get("articulated_selection_artifact"),
        SceneAssemblySourceArtifactType.ARTICULATED_SELECTION,
    )
    selection = _load(
        root,
        selection_ref,
        SceneAssemblySourceArtifactType.ARTICULATED_SELECTION,
        ArticulatedCandidateSelection,
    )
    candidate_ref = _reference(
        item.get("articulated_candidate_manifest"),
        SceneAssemblySourceArtifactType.ARTICULATED_CANDIDATE_MANIFEST,
    )
    evaluation_ref = _reference(
        item.get("articulated_evaluation_artifact"),
        SceneAssemblySourceArtifactType.ARTICULATED_EVALUATION,
    )
    fitting_ref = _reference(
        item.get("articulated_fitting_artifact"),
        SceneAssemblySourceArtifactType.ARTICULATED_FITTING,
    )
    assignment_ref = _reference(
        item.get("articulated_link_assignment_artifact"),
        SceneAssemblySourceArtifactType.ARTICULATED_LINK_ASSIGNMENT,
    )
    candidate_manifest = _load(
        root,
        candidate_ref,
        SceneAssemblySourceArtifactType.ARTICULATED_CANDIDATE_MANIFEST,
        ArticulatedCandidateManifest,
    )
    evaluation_manifest = _load(
        root,
        evaluation_ref,
        SceneAssemblySourceArtifactType.ARTICULATED_EVALUATION,
        ArticulatedEvaluationManifest,
    )
    fitting_manifest = _load(
        root,
        fitting_ref,
        SceneAssemblySourceArtifactType.ARTICULATED_FITTING,
        ArticulationFittingManifest,
    )
    assignments = _load(
        root,
        assignment_ref,
        SceneAssemblySourceArtifactType.ARTICULATED_LINK_ASSIGNMENT,
        ArticulatedLinkAssignmentManifest,
    )
    candidate_sha = SceneAssemblySourceReference.model_validate(candidate_ref).sha256
    fitting_sha = SceneAssemblySourceReference.model_validate(fitting_ref).sha256
    assignment_sha = SceneAssemblySourceReference.model_validate(assignment_ref).sha256
    if (
        evaluation_manifest.candidate_manifest_sha256 != candidate_sha
        or fitting_manifest.candidate_manifest_sha256 != candidate_sha
        or assignments.candidate_manifest_sha256 != candidate_sha
        or evaluation_manifest.fitting_manifest_sha256 != fitting_sha
        or evaluation_manifest.link_assignments_sha256 != assignment_sha
    ):
        raise ValueError("articulated selection inputs do not form one artifact identity chain")
    object_id = str(item["object_id"])
    selected = next(
        (value for value in selection.objects if value.articulated_object_id == object_id),
        None,
    )
    if selected is None:
        raise ValueError(f"articulated selection has no object {object_id!r}")
    if selected.candidate_manifest_sha256 != candidate_sha:
        raise ValueError("articulated selection candidate-manifest hash mismatch")
    _assert_or_set(item, "upstream_status", selected.status.value, label="articulated status")
    _assert_or_set(
        item,
        "preferred_research_candidate_id",
        selected.best_research_articulated_candidate,
        label="articulated research selection",
    )
    _assert_or_set(
        item,
        "preferred_deployment_candidate_id",
        selected.best_production_eligible_articulated_candidate,
        label="articulated deployment selection",
    )
    candidate_by_id = {value.candidate_id: value for value in candidate_manifest.candidates}
    evaluation_by_id = {value.candidate_id: value for value in evaluation_manifest.evaluations}
    fitting_by_id = {value.candidate_id: value for value in fitting_manifest.fittings}
    assignment_by_id = {value.candidate_id: value for value in assignments.assignments}
    preferred: set[str] = {
        value
        for value in (
            selected.best_research_articulated_candidate,
            selected.best_production_eligible_articulated_candidate,
        )
        if value is not None
    }
    for candidate_id in preferred:
        candidate = candidate_by_id.get(candidate_id)
        evaluation = evaluation_by_id.get(candidate_id)
        fitted = fitting_by_id.get(candidate_id)
        assignment = assignment_by_id.get(candidate_id)
        if (
            candidate is None
            or evaluation is None
            or fitted is None
            or assignment is None
            or not evaluation.passed_hard_gates
            or fitted.status == "failed"
            or fitted.fitted_model is None
        ):
            raise ValueError(
                "articulated selection names a candidate without a passing fitted identity"
            )
    selected_bundle: SceneAssemblySourceReference | None = None
    if selected.selected_candidate_id is not None:
        candidate = candidate_by_id.get(selected.selected_candidate_id)
        evaluation = evaluation_by_id.get(selected.selected_candidate_id)
        fitted = fitting_by_id.get(selected.selected_candidate_id)
        assignment = assignment_by_id.get(selected.selected_candidate_id)
        if (
            candidate is None
            or evaluation is None
            or fitted is None
            or fitted.fitted_model is None
            or assignment is None
        ):
            raise ValueError("selected articulation lacks candidate/evaluation/fitting identity")
        selected_bundle = _validate_selected_articulated_identity(
            root,
            item,
            selected,
            candidate,
            evaluation,
            fitted.fitted_model,
            assignment,
        )
    normalized_visuals_by_candidate: dict[str, set[tuple[str, str]]] = {}
    for asset_id in item.get("candidate_asset_ids", []):
        asset = assets_by_id[str(asset_id)]
        candidate_id = str(asset.get("candidate_id"))
        candidate = candidate_by_id.get(candidate_id)
        evaluation = evaluation_by_id.get(candidate_id)
        fitted = fitting_by_id.get(candidate_id)
        assignment = assignment_by_id.get(candidate_id)
        if candidate is None or evaluation is None or fitted is None or assignment is None:
            raise ValueError("articulated candidate asset lacks exact source identity")
        if candidate.articulated_object_id != object_id:
            raise ValueError("articulated candidate object identity mismatch")
        if (
            candidate.license_record.source_family is not candidate.source_family
            or candidate.production_selectable != candidate.license_record.production_selectable
        ):
            raise ValueError("articulated candidate and license identities mismatch")
        if (
            evaluation.candidate_sha256 != stable_digest(candidate.model_dump(mode="json"))
            or fitted.fitted_model_sha256 != evaluation.fitted_model_sha256
            or evaluation.link_assignment_sha256
            != stable_digest(assignment.model_dump(mode="json"))
        ):
            raise ValueError("articulated candidate/evaluation/fitting identity mismatch")
        if candidate.source_family is ArticulatedSourceFamily.MEASURED_MOTION:
            raise ValueError(
                "measured-motion analytic geometry must remain a measured anchor, "
                "not an articulated visual candidate"
            )
        link = next(
            (value for value in candidate.links if value.link_id == asset.get("link_id")), None
        )
        if link is None:
            raise ValueError("articulated asset link identity mismatch")
        source_path = asset.get("source_native_asset_path") or asset.get("representation_id")
        if source_path is None and len(link.visual_asset_paths) == 1:
            source_path = link.visual_asset_paths[0]
        if source_path not in link.visual_asset_paths:
            raise ValueError("articulated asset link/path identity mismatch")
        if asset.get("asset_sha256") != link.visual_asset_hashes[source_path]:
            raise ValueError("articulated selected asset path/hash mismatch")
        _assert_or_set(
            asset,
            "source_native_asset_path",
            source_path,
            label="articulated native asset path",
        )
        _assert_or_set(
            asset,
            "representation_id",
            source_path,
            label="articulated representation",
        )
        if fitted.matrix_reference_world_from_candidate_base is None:
            raise ValueError("articulated candidate lacks a fitted base transform")
        visual_space = link.visual_asset_spaces[source_path]
        if visual_space is ArticulatedAssetSpace.REFERENCE_WORLD:
            raise ValueError("generated/retrieved articulated visual cannot be reference-world")
        model_source = (
            selected_bundle
            if candidate_id == selected.selected_candidate_id
            else SceneAssemblySourceReference.model_validate(fitting_ref)
        )
        if model_source is None:
            raise ValueError("selected articulated candidate lacks its kinematic source")
        geometry_source = (
            GeometrySourceType.GENERATED
            if candidate.source_family is ArticulatedSourceFamily.PARTICULATE
            else GeometrySourceType.RETRIEVED
        )
        derived = {
            "selected_upstream": candidate_id in preferred,
            "observation_validation_passed": evaluation.passed_hard_gates,
            "candidate_selection": selection_ref,
            "candidate_evaluation": evaluation_ref,
            "candidate_generation": candidate_ref,
            "asset_native_space": visual_space.value,
            "asset_to_object": list(link.visual_asset_transforms_candidate_base[source_path]),
            "object_to_source_world": list(fitted.matrix_reference_world_from_candidate_base),
            "articulation_id": object_id,
            "kinematic_bundle": model_source.model_dump(mode="json"),
            "license": _articulated_license(candidate.license_record, candidate_ref),
            "license_source_record": candidate_ref,
            "source": geometry_source.value,
        }
        for key, value in derived.items():
            _assert_or_set(
                asset,
                key,
                value,
                label=f"articulated candidate {key.replace('_', ' ')}",
            )
        normalized_visuals_by_candidate.setdefault(candidate_id, set()).add(
            (link.link_id, source_path)
        )
    for candidate_id in preferred:
        candidate = candidate_by_id[candidate_id]
        if candidate.source_family is ArticulatedSourceFamily.MEASURED_MOTION:
            continue
        expected_visuals = {
            (link.link_id, path) for link in candidate.links for path in link.visual_asset_paths
        }
        if normalized_visuals_by_candidate.get(candidate_id, set()) != expected_visuals:
            raise ValueError(
                "assembly manifest omits a preferred articulated candidate visual link"
            )


def _normalize_calibration(raw: dict[str, Any], root: Path) -> None:
    artifact_ref = raw.get("calibration_artifact")
    wrapper_ref = raw.get("canonical_wrapper")
    if artifact_ref is None and wrapper_ref is None:
        if (
            raw.get("calibration_status") is not None
            or raw.get("source_world_to_assembly_world") is not None
        ):
            raise ValueError("local assembly manifest cannot declare calibration without sources")
        return
    artifact = _load(
        root,
        artifact_ref,
        SceneAssemblySourceArtifactType.WORLD_CALIBRATION,
        WorldCalibrationArtifact,
    )
    wrapper = _load(
        root,
        wrapper_ref,
        SceneAssemblySourceArtifactType.CANONICAL_WRAPPER,
        CanonicalSceneWrapper,
    )
    _assert_or_set(raw, "calibration_status", artifact.status.value, label="calibration status")
    transform = (
        list(artifact.accepted_transform.matrix_canonical_from_colmap)
        if artifact.accepted_transform is not None
        else None
    )
    if artifact.status not in {
        WorldCalibrationStatus.ACCEPTED_FULL_CANONICAL,
        WorldCalibrationStatus.ACCEPTED_METRIC_ONLY,
        WorldCalibrationStatus.ACCEPTED_GRAVITY_ONLY,
    }:
        transform = None
    _assert_or_set(
        raw,
        "source_world_to_assembly_world",
        transform,
        label="calibration transform",
    )
    artifact_reference = SceneAssemblySourceReference.model_validate(artifact_ref)
    primary = next(
        value for value in raw["lineages"] if value["lineage_id"] == raw["primary_lineage_id"]
    )
    if (
        wrapper.calibration_artifact_sha256 != artifact_reference.sha256
        or wrapper.source_scene_ir_sha256 != primary["source_scene_ir"]["sha256"]
        or wrapper.source_camera_reconstruction_sha256 != primary["camera_reconstruction"]["sha256"]
        or wrapper.calibration_status is not artifact.status
    ):
        raise ValueError("canonical wrapper source identity mismatches the primary lineage")
    if wrapper.accepted_transform != artifact.accepted_transform:
        raise ValueError("canonical wrapper transform mismatches calibration artifact")


def normalize_assembly_manifest(raw: dict[str, Any], root: Path) -> SceneAssemblyInputManifest:
    normalized = dict(raw)
    normalized["schema_version"] = "0.3.0"
    _normalize_lineages(normalized, root)
    primary = next(
        (
            value
            for value in normalized["lineages"]
            if value["lineage_id"] == normalized.get("primary_lineage_id")
        ),
        None,
    )
    if primary is None:
        raise ValueError("assembly primary lineage is not declared")
    _assert_or_set(
        normalized,
        "source_scene_ir",
        primary["source_scene_ir"],
        label="primary source Scene IR",
    )
    assets = normalized.get("assets")
    objects = normalized.get("objects")
    if not isinstance(assets, list) or not isinstance(objects, list):
        raise ValueError("assembly assets and objects must be lists")
    assets_by_id = {str(value["asset_id"]): value for value in assets}
    _normalize_measured_assets(normalized, root, assets_by_id)
    _normalize_global_context_assets(normalized, root, assets_by_id)
    for item in objects:
        if not isinstance(item, dict):
            raise ValueError("assembly object input must be a mapping")
        if item.get("asset_type") == "articulated":
            _normalize_articulated_object(item, root, assets_by_id)
        elif item.get("rigid_selection_artifact") is not None:
            _normalize_rigid_object(item, root, assets_by_id)
    _normalize_calibration(normalized, root)
    return SceneAssemblyInputManifest.model_validate(normalized)


__all__ = ["normalize_assembly_manifest"]
