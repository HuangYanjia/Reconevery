from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageOps

from recon2sim.adapters.base import (
    HealthcheckResult,
    InputSpec,
    OutputSpec,
    StageContext,
    StageResult,
)
from recon2sim.adapters.completion_registration import GENERATION_MANIFESTS
from recon2sim.artifacts import (
    CameraReconstruction,
    CandidateEvaluationManifest,
    CandidateGenerationManifest,
    CandidateHeldoutEvaluation,
    CandidateRegistrationManifest,
    CandidateRegistrationRequest,
    CandidateSelectionArtifact,
    CompletionCropManifest,
    CompletionDiagnostics,
    CompletionEligibilityArtifact,
    CompletionEligibilityStatus,
    CompletionEvidencePackage,
    CompletionEvidenceSplit,
    CompletionLicenseMode,
    DenseDepthManifest,
    EndToEndConsistencyCheck,
    IngestManifest,
    MeasuredObjectGeometryArtifact,
    ObjectCompletionCandidate,
    Phase5BConsistencyReport,
    SegmentationTrackingArtifact,
    SelectedVisualCompletion,
)
from recon2sim.completion import (
    pareto_front,
    positive_scale_sim3,
    sha256_file,
    stable_digest,
)
from recon2sim.ir import (
    AssetType,
    ConfidenceRecord,
    GeometryAsset,
    GeometrySourceType,
    ObjectInstance,
    ProvenanceRecord,
    SceneIR,
)
from recon2sim.storage import atomic_write_json


class CompletionSelectionAdapter:
    name = "completion_candidate_selection"
    version = "0.1.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        specs = [
            InputSpec("reconstruction/completion/eligibility.json", "completion_eligibility"),
            InputSpec(
                "reconstruction/completion/evaluation_manifest.json",
                "candidate_evaluation_manifest",
            ),
            InputSpec(
                "reconstruction/completion/registration_manifest.json",
                "candidate_registration_manifest",
            ),
            InputSpec(
                "reconstruction/measured_objects/geometry_manifest.json",
                "measured_object_geometry",
            ),
            InputSpec("scene_ir/phase5a_scene.json", "scene_ir"),
        ]
        evaluation = CandidateEvaluationManifest.model_validate_json(
            context.canonical_path(
                "reconstruction", "completion", "evaluation_manifest.json"
            ).read_text(encoding="utf-8")
        )
        specs.extend(
            InputSpec(path, "candidate_heldout_render")
            for item in evaluation.evaluations
            for path in item.render_paths.values()
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
                            "measured_object_geometry_file"
                            if candidate.backend.value == "measured_partial_baseline"
                            else "completion_candidate_file"
                        ),
                        materialization_mode="reflink_or_copy",
                    )
                    for asset in candidate.native_assets
                )
        return [replace(spec, include_producer_signature=False) for spec in specs]

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return HealthcheckResult(True, "deterministic license-aware completion selection available")

    def prepare(self, context: StageContext) -> None:
        for path in (
            context.path("reconstruction", "completion", "selected"),
            context.path("reconstruction", "completion", "previews", "objects"),
            context.path("scene_ir"),
        ):
            path.mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        root = "reconstruction/completion"
        outputs = [
            OutputSpec(
                f"{root}/selection.json",
                "candidate_selection",
                "application/json",
                "completion_selection",
                validation="json",
                model=CandidateSelectionArtifact,
            ),
            OutputSpec(
                f"{root}/diagnostics.json",
                "completion_diagnostics",
                "application/json",
                "completion_selection",
                validation="json",
                model=CompletionDiagnostics,
            ),
            OutputSpec(
                "scene_ir/phase5b_scene.json",
                "scene_ir",
                "application/json",
                "completion_selection",
                validation="scene_ir",
                model=SceneIR,
            ),
        ]
        for name in (
            "candidate_grid",
            "registration_contact_sheet",
            "heldout_evaluation_contact_sheet",
            "measured_vs_candidates",
            "selected_object_contact_sheet",
        ):
            outputs.append(
                OutputSpec(
                    f"{root}/previews/{name}.png",
                    "completion_preview",
                    "image/png",
                    "completion_selection",
                    validation="png",
                )
            )
        return outputs

    def run(self, context: StageContext) -> StageResult:
        license_mode = CompletionLicenseMode(
            context.config.adapter.config.get("license_mode", "research_evaluation")
        )
        root = context.path("reconstruction", "completion")
        eligibility = CompletionEligibilityArtifact.model_validate_json(
            (root / "eligibility.json").read_text(encoding="utf-8")
        )
        evaluation = CandidateEvaluationManifest.model_validate_json(
            (root / "evaluation_manifest.json").read_text(encoding="utf-8")
        )
        registration = CandidateRegistrationManifest.model_validate_json(
            (root / "registration_manifest.json").read_text(encoding="utf-8")
        )
        measured = MeasuredObjectGeometryArtifact.model_validate_json(
            context.path("reconstruction", "measured_objects", "geometry_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        generations = [
            CandidateGenerationManifest.model_validate_json(
                context.path(*Path(path).parts).read_text(encoding="utf-8")
            )
            for path in GENERATION_MANIFESTS.values()
        ]
        candidates = {
            candidate.candidate_id: candidate
            for generation in generations
            for candidate in generation.candidates
        }
        measured_paths = {
            item.object_id: (
                item.point_cloud.relative_path if item.point_cloud is not None else None
            )
            for item in measured.hypotheses
        }
        evaluations_by_object: dict[str, list[CandidateHeldoutEvaluation]] = {}
        for item in evaluation.evaluations:
            evaluations_by_object.setdefault(item.object_id, []).append(item)
        selected: list[SelectedVisualCompletion] = []
        for eligibility_record in eligibility.records:
            if eligibility_record.status not in {
                CompletionEligibilityStatus.ELIGIBLE_RIGID,
                CompletionEligibilityStatus.ELIGIBLE_STATIC,
            }:
                selected.append(
                    SelectedVisualCompletion(
                        object_id=eligibility_record.object_id,
                        status="deferred_object_type",
                        measured_anchor_asset_path=measured_paths.get(eligibility_record.object_id),
                        selection_rationale=[eligibility_record.reason],
                    )
                )
                continue
            generated_evaluations = [
                item
                for item in evaluations_by_object.get(eligibility_record.object_id, [])
                if item.backend.value != "measured_partial_baseline"
            ]
            research_front = pareto_front(
                [
                    item
                    for item in generated_evaluations
                    if item.license_record.research_evaluation_allowed
                ]
            )
            production_front = pareto_front(
                [
                    item
                    for item in generated_evaluations
                    if item.license_record.production_selectable
                ]
            )
            research_id = research_front[0].candidate_id if research_front else None
            production_id = production_front[0].candidate_id if production_front else None
            chosen = (
                research_id
                if license_mode is CompletionLicenseMode.RESEARCH_EVALUATION
                else production_id
            )
            if chosen is None:
                if (
                    research_id is not None
                    and license_mode is CompletionLicenseMode.PRODUCTION_CANDIDATE
                ):
                    status: Literal[
                        "license_blocked",
                        "rejected_inconsistent",
                        "unresolved_no_candidate",
                    ] = "license_blocked"
                    rationale = [
                        "a research candidate passed geometry gates but no candidate is "
                        "production-selectable under the configured policy"
                    ]
                elif generated_evaluations:
                    status = "rejected_inconsistent"
                    rationale = ["no generated candidate passed all preconfigured held-out gates"]
                else:
                    status = "unresolved_no_candidate"
                    rationale = ["no generated completion candidate was available"]
                selected.append(
                    SelectedVisualCompletion(
                        object_id=eligibility_record.object_id,
                        status=status,
                        best_research_candidate=research_id,
                        best_production_eligible_candidate=production_id,
                        measured_anchor_asset_path=measured_paths.get(eligibility_record.object_id),
                        selection_rationale=rationale,
                    )
                )
                continue
            chosen_candidate = candidates[chosen]
            accepted_status: Literal[
                "accepted_visual_completion",
                "ambiguous_multiple_candidates",
            ] = (
                "ambiguous_multiple_candidates"
                if len(
                    research_front
                    if license_mode.value == "research_evaluation"
                    else production_front
                )
                > 1
                else "accepted_visual_completion"
            )
            selected.append(
                SelectedVisualCompletion(
                    object_id=eligibility_record.object_id,
                    status=accepted_status,
                    best_research_candidate=research_id,
                    best_production_eligible_candidate=production_id,
                    selected_candidate=chosen,
                    measured_anchor_asset_path=measured_paths.get(eligibility_record.object_id),
                    selected_native_asset_path=chosen_candidate.native_assets[0].relative_path,
                    selection_rationale=[
                        "passed hard held-out validity gates",
                        "survived Pareto filtering",
                        "won deterministic geometry-first ranking",
                        f"selected under {license_mode.value} license policy",
                    ],
                    geometry_status="complete_visual_candidate",
                )
            )
        selection = CandidateSelectionArtifact(
            evaluation_manifest_sha256=sha256_file(root / "evaluation_manifest.json"),
            license_mode=license_mode,
            ranking_policy="hard_gates_pareto_deterministic_v1",
            objects=selected,
            deterministic_selection_digest=stable_digest(
                {
                    "license_mode": license_mode.value,
                    "objects": [item.model_dump(mode="json") for item in selected],
                }
            ),
        )
        atomic_write_json(root / "selection.json", selection)
        diagnostics = CompletionDiagnostics(
            eligible_object_count=sum(
                item.status
                in {
                    CompletionEligibilityStatus.ELIGIBLE_RIGID,
                    CompletionEligibilityStatus.ELIGIBLE_STATIC,
                }
                for item in eligibility.records
            ),
            deferred_object_count=sum(
                item.status
                not in {
                    CompletionEligibilityStatus.ELIGIBLE_RIGID,
                    CompletionEligibilityStatus.ELIGIBLE_STATIC,
                }
                for item in eligibility.records
            ),
            candidate_count_by_backend={
                generation.backend.value: len(generation.candidates) for generation in generations
            },
            registered_candidate_count=sum(
                item.status != "registration_failed" for item in registration.registrations
            ),
            evaluated_candidate_count=len(evaluation.evaluations),
            passing_candidate_count=sum(item.passed_hard_gates for item in evaluation.evaluations),
            selected_research_count=sum(
                item.best_research_candidate is not None for item in selected
            ),
            selected_production_count=sum(
                item.best_production_eligible_candidate is not None for item in selected
            ),
            total_runtime_seconds=(
                sum(generation.runtime_seconds for generation in generations)
                + registration.runtime_seconds
                + evaluation.runtime_seconds
            ),
            peak_gpu_memory_bytes=max(
                value
                for value in (
                    *(
                        candidate.peak_gpu_memory_bytes
                        for generation in generations
                        for candidate in generation.candidates
                    ),
                    registration.peak_gpu_memory_bytes,
                    evaluation.peak_gpu_memory_bytes,
                    0,
                )
                if value is not None
            ),
            peak_host_memory_bytes=max(
                value
                for value in (
                    registration.peak_host_memory_bytes,
                    evaluation.peak_host_memory_bytes,
                    0,
                )
                if value is not None
            ),
            warnings=["selected assets are visual candidates and have no physical validation"],
        )
        atomic_write_json(root / "diagnostics.json", diagnostics)
        source_scene = SceneIR.model_validate_json(
            context.path("scene_ir", "phase5a_scene.json").read_text(encoding="utf-8")
        )
        atomic_write_json(
            context.path("scene_ir", "phase5b_scene.json"),
            self._integrate_scene(source_scene, selection, candidates),
        )
        self._write_previews(root / "previews", selection, evaluation, context.run_dir)
        return StageResult(
            metrics={
                "selected_research": diagnostics.selected_research_count,
                "selected_production": diagnostics.selected_production_count,
                "passing_candidates": diagnostics.passing_candidate_count,
            }
        )

    def _integrate_scene(
        self,
        scene: SceneIR,
        selection: CandidateSelectionArtifact,
        candidates: Mapping[str, ObjectCompletionCandidate],
    ) -> SceneIR:
        existing = {item.asset_id: item for item in scene.geometry_assets}
        objects = {item.object_id: item for item in scene.objects}
        for item in selection.objects:
            if item.selected_candidate is None or item.selected_native_asset_path is None:
                continue
            candidate = candidates[item.selected_candidate]
            asset_id = f"visual_completion_{item.object_id}"
            provenance = ProvenanceRecord(
                adapter_name=self.name,
                adapter_version=self.version,
                configuration={
                    "candidate_id": item.selected_candidate,
                    "selection_policy": "hard_gates_pareto_deterministic_v1",
                },
                input_artifact_paths=[
                    "reconstruction/completion/evaluation_manifest.json",
                    item.selected_native_asset_path,
                ],
                output_artifact_paths=["scene_ir/phase5b_scene.json"],
                timestamp=datetime.now(UTC),
                confidence=ConfidenceRecord(
                    score=0.5,
                    method="heldout_observation_validation",
                    notes="visual completion confidence; no physical validation",
                ),
                source=GeometrySourceType.GENERATED,
            )
            suffix = Path(item.selected_native_asset_path).suffix.lower()
            existing[asset_id] = GeometryAsset(
                asset_id=asset_id,
                asset_type=AssetType.UNCLASSIFIED,
                uri=item.selected_native_asset_path,
                format="glb" if suffix == ".glb" else "ply",
                source=GeometrySourceType.GENERATED,
                coordinate_convention=scene.metadata.coordinate_convention,
                scale_status=scene.metadata.coordinate_convention.scale_status,
                geometry_status="complete_visual_candidate",
                completion_status="selected_by_observation_validation",
                asset_role="visual_completion_candidate",
                observation_grounded=True,
                physical_validation="not_implemented",
                collision_ready=False,
                usage_policy=selection.license_mode.value,
                license_record_path="reconstruction/completion/selection.json",
                production_selectable=candidate.license_record.production_selectable,
                sim_ready=False,
                provenance=provenance,
            )
            prior = objects.get(item.object_id)
            geometry = list(prior.geometry_asset_ids) if prior is not None else []
            if asset_id not in geometry:
                geometry.append(asset_id)
            objects[item.object_id] = ObjectInstance(
                object_id=item.object_id,
                name=prior.name if prior is not None else item.object_id,
                asset_type=prior.asset_type if prior is not None else AssetType.UNCLASSIFIED,
                geometry_asset_ids=geometry,
                geometry_status="complete_visual_candidate",
                completion_status="selected_by_observation_validation",
                observation_grounded=True,
                physical_validation="not_implemented",
                sim_ready=False,
                confidence=prior.confidence if prior is not None else provenance.confidence,
                provenance=[
                    *(prior.provenance if prior is not None else []),
                    provenance,
                ],
            )
        scene.geometry_assets = sorted(existing.values(), key=lambda value: value.asset_id)
        scene.objects = sorted(objects.values(), key=lambda value: value.object_id)
        scene.schema_version = "0.1.5"
        return SceneIR.model_validate(scene.model_dump(mode="json"))

    @staticmethod
    def _write_previews(
        preview_root: Path,
        selection: CandidateSelectionArtifact,
        evaluation: CandidateEvaluationManifest,
        artifact_root: Path | None = None,
    ) -> None:
        preview_root.mkdir(parents=True, exist_ok=True)
        run_root = artifact_root or preview_root.parents[2]
        evaluations_by_object: dict[str, list[CandidateHeldoutEvaluation]] = {}
        all_entries: list[tuple[str, Path]] = []
        heldout_entries: list[tuple[str, Path]] = []
        for item in evaluation.evaluations:
            evaluations_by_object.setdefault(item.object_id, []).append(item)
            render_items = sorted(item.render_paths.items())
            if render_items:
                frame_id, path = render_items[0]
                all_entries.append(
                    (
                        f"{item.candidate_id} | {frame_id} | IoU {item.metrics.mask_iou:.3f}",
                        run_root / path,
                    )
                )
            heldout_entries.extend(
                (
                    f"{item.candidate_id} | {frame_id} | "
                    f"{'PASS' if item.passed_hard_gates else 'REJECT'}",
                    run_root / path,
                )
                for frame_id, path in render_items
            )
        comparison_entries: list[tuple[str, Path]] = []
        selected_entries: list[tuple[str, Path]] = []
        selection_by_object = {item.object_id: item for item in selection.objects}
        for object_id, object_evaluations in sorted(evaluations_by_object.items()):
            baseline = next(
                (
                    item
                    for item in object_evaluations
                    if item.backend.value == "measured_partial_baseline"
                ),
                None,
            )
            generated = sorted(
                (
                    item
                    for item in object_evaluations
                    if item.backend.value != "measured_partial_baseline"
                ),
                key=lambda item: (-item.metrics.mask_iou, item.candidate_id),
            )
            for item in ([baseline] if baseline is not None else []) + generated[:1]:
                if item is None or not item.render_paths:
                    continue
                frame_id, path = sorted(item.render_paths.items())[0]
                comparison_entries.append(
                    (
                        f"{object_id} | {item.backend.value} | IoU {item.metrics.mask_iou:.3f}",
                        run_root / path,
                    )
                )
            selected_id = selection_by_object[object_id].selected_candidate
            shown = next(
                (item for item in object_evaluations if item.candidate_id == selected_id),
                generated[0] if generated else baseline,
            )
            if shown is not None and shown.render_paths:
                frame_id, path = sorted(shown.render_paths.items())[0]
                selected_entries.append(
                    (
                        f"{object_id} | {selection_by_object[object_id].status} | {frame_id}",
                        run_root / path,
                    )
                )
        CompletionSelectionAdapter._write_contact_sheet(
            preview_root / "candidate_grid.png",
            "Phase 5B candidate grid",
            all_entries,
        )
        CompletionSelectionAdapter._write_contact_sheet(
            preview_root / "registration_contact_sheet.png",
            "Registered candidates in held-out cameras",
            all_entries,
        )
        CompletionSelectionAdapter._write_contact_sheet(
            preview_root / "heldout_evaluation_contact_sheet.png",
            "Held-out RGB, masks, visibility, and errors",
            heldout_entries,
        )
        CompletionSelectionAdapter._write_contact_sheet(
            preview_root / "measured_vs_candidates.png",
            "Measured baseline versus generated candidates",
            comparison_entries,
        )
        CompletionSelectionAdapter._write_contact_sheet(
            preview_root / "selected_object_contact_sheet.png",
            "Selected candidates or strongest rejected diagnostics",
            selected_entries,
        )

    @staticmethod
    def _write_contact_sheet(
        path: Path,
        title: str,
        entries: list[tuple[str, Path]],
    ) -> None:
        cell_width, cell_height = 560, 380
        columns = 2
        rows = max(1, (len(entries) + columns - 1) // columns)
        image = Image.new(
            "RGB",
            (columns * cell_width, 52 + rows * cell_height + 30),
            (245, 247, 249),
        )
        draw = ImageDraw.Draw(image)
        draw.text((20, 18), title, fill=(20, 30, 45))
        for index, (label, source) in enumerate(entries):
            x = (index % columns) * cell_width
            y = 52 + (index // columns) * cell_height
            draw.text((x + 12, y + 8), label, fill=(35, 48, 62))
            if source.is_file():
                with Image.open(source) as opened:
                    preview = ImageOps.contain(opened.convert("RGB"), (536, 330))
                image.paste(preview, (x + 12, y + 36))
            else:
                draw.text((x + 12, y + 46), f"missing: {source.name}", fill=(180, 45, 45))
        if not entries:
            draw.text((20, 80), "No candidate renders available.", fill=(120, 55, 45))
        draw.text(
            (20, image.height - 22),
            "Diagnostic visualization only; no physical validation.",
            fill=(150, 45, 35),
        )
        image.save(path, format="PNG", compress_level=9)


class Phase5BConsistencyValidationAdapter:
    name = "phase5b_consistency_validation"
    version = "0.1.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        specs = [
            InputSpec("inputs/manifest.json", "ingest_manifest"),
            InputSpec("camera/reconstruction.json", "camera_reconstruction"),
            InputSpec("observations/object_tracks.json", "segmentation_tracking"),
            InputSpec("reconstruction/dense/depth_manifest.json", "dense_depth_manifest"),
            InputSpec(
                "reconstruction/measured_objects/geometry_manifest.json",
                "measured_object_geometry",
            ),
            InputSpec("reconstruction/completion/eligibility.json", "completion_eligibility"),
            InputSpec(
                "reconstruction/completion/evidence_split.json",
                "completion_evidence_split",
            ),
            InputSpec(
                "reconstruction/completion/crop_manifest.json",
                "completion_crop_manifest",
            ),
            InputSpec(
                "reconstruction/completion/evidence/evidence_package.json",
                "completion_evidence_package",
            ),
            InputSpec(
                "reconstruction/completion/registration_request.json",
                "candidate_registration_request",
            ),
            InputSpec(
                "reconstruction/completion/registration_manifest.json",
                "candidate_registration_manifest",
            ),
            InputSpec(
                "reconstruction/completion/evaluation_manifest.json",
                "candidate_evaluation_manifest",
            ),
            InputSpec("reconstruction/completion/selection.json", "candidate_selection"),
            InputSpec("scene_ir/phase5b_scene.json", "scene_ir"),
        ]
        crop = CompletionCropManifest.model_validate_json(
            context.canonical_path("reconstruction", "completion", "crop_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        specs.extend(
            spec
            for anchor in crop.anchors
            for spec in (
                InputSpec(anchor.crop_path, "completion_evidence_file"),
                InputSpec(anchor.crop_metadata_path, "completion_evidence_file"),
            )
        )
        for path in GENERATION_MANIFESTS.values():
            specs.append(InputSpec(path, "candidate_generation_manifest"))
            generation = CandidateGenerationManifest.model_validate_json(
                context.canonical_path(*Path(path).parts).read_text(encoding="utf-8")
            )
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
                for candidate in generation.candidates
                for asset in candidate.native_assets
            )
        return [replace(spec, include_producer_signature=False) for spec in specs]

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return HealthcheckResult(True, "Phase 5B consistency validation available")

    def prepare(self, context: StageContext) -> None:
        context.path("validation").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                "validation/phase5b_rigid_completion.json",
                "phase5b_consistency_report",
                "application/json",
                "phase5b_validation",
                validation="json",
                model=Phase5BConsistencyReport,
            )
        ]

    def run(self, context: StageContext) -> StageResult:
        root = context.path("reconstruction", "completion")
        manifest_path = context.path("inputs", "manifest.json")
        camera_path = context.path("camera", "reconstruction.json")
        tracks_path = context.path("observations", "object_tracks.json")
        depth_path = context.path("reconstruction", "dense", "depth_manifest.json")
        measured_path = context.path("reconstruction", "measured_objects", "geometry_manifest.json")
        manifest = IngestManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        camera = CameraReconstruction.model_validate_json(camera_path.read_text(encoding="utf-8"))
        tracks = SegmentationTrackingArtifact.model_validate_json(
            tracks_path.read_text(encoding="utf-8")
        )
        DenseDepthManifest.model_validate_json(depth_path.read_text(encoding="utf-8"))
        measured = MeasuredObjectGeometryArtifact.model_validate_json(
            measured_path.read_text(encoding="utf-8")
        )
        eligibility = CompletionEligibilityArtifact.model_validate_json(
            (root / "eligibility.json").read_text(encoding="utf-8")
        )
        split = CompletionEvidenceSplit.model_validate_json(
            (root / "evidence_split.json").read_text(encoding="utf-8")
        )
        crop = CompletionCropManifest.model_validate_json(
            (root / "crop_manifest.json").read_text(encoding="utf-8")
        )
        evidence = CompletionEvidencePackage.model_validate_json(
            (root / "evidence" / "evidence_package.json").read_text(encoding="utf-8")
        )
        evaluation = CandidateEvaluationManifest.model_validate_json(
            (root / "evaluation_manifest.json").read_text(encoding="utf-8")
        )
        registration = CandidateRegistrationManifest.model_validate_json(
            (root / "registration_manifest.json").read_text(encoding="utf-8")
        )
        registration_request = CandidateRegistrationRequest.model_validate_json(
            (root / "registration_request.json").read_text(encoding="utf-8")
        )
        selection = CandidateSelectionArtifact.model_validate_json(
            (root / "selection.json").read_text(encoding="utf-8")
        )
        scene = SceneIR.model_validate_json(
            context.path("scene_ir", "phase5b_scene.json").read_text(encoding="utf-8")
        )
        generations = [
            CandidateGenerationManifest.model_validate_json(
                context.path(*Path(path).parts).read_text(encoding="utf-8")
            )
            for path in GENERATION_MANIFESTS.values()
        ]
        candidates = {
            candidate.candidate_id: candidate
            for generation in generations
            for candidate in generation.candidates
        }
        registrations = {item.candidate_id: item for item in registration.registrations}
        checks: list[EndToEndConsistencyCheck] = []

        def check(identifier: str, passed: bool, message: str) -> None:
            checks.append(
                EndToEndConsistencyCheck(
                    check_id=identifier,
                    passed=passed,
                    message=message,
                )
            )

        check(
            "shared_lineage",
            manifest.frame_sequence_digest is not None
            and len(
                {
                    manifest.frame_sequence_digest,
                    camera.frame_sequence_digest,
                    tracks.frame_sequence_digest,
                    measured.frame_sequence_digest,
                    eligibility.frame_sequence_digest,
                    split.frame_sequence_digest,
                    evidence.frame_sequence_digest,
                }
            )
            == 1
            and eligibility.manifest_sha256 == sha256_file(manifest_path)
            and evidence.manifest_sha256 == sha256_file(manifest_path)
            and evidence.camera_reconstruction_sha256 == sha256_file(camera_path)
            and evidence.segmentation_tracking_sha256 == sha256_file(tracks_path)
            and evidence.dense_depth_manifest_sha256 == sha256_file(depth_path)
            and evidence.measured_geometry_sha256 == sha256_file(measured_path),
            "tracks, cameras, dense maps, and measured geometry share one observation lineage",
        )
        check(
            "disjoint_evidence",
            all(
                not (
                    set(item.generation_anchor_frames) & set(item.registration_fitting_frames)
                    or set(item.generation_anchor_frames) & set(item.heldout_validation_frames)
                    or set(item.registration_fitting_frames) & set(item.heldout_validation_frames)
                )
                for item in split.objects
            ),
            "generation, fitting, and held-out evidence are disjoint",
        )
        check(
            "heldout_evaluation_exact",
            all(
                item.candidate_id in registrations
                and item.heldout_frame_ids == registrations[item.candidate_id].heldout_frame_ids
                and set(item.metrics.per_frame_iou) == set(item.heldout_frame_ids)
                and set(item.render_paths) == set(item.heldout_frame_ids)
                for item in evaluation.evaluations
            ),
            "held-out evaluation uses exactly the declared frozen validation frames",
        )
        check(
            "frozen_transforms",
            evaluation.transforms_frozen_before_heldout_evaluation,
            "candidate transforms were frozen before held-out evaluation",
        )
        check(
            "heldout_not_registered",
            all(
                not set(item.fitting_frame_ids) & set(item.heldout_frame_ids)
                for item in registration.registrations
            )
            and all(
                values.get("frame_ids")
                == next(
                    item.registration_fitting_frames
                    for item in split.objects
                    if item.object_id == object_id
                )
                for object_id, values in registration_request.fitting_inputs.items()
            ),
            "registration request contains only declared fitting frames and no held-out frames",
        )
        check(
            "proper_sim3",
            all(
                item.frozen_transform is None
                or positive_scale_sim3(item.frozen_transform.matrix_world_from_candidate)
                for item in registration.registrations
            ),
            "every successful candidate transform is a finite proper positive-scale Sim(3)",
        )
        check(
            "crop_integrity",
            all(
                sha256_file(context.path(*Path(item.crop_path).parts)) == item.crop_sha256
                and context.path(*Path(item.crop_metadata_path).parts).is_file()
                for item in crop.anchors
            ),
            "anchor crop bytes and metadata match the immutable crop manifest",
        )
        expected_identities = {
            "sam3d_objects": (
                "https://github.com/facebookresearch/sam-3d-objects",
                "f91db411c50efee93d8db7aeb323885650f6f722",
                "facebook/sam-3d-objects",
                "05929e2a63f234014031f9941f4aabefea5f382e",
            ),
            "trellis2": (
                "https://github.com/microsoft/TRELLIS.2",
                "75fbf0183001ed9876c8dbb35de6b68552ee08bd",
                "microsoft/TRELLIS.2-4B",
                "af44b45f2e35a493886929c6d786e563ec68364d",
            ),
        }
        check(
            "official_backend_pins",
            all(
                generation.backend.value == "measured_partial_baseline"
                or (
                    generation.official_repository,
                    generation.official_code_commit,
                    generation.checkpoint_repository,
                    generation.checkpoint_revision,
                )
                == expected_identities[generation.backend.value]
                for generation in generations
            ),
            "official candidate code and checkpoint identities match exact reviewed pins",
        )
        check(
            "native_candidate_hashes",
            all(
                (path := context.path(*Path(asset.relative_path).parts)).is_file()
                and path.stat().st_size == asset.size_bytes
                and sha256_file(path) == asset.sha256
                for candidate in candidates.values()
                for asset in candidate.native_assets
            ),
            "every native candidate asset size and SHA-256 matches its manifest",
        )
        check(
            "evaluation_registration",
            all(item.candidate_id in registrations for item in evaluation.evaluations),
            "every evaluated candidate has a registration record",
        )
        check(
            "acceptance_gates",
            all(
                self._expected_failed_gates(item, evaluation.evaluation_configuration)
                == set(item.failed_gates)
                for item in evaluation.evaluations
            ),
            "candidate pass/fail results exactly match the preconfigured held-out gates",
        )
        check(
            "license_policy",
            all(
                item.best_production_eligible_candidate is None
                or next(
                    evaluation_item
                    for evaluation_item in evaluation.evaluations
                    if evaluation_item.candidate_id == item.best_production_eligible_candidate
                ).license_record.production_selectable
                for item in selection.objects
            ),
            "production selections satisfy typed license policy",
        )
        check(
            "selection_determinism",
            selection.deterministic_selection_digest
            == stable_digest(
                {
                    "license_mode": selection.license_mode.value,
                    "objects": [item.model_dump(mode="json") for item in selection.objects],
                }
            ),
            "selection digest matches deterministic geometry-first ranking output",
        )
        selected_assets = {
            asset.uri: asset
            for asset in scene.geometry_assets
            if asset.geometry_status == "complete_visual_candidate"
        }
        check(
            "scene_selection",
            all(
                item.selected_native_asset_path is None
                or (
                    item.selected_candidate in candidates
                    and item.selected_native_asset_path in selected_assets
                    and selected_assets[item.selected_native_asset_path].observation_grounded
                    and selected_assets[item.selected_native_asset_path].sim_ready is False
                )
                for item in selection.objects
            ),
            "Scene IR visual candidates exactly match observation-validated selections",
        )
        check(
            "measured_assets_retained",
            any(
                asset.asset_role == "measured_anchor" or asset.geometry_status == "partial_measured"
                for asset in scene.geometry_assets
            ),
            "Scene IR retains measured partial geometry",
        )
        check(
            "eligibility_policy",
            all(
                item.status
                not in {
                    CompletionEligibilityStatus.ELIGIBLE_RIGID,
                    CompletionEligibilityStatus.ELIGIBLE_STATIC,
                }
                or item.asset_type_hint
                in {
                    AssetType.RIGID,
                    AssetType.STATIC_STRUCTURE,
                    AssetType.UNCLASSIFIED,
                    None,
                }
                or item.explicitly_overridden
                for item in eligibility.records
            ),
            "only rigid/static/unclassified or explicitly overridden objects are eligible",
        )
        check(
            "no_collision",
            not scene.collision_assets,
            "Phase 5B produces no collision assets",
        )
        check(
            "no_articulation",
            all(item.articulation is None for item in scene.objects),
            "Phase 5B produces no articulated reconstruction",
        )
        check(
            "non_sim_ready",
            all(asset.sim_ready is not True for asset in scene.geometry_assets),
            "selected visual assets remain non-simulation-ready",
        )
        check(
            "raw_coordinate_semantics",
            scene.metadata.coordinate_convention.world_frame.value == "colmap_arbitrary"
            and scene.metadata.coordinate_convention.alignment_status.value == "unoriented"
            and scene.metadata.coordinate_convention.linear_units.value == "arbitrary_units"
            and scene.metadata.coordinate_convention.scale_status.value == "scale_ambiguous",
            "Phase 5B preserves arbitrary, unoriented, scale-ambiguous COLMAP coordinates",
        )
        check(
            "selective_materialization",
            not any(
                context.path(*Path(path).parts).exists()
                for path in (
                    "camera/colmap/database.db",
                    "observations/raw",
                    "reconstruction/global/raw",
                )
            ),
            "validator workspace excludes raw model workspaces",
        )
        report = Phase5BConsistencyReport(
            passed=all(item.passed for item in checks),
            checks=checks,
            generated_hidden_geometry_used=any(
                item.selected_candidate is not None for item in selection.objects
            ),
            warnings=["visual completion is observation-grounded but not physically validated"],
        )
        atomic_write_json(
            context.path("validation", "phase5b_rigid_completion.json"),
            report,
        )
        return StageResult(
            metrics={
                "passed": report.passed,
                "checks": len(report.checks),
            }
        )

    @staticmethod
    def _expected_failed_gates(
        item: CandidateHeldoutEvaluation,
        config: dict[str, object],
    ) -> set[str]:
        minimum_validation_views = _configured_int(config, "minimum_validation_views")
        if item.backend.value == "measured_partial_baseline":
            return (
                {"minimum_validation_views"}
                if len(item.heldout_frame_ids) < minimum_validation_views
                else set()
            )
        metrics = item.metrics
        gains = item.completion_gain
        checks = {
            "minimum_validation_views": len(item.heldout_frame_ids) >= minimum_validation_views,
            "minimum_mask_iou": metrics.mask_iou >= _configured_float(config, "minimum_mask_iou"),
            "minimum_mask_precision": metrics.mask_precision
            >= _configured_float(config, "minimum_mask_precision"),
            "maximum_median_relative_depth_residual": (
                metrics.dense_depth_relative_residual
                <= _configured_float(config, "maximum_median_relative_depth_residual")
            ),
            "minimum_depth_inlier_fraction": metrics.depth_inlier_fraction
            >= _configured_float(config, "minimum_depth_inlier_fraction"),
            "maximum_negative_space_violation_ratio": (
                metrics.negative_space_violation_ratio
                <= _configured_float(config, "maximum_negative_space_violation_ratio")
            ),
            "maximum_front_of_scene_violation_ratio": (
                metrics.front_of_scene_violation_ratio
                <= _configured_float(config, "maximum_front_of_scene_violation_ratio")
            ),
            "minimum_recall_gain_over_measured_baseline": (
                gains.recall_gain_vs_measured_baseline
                >= _configured_float(config, "minimum_recall_gain_over_measured_baseline")
            ),
            "maximum_precision_drop_from_measured_baseline": (
                gains.precision_change_vs_measured_baseline
                >= -_configured_float(config, "maximum_precision_drop_from_measured_baseline")
            ),
        }
        return {name for name, passed in checks.items() if not passed}


def _configured_float(config: dict[str, object], key: str) -> float:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _configured_int(config: dict[str, object], key: str) -> int:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value
