from __future__ import annotations

from collections import defaultdict
from typing import Literal

from recon2sim.adapters.base import (
    HealthcheckResult,
    InputSpec,
    OutputSpec,
    StageContext,
    StageResult,
)
from recon2sim.artifacts import (
    BundleObjectAssemblyDecision,
    EndToEndConsistencyCheck,
    ObjectAssemblyDecision,
    ObjectAssemblyDecisionSet,
    ObjectAssemblyDecisionStatus,
    Phase6BConsistencyReport,
    PlannedAssemblyAsset,
    SceneAssemblyArtifactReference,
    SceneAssemblyAssetRecord,
    SceneAssemblyAssetRole,
    SceneAssemblyBundle,
    SceneAssemblyBundleKind,
    SceneAssemblyCompilerManifest,
    SceneAssemblyInputManifest,
    SceneAssemblyLayer,
    SceneAssemblyLicenseSummary,
    SceneAssemblyLineageReport,
    SceneAssemblyObjectInput,
    SceneAssemblyOverlapDiagnostic,
    SceneAssemblyOverlapReport,
    SceneAssemblyPlan,
    SceneAssemblyPreviewManifest,
    SceneAssemblySourceReference,
)
from recon2sim.assembly import (
    IDENTITY_MATRIX4,
    bounds_center_distance,
    bounds_overlap_ratio,
    connected_lineages,
    multiply_matrix4,
    resolve_world,
    stable_digest,
    transformed_bounds,
    validate_proper_sim3,
)
from recon2sim.calibration import sha256_file
from recon2sim.ir import (
    AlignmentStatus,
    LinearUnits,
    ScaleStatus,
    SceneAssemblySceneReference,
    SceneIR,
    WorldFrame,
)
from recon2sim.storage import atomic_write_json


def _load_model(context: StageContext, path: str, model: type[SceneIR]) -> SceneIR:
    return model.model_validate_json(context.path(*path.split("/")).read_text(encoding="utf-8"))


def _artifact_reference(context: StageContext, path: str) -> SceneAssemblyArtifactReference:
    return SceneAssemblyArtifactReference(
        path=path,
        sha256=sha256_file(context.path(*path.split("/"))),
    )


def _is_identity(matrix: tuple[float, ...], *, tolerance: float = 1e-9) -> bool:
    return (
        max(abs(value - expected) for value, expected in zip(matrix, IDENTITY_MATRIX4, strict=True))
        <= tolerance
    )


def _lineage_transform(
    manifest: SceneAssemblyInputManifest,
    lineage_id: str,
) -> tuple[float, ...]:
    records = {item.lineage_id: item for item in manifest.lineages}

    def visit(current: str, active: set[str]) -> tuple[float, ...]:
        if current == manifest.primary_lineage_id:
            return IDENTITY_MATRIX4
        if current in active:
            raise ValueError(f"lineage connection cycle reaches {current!r}")
        record = records[current]
        if (
            record.connected_to_lineage_id is None
            or record.accepted_alignment is None
            or record.transform_connected_from_lineage is None
        ):
            raise ValueError(f"lineage {current!r} lacks an accepted typed alignment")
        parent_from_current = record.transform_connected_from_lineage
        primary_from_parent = visit(
            record.connected_to_lineage_id,
            {*active, current},
        )
        return multiply_matrix4(primary_from_parent, parent_from_current)

    return visit(lineage_id, set())


def _candidate_assets(
    item: SceneAssemblyObjectInput,
    assets: dict[str, SceneAssemblyAssetRecord],
    candidate_id: str | None,
) -> list[SceneAssemblyAssetRecord]:
    if candidate_id is None:
        return []
    return sorted(
        (
            assets[asset_id]
            for asset_id in item.candidate_asset_ids
            if assets[asset_id].candidate_id == candidate_id
        ),
        key=lambda asset: asset.asset_id,
    )


def _object_decision(
    item: SceneAssemblyObjectInput,
    assets: dict[str, SceneAssemblyAssetRecord],
) -> ObjectAssemblyDecisionSet:
    measured_ids = sorted(item.measured_anchor_asset_ids)

    def articulated_source(
        values: list[SceneAssemblyAssetRecord],
    ) -> SceneAssemblySourceReference | None:
        references = [asset.kinematic_bundle for asset in values if asset.kinematic_bundle]
        identities = {(reference.path, reference.sha256) for reference in references}
        if len(identities) > 1:
            raise ValueError("candidate visual links disagree on their kinematic source")
        return references[0] if references else None

    def unresolved(*, deployment: bool) -> BundleObjectAssemblyDecision:
        named_candidate_id = (
            item.preferred_deployment_candidate_id
            if deployment
            else item.preferred_research_candidate_id
        )
        named_candidates = _candidate_assets(item, assets, named_candidate_id)
        if named_candidates and any(
            not (
                asset.license.production_selectable
                if deployment
                else asset.license.research_evaluation_allowed
            )
            for asset in named_candidates
        ):
            status = ObjectAssemblyDecisionStatus.DEFERRED_LICENSE_BLOCKED
            reason = "candidate license blocks the requested bundle"
        elif item.asset_type.value == "articulated":
            status = ObjectAssemblyDecisionStatus.DEFERRED_ARTICULATED_UNRESOLVED
            reason = "no articulated visual candidate passed frozen held-out validation"
        elif measured_ids:
            status = ObjectAssemblyDecisionStatus.MEASURED_ONLY
            reason = "measured observation anchor retained without a valid visual completion"
        elif item.global_context_asset_ids:
            status = ObjectAssemblyDecisionStatus.GLOBAL_CONTEXT_ONLY
            reason = "object is represented only by immutable global context"
        else:
            status = ObjectAssemblyDecisionStatus.DEFERRED_NO_VALID_CANDIDATE
            reason = "no validated visual candidate or measured anchor is available"
        return BundleObjectAssemblyDecision(
            status=status,
            rationale=[reason, f"upstream status: {item.upstream_status}"],
        )

    if item.ignored:
        ignored = BundleObjectAssemblyDecision(
            status=ObjectAssemblyDecisionStatus.IGNORED,
            rationale=["object is explicitly ignored by the promoted assembly input"],
        )
        return ObjectAssemblyDecisionSet(
            object_id=item.object_id,
            measured_anchor_asset_ids=measured_ids,
            research_decision=ignored,
            deployment_decision=ignored,
            measured_motion=item.measured_motion,
        )

    deployment = _candidate_assets(item, assets, item.preferred_deployment_candidate_id)
    deployment_decision = unresolved(deployment=True)
    if deployment and all(
        asset.selected_upstream
        and asset.observation_validation_passed
        and asset.license.production_selectable
        for asset in deployment
    ):
        deployment_decision = BundleObjectAssemblyDecision(
            status=ObjectAssemblyDecisionStatus.SELECTED_DEPLOYMENT_CANDIDATE,
            selected_candidate_id=item.preferred_deployment_candidate_id,
            selected_visual_asset_ids=[asset.asset_id for asset in deployment],
            articulated_model_source=articulated_source(deployment),
            rationale=[
                "upstream observation validation passed",
                "all selected representations are production-selectable",
            ],
        )

    research = _candidate_assets(item, assets, item.preferred_research_candidate_id)
    research_decision = unresolved(deployment=False)
    if research and all(
        asset.selected_upstream
        and asset.observation_validation_passed
        and asset.license.research_evaluation_allowed
        for asset in research
    ):
        research_decision = BundleObjectAssemblyDecision(
            status=ObjectAssemblyDecisionStatus.SELECTED_RESEARCH_CANDIDATE,
            selected_candidate_id=item.preferred_research_candidate_id,
            selected_visual_asset_ids=[asset.asset_id for asset in research],
            articulated_model_source=articulated_source(research),
            rationale=[
                "upstream observation validation passed",
                "candidate is selected independently for the research visual bundle",
            ],
        )

    # A production candidate is valid in research when the upstream research selector
    # did not nominate a different candidate and research use is permitted.
    if (
        item.preferred_research_candidate_id is None
        and not research
        and deployment
        and all(
            asset.selected_upstream
            and asset.observation_validation_passed
            and asset.license.research_evaluation_allowed
            for asset in deployment
        )
    ):
        research_decision = BundleObjectAssemblyDecision(
            status=ObjectAssemblyDecisionStatus.SELECTED_RESEARCH_CANDIDATE,
            selected_candidate_id=item.preferred_deployment_candidate_id,
            selected_visual_asset_ids=[asset.asset_id for asset in deployment],
            articulated_model_source=articulated_source(deployment),
            rationale=[
                "upstream production candidate also permits research evaluation",
            ],
        )

    return ObjectAssemblyDecisionSet(
        object_id=item.object_id,
        measured_anchor_asset_ids=measured_ids,
        research_decision=research_decision,
        deployment_decision=deployment_decision,
        measured_motion=item.measured_motion,
    )


class SceneAssemblyPlanAdapter:
    name = "scene_assembly_plan"
    version = "0.1.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        return [
            InputSpec(
                "assembly/input_manifest.json",
                "scene_assembly_input_manifest",
                source_artifact_path="assembly/input_manifest.json",
            )
        ]

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return HealthcheckResult(True, "deterministic layered scene planning available")

    def prepare(self, context: StageContext) -> None:
        context.path("assembly").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                "assembly/calibration_policy.json",
                "scene_assembly_calibration_policy",
                "application/json",
                self.name,
                validation="json",
            ),
            OutputSpec(
                "assembly/lineage_report.json",
                "scene_assembly_lineage_report",
                "application/json",
                self.name,
                validation="json",
                model=SceneAssemblyLineageReport,
            ),
            OutputSpec(
                "assembly/asset_manifest.json",
                "scene_assembly_asset_manifest",
                "application/json",
                self.name,
                validation="json",
            ),
            OutputSpec(
                "assembly/assembly_plan.json",
                "scene_assembly_plan",
                "application/json",
                self.name,
                validation="json",
                model=SceneAssemblyPlan,
            ),
        ]

    def run(self, context: StageContext) -> StageResult:
        manifest = SceneAssemblyInputManifest.model_validate_json(
            context.path("assembly/input_manifest.json").read_text(encoding="utf-8")
        )
        connected = connected_lineages(manifest)
        used_lineages = {asset.lineage_id for asset in manifest.assets} | {
            item.lineage_id for item in manifest.objects
        }
        disconnected = sorted(used_lineages - connected)
        if disconnected:
            raise ValueError(
                f"assembly rejects unconnected reconstruction lineages: {disconnected}"
            )
        for lineage in manifest.lineages:
            for reference in (
                lineage.camera_reconstruction,
                lineage.source_scene_ir,
                lineage.accepted_alignment,
            ):
                if reference is None:
                    continue
                source = context.canonical_path(*reference.path.split("/"))
                if not source.is_file() or sha256_file(source) != reference.sha256:
                    raise ValueError(f"lineage artifact identity mismatch: {reference.path}")
        lineage_report = SceneAssemblyLineageReport(
            primary_lineage_id=manifest.primary_lineage_id,
            lineage_ids=sorted(used_lineages),
            accepted_connection_ids=sorted(used_lineages - {manifest.primary_lineage_id}),
            coherent=True,
        )
        atomic_write_json(context.path("assembly/lineage_report.json"), lineage_report)
        world = resolve_world(manifest)
        atomic_write_json(
            context.path("assembly/calibration_policy.json"),
            {
                "schema_version": "0.1.0",
                "policy": manifest.calibration_policy,
                "calibration_status": manifest.calibration_status,
                "resolved_world": world,
            },
        )
        by_id = {asset.asset_id: asset for asset in manifest.assets}
        decisions = [
            _object_decision(item, by_id)
            for item in sorted(
                manifest.objects,
                key=lambda value: value.object_id,
            )
        ]
        research_selected_ids = {
            asset_id
            for decision in decisions
            for asset_id in decision.research_decision.selected_visual_asset_ids
        }
        deployment_selected_ids = {
            asset_id
            for decision in decisions
            for asset_id in decision.deployment_decision.selected_visual_asset_ids
        }
        selected_ids = research_selected_ids | deployment_selected_ids
        measured_assets = {
            asset.asset_id
            for asset in manifest.assets
            if asset.role is SceneAssemblyAssetRole.MEASURED_ANCHOR
        }
        global_assets = {
            asset.asset_id
            for asset in manifest.assets
            if asset.role is SceneAssemblyAssetRole.GLOBAL_CONTEXT
        }
        planned: list[PlannedAssemblyAsset] = []
        for asset in sorted(manifest.assets, key=lambda value: value.asset_id):
            validate_proper_sim3(asset.asset_to_object)
            validate_proper_sim3(asset.object_to_source_world)
            if asset.asset_native_space.value in {"reference_world", "global_context"} and (
                not _is_identity(asset.asset_to_object)
                or not _is_identity(asset.object_to_source_world)
            ):
                raise ValueError(
                    f"{asset.asset_id!r} is world-space evidence but declares an object transform"
                )
            lineage_transform = _lineage_transform(manifest, asset.lineage_id)
            source_from_asset = multiply_matrix4(
                lineage_transform,
                multiply_matrix4(asset.object_to_source_world, asset.asset_to_object),
            )
            assembly_from_asset = multiply_matrix4(
                world.source_world_to_assembly_world,
                source_from_asset,
            )
            validate_proper_sim3(assembly_from_asset)
            included_research = (
                asset.asset_id in measured_assets
                or (asset.asset_id in global_assets and asset.license.research_evaluation_allowed)
                or (
                    asset.asset_id in research_selected_ids
                    and asset.license.research_evaluation_allowed
                )
            )
            included_deployment = (
                asset.asset_id in measured_assets
                or (asset.asset_id in global_assets and asset.license.production_selectable)
                or (
                    asset.asset_id in deployment_selected_ids
                    and asset.license.production_selectable
                )
            )
            reasons: list[str] = []
            if (
                asset.asset_id not in measured_assets
                and asset.asset_id not in global_assets
                and asset.asset_id not in selected_ids
            ):
                reasons.append("asset was not the deterministic selected representation")
            if asset.asset_id in global_assets and not asset.license.production_selectable:
                reasons.append("global-context license blocks deployment selection")
            if asset.asset_id in selected_ids and not asset.license.research_evaluation_allowed:
                reasons.append("license blocks research evaluation")
            if asset.asset_id in selected_ids and not asset.license.production_selectable:
                reasons.append("license blocks deployment selection")
            planned.append(
                PlannedAssemblyAsset(
                    asset=asset,
                    asset_to_assembly_world=assembly_from_asset,
                    included_in_research=included_research,
                    included_in_deployment=included_deployment,
                    exclusion_reasons=reasons,
                )
            )
        layers = [
            SceneAssemblyLayer(
                layer_id=f"{role.value}_layer",
                role=role,
                asset_ids=[item.asset.asset_id for item in planned if item.asset.role is role],
                included_in_research=any(
                    item.included_in_research for item in planned if item.asset.role is role
                ),
                included_in_deployment=any(
                    item.included_in_deployment for item in planned if item.asset.role is role
                ),
            )
            for role in SceneAssemblyAssetRole
            if any(item.asset.role is role for item in planned)
        ]
        atomic_write_json(
            context.path("assembly/asset_manifest.json"),
            {
                "schema_version": "0.1.0",
                "assets": [item.model_dump(mode="json") for item in planned],
            },
        )
        plan_payload = {
            "schema_version": "0.2.0",
            "input_manifest": _artifact_reference(
                context,
                "assembly/input_manifest.json",
            ),
            "world": world,
            "lineage_report": _artifact_reference(context, "assembly/lineage_report.json"),
            "decisions": decisions,
            "assets": planned,
            "layers": layers,
            "global_scene_policy": manifest.global_scene_policy,
        }
        digest = stable_digest(
            {
                key: (
                    value.model_dump(mode="json")
                    if hasattr(value, "model_dump")
                    else [
                        item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                        for item in value
                    ]
                    if isinstance(value, list)
                    else value
                )
                for key, value in plan_payload.items()
            }
        )
        plan = SceneAssemblyPlan.model_validate(
            {**plan_payload, "deterministic_plan_digest": digest}
        )
        atomic_write_json(context.path("assembly/assembly_plan.json"), plan)
        return StageResult(
            metrics={
                "objects": len(decisions),
                "assets": len(planned),
                "world_mode": world.world_mode.value,
                "selected_candidates": len(selected_ids),
            }
        )


def _license_summary(
    assets: list[PlannedAssemblyAsset],
    *,
    deployment: bool,
) -> SceneAssemblyLicenseSummary:
    included = (
        [item for item in assets if item.included_in_deployment]
        if deployment
        else [item for item in assets if item.included_in_research]
    )
    included_ids = {item.asset.asset_id for item in included}
    return SceneAssemblyLicenseSummary(
        included_license_ids=sorted({item.asset.license.license_id for item in included}),
        excluded_asset_reasons={
            item.asset.asset_id: "; ".join(item.exclusion_reasons) or "bundle policy exclusion"
            for item in assets
            if item.asset.asset_id not in included_ids
        },
        research_only_asset_ids=sorted(
            item.asset.asset_id for item in included if not item.asset.license.production_selectable
        ),
        production_asset_ids=sorted(
            item.asset.asset_id for item in included if item.asset.license.production_selectable
        ),
    )


def _bundle(
    context: StageContext,
    plan: SceneAssemblyPlan,
    manifest: SceneAssemblyInputManifest,
    *,
    kind: SceneAssemblyBundleKind,
) -> SceneAssemblyBundle:
    deployment = kind is SceneAssemblyBundleKind.DEPLOYMENT_ELIGIBLE
    selected_assets = (
        [item for item in plan.assets if item.included_in_deployment]
        if deployment
        else [item for item in plan.assets if item.included_in_research]
    )
    selected_ids = {item.asset.asset_id for item in selected_assets}
    layers = [
        layer.model_copy(
            update={
                "asset_ids": [asset_id for asset_id in layer.asset_ids if asset_id in selected_ids]
            }
        )
        for layer in plan.layers
        if any(asset_id in selected_ids for asset_id in layer.asset_ids)
    ]
    decisions = [
        ObjectAssemblyDecision(
            object_id=item.object_id,
            measured_anchor_asset_ids=item.measured_anchor_asset_ids,
            decision=(item.deployment_decision if deployment else item.research_decision),
            measured_motion=item.measured_motion,
        )
        for item in plan.decisions
    ]
    unresolved = [
        item.object_id
        for item in decisions
        if item.decision.status
        in {
            ObjectAssemblyDecisionStatus.DEFERRED_NO_VALID_CANDIDATE,
            ObjectAssemblyDecisionStatus.DEFERRED_LICENSE_BLOCKED,
            ObjectAssemblyDecisionStatus.DEFERRED_ARTICULATED_UNRESOLVED,
        }
    ]
    return SceneAssemblyBundle(
        bundle_id=f"{manifest.assembly_id}_{kind.value}",
        bundle_kind=kind,
        assembly_plan=_artifact_reference(context, "assembly/assembly_plan.json"),
        world=plan.world,
        lineage_id=manifest.primary_lineage_id,
        asset_ids=sorted(selected_ids),
        object_decisions=decisions,
        layers=layers,
        license_summary=_license_summary(plan.assets, deployment=deployment),
        unresolved_object_ids=sorted(unresolved),
    )


def _overlap_report(plan: SceneAssemblyPlan) -> SceneAssemblyOverlapReport:
    by_object: dict[str, list[PlannedAssemblyAsset]] = defaultdict(list)
    for item in plan.assets:
        if item.asset.object_id is not None:
            by_object[item.asset.object_id].append(item)
    global_bounds_items = [
        transformed_bounds(item.asset.bounds_native, item.asset_to_assembly_world)
        for item in plan.assets
        if item.asset.role is SceneAssemblyAssetRole.GLOBAL_CONTEXT
    ]
    valid_global_bounds = [item for item in global_bounds_items if item is not None]
    global_bounds = (
        (
            min(item[0] for item in valid_global_bounds),
            min(item[1] for item in valid_global_bounds),
            min(item[2] for item in valid_global_bounds),
            max(item[3] for item in valid_global_bounds),
            max(item[4] for item in valid_global_bounds),
            max(item[5] for item in valid_global_bounds),
        )
        if valid_global_bounds
        else None
    )
    diagnostics: list[SceneAssemblyOverlapDiagnostic] = []
    units: Literal["meters", "object_relative", "scene_relative"] = (
        "meters" if plan.world.metric_scale_known else "object_relative"
    )

    def union_bounds(
        values: list[tuple[float, float, float, float, float, float]],
    ) -> tuple[float, float, float, float, float, float] | None:
        if not values:
            return None
        return (
            min(value[0] for value in values),
            min(value[1] for value in values),
            min(value[2] for value in values),
            max(value[3] for value in values),
            max(value[4] for value in values),
            max(value[5] for value in values),
        )

    for decision_set in plan.decisions:
        items = by_object[decision_set.object_id]
        anchors = [
            item for item in items if item.asset.role is SceneAssemblyAssetRole.MEASURED_ANCHOR
        ]
        selected_visual_ids = sorted(
            set(decision_set.research_decision.selected_visual_asset_ids)
            | set(decision_set.deployment_decision.selected_visual_asset_ids)
        )
        candidates = [item for item in items if item.asset.asset_id in set(selected_visual_ids)]
        anchor_bounds = [
            value
            for item in anchors
            if (
                value := transformed_bounds(
                    item.asset.bounds_native,
                    item.asset_to_assembly_world,
                )
            )
            is not None
        ]
        candidate_asset_bounds = {
            item.asset.asset_id: value
            for item in candidates
            if (
                value := transformed_bounds(
                    item.asset.bounds_native,
                    item.asset_to_assembly_world,
                )
            )
            is not None
        }
        measured_bounds = union_bounds(anchor_bounds)
        candidate_bounds = union_bounds(list(candidate_asset_bounds.values()))
        overlap = bounds_overlap_ratio(candidate_bounds, measured_bounds)
        global_overlap = bounds_overlap_ratio(candidate_bounds, global_bounds)
        distance = bounds_center_distance(candidate_bounds, measured_bounds)
        if distance is not None and measured_bounds is not None and units == "object_relative":
            diagonal = (
                sum(
                    (measured_bounds[index + 3] - measured_bounds[index]) ** 2 for index in range(3)
                )
                ** 0.5
            )
            distance = distance / diagonal if diagonal > 0 else None
        diagnostics.append(
            SceneAssemblyOverlapDiagnostic(
                object_id=decision_set.object_id,
                candidate_asset_id=selected_visual_ids[0]
                if len(selected_visual_ids) == 1
                else None,
                candidate_asset_ids=selected_visual_ids,
                measured_anchor_asset_ids=[item.asset.asset_id for item in anchors],
                candidate_bounds_assembly=candidate_bounds,
                measured_bounds_assembly=measured_bounds,
                candidate_measured_overlap_ratio=overlap,
                global_context_intersection_ratio=global_overlap,
                potential_duplicate_geometry_ratio=(
                    max(value for value in (overlap, global_overlap) if value is not None)
                    if overlap is not None or global_overlap is not None
                    else None
                ),
                measured_candidate_distance=distance,
                units=units,
                warning=(
                    "layered_no_carve_v1 retains possible duplicate global/object geometry"
                    if candidates
                    else None
                ),
                per_asset_overlap={
                    asset_id: bounds_overlap_ratio(bounds, measured_bounds)
                    for asset_id, bounds in sorted(candidate_asset_bounds.items())
                },
                unresolved_part_asset_ids=[
                    asset_id
                    for asset_id in selected_visual_ids
                    if asset_id not in candidate_asset_bounds
                ],
            )
        )
    return SceneAssemblyOverlapReport(diagnostics=diagnostics)


class LayeredSceneBundleAdapter:
    name = "layered_scene_bundle"
    version = "0.1.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        manifest = SceneAssemblyInputManifest.model_validate_json(
            context.canonical_path("assembly/input_manifest.json").read_text(encoding="utf-8")
        )
        return [
            InputSpec(
                "assembly/input_manifest.json",
                "scene_assembly_input_manifest",
                source_artifact_path="assembly/input_manifest.json",
            ),
            InputSpec(
                "assembly/assembly_plan.json",
                "scene_assembly_plan",
                source_artifact_path="assembly/assembly_plan.json",
            ),
            InputSpec(
                manifest.source_scene_ir.path,
                "scene_assembly_source",
                expected_sha256=manifest.source_scene_ir.sha256,
                include_producer_signature=False,
            ),
        ]

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return HealthcheckResult(True, "visual-only layered bundle assembly available")

    def prepare(self, context: StageContext) -> None:
        context.path("assembly").mkdir(parents=True, exist_ok=True)
        context.path("scene_ir").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                "assembly/research_visual_bundle.json",
                "scene_assembly_bundle",
                "application/json",
                self.name,
                validation="json",
                model=SceneAssemblyBundle,
            ),
            OutputSpec(
                "assembly/deployment_eligible_visual_bundle.json",
                "scene_assembly_bundle",
                "application/json",
                self.name,
                validation="json",
                model=SceneAssemblyBundle,
            ),
            OutputSpec(
                "assembly/compiler_input_manifest.json",
                "scene_assembly_compiler_manifest",
                "application/json",
                self.name,
                validation="json",
                model=SceneAssemblyCompilerManifest,
            ),
            OutputSpec(
                "assembly/diagnostics.json",
                "scene_assembly_diagnostics",
                "application/json",
                self.name,
                validation="json",
            ),
            OutputSpec(
                "assembly/overlap_diagnostics.json",
                "scene_assembly_overlap_diagnostics",
                "application/json",
                self.name,
                validation="json",
                model=SceneAssemblyOverlapReport,
            ),
            OutputSpec(
                "scene_ir/phase6b_layered_scene.json",
                "scene_ir",
                "application/json",
                self.name,
                validation="scene_ir",
                model=SceneIR,
            ),
        ]

    def run(self, context: StageContext) -> StageResult:
        manifest = SceneAssemblyInputManifest.model_validate_json(
            context.path("assembly/input_manifest.json").read_text(encoding="utf-8")
        )
        plan = SceneAssemblyPlan.model_validate_json(
            context.path("assembly/assembly_plan.json").read_text(encoding="utf-8")
        )
        research = _bundle(
            context,
            plan,
            manifest,
            kind=SceneAssemblyBundleKind.RESEARCH,
        )
        deployment = _bundle(
            context,
            plan,
            manifest,
            kind=SceneAssemblyBundleKind.DEPLOYMENT_ELIGIBLE,
        )
        atomic_write_json(context.path("assembly/research_visual_bundle.json"), research)
        atomic_write_json(
            context.path("assembly/deployment_eligible_visual_bundle.json"),
            deployment,
        )
        research_articulated = {
            decision.object_id: decision.research_decision.articulated_model_source
            for decision in plan.decisions
            if decision.research_decision.articulated_model_source is not None
        }
        deployment_articulated = {
            decision.object_id: decision.deployment_decision.articulated_model_source
            for decision in plan.decisions
            if decision.deployment_decision.articulated_model_source is not None
        }
        unresolved = sorted(
            set(research.unresolved_object_ids) | set(deployment.unresolved_object_ids)
        )
        compiler = SceneAssemblyCompilerManifest(
            world=plan.world,
            research_bundle=_artifact_reference(
                context,
                "assembly/research_visual_bundle.json",
            ),
            deployment_bundle=_artifact_reference(
                context,
                "assembly/deployment_eligible_visual_bundle.json",
            ),
            assets=[
                item
                for item in plan.assets
                if item.included_in_research or item.included_in_deployment
            ],
            research_object_instances=research.object_decisions,
            deployment_object_instances=deployment.object_decisions,
            research_articulated_hierarchies=research_articulated,
            deployment_articulated_hierarchies=deployment_articulated,
            unresolved_objects=unresolved,
            missing_collision_assets=sorted(item.object_id for item in plan.decisions),
            missing_physical_properties=sorted(item.object_id for item in plan.decisions),
        )
        atomic_write_json(context.path("assembly/compiler_input_manifest.json"), compiler)
        overlap = _overlap_report(plan)
        atomic_write_json(context.path("assembly/overlap_diagnostics.json"), overlap)
        atomic_write_json(
            context.path("assembly/diagnostics.json"),
            {
                "schema_version": "0.1.0",
                "assembly_id": manifest.assembly_id,
                "world_mode": plan.world.world_mode,
                "objects": len(plan.decisions),
                "research_assets": len(research.asset_ids),
                "deployment_assets": len(deployment.asset_ids),
                "unresolved_objects": unresolved,
                "global_scene_policy": plan.global_scene_policy,
                "source_geometry_modified": False,
                "visual_only": True,
                "collision_ready": False,
                "physical_validation": "not_implemented",
                "sim_ready": False,
            },
        )
        source = _load_model(context, manifest.source_scene_ir.path, SceneIR)
        scene = source.model_copy(deep=True)
        scene.schema_version = "0.1.9"
        if plan.world.world_mode.value == "canonical_metric":
            scene.metadata.coordinate_convention.world_frame = (
                WorldFrame.CANONICAL_X_FORWARD_Y_LEFT_Z_UP
            )
            scene.metadata.coordinate_convention.linear_units = LinearUnits.METERS
            scene.metadata.coordinate_convention.alignment_status = AlignmentStatus.CANONICAL
            scene.metadata.coordinate_convention.scale_status = ScaleStatus.METRIC_SCALE_KNOWN
        elif plan.world.world_mode.value == "metric_unoriented":
            scene.metadata.coordinate_convention.linear_units = LinearUnits.METERS
            scene.metadata.coordinate_convention.alignment_status = AlignmentStatus.UNORIENTED
            scene.metadata.coordinate_convention.scale_status = ScaleStatus.METRIC_SCALE_KNOWN
        elif plan.world.world_mode.value == "gravity_aligned_arbitrary_scale":
            scene.metadata.coordinate_convention.linear_units = LinearUnits.ARBITRARY_UNITS
            scene.metadata.coordinate_convention.alignment_status = AlignmentStatus.GRAVITY_ALIGNED
            scene.metadata.coordinate_convention.scale_status = ScaleStatus.SCALE_AMBIGUOUS
        scene.metadata.scene_assembly = SceneAssemblySceneReference(
            assembly_plan_path="assembly/assembly_plan.json",
            assembly_plan_sha256=sha256_file(context.path("assembly/assembly_plan.json")),
            research_bundle_path="assembly/research_visual_bundle.json",
            research_bundle_sha256=sha256_file(
                context.path("assembly/research_visual_bundle.json")
            ),
            deployment_bundle_path="assembly/deployment_eligible_visual_bundle.json",
            deployment_bundle_sha256=sha256_file(
                context.path("assembly/deployment_eligible_visual_bundle.json")
            ),
            compiler_manifest_path="assembly/compiler_input_manifest.json",
            compiler_manifest_sha256=sha256_file(
                context.path("assembly/compiler_input_manifest.json")
            ),
            lineage_id=manifest.primary_lineage_id,
            world_mode=plan.world.world_mode.value,
            calibration_status=(
                manifest.calibration_status.value
                if manifest.calibration_status is not None
                else None
            ),
        )
        atomic_write_json(context.path("scene_ir/phase6b_layered_scene.json"), scene)
        return StageResult(
            metrics={
                "research_assets": len(research.asset_ids),
                "deployment_assets": len(deployment.asset_ids),
                "unresolved_objects": len(unresolved),
                "overlap_records": len(overlap.diagnostics),
            }
        )


def _check(check_id: str, passed: bool, message: str) -> EndToEndConsistencyCheck:
    return EndToEndConsistencyCheck(check_id=check_id, passed=passed, message=message)


class Phase6BConsistencyValidationAdapter:
    name = "phase6b_consistency_validation"
    version = "0.1.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        manifest = SceneAssemblyInputManifest.model_validate_json(
            context.canonical_path("assembly/input_manifest.json").read_text(encoding="utf-8")
        )
        preview_manifest = SceneAssemblyPreviewManifest.model_validate_json(
            context.canonical_path("assembly/preview_manifest.json").read_text(encoding="utf-8")
        )
        specs = [
            InputSpec("assembly/input_manifest.json", "scene_assembly_input_manifest"),
            InputSpec("assembly/lineage_report.json", "scene_assembly_lineage_report"),
            InputSpec("assembly/assembly_plan.json", "scene_assembly_plan"),
            InputSpec("assembly/research_visual_bundle.json", "scene_assembly_bundle"),
            InputSpec(
                "assembly/deployment_eligible_visual_bundle.json",
                "scene_assembly_bundle",
            ),
            InputSpec(
                "assembly/compiler_input_manifest.json",
                "scene_assembly_compiler_manifest",
            ),
            InputSpec(
                "assembly/overlap_diagnostics.json",
                "scene_assembly_overlap_diagnostics",
            ),
            InputSpec("assembly/preview_manifest.json", "scene_assembly_preview_manifest"),
            InputSpec("scene_ir/phase6b_layered_scene.json", "scene_ir"),
        ]
        specs.extend(
            InputSpec(path, "scene_assembly_preview")
            for path in preview_manifest.preview_paths.values()
        )
        specs.extend(
            InputSpec(path, "scene_assembly_preview_glb")
            for path in preview_manifest.preview_asset_paths.values()
        )
        for asset in manifest.assets:
            specs.append(
                InputSpec(
                    asset.asset_path,
                    "scene_assembly_visual_asset",
                    expected_sha256=asset.asset_sha256,
                    include_producer_signature=False,
                )
            )
        references = [
            manifest.source_scene_ir,
            manifest.calibration_artifact,
            manifest.canonical_wrapper,
            *(
                reference
                for lineage in manifest.lineages
                for reference in (
                    lineage.camera_reconstruction,
                    lineage.source_scene_ir,
                    lineage.accepted_alignment,
                )
            ),
            *(
                reference
                for asset in manifest.assets
                for reference in (
                    asset.candidate_selection,
                    asset.candidate_evaluation,
                    asset.candidate_generation,
                    asset.measured_geometry,
                    asset.kinematic_bundle,
                    asset.license_source_record,
                    asset.license.source_record,
                )
            ),
            *(
                reference
                for item in manifest.objects
                for reference in (
                    item.rigid_selection_artifact,
                    item.rigid_evaluation_artifact,
                    item.rigid_registration_artifact,
                    *item.rigid_generation_artifacts,
                    *item.representation_parity_artifacts,
                    item.articulated_selection_artifact,
                    item.articulated_candidate_manifest,
                    item.articulated_evaluation_artifact,
                    item.articulated_fitting_artifact,
                    item.articulated_link_assignment_artifact,
                    item.selected_identity_manifest,
                    item.measured_motion,
                    item.kinematic_bundle,
                )
            ),
        ]
        for reference in references:
            if reference is not None and all(
                item.relative_path != reference.path for item in specs
            ):
                specs.append(
                    InputSpec(
                        reference.path,
                        "scene_assembly_source",
                        expected_sha256=reference.sha256,
                        include_producer_signature=False,
                    )
                )
        return specs

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return HealthcheckResult(True, "Phase 6B consistency validation available")

    def prepare(self, context: StageContext) -> None:
        context.path("validation").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                "validation/phase6b_layered_scene_assembly.json",
                "phase6b_consistency_report",
                "application/json",
                self.name,
                validation="json",
                model=Phase6BConsistencyReport,
            )
        ]

    def run(self, context: StageContext) -> StageResult:
        manifest = SceneAssemblyInputManifest.model_validate_json(
            context.path("assembly/input_manifest.json").read_text(encoding="utf-8")
        )
        lineage = SceneAssemblyLineageReport.model_validate_json(
            context.path("assembly/lineage_report.json").read_text(encoding="utf-8")
        )
        plan = SceneAssemblyPlan.model_validate_json(
            context.path("assembly/assembly_plan.json").read_text(encoding="utf-8")
        )
        research = SceneAssemblyBundle.model_validate_json(
            context.path("assembly/research_visual_bundle.json").read_text(encoding="utf-8")
        )
        deployment = SceneAssemblyBundle.model_validate_json(
            context.path("assembly/deployment_eligible_visual_bundle.json").read_text(
                encoding="utf-8"
            )
        )
        compiler = SceneAssemblyCompilerManifest.model_validate_json(
            context.path("assembly/compiler_input_manifest.json").read_text(encoding="utf-8")
        )
        overlap = SceneAssemblyOverlapReport.model_validate_json(
            context.path("assembly/overlap_diagnostics.json").read_text(encoding="utf-8")
        )
        previews = SceneAssemblyPreviewManifest.model_validate_json(
            context.path("assembly/preview_manifest.json").read_text(encoding="utf-8")
        )
        scene = SceneIR.model_validate_json(
            context.path("scene_ir/phase6b_layered_scene.json").read_text(encoding="utf-8")
        )
        source_hashes = all(
            context.path(*asset.asset_path.split("/")).is_file()
            and sha256_file(context.path(*asset.asset_path.split("/"))) == asset.asset_sha256
            for asset in manifest.assets
        )
        declared_references = [
            manifest.source_scene_ir,
            manifest.calibration_artifact,
            manifest.canonical_wrapper,
            *(
                reference
                for lineage_record in manifest.lineages
                for reference in (
                    lineage_record.camera_reconstruction,
                    lineage_record.source_scene_ir,
                    lineage_record.accepted_alignment,
                )
            ),
            *(
                reference
                for asset in manifest.assets
                for reference in (
                    asset.candidate_selection,
                    asset.candidate_evaluation,
                    asset.candidate_generation,
                    asset.measured_geometry,
                    asset.kinematic_bundle,
                    asset.license_source_record,
                    asset.license.source_record,
                )
            ),
            *(
                reference
                for object_input in manifest.objects
                for reference in (
                    object_input.rigid_selection_artifact,
                    object_input.rigid_evaluation_artifact,
                    object_input.rigid_registration_artifact,
                    *object_input.rigid_generation_artifacts,
                    *object_input.representation_parity_artifacts,
                    object_input.articulated_selection_artifact,
                    object_input.articulated_candidate_manifest,
                    object_input.articulated_evaluation_artifact,
                    object_input.articulated_fitting_artifact,
                    object_input.articulated_link_assignment_artifact,
                    object_input.selected_identity_manifest,
                    object_input.measured_motion,
                    object_input.kinematic_bundle,
                )
            ),
        ]
        source_hashes = source_hashes and all(
            reference is None
            or (
                context.path(*reference.path.split("/")).is_file()
                and sha256_file(context.path(*reference.path.split("/"))) == reference.sha256
            )
            for reference in declared_references
        )
        connected = connected_lineages(manifest)
        used_lineages = {item.lineage_id for item in manifest.assets} | {
            item.lineage_id for item in manifest.objects
        }
        world = resolve_world(manifest)
        world_matches = (
            plan.world == world and research.world == world and deployment.world == world
        )
        measured = {
            item.asset.asset_id
            for item in plan.assets
            if item.asset.role is SceneAssemblyAssetRole.MEASURED_ANCHOR
        }
        measured_retained = measured <= set(research.asset_ids) and measured <= set(
            deployment.asset_ids
        )
        selected_assets = {
            asset_id
            for decision in plan.decisions
            for asset_id in (
                decision.research_decision.selected_visual_asset_ids
                + decision.deployment_decision.selected_visual_asset_ids
            )
        }
        by_id = {item.asset.asset_id: item for item in plan.assets}
        selected_valid = all(
            by_id[asset_id].asset.selected_upstream
            and by_id[asset_id].asset.observation_validation_passed
            for asset_id in selected_assets
        )
        rejected_excluded = all(
            item.asset.asset_id not in research.asset_ids
            and item.asset.asset_id not in deployment.asset_ids
            for item in plan.assets
            if item.asset.role
            in {
                SceneAssemblyAssetRole.VISUAL_COMPLETION,
                SceneAssemblyAssetRole.ARTICULATED_VISUAL,
            }
            and item.asset.asset_id not in selected_assets
        )
        deployment_license_ok = all(
            by_id[asset_id].asset.license.production_selectable
            or by_id[asset_id].asset.role is SceneAssemblyAssetRole.MEASURED_ANCHOR
            for asset_id in deployment.asset_ids
        )
        research_decisions_match = compiler.research_object_instances == research.object_decisions
        deployment_decisions_match = (
            compiler.deployment_object_instances == deployment.object_decisions
        )
        manifest_assets = {item.asset_id: item for item in manifest.assets}
        source_derived_decisions = [
            _object_decision(item, manifest_assets)
            for item in sorted(manifest.objects, key=lambda value: value.object_id)
        ]
        plan_decisions_source_bound = plan.decisions == source_derived_decisions
        expected_research_articulated = {
            item.object_id: item.research_decision.articulated_model_source
            for item in source_derived_decisions
            if item.research_decision.articulated_model_source is not None
        }
        expected_deployment_articulated = {
            item.object_id: item.deployment_decision.articulated_model_source
            for item in source_derived_decisions
            if item.deployment_decision.articulated_model_source is not None
        }
        compiler_articulated_sources_match = (
            compiler.research_articulated_hierarchies == expected_research_articulated
            and compiler.deployment_articulated_hierarchies == expected_deployment_articulated
        )
        objects_by_id = {item.object_id: item for item in manifest.objects}
        object_asset_binding = all(
            asset.object_id is None
            or (
                asset.object_id in objects_by_id
                and asset.lineage_id in connected
                and objects_by_id[asset.object_id].lineage_id in connected
            )
            for asset in manifest.assets
        )
        candidate_source_binding = all(
            (
                asset.candidate_selection is not None
                and asset.candidate_evaluation is not None
                and asset.candidate_generation is not None
                and asset.representation_id is not None
            )
            if asset.role
            in {
                SceneAssemblyAssetRole.VISUAL_COMPLETION,
                SceneAssemblyAssetRole.ARTICULATED_VISUAL,
            }
            else True
            for asset in manifest.assets
        )
        license_source_binding = all(
            asset.license.source_record == asset.license_source_record
            for asset in manifest.assets
            if asset.license_source_record is not None
        )
        overlap_by_object = {item.object_id: item for item in overlap.diagnostics}
        derived_by_object = {item.object_id: item for item in source_derived_decisions}
        overlap_aggregation_complete = all(
            item.object_id in overlap_by_object
            and overlap_by_object[item.object_id].measured_anchor_asset_ids
            == sorted(item.measured_anchor_asset_ids)
            and overlap_by_object[item.object_id].candidate_asset_ids
            == sorted(
                set(derived_by_object[item.object_id].research_decision.selected_visual_asset_ids)
                | set(
                    derived_by_object[item.object_id].deployment_decision.selected_visual_asset_ids
                )
            )
            and (
                set(overlap_by_object[item.object_id].per_asset_overlap)
                | set(overlap_by_object[item.object_id].unresolved_part_asset_ids)
            )
            == set(overlap_by_object[item.object_id].candidate_asset_ids)
            for item in manifest.objects
        )
        transforms_ok = True
        no_double_transform = True
        for item in plan.assets:
            try:
                validate_proper_sim3(item.asset_to_assembly_world)
            except ValueError:
                transforms_ok = False
            if item.asset.asset_native_space.value in {"reference_world", "global_context"}:
                no_double_transform = (
                    no_double_transform
                    and _is_identity(item.asset.asset_to_object)
                    and _is_identity(item.asset.object_to_source_world)
                )
        articulation_ok = all(
            reference is None
            or (
                context.path(*reference.path.split("/")).is_file()
                and sha256_file(context.path(*reference.path.split("/"))) == reference.sha256
            )
            for decision in plan.decisions
            for reference in (
                decision.research_decision.articulated_model_source,
                decision.deployment_decision.articulated_model_source,
            )
        )
        compiler_refs = (
            compiler.research_bundle.sha256
            == sha256_file(context.path("assembly/research_visual_bundle.json"))
            and compiler.deployment_bundle.sha256
            == sha256_file(context.path("assembly/deployment_eligible_visual_bundle.json"))
            and compiler.world == plan.world
            and {item.asset.asset_id for item in compiler.assets}
            == set(research.asset_ids) | set(deployment.asset_ids)
            and all(
                item.asset.asset_id in research.asset_ids
                for item in compiler.assets
                if not item.included_in_deployment
            )
            and research_decisions_match
            and deployment_decisions_match
        )
        assembly_references_ok = (
            plan.input_manifest.sha256 == sha256_file(context.path(plan.input_manifest.path))
            and plan.lineage_report.sha256 == sha256_file(context.path(plan.lineage_report.path))
            and research.assembly_plan.sha256
            == sha256_file(context.path(research.assembly_plan.path))
            and deployment.assembly_plan.sha256
            == sha256_file(context.path(deployment.assembly_plan.path))
        )
        scene_reference = scene.metadata.scene_assembly
        scene_refs_ok = scene_reference is not None and all(
            (
                scene_reference.assembly_plan_sha256
                == sha256_file(context.path(scene_reference.assembly_plan_path)),
                scene_reference.research_bundle_sha256
                == sha256_file(context.path(scene_reference.research_bundle_path)),
                scene_reference.deployment_bundle_sha256
                == sha256_file(context.path(scene_reference.deployment_bundle_path)),
                scene_reference.compiler_manifest_sha256
                == sha256_file(context.path(scene_reference.compiler_manifest_path)),
            )
        )
        no_collisions = not scene.collision_assets and all(
            not item.collision_asset_ids for item in scene.objects
        )
        no_physics = all(
            item.physics.mass_kg is None
            and item.physics.friction is None
            and item.physics.restitution is None
            for item in scene.objects
        )
        preview_paths_ok = all(
            context.path(*path.split("/")).is_file()
            for path in [*previews.preview_paths.values(), *previews.preview_asset_paths.values()]
        )
        checks = [
            _check(
                "coherent_lineage",
                lineage.coherent and used_lineages <= connected,
                "all assets belong to one connected reconstruction lineage",
            ),
            _check(
                "exact_source_hashes",
                source_hashes,
                "all promoted visual assets match exact source hashes",
            ),
            _check(
                "calibration_world_mode",
                world_matches,
                "calibration status and assembly world mode agree",
            ),
            _check(
                "no_false_metric_claim",
                plan.world.metric_scale_known
                == (plan.world.world_mode.value in {"canonical_metric", "metric_unoriented"}),
                "meters are claimed only with accepted metric evidence",
            ),
            _check(
                "no_false_gravity_claim",
                plan.world.gravity_alignment_known
                == (
                    plan.world.world_mode.value
                    in {"canonical_metric", "gravity_aligned_arbitrary_scale"}
                ),
                "gravity alignment is claimed only with accepted evidence",
            ),
            _check(
                "measured_anchors_retained",
                measured_retained,
                "measured anchors remain in research and deployment bundles",
            ),
            _check(
                "selected_candidates_validated",
                selected_valid,
                "selected candidates passed their upstream observation validation",
            ),
            _check(
                "rejected_candidates_excluded",
                rejected_excluded,
                "unselected and rejected candidates are absent from both bundles",
            ),
            _check(
                "deployment_license_policy",
                deployment_license_ok,
                "deployment bundle contains no unapproved candidate assets",
            ),
            _check(
                "source_derived_bundle_decisions",
                plan_decisions_source_bound,
                "research and deployment choices are independently rederived from source inputs",
            ),
            _check(
                "compiler_research_decisions",
                research_decisions_match,
                "compiler research decisions exactly match the research bundle",
            ),
            _check(
                "compiler_deployment_decisions",
                deployment_decisions_match,
                "compiler deployment decisions exactly match the deployment bundle",
            ),
            _check(
                "compiler_articulated_sources",
                compiler_articulated_sources_match,
                "compiler articulated hierarchies use each bundle's exact fitted source",
            ),
            _check(
                "object_asset_source_binding",
                object_asset_binding,
                "object assets match their normalized object and lineage identities",
            ),
            _check(
                "candidate_source_binding",
                candidate_source_binding,
                "candidate validation and representation identities have typed source references",
            ),
            _check(
                "license_source_binding",
                license_source_binding,
                "normalized license policy matches its exact source record",
            ),
            _check(
                "global_mesh_immutable",
                plan.source_geometry_immutable and overlap.source_geometry_modified is False,
                "global context remains immutable under layered_no_carve_v1",
            ),
            _check(
                "no_destructive_carve",
                not plan.destructive_object_removal and not plan.background_hole_filling,
                "assembly performs no face removal or background hole filling",
            ),
            _check(
                "finite_proper_transforms",
                transforms_ok,
                "all asset-to-assembly transforms are proper finite Sim(3)",
            ),
            _check(
                "no_double_world_transform",
                no_double_transform,
                "reference-world evidence receives the world wrapper exactly once",
            ),
            _check(
                "articulated_local_quantities_preserved",
                articulation_ok,
                "typed kinematic bundles are retained without rewriting local joints or q",
            ),
            _check(
                "overlap_diagnostics",
                overlap_aggregation_complete,
                "every object's overlap diagnostic covers all anchors and selected visual parts",
            ),
            _check(
                "compiler_manifest_identity",
                compiler_refs,
                "compiler manifest matches the exact dual bundles and assembled assets",
            ),
            _check(
                "assembly_artifact_references",
                assembly_references_ok,
                "plan and bundles reference exact input, lineage, and plan bytes",
            ),
            _check(
                "scene_ir_identity",
                scene_refs_ok,
                "Phase 6B Scene IR references exact assembly artifacts",
            ),
            _check(
                "preview_outputs",
                preview_paths_ok and previews.diagnostic_only,
                "all preview PNGs and diagnostic GLBs exist",
            ),
            _check(
                "preview_source_immutable",
                not previews.source_geometry_modified,
                "preview generation did not rewrite source geometry",
            ),
            _check(
                "no_collision_assets",
                no_collisions,
                "Phase 6B creates no collision assets",
            ),
            _check(
                "no_physics_properties",
                no_physics,
                "Phase 6B identifies no physical properties",
            ),
            _check(
                "visual_only_bundles",
                research.visual_only
                and deployment.visual_only
                and not research.sim_ready
                and not deployment.sim_ready,
                "both bundles remain visual-only and non-simulation-ready",
            ),
            _check(
                "source_artifacts_immutable",
                manifest.source_artifacts_immutable and source_hashes,
                "all promoted upstream artifacts remain byte-identical",
            ),
        ]
        report = Phase6BConsistencyReport(
            passed=all(item.passed for item in checks),
            checks=checks,
            visual_scene_assembled=bool(research.asset_ids or deployment.asset_ids),
            full_canonical_world_used=plan.world.full_canonical_world_used,
            metric_scale_known=plan.world.metric_scale_known,
            gravity_alignment_known=plan.world.gravity_alignment_known,
            warnings=previews.representation_warnings,
        )
        atomic_write_json(
            context.path("validation/phase6b_layered_scene_assembly.json"),
            report,
        )
        if not report.passed:
            failed = [item.check_id for item in checks if not item.passed]
            raise RuntimeError(f"Phase 6B consistency checks failed: {failed}")
        return StageResult(
            metrics={
                "checks": len(checks),
                "passed": report.passed,
                "visual_scene_assembled": report.visual_scene_assembled,
            }
        )


__all__ = [
    "LayeredSceneBundleAdapter",
    "Phase6BConsistencyValidationAdapter",
    "SceneAssemblyPlanAdapter",
]
