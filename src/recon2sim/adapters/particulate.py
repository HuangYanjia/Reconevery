from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

from pydantic import Field, model_validator

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
    ArticulatedAssetSpace,
    ArticulatedCandidate,
    ArticulatedCandidateManifest,
    ArticulatedJoint,
    ArticulatedLicenseRecord,
    ArticulatedLink,
    ArticulatedPartStateGeometryManifest,
    ArticulatedRetrievalResult,
    ArticulatedSourceFamily,
    ArticulatedState,
    MeasuredPartMotionArtifact,
    ParticulateCandidateRequest,
)
from recon2sim.ir import ConfidenceRecord, GeometrySourceType, ProvenanceRecord
from recon2sim.storage import atomic_write_json

PARTICULATE_REPOSITORY: Final = "https://github.com/RuiningLi/particulate"
PARTICULATE_COMMIT: Final = "dee37a75c449f324d9989993461ee09eaccc1686"
PARTICULATE_CHECKPOINT_REPOSITORY: Final = "rayli/Particulate"
PARTICULATE_CHECKPOINT_REVISION: Final = "096167e661feb92a443535d15916323ec8a01613"
PARTICULATE_MODEL_SHA256 = "ad6f14067dadf85335119199b94e8249401376d5700c9b627c3608594ea99b5c"
PARTFIELD_REPOSITORY = "mikaelaangel/partfield-ckpt"
PARTFIELD_REVISION = "90b9b1e08b6a12fdcb6ee26b4854a26235e1765f"
PARTFIELD_MODEL_SHA256 = "463efc8a3afd3913142aa025e0125c00f16ef452b8de6a132ebe32bbe7877ee4"


class ParticulateAdapterConfig(ArticulationWorkerConfig):
    worker_module: str = "particulate_worker"
    docker_image: str = "reconevery/particulate:phase5c"
    official_repository: str = PARTICULATE_REPOSITORY
    official_code_commit: str = PARTICULATE_COMMIT
    checkpoint_repository: str = PARTICULATE_CHECKPOINT_REPOSITORY
    checkpoint_revision: str = PARTICULATE_CHECKPOINT_REVISION
    checkpoint_hashes: dict[str, str] = Field(default_factory=dict)
    runtime_model_revisions: dict[str, str] = Field(default_factory=dict)
    runtime_model_hashes: dict[str, str] = Field(default_factory=dict)
    official_repository_path: str | None = None
    checkpoint_path: str | None = None
    partfield_checkpoint_path: str | None = None
    source_meshes: dict[str, str] = Field(default_factory=dict)
    maximum_candidates: int = Field(default=4, ge=1, le=12)
    working_axis_hint: Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"] | None = None
    generation_configuration: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def verified_official_identity(self) -> ParticulateAdapterConfig:
        if (
            self.official_repository != PARTICULATE_REPOSITORY
            or self.official_code_commit != PARTICULATE_COMMIT
            or self.checkpoint_repository != PARTICULATE_CHECKPOINT_REPOSITORY
            or self.checkpoint_revision != PARTICULATE_CHECKPOINT_REVISION
        ):
            raise ValueError("Particulate source identity differs from the reviewed official pin")
        if self.execution_mode == "fake_worker":
            return self
        if self.working_axis_hint is None:
            raise ValueError(
                "real Particulate execution requires an explicit working_axis_hint; "
                "Phase 5C does not assume gravity"
            )
        if self.checkpoint_hashes != {"model.pt": PARTICULATE_MODEL_SHA256}:
            raise ValueError("Particulate checkpoint hash does not match the official pin")
        if self.runtime_model_revisions != {PARTFIELD_REPOSITORY: PARTFIELD_REVISION}:
            raise ValueError("PartField checkpoint revision does not match the official pin")
        if self.runtime_model_hashes != {"model_objaverse.ckpt": PARTFIELD_MODEL_SHA256}:
            raise ValueError("PartField checkpoint hash does not match the official pin")
        if not self.source_meshes:
            raise ValueError("real Particulate execution requires at least one source mesh")
        if self.execution_mode == "local_worker" and (
            self.official_repository_path is None
            or self.checkpoint_path is None
            or self.partfield_checkpoint_path is None
        ):
            raise ValueError("local Particulate execution requires repository and checkpoint paths")
        return self


def particulate_license() -> ArticulatedLicenseRecord:
    return ArticulatedLicenseRecord(
        source_family=ArticulatedSourceFamily.PARTICULATE,
        code_license="Apache-2.0 (declared by official model card)",
        checkpoint_license="CC-BY-4.0",
        dependency_licenses={
            "PartField code": "NVIDIA Source Code License",
            "PartField checkpoint": "NVIDIA non-commercial research license",
        },
        asset_license="generated output requires project legal review",
        training_data_notes=[
            "official repository evaluates on PartNet-Mobility and Lightwheel",
            "PartField runtime checkpoint is non-commercial research only",
        ],
        commercial_review_status="research_only",
        research_evaluation_allowed=True,
        production_selectable=False,
    )


def measured_motion_license() -> ArticulatedLicenseRecord:
    return ArticulatedLicenseRecord(
        source_family=ArticulatedSourceFamily.MEASURED_MOTION,
        code_license="Reconevery project license",
        checkpoint_license="not_applicable",
        asset_license="source observation license",
        commercial_review_status="approved_by_project_policy",
        research_evaluation_allowed=True,
        production_selectable=True,
    )


class ParticulateAdapter:
    name = "particulate_candidates"
    version = "0.1.0"

    @staticmethod
    def _retrieved_candidate_specs(context: StageContext) -> list[InputSpec]:
        specs: list[InputSpec] = []
        for filename in ("artvip_retrieval.json", "partnet_retrieval.json"):
            result = ArticulatedRetrievalResult.model_validate_json(
                context.canonical_path("reconstruction", "articulation", filename).read_text(
                    encoding="utf-8"
                )
            )
            for candidate in result.candidates:
                if candidate.candidate_bundle_path is None:
                    continue
                specs.append(
                    InputSpec(
                        candidate.candidate_bundle_path,
                        "articulated_retrieved_candidate_bundle",
                        expected_sha256=candidate.candidate_bundle_sha256,
                        materialization_mode="copy",
                    )
                )
                specs.extend(
                    InputSpec(
                        path,
                        "articulated_candidate_visual_link",
                        expected_sha256=expected_hash,
                        materialization_mode="reflink_or_copy",
                    )
                    for path, expected_hash in sorted(candidate.visual_asset_hashes.items())
                )
        return specs

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        config = ParticulateAdapterConfig.model_validate(context.config.adapter.config)
        specs = [
            InputSpec(
                "reconstruction/articulation/measured_motion.json",
                "measured_part_motion",
            ),
            InputSpec(
                "reconstruction/articulation/measured_states/manifest.json",
                "articulated_part_state_geometry_manifest",
            ),
            InputSpec(
                "reconstruction/articulation/artvip_retrieval.json",
                "articulated_retrieval_result",
            ),
            InputSpec(
                "reconstruction/articulation/partnet_retrieval.json",
                "articulated_retrieval_result",
            ),
        ]
        specs.extend(self._retrieved_candidate_specs(context))
        for source_id, value in sorted(config.source_meshes.items()):
            specs.append(
                InputSpec(
                    f"reconstruction/articulation/source_meshes/{source_id}/mesh.glb",
                    "articulation_source_mesh",
                    source_path=Path(value).expanduser().resolve(),
                    materialization_mode="reflink_or_copy",
                )
            )
        if config.execution_mode == "fake_worker":
            geometry = ArticulatedPartStateGeometryManifest.model_validate_json(
                context.canonical_path(
                    "reconstruction", "articulation", "measured_states", "manifest.json"
                ).read_text(encoding="utf-8")
            )
            specs.append(
                InputSpec(
                    geometry.geometries[-1].measured_point_cloud_path,
                    "measured_articulated_part_point_cloud",
                )
            )
        return specs

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return articulation_healthcheck(
            context,
            ParticulateAdapterConfig,
            worker_name=self.name,
        )

    def prepare(self, context: StageContext) -> None:
        context.path("reconstruction", "articulation", "candidates").mkdir(
            parents=True, exist_ok=True
        )
        context.path("reconstruction", "articulation", "raw", "logs").mkdir(
            parents=True, exist_ok=True
        )

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                "reconstruction/articulation/candidate_manifest.json",
                "articulated_candidate_manifest",
                "application/json",
                self.name,
                validation="json",
                model=ArticulatedCandidateManifest,
            )
        ]

    def run(self, context: StageContext) -> StageResult:
        config = ParticulateAdapterConfig.model_validate(context.config.adapter.config)
        root = context.path("reconstruction", "articulation")
        artvip = ArticulatedRetrievalResult.model_validate_json(
            (root / "artvip_retrieval.json").read_text(encoding="utf-8")
        )
        partnet = ArticulatedRetrievalResult.model_validate_json(
            (root / "partnet_retrieval.json").read_text(encoding="utf-8")
        )
        geometries = ArticulatedPartStateGeometryManifest.model_validate_json(
            (root / "measured_states/manifest.json").read_text(encoding="utf-8")
        )
        measured = MeasuredPartMotionArtifact.model_validate_json(
            (root / "measured_motion.json").read_text(encoding="utf-8")
        )
        direct_candidates: list[ArticulatedCandidate] = []
        for retrieval in (artvip, partnet):
            for retrieved in retrieval.candidates:
                if retrieved.candidate_bundle_path is None:
                    continue
                bundle = context.path(*Path(retrieved.candidate_bundle_path).parts)
                if sha256_file(bundle) != retrieved.candidate_bundle_sha256:
                    raise RuntimeError("retrieved candidate bundle changed before Particulate")
                candidate = ArticulatedCandidate.model_validate_json(
                    bundle.read_text(encoding="utf-8")
                )
                if (
                    candidate.candidate_id != retrieved.candidate_id
                    or candidate.source_family is not retrieved.source_family
                ):
                    raise RuntimeError("retrieved candidate identity mismatch")
                direct_candidates.append(candidate)
        source_items: list[tuple[str, str, str]] = []
        if config.source_meshes:
            source_items.extend(
                (
                    source_id,
                    f"reconstruction/articulation/source_meshes/{source_id}/mesh.glb",
                    "configured_external_visual_mesh",
                )
                for source_id in sorted(config.source_meshes)
            )
        source_items.extend(
            (
                candidate.candidate_id,
                candidate.links[0].visual_asset_paths[0],
                (
                    "retrieved_artvip_static_geometry"
                    if candidate.source_family is ArticulatedSourceFamily.ARTVIP
                    else "retrieved_partnet_static_geometry"
                ),
            )
            for candidate in direct_candidates
            if candidate.links and candidate.links[0].visual_asset_paths
        )
        if not source_items:
            measured_geometry = geometries.geometries[-1]
            source_items.append(
                (
                    "measured_observed_surface",
                    measured_geometry.measured_point_cloud_path,
                    "measured_observed_surface",
                )
            )
        requests: list[ParticulateCandidateRequest] = []
        working_hint = config.working_axis_hint or "+Z"
        for index, (source_id, source_path, representation) in enumerate(
            source_items[: config.maximum_candidates]
        ):
            source = context.path(*Path(source_path).parts)
            requests.append(
                ParticulateCandidateRequest(
                    candidate_id=f"{artvip.articulated_object_id}__particulate__{index:02d}",
                    articulated_object_id=artvip.articulated_object_id,
                    source_mesh_path=source_path,
                    source_mesh_sha256=sha256_file(source),
                    source_backend=source_id,
                    source_representation=representation,
                    source_license=particulate_license(),
                    visual_completeness_status=(
                        "partial_measured"
                        if representation == "measured_observed_surface"
                        else "complete_visual_candidate"
                    ),
                    official_repository=PARTICULATE_REPOSITORY,
                    official_code_commit=PARTICULATE_COMMIT,
                    checkpoint_repository=PARTICULATE_CHECKPOINT_REPOSITORY,
                    checkpoint_revision=PARTICULATE_CHECKPOINT_REVISION,
                    checkpoint_hashes=(
                        config.checkpoint_hashes or {"model.pt": PARTICULATE_MODEL_SHA256}
                    ),
                    runtime_model_revisions=(
                        config.runtime_model_revisions or {PARTFIELD_REPOSITORY: PARTFIELD_REVISION}
                    ),
                    runtime_model_hashes=(
                        config.runtime_model_hashes
                        or {"model_objaverse.ckpt": PARTFIELD_MODEL_SHA256}
                    ),
                    working_frame_hypothesis=working_hint,
                    hypotheses_evaluated=[working_hint],
                    hypothesis_selection_evidence=(
                        "explicit configured up-axis prior; no gravity claim and no "
                        "alternate hypothesis evaluation"
                    ),
                    generation_configuration=config.generation_configuration,
                    output_directory=(
                        "reconstruction/articulation/candidates/"
                        f"{artvip.articulated_object_id}__particulate__{index:02d}"
                    ),
                    seed=context.seed + index,
                )
            )
        request_path = root / "raw" / "particulate_request.json"
        atomic_write_json(
            request_path,
            {
                "schema_version": "0.1.0",
                "requests": [item.model_dump(mode="json") for item in requests],
                "measured_motion_sha256": artvip.measured_motion_sha256,
                "retrieval_manifest_sha256": sha256_file(root / "artvip_retrieval.json"),
                "partnet_retrieval_manifest_sha256": sha256_file(root / "partnet_retrieval.json"),
                "license_policy": particulate_license().model_dump(mode="json"),
                "official_repository_path": config.official_repository_path,
                "checkpoint_path": config.checkpoint_path,
                "partfield_checkpoint_path": config.partfield_checkpoint_path,
                "output_path": "reconstruction/articulation/candidate_manifest.json",
                "fake_mode": config.fake_mode,
            },
        )
        run_articulation_worker(
            context,
            config,
            action="generate",
            request_path=request_path.relative_to(context.run_dir).as_posix(),
            output_directory="reconstruction/articulation",
            log_name="particulate",
        )
        result = ArticulatedCandidateManifest.model_validate_json(
            (root / "candidate_manifest.json").read_text(encoding="utf-8")
        )
        reference_state = min(
            {item.state_id for item in geometries.geometries},
            key=lambda state_id: next(
                index
                for index, item in enumerate(geometries.geometries)
                if item.state_id == state_id
            ),
        )
        reference_geometries = [
            item for item in geometries.geometries if item.state_id == reference_state
        ]
        measured_candidate = ArticulatedCandidate(
            candidate_id=f"{measured.articulated_object_id}__measured_motion__baseline",
            articulated_object_id=measured.articulated_object_id,
            source_family=ArticulatedSourceFamily.MEASURED_MOTION,
            source_asset_id="phase5a_measured_part_geometry",
            links=[
                ArticulatedLink(
                    link_id=item.part_id,
                    name=item.semantic_label,
                    visual_asset_paths=[item.measured_point_cloud_path],
                    visual_asset_hashes={
                        item.measured_point_cloud_path: item.measured_point_cloud_sha256
                    },
                    visual_asset_spaces={
                        item.measured_point_cloud_path: ArticulatedAssetSpace.REFERENCE_WORLD
                    },
                    visual_asset_transforms_candidate_base={
                        item.measured_point_cloud_path: (
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
                    },
                    native_bounds_min=(0.0, 0.0, 0.0),
                    native_bounds_max=(0.0, 0.0, 0.0),
                )
                for item in reference_geometries
            ],
            joints=[
                ArticulatedJoint(
                    joint_id=item.joint_id,
                    parent_link_id=item.parent_part_id,
                    child_link_id=item.child_part_id,
                    joint_type=item.joint_type,
                    axis=item.axis or (1.0, 0.0, 0.0),
                    pivot=item.pivot,
                    candidate_limit_lower=item.observed_position_min,
                    candidate_limit_upper=item.observed_position_max,
                    limit_source="observed_range",
                )
                for item in measured.joint_hypotheses
                if item.joint_type.value in {"fixed", "prismatic", "revolute"}
            ],
            states=[
                ArticulatedState(
                    state_id=state.state_id,
                    joint_positions={
                        joint.joint_id: next(
                            measured_state.position
                            for measured_state in joint.states
                            if measured_state.state_id == state.state_id
                        )
                        for joint in measured.joint_hypotheses
                        if any(
                            measured_state.state_id == state.state_id
                            for measured_state in joint.states
                        )
                    },
                    link_transforms={},
                )
                for state in measured.joint_hypotheses[0].states
            ]
            if measured.joint_hypotheses
            else [],
            native_coordinate_convention="reference state raw COLMAP arbitrary frame",
            native_units="arbitrary_units",
            license_record=measured_motion_license(),
            production_selectable=True,
            provenance=ProvenanceRecord(
                adapter_name=self.name,
                adapter_version=self.version,
                input_artifact_paths=[
                    "reconstruction/articulation/measured_motion.json",
                    "reconstruction/articulation/measured_states/manifest.json",
                ],
                output_artifact_paths=["reconstruction/articulation/candidate_manifest.json"],
                timestamp=datetime.now(UTC),
                confidence=ConfidenceRecord(
                    score=max(
                        (item.confidence for item in measured.joint_hypotheses),
                        default=0.0,
                    ),
                    method="measured_motion_analytic_baseline",
                ),
                source=GeometrySourceType.MEASURED,
            ),
            warnings=[
                "measured analytic candidate contains observed surfaces only",
                "candidate limits equal observed range and are not mechanical limits",
            ],
        )
        result = result.model_copy(
            update={
                "candidates": [
                    measured_candidate,
                    *direct_candidates,
                    *result.candidates,
                ]
            }
        )
        atomic_write_json(root / "candidate_manifest.json", result)
        if any(item.production_selectable for item in result.candidates):
            if any(
                item.production_selectable
                for item in result.candidates
                if item.source_family is ArticulatedSourceFamily.PARTICULATE
            ):
                raise RuntimeError(
                    "Particulate candidate cannot be production-selectable under PartField license"
                )
        outputs: list[OutputSpec] = []
        for candidate in result.candidates:
            if candidate.source_family is not ArticulatedSourceFamily.PARTICULATE:
                continue
            outputs.extend(
                OutputSpec(
                    path,
                    "particulate_native_output",
                    (
                        "model/gltf-binary"
                        if path.lower().endswith(".glb")
                        else "application/octet-stream"
                    ),
                    self.name,
                )
                for path in candidate.native_output_paths
            )
            for link in candidate.links:
                outputs.extend(
                    OutputSpec(
                        path,
                        "articulated_candidate_visual_link",
                        "model/ply",
                        self.name,
                    )
                    for path in link.visual_asset_paths
                )
        return StageResult(
            outputs=outputs,
            metrics={
                "particulate_candidates": len(result.candidates),
                "retrieval_candidates": len(artvip.candidates) + len(partnet.candidates),
            },
        )


__all__ = [
    "PARTFIELD_MODEL_SHA256",
    "PARTFIELD_REVISION",
    "PARTICULATE_CHECKPOINT_REVISION",
    "PARTICULATE_COMMIT",
    "ParticulateAdapter",
    "ParticulateAdapterConfig",
    "measured_motion_license",
    "particulate_license",
]
