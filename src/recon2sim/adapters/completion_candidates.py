from __future__ import annotations

import re
import struct
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

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
    CandidateNativeAsset,
    CandidateNativeFormat,
    CandidateRenderCapability,
    CompletionBackend,
    CompletionCropManifest,
    CompletionEligibilityArtifact,
    CompletionEligibilityStatus,
    CompletionEvidencePackage,
    CompletionEvidenceSplit,
    CompletionLicenseRecord,
    CompletionWorkerManifest,
    MeasuredObjectGeometryArtifact,
    ObjectCompletionCandidate,
    ObjectCompletionCandidateRequest,
)
from recon2sim.completion import candidate_id, sha256_file
from recon2sim.ir import ConfidenceRecord, GeometrySourceType, ProvenanceRecord
from recon2sim.storage import atomic_write_json

SAM3D_REPOSITORY = "https://github.com/facebookresearch/sam-3d-objects"
SAM3D_COMMIT = "f91db411c50efee93d8db7aeb323885650f6f722"
SAM3D_CHECKPOINT_REPOSITORY = "facebook/sam-3d-objects"
SAM3D_CHECKPOINT_REVISION = "05929e2a63f234014031f9941f4aabefea5f382e"
TRELLIS2_REPOSITORY = "https://github.com/microsoft/TRELLIS.2"
TRELLIS2_COMMIT = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
TRELLIS2_CHECKPOINT_REPOSITORY = "microsoft/TRELLIS.2-4B"
TRELLIS2_CHECKPOINT_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"
TRELLIS2_RUNTIME_REVISIONS = {
    "facebook/dinov3-vitl16-pretrain-lvd1689m": ("ea8dc2863c51be0a264bab82070e3e8836b02d51"),
    "microsoft/TRELLIS-image-large": "25e0d31ffbebe4b5a97464dd851910efc3002d96",
}


def validate_native_candidate_asset(path: Path, format_name: str) -> None:
    if format_name in {"mesh_glb", "pbr_glb"}:
        with path.open("rb") as file:
            header = file.read(12)
        if len(header) != 12 or header[:4] != b"glTF":
            raise ValueError(f"candidate GLB has an invalid header: {path}")
        version, declared_size = struct.unpack("<II", header[4:12])
        if version != 2 or declared_size != path.stat().st_size:
            raise ValueError(f"candidate GLB version or declared size is invalid: {path}")
        return
    if format_name in {"mesh_ply", "gaussian_splat_ply"}:
        with path.open("rb") as file:
            header = file.read(64 * 1024)
        end = header.find(b"end_header")
        if not header.startswith((b"ply\n", b"ply\r\n")) or end < 0:
            raise ValueError(f"candidate PLY has an invalid header: {path}")
        match = re.search(rb"(?:^|\n)element vertex ([0-9]+)(?:\r?\n)", header[:end])
        if match is None or int(match.group(1)) <= 0:
            raise ValueError(f"candidate PLY contains no declared vertices: {path}")


def validate_worker_model_identity(
    worker: CompletionWorkerManifest,
    request: ObjectCompletionCandidateRequest,
) -> None:
    if (
        worker.official_repository != request.official_repository
        or worker.official_code_commit != request.official_code_commit
        or worker.checkpoint_repository != request.checkpoint_repository
        or worker.checkpoint_revision != request.checkpoint_revision
        or worker.checkpoint_hashes != request.checkpoint_hashes
        or worker.runtime_model_revisions != request.runtime_model_revisions
        or worker.runtime_model_hashes != request.runtime_model_hashes
    ):
        raise RuntimeError("candidate worker model identity mismatch")


class CandidateBackendConfig(CompletionWorkerConfig):
    backend: Literal["sam3d_objects", "trellis2"]
    official_repository: str
    official_code_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    checkpoint_repository: str
    checkpoint_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    checkpoint_hashes: dict[str, str] = Field(default_factory=dict)
    runtime_model_revisions: dict[str, str] = Field(default_factory=dict)
    runtime_model_hashes: dict[str, dict[str, str]] = Field(default_factory=dict)
    object_ids: list[str] | None = None
    seeds_per_anchor: int = Field(default=1, ge=1, le=8)
    maximum_candidates_per_object: int = Field(default=6, ge=1, le=24)
    generation_configuration: dict[str, object] = Field(default_factory=dict)


class Sam3DObjectsAdapterConfig(CandidateBackendConfig):
    backend: Literal["sam3d_objects"] = "sam3d_objects"
    worker_module: str = "sam3d_objects_worker"
    docker_image: str = "reconevery/sam3d-objects:phase5b"
    official_repository: Literal["https://github.com/facebookresearch/sam-3d-objects"] = (
        "https://github.com/facebookresearch/sam-3d-objects"
    )
    official_code_commit: Literal["f91db411c50efee93d8db7aeb323885650f6f722"] = (
        "f91db411c50efee93d8db7aeb323885650f6f722"
    )
    checkpoint_repository: Literal["facebook/sam-3d-objects"] = "facebook/sam-3d-objects"
    checkpoint_revision: Literal["05929e2a63f234014031f9941f4aabefea5f382e"] = (
        "05929e2a63f234014031f9941f4aabefea5f382e"
    )
    seeds_per_anchor: int = Field(default=1, ge=1, le=8)

    @model_validator(mode="after")
    def official_checkpoint_is_immutable(self) -> Sam3DObjectsAdapterConfig:
        if self.execution_mode != "fake_worker" and not self.checkpoint_hashes:
            raise ValueError("real SAM 3D Objects requires exact checkpoint file hashes")
        return self


class Trellis2ObjectsAdapterConfig(CandidateBackendConfig):
    backend: Literal["trellis2"] = "trellis2"
    worker_module: str = "trellis2_objects_worker"
    docker_image: str = "reconevery/trellis2-objects:phase5b"
    official_repository: Literal["https://github.com/microsoft/TRELLIS.2"] = (
        "https://github.com/microsoft/TRELLIS.2"
    )
    official_code_commit: Literal["75fbf0183001ed9876c8dbb35de6b68552ee08bd"] = (
        "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    checkpoint_repository: Literal["microsoft/TRELLIS.2-4B"] = "microsoft/TRELLIS.2-4B"
    checkpoint_revision: Literal["af44b45f2e35a493886929c6d786e563ec68364d"] = (
        "af44b45f2e35a493886929c6d786e563ec68364d"
    )
    seeds_per_anchor: int = Field(default=2, ge=1, le=8)

    @model_validator(mode="after")
    def official_runtime_models_are_immutable(self) -> Trellis2ObjectsAdapterConfig:
        if self.execution_mode == "fake_worker":
            return self
        if not self.checkpoint_hashes:
            raise ValueError("real TRELLIS.2 requires exact checkpoint file hashes")
        if self.runtime_model_revisions != TRELLIS2_RUNTIME_REVISIONS:
            raise ValueError("real TRELLIS.2 runtime model revisions do not match reviewed pins")
        if set(self.runtime_model_hashes) != set(TRELLIS2_RUNTIME_REVISIONS) or any(
            not hashes for hashes in self.runtime_model_hashes.values()
        ):
            raise ValueError("real TRELLIS.2 requires exact hashes for every runtime model")
        return self


def sam3d_license() -> CompletionLicenseRecord:
    return CompletionLicenseRecord(
        backend=CompletionBackend.SAM3D_OBJECTS,
        code_license="SAM License",
        checkpoint_license="SAM License and gated checkpoint terms",
        dependency_licenses={},
        asset_license="generated asset usage requires project legal review",
        access_conditions=[
            "official gated access must be accepted by the user",
            "authentication is supplied only through the worker environment or mounted cache",
        ],
        commercial_use_review_status="research_only",
        research_evaluation_allowed=True,
        production_selectable=False,
    )


def trellis2_license() -> CompletionLicenseRecord:
    return CompletionLicenseRecord(
        backend=CompletionBackend.TRELLIS2,
        code_license="MIT",
        checkpoint_license="MIT",
        dependency_licenses={
            "O-Voxel": "license inventory required",
            "Kaolin": "Apache-2.0",
            "nvdiffrast": "NVIDIA Source Code License",
            "nvdiffrec": "NVIDIA Source Code License",
        },
        asset_license="generated asset usage requires transitive dependency review",
        access_conditions=["official checkpoint repository only", "offline after prefetch"],
        commercial_use_review_status="not_reviewed",
        research_evaluation_allowed=True,
        production_selectable=False,
    )


class _CandidateGenerationAdapter:
    backend: CompletionBackend
    config_type: type[CandidateBackendConfig]
    manifest_name: str
    license_record: CompletionLicenseRecord
    name: str
    version = "0.1.1"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        crop = CompletionCropManifest.model_validate_json(
            context.canonical_path("reconstruction", "completion", "crop_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        specs = [
            InputSpec("reconstruction/completion/eligibility.json", "completion_eligibility"),
            InputSpec("reconstruction/completion/evidence_split.json", "completion_evidence_split"),
            InputSpec("reconstruction/completion/crop_manifest.json", "completion_crop_manifest"),
        ]
        for anchor in crop.anchors:
            specs.extend(
                [
                    InputSpec(anchor.crop_path, "completion_evidence_file"),
                    InputSpec(anchor.crop_metadata_path, "completion_evidence_file"),
                ]
            )
        return [replace(spec, include_producer_signature=False) for spec in specs]

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return completion_healthcheck(context, self.config_type, worker_name=self.name)

    def prepare(self, context: StageContext) -> None:
        context.path("reconstruction", "completion", "candidates").mkdir(
            parents=True, exist_ok=True
        )
        context.path("reconstruction", "completion", "raw", f"{self.name}_logs").mkdir(
            parents=True, exist_ok=True
        )

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                f"reconstruction/completion/{self.manifest_name}",
                "candidate_generation_manifest",
                "application/json",
                self.name,
                validation="json",
                model=CandidateGenerationManifest,
            )
        ]

    def run(self, context: StageContext) -> StageResult:
        config = self.config_type.model_validate(context.config.adapter.config)
        root = context.path("reconstruction", "completion")
        eligibility = CompletionEligibilityArtifact.model_validate_json(
            (root / "eligibility.json").read_text(encoding="utf-8")
        )
        split = CompletionEvidenceSplit.model_validate_json(
            (root / "evidence_split.json").read_text(encoding="utf-8")
        )
        crop = CompletionCropManifest.model_validate_json(
            (root / "crop_manifest.json").read_text(encoding="utf-8")
        )
        eligible = {
            record.object_id: record
            for record in eligibility.records
            if record.status
            in {
                CompletionEligibilityStatus.ELIGIBLE_RIGID,
                CompletionEligibilityStatus.ELIGIBLE_STATIC,
            }
        }
        requests: list[ObjectCompletionCandidateRequest] = []
        candidates: list[ObjectCompletionCandidate] = []
        failures: list[str] = []
        per_object_count: dict[str, int] = {}
        split_ids = {item.object_id for item in split.objects}
        for anchor in crop.anchors:
            record = eligible.get(anchor.object_id)
            if record is None or anchor.object_id not in split_ids:
                continue
            if config.object_ids is not None and anchor.object_id not in config.object_ids:
                continue
            for offset in range(config.seeds_per_anchor):
                if (
                    per_object_count.get(anchor.object_id, 0)
                    >= config.maximum_candidates_per_object
                ):
                    break
                seed = context.seed + offset
                identifier = candidate_id(
                    anchor.object_id, self.backend.value, anchor.frame_id, seed
                )
                output_dir = f"reconstruction/completion/candidates/{anchor.object_id}/{identifier}"
                request = ObjectCompletionCandidateRequest(
                    run_id=context.canonical_run_dir.name,
                    object_id=anchor.object_id,
                    semantic_label=record.semantic_label,
                    asset_type_hint=record.asset_type_hint,
                    eligibility_status=record.status,
                    backend=self.backend,
                    official_repository=config.official_repository,
                    official_code_commit=config.official_code_commit,
                    checkpoint_repository=config.checkpoint_repository,
                    checkpoint_revision=config.checkpoint_revision,
                    checkpoint_hashes=config.checkpoint_hashes,
                    runtime_model_revisions=config.runtime_model_revisions,
                    runtime_model_hashes=config.runtime_model_hashes,
                    license_policy=self.license_record,
                    anchor_frame_id=anchor.frame_id,
                    anchor_crop_path=anchor.crop_path,
                    anchor_crop_sha256=anchor.crop_sha256,
                    anchor_crop_transform=anchor.crop_to_source_transform,
                    source_frame_sha256=anchor.source_frame_sha256,
                    source_mask_sha256=anchor.source_mask_sha256,
                    generation_seed=seed,
                    generation_configuration={
                        **config.generation_configuration,
                        "candidate_id": identifier,
                        "fake_mode": config.fake_mode,
                    },
                    output_directory=output_dir,
                )
                request_path = root / "candidates" / anchor.object_id / identifier / "request.json"
                atomic_write_json(request_path, request)
                requests.append(request)
                try:
                    run_process(
                        worker_command(
                            context,
                            config,
                            "generate",
                            request_path.relative_to(context.run_dir).as_posix(),
                            output_dir,
                        ),
                        context=context,
                        name=f"{self.name}_{identifier}",
                        log_directory=f"reconstruction/completion/raw/{self.name}_logs",
                    )
                    candidate = ObjectCompletionCandidate.model_validate_json(
                        (request_path.parent / "candidate.json").read_text(encoding="utf-8")
                    )
                    worker = CompletionWorkerManifest.model_validate_json(
                        (request_path.parent / "worker_manifest.json").read_text(encoding="utf-8")
                    )
                    if candidate.candidate_id != identifier:
                        raise RuntimeError("candidate worker changed deterministic candidate ID")
                    if worker.request_sha256 != sha256_file(request_path):
                        raise RuntimeError("candidate worker request hash mismatch")
                    validate_worker_model_identity(worker, request)
                    for asset in candidate.native_assets:
                        asset_path = context.path(*Path(asset.relative_path).parts)
                        if (
                            not asset_path.is_file()
                            or sha256_file(asset_path) != asset.sha256
                            or asset_path.stat().st_size != asset.size_bytes
                        ):
                            raise RuntimeError(
                                f"candidate native asset is inconsistent: {asset.relative_path}"
                            )
                        validate_native_candidate_asset(asset_path, asset.format.value)
                    candidates.append(candidate)
                except (ProcessExecutionError, OSError, ValueError, RuntimeError):
                    failures.append(identifier)
                per_object_count[anchor.object_id] = per_object_count.get(anchor.object_id, 0) + 1
        generation = CandidateGenerationManifest(
            backend=self.backend,
            official_repository=config.official_repository,
            official_code_commit=config.official_code_commit,
            checkpoint_repository=config.checkpoint_repository,
            checkpoint_revision=config.checkpoint_revision,
            checkpoint_hashes=config.checkpoint_hashes,
            runtime_model_revisions=config.runtime_model_revisions,
            runtime_model_hashes=config.runtime_model_hashes,
            evidence_split_sha256=sha256_file(root / "evidence_split.json"),
            crop_manifest_sha256=sha256_file(root / "crop_manifest.json"),
            requests=requests,
            candidates=sorted(candidates, key=lambda item: item.candidate_id),
            failed_candidate_ids=sorted(failures),
            runtime_seconds=sum(item.generation_runtime_seconds for item in candidates),
            warnings=(
                [f"{len(failures)} candidate generation attempts failed"] if failures else []
            ),
        )
        atomic_write_json(root / self.manifest_name, generation)
        fixed = {self.manifest_name}
        dynamic = [
            OutputSpec(
                path.relative_to(context.run_dir).as_posix(),
                "completion_candidate_file",
                (
                    "model/gltf-binary"
                    if path.suffix == ".glb"
                    else "model/ply"
                    if path.suffix == ".ply"
                    else "application/json"
                ),
                self.name,
                validation="json" if path.suffix == ".json" else "exists",
            )
            for path in sorted((root / "candidates").rglob("*"))
            if path.is_file()
            and path.relative_to(root).as_posix().split("/")[0] == "candidates"
            and path.name not in fixed
            and self.backend.value in path.relative_to(root).as_posix()
        ]
        return StageResult(
            outputs=dynamic,
            metrics={"candidates": len(candidates), "failed_candidates": len(failures)},
        )


class Sam3DObjectsCandidateAdapter(_CandidateGenerationAdapter):
    name = "sam3d_object_candidates"
    version = "0.1.2"
    backend = CompletionBackend.SAM3D_OBJECTS
    config_type = Sam3DObjectsAdapterConfig
    manifest_name = "sam3d_generation_manifest.json"
    license_record = sam3d_license()


class Trellis2ObjectCandidateAdapter(_CandidateGenerationAdapter):
    name = "trellis2_object_candidates"
    backend = CompletionBackend.TRELLIS2
    config_type = Trellis2ObjectsAdapterConfig
    manifest_name = "trellis2_generation_manifest.json"
    license_record = trellis2_license()


class MeasuredOnlyCandidateAdapter:
    name = "measured_only_candidates"
    version = "0.1.2"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        measured = MeasuredObjectGeometryArtifact.model_validate_json(
            context.canonical_path(
                "reconstruction", "measured_objects", "geometry_manifest.json"
            ).read_text(encoding="utf-8")
        )
        specs = [
            InputSpec("reconstruction/completion/eligibility.json", "completion_eligibility"),
            InputSpec("reconstruction/completion/evidence_split.json", "completion_evidence_split"),
            InputSpec("reconstruction/completion/crop_manifest.json", "completion_crop_manifest"),
            InputSpec(
                "reconstruction/measured_objects/geometry_manifest.json",
                "measured_object_geometry",
            ),
            InputSpec(
                "reconstruction/completion/evidence/evidence_package.json",
                "completion_evidence_package",
            ),
        ]
        evidence = CompletionEvidencePackage.model_validate_json(
            context.canonical_path(
                "reconstruction",
                "completion",
                "evidence",
                "evidence_package.json",
            ).read_text(encoding="utf-8")
        )
        for evidence_item in evidence.objects:
            if evidence_item.renderer_control_mesh_path is not None:
                specs.append(
                    InputSpec(
                        evidence_item.renderer_control_mesh_path,
                        "completion_evidence_file",
                        materialization_mode="reflink_or_copy",
                    )
                )
        for hypothesis in measured.hypotheses:
            if hypothesis.point_cloud is not None:
                specs.append(
                    InputSpec(
                        hypothesis.point_cloud.relative_path,
                        "measured_object_geometry_file",
                        materialization_mode="reflink_or_copy",
                    )
                )
        return [replace(spec, include_producer_signature=False) for spec in specs]

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return HealthcheckResult(True, "measured-only completion baseline available")

    def prepare(self, context: StageContext) -> None:
        context.path("reconstruction", "completion").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                "reconstruction/completion/measured_generation_manifest.json",
                "candidate_generation_manifest",
                "application/json",
                "measured_only_candidates",
                validation="json",
                model=CandidateGenerationManifest,
            )
        ]

    def run(self, context: StageContext) -> StageResult:
        root = context.path("reconstruction", "completion")
        measured = MeasuredObjectGeometryArtifact.model_validate_json(
            context.path("reconstruction", "measured_objects", "geometry_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        split = CompletionEvidenceSplit.model_validate_json(
            (root / "evidence_split.json").read_text(encoding="utf-8")
        )
        evidence = CompletionEvidencePackage.model_validate_json(
            (root / "evidence" / "evidence_package.json").read_text(encoding="utf-8")
        )
        evidence_by_id = {item.object_id: item for item in evidence.objects}
        split_by_id = {item.object_id: item for item in split.objects}
        eligible_ids = {item.object_id for item in split.objects}
        license_record = CompletionLicenseRecord(
            backend=CompletionBackend.MEASURED_PARTIAL_BASELINE,
            code_license="Reconevery project license",
            checkpoint_license="not applicable",
            asset_license="inherits measured input data rights",
            commercial_use_review_status="approved_by_project_policy",
            research_evaluation_allowed=True,
            production_selectable=False,
        )
        candidates: list[ObjectCompletionCandidate] = []
        for item in measured.hypotheses:
            if item.object_id not in eligible_ids or item.point_cloud is None:
                continue
            training = evidence_by_id[item.object_id]
            control_relative = training.renderer_control_mesh_path
            control_path = (
                context.path(*Path(control_relative).parts)
                if control_relative is not None
                else None
            )
            asset_path = context.path(*Path(item.point_cloud.relative_path).parts)
            native_assets = [
                CandidateNativeAsset(
                    asset_id="measured_partial_point_cloud",
                    relative_path=item.point_cloud.relative_path,
                    sha256=sha256_file(asset_path),
                    format=CandidateNativeFormat.MESH_PLY,
                    size_bytes=asset_path.stat().st_size,
                    role="phase5a_all_view_diagnostic_only",
                )
            ]
            selected_id = "measured_partial_point_cloud"
            selected_path = item.point_cloud.relative_path
            renderer = "measured_point_splat"
            if control_path is not None and control_relative is not None:
                native_assets.append(
                    CandidateNativeAsset(
                        asset_id="fitting_measured_renderer_control",
                        relative_path=control_relative,
                        sha256=sha256_file(control_path),
                        format=CandidateNativeFormat.MESH_PLY,
                        size_bytes=control_path.stat().st_size,
                        role="fitting_only_open_surface_renderer_control",
                    )
                )
                selected_id = "fitting_measured_renderer_control"
                selected_path = control_relative
                renderer = "nvdiffrast_open_measured_control"
            candidates.append(
                ObjectCompletionCandidate(
                    candidate_id=f"{item.object_id}__measured_partial_baseline__measured",
                    object_id=item.object_id,
                    semantic_label=item.semantic_label,
                    backend=CompletionBackend.MEASURED_PARTIAL_BASELINE,
                    anchor_frame_id=split_by_id[item.object_id].generation_anchor_frames[0],
                    generation_seed=context.seed,
                    native_assets=native_assets,
                    registration_asset_id=selected_id,
                    registration_asset_path=selected_path,
                    evaluation_asset_id=selected_id,
                    evaluation_asset_path=selected_path,
                    selection_asset_id=selected_id,
                    selection_asset_path=selected_path,
                    native_coordinate_convention="colmap_arbitrary",
                    vertex_count=item.point_cloud.point_count,
                    render_capability=CandidateRenderCapability(
                        renderer=renderer,
                        supports_rgba=True,
                        supports_depth=True,
                        camera_axes="x_right_y_down_z_forward",
                    ),
                    sampling_method="fitting_only_measured_open_surface_control",
                    generation_runtime_seconds=0.0,
                    license_record=license_record,
                    provenance=ProvenanceRecord(
                        adapter_name=self.name,
                        adapter_version=self.version,
                        configuration={"baseline": True},
                        input_artifact_paths=[item.point_cloud.relative_path],
                        output_artifact_paths=[],
                        timestamp=datetime.now(UTC),
                        confidence=ConfidenceRecord(
                            score=item.measurement_confidence,
                            method="measured_phase5a_geometry",
                        ),
                        source=GeometrySourceType.MEASURED,
                    ),
                )
            )
        manifest = CandidateGenerationManifest(
            backend=CompletionBackend.MEASURED_PARTIAL_BASELINE,
            official_repository="https://github.com/HuangYanjia/Reconevery",
            official_code_commit="0" * 40,
            checkpoint_repository="not-applicable/measured-partial-baseline",
            checkpoint_revision="0" * 40,
            checkpoint_hashes={},
            evidence_split_sha256=sha256_file(root / "evidence_split.json"),
            crop_manifest_sha256=sha256_file(root / "crop_manifest.json"),
            requests=[],
            candidates=sorted(candidates, key=lambda item: item.candidate_id),
            runtime_seconds=0,
        )
        atomic_write_json(root / "measured_generation_manifest.json", manifest)
        return StageResult(metrics={"measured_baselines": len(candidates)})
