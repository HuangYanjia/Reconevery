from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Literal

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
    ArticulatedAssetIndex,
    ArticulatedAssetIndexRecord,
    ArticulatedCandidate,
    ArticulatedRetrievalResult,
    ArticulationPartPromptManifest,
    MeasuredPartMotionArtifact,
)
from recon2sim.storage import atomic_write_json


class ArticulatedRetrievalConfig(ArticulationWorkerConfig):
    worker_module: str = "articulated_retrieval_worker"
    docker_image: str = "reconevery/articulated-retrieval:phase5c"
    source_family: Literal["artvip", "partnet_mobility"]
    index_path: str | None = None
    maximum_candidates: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def real_retrieval_requires_local_index(self) -> ArticulatedRetrievalConfig:
        if self.execution_mode != "fake_worker" and self.index_path is None:
            raise ValueError("real articulated retrieval requires a local immutable index")
        return self


class _ArticulatedRetrievalAdapter:
    source_family: Literal["artvip", "partnet_mobility"]
    manifest_name: str
    name: str
    version = "0.1.0"

    def _selected_records(
        self,
        context: StageContext,
        config: ArticulatedRetrievalConfig,
    ) -> list[ArticulatedAssetIndexRecord]:
        if config.index_path is None:
            return []
        index = ArticulatedAssetIndex.model_validate_json(
            Path(config.index_path).expanduser().resolve().read_text(encoding="utf-8")
        )
        measured = MeasuredPartMotionArtifact.model_validate_json(
            context.canonical_path(
                "reconstruction", "articulation", "measured_motion.json"
            ).read_text(encoding="utf-8")
        )
        prompts = ArticulationPartPromptManifest.model_validate_json(
            context.canonical_path(
                "reconstruction", "articulation", "part_prompt_manifest.json"
            ).read_text(encoding="utf-8")
        )
        prompt = next(
            (
                item
                for item in prompts.objects
                if item.articulated_object_id == measured.articulated_object_id
            ),
            None,
        )
        if prompt is None:
            raise ValueError("retrieval prompt manifest does not contain the measured object")
        observed_joint_types = {item.joint_type.value for item in measured.joint_hypotheses}

        def score(record: ArticulatedAssetIndexRecord) -> tuple[float, str]:
            category = record.category.casefold()
            label = prompt.semantic_label.casefold()
            semantic = (
                1.0 if category == label else 0.6 if category in label or label in category else 0.0
            )
            candidate_joint_types = {item.value for item in record.joint_types}
            joint = (
                len(candidate_joint_types & observed_joint_types)
                / max(len(observed_joint_types), 1)
                if observed_joint_types
                else 0.5
            )
            part = 1.0 / (1.0 + abs(record.link_count - (len(measured.joint_hypotheses) + 1)))
            return 0.50 * semantic + 0.35 * joint + 0.15 * part, record.asset_id

        eligible = [item for item in index.records if item.candidate_bundle_path is not None]
        return sorted(eligible, key=lambda item: (-score(item)[0], score(item)[1]))[
            : config.maximum_candidates
        ]

    def _selected_asset_specs(
        self,
        context: StageContext,
        config: ArticulatedRetrievalConfig,
    ) -> list[InputSpec]:
        if config.index_path is None:
            return []
        index_root = Path(config.index_path).expanduser().resolve().parent
        specs: list[InputSpec] = []
        for record in self._selected_records(context, config):
            asset_id = record.asset_id
            if PurePosixPath(asset_id).name != asset_id or asset_id in {"", ".", ".."}:
                raise ValueError(f"unsafe articulated asset ID: {asset_id!r}")
            if record.candidate_bundle_path is None or record.candidate_bundle_sha256 is None:
                continue
            specs.append(
                InputSpec(
                    (
                        "reconstruction/articulation/catalogs/"
                        f"{self.source_family}/selected/{asset_id}/source_candidate.json"
                    ),
                    "articulated_catalog_candidate_bundle",
                    expected_sha256=record.candidate_bundle_sha256,
                    source_path=index_root / record.candidate_bundle_path,
                    materialization_mode="copy",
                )
            )
            for position, source_relative in enumerate(record.visual_asset_paths):
                source = index_root / source_relative
                suffix = source.suffix.lower() or ".bin"
                specs.append(
                    InputSpec(
                        (
                            "reconstruction/articulation/catalogs/"
                            f"{self.source_family}/selected/{asset_id}/visuals/"
                            f"{position:03d}{suffix}"
                        ),
                        "articulated_catalog_visual_asset",
                        expected_sha256=record.file_hashes[source_relative],
                        source_path=source,
                        materialization_mode="reflink_or_copy",
                    )
                )
        return specs

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        config = ArticulatedRetrievalConfig.model_validate(context.config.adapter.config)
        specs = [
            InputSpec(
                "reconstruction/articulation/measured_motion.json",
                "measured_part_motion",
            ),
            InputSpec(
                "reconstruction/articulation/part_prompt_manifest.json",
                "articulation_part_prompt_manifest",
            ),
        ]
        if config.index_path is not None:
            specs.append(
                InputSpec(
                    f"reconstruction/articulation/catalogs/{self.source_family}_index.json",
                    "articulated_asset_index",
                    source_path=Path(config.index_path).expanduser().resolve(),
                    materialization_mode="copy",
                )
            )
        specs.extend(self._selected_asset_specs(context, config))
        return specs

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        if context is not None:
            try:
                config = ArticulatedRetrievalConfig.model_validate(context.config.adapter.config)
                if config.source_family != self.source_family:
                    raise ValueError("retrieval adapter source_family mismatch")
                if config.index_path is not None:
                    path = Path(config.index_path).expanduser()
                    if not path.is_file():
                        raise ValueError(f"articulated asset index does not exist: {path}")
                    ArticulatedAssetIndex.model_validate_json(path.read_text(encoding="utf-8"))
            except ValueError as exc:
                return HealthcheckResult(False, str(exc))
        return articulation_healthcheck(
            context,
            ArticulatedRetrievalConfig,
            worker_name=self.name,
        )

    def prepare(self, context: StageContext) -> None:
        context.path("reconstruction", "articulation", "raw", "logs").mkdir(
            parents=True, exist_ok=True
        )

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                f"reconstruction/articulation/{self.manifest_name}",
                "articulated_retrieval_result",
                "application/json",
                self.name,
                validation="json",
                model=ArticulatedRetrievalResult,
            )
        ]

    def run(self, context: StageContext) -> StageResult:
        config = ArticulatedRetrievalConfig.model_validate(context.config.adapter.config)
        root = context.path("reconstruction", "articulation")
        measured = MeasuredPartMotionArtifact.model_validate_json(
            (root / "measured_motion.json").read_text(encoding="utf-8")
        )
        index_relative = f"reconstruction/articulation/catalogs/{self.source_family}_index.json"
        index_path = context.path(*Path(index_relative).parts)
        selected_assets = []
        for record in self._selected_records(context, config):
            if record.candidate_bundle_path is None:
                continue
            asset_id = record.asset_id
            selected_assets.append(
                {
                    "asset_id": asset_id,
                    "source_candidate_path": (
                        "reconstruction/articulation/catalogs/"
                        f"{self.source_family}/selected/{asset_id}/source_candidate.json"
                    ),
                    "visual_path_mapping": {
                        source_relative: (
                            "reconstruction/articulation/catalogs/"
                            f"{self.source_family}/selected/{asset_id}/visuals/"
                            f"{position:03d}{Path(source_relative).suffix.lower() or '.bin'}"
                        )
                        for position, source_relative in enumerate(record.visual_asset_paths)
                    },
                }
            )
        request_path = root / "raw" / f"{self.source_family}_retrieval_request.json"
        atomic_write_json(
            request_path,
            {
                "schema_version": "0.1.0",
                "source_family": self.source_family,
                "articulated_object_id": measured.articulated_object_id,
                "measured_motion_path": "reconstruction/articulation/measured_motion.json",
                "measured_motion_sha256": sha256_file(root / "measured_motion.json"),
                "part_prompt_manifest_path": (
                    "reconstruction/articulation/part_prompt_manifest.json"
                ),
                "part_prompt_manifest_sha256": sha256_file(root / "part_prompt_manifest.json"),
                "asset_index_path": index_relative if index_path.is_file() else None,
                "asset_index_sha256": (sha256_file(index_path) if index_path.is_file() else None),
                "selected_assets": selected_assets,
                "maximum_candidates": config.maximum_candidates,
                "output_path": f"reconstruction/articulation/{self.manifest_name}",
                "seed": context.seed,
                "fake_mode": config.fake_mode,
            },
        )
        run_articulation_worker(
            context,
            config,
            action="retrieve",
            request_path=request_path.relative_to(context.run_dir).as_posix(),
            output_directory="reconstruction/articulation",
            log_name=f"{self.source_family}_retrieval",
        )
        result = ArticulatedRetrievalResult.model_validate_json(
            (root / self.manifest_name).read_text(encoding="utf-8")
        )
        if result.measured_motion_sha256 != sha256_file(root / "measured_motion.json"):
            raise RuntimeError("articulated retrieval measured-motion hash mismatch")
        if any(item.source_family.value != self.source_family for item in result.candidates):
            raise RuntimeError("articulated retrieval returned the wrong source family")
        outputs: list[OutputSpec] = []
        for candidate in result.candidates:
            if candidate.candidate_bundle_path is None:
                continue
            bundle = context.path(*Path(candidate.candidate_bundle_path).parts)
            if sha256_file(bundle) != candidate.candidate_bundle_sha256:
                raise RuntimeError("retrieved articulated candidate bundle hash mismatch")
            ArticulatedCandidate.model_validate_json(bundle.read_text(encoding="utf-8"))
            outputs.append(
                OutputSpec(
                    candidate.candidate_bundle_path,
                    "articulated_retrieved_candidate_bundle",
                    "application/json",
                    self.name,
                    validation="json",
                    model=ArticulatedCandidate,
                )
            )
            for path, expected_hash in candidate.visual_asset_hashes.items():
                asset = context.path(*Path(path).parts)
                if sha256_file(asset) != expected_hash:
                    raise RuntimeError("retrieved articulated visual-asset hash mismatch")
                outputs.append(
                    OutputSpec(
                        path,
                        "articulated_candidate_visual_link",
                        (
                            "model/gltf-binary"
                            if path.lower().endswith(".glb")
                            else "model/ply"
                            if path.lower().endswith(".ply")
                            else "application/octet-stream"
                        ),
                        self.name,
                    )
                )
        return StageResult(
            outputs=outputs,
            metrics={"retrieval_candidates": len(result.candidates)},
        )


class ArtVIPRetrievalAdapter(_ArticulatedRetrievalAdapter):
    source_family = "artvip"
    manifest_name = "artvip_retrieval.json"
    name = "artvip_retrieval"


class PartNetRetrievalAdapter(_ArticulatedRetrievalAdapter):
    source_family = "partnet_mobility"
    manifest_name = "partnet_retrieval.json"
    name = "partnet_retrieval"


__all__ = [
    "ArtVIPRetrievalAdapter",
    "ArticulatedRetrievalConfig",
    "PartNetRetrievalAdapter",
]
