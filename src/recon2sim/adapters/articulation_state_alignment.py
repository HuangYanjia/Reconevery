from __future__ import annotations

from dataclasses import replace

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
    ArticulatedPartStateGeometryManifest,
    ArticulationCaptureManifest,
    ArticulationPartPromptManifest,
    ArticulationStateAlignmentArtifact,
)
from recon2sim.storage import atomic_write_json


class ArticulationStateAlignmentConfig(ArticulationWorkerConfig):
    worker_module: str = "articulation_alignment_worker"
    docker_image: str = "reconevery/articulation-alignment:phase5c"
    minimum_static_correspondences: int = 200
    maximum_static_median_residual_scene_diagonal: float = 0.02
    maximum_static_p90_residual_scene_diagonal: float = 0.05
    minimum_heldout_static_depth_inlier_fraction: float = 0.60


class ArticulationStateAlignmentAdapter:
    name = "articulation_state_alignment"
    version = "0.1.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        geometry = ArticulatedPartStateGeometryManifest.model_validate_json(
            context.canonical_path(
                "reconstruction", "articulation", "measured_states", "manifest.json"
            ).read_text(encoding="utf-8")
        )
        specs = [
            InputSpec(
                "reconstruction/articulation/capture_manifest.json",
                "articulation_capture_manifest",
            ),
            InputSpec(
                "reconstruction/articulation/part_prompt_manifest.json",
                "articulation_part_prompt_manifest",
            ),
            InputSpec(
                "reconstruction/articulation/measured_states/manifest.json",
                "articulated_part_state_geometry_manifest",
            ),
        ]
        specs.extend(
            InputSpec(
                item.measured_point_cloud_path,
                "measured_articulated_part_point_cloud",
                materialization_mode="reflink_or_copy",
            )
            for item in geometry.geometries
        )
        return [replace(spec, include_producer_signature=False) for spec in specs]

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return articulation_healthcheck(
            context,
            ArticulationStateAlignmentConfig,
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
        return [
            OutputSpec(
                "reconstruction/articulation/state_alignment.json",
                "articulation_state_alignment",
                "application/json",
                self.name,
                validation="json",
                model=ArticulationStateAlignmentArtifact,
            ),
            OutputSpec(
                "reconstruction/articulation/previews/state_alignment.png",
                "articulation_preview",
                "image/png",
                self.name,
                validation="png",
            ),
        ]

    def run(self, context: StageContext) -> StageResult:
        config = ArticulationStateAlignmentConfig.model_validate(context.config.adapter.config)
        root = context.path("reconstruction", "articulation")
        capture = ArticulationCaptureManifest.model_validate_json(
            (root / "capture_manifest.json").read_text(encoding="utf-8")
        )
        prompt = ArticulationPartPromptManifest.model_validate_json(
            (root / "part_prompt_manifest.json").read_text(encoding="utf-8")
        )
        geometry = ArticulatedPartStateGeometryManifest.model_validate_json(
            (root / "measured_states/manifest.json").read_text(encoding="utf-8")
        )
        request_path = root / "raw" / "state_alignment_request.json"
        atomic_write_json(
            request_path,
            {
                "schema_version": "0.1.0",
                "capture_manifest_path": "reconstruction/articulation/capture_manifest.json",
                "capture_manifest_sha256": sha256_file(root / "capture_manifest.json"),
                "part_prompt_manifest_path": (
                    "reconstruction/articulation/part_prompt_manifest.json"
                ),
                "part_prompt_manifest_sha256": sha256_file(root / "part_prompt_manifest.json"),
                "measured_states_manifest_path": (
                    "reconstruction/articulation/measured_states/manifest.json"
                ),
                "measured_states_manifest_sha256": sha256_file(
                    root / "measured_states/manifest.json"
                ),
                "reference_state_id": capture.reference_state_id,
                "state_ids": [state.state_id for state in capture.states],
                "base_part_id": prompt.objects[0].base.part_id,
                "movable_part_ids": [
                    part.part_id for part in prompt.objects[0].movable_parts if part.include
                ],
                "geometry_paths": [item.measured_point_cloud_path for item in geometry.geometries],
                "acceptance_configuration": {
                    "minimum_static_correspondences": config.minimum_static_correspondences,
                    "maximum_static_median_residual_scene_diagonal": (
                        config.maximum_static_median_residual_scene_diagonal
                    ),
                    "maximum_static_p90_residual_scene_diagonal": (
                        config.maximum_static_p90_residual_scene_diagonal
                    ),
                    "minimum_heldout_static_depth_inlier_fraction": (
                        config.minimum_heldout_static_depth_inlier_fraction
                    ),
                },
                "output_directory": "reconstruction/articulation",
                "seed": context.seed,
                "fake_mode": config.fake_mode,
            },
        )
        run_articulation_worker(
            context,
            config,
            action="align",
            request_path=request_path.relative_to(context.run_dir).as_posix(),
            output_directory="reconstruction/articulation",
            log_name="state_alignment",
        )
        result = ArticulationStateAlignmentArtifact.model_validate_json(
            (root / "state_alignment.json").read_text(encoding="utf-8")
        )
        if result.capture_manifest_sha256 != sha256_file(root / "capture_manifest.json"):
            raise RuntimeError("state alignment capture hash mismatch")
        if {item.state_id for item in result.transforms} != {
            state.state_id for state in capture.states
        }:
            raise RuntimeError("state alignment did not return every capture state")
        return StageResult(
            metrics={
                "aligned_states": sum(item.accepted for item in result.transforms),
                "state_count": len(result.transforms),
            }
        )


__all__ = [
    "ArticulationStateAlignmentAdapter",
    "ArticulationStateAlignmentConfig",
]
