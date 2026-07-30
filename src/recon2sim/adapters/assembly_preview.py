from __future__ import annotations

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
from recon2sim.adapters.ingest import run_process
from recon2sim.artifacts import (
    SceneAssemblyPlan,
    SceneAssemblyPreviewManifest,
)
from recon2sim.calibration import sha256_file
from recon2sim.storage import atomic_write_json


class AssemblyPreviewConfig(CompletionWorkerConfig):
    worker_module: str = "scene_assembly_worker"
    docker_image: str = "reconevery/scene-assembly:phase6b"
    image_width: int = 960
    image_height: int = 640
    background_rgb: tuple[int, int, int] = (245, 246, 248)
    articulated_snapshot_policy: str = "reference_state"


class AssemblyPreviewAdapter:
    name = "assembly_previews"
    version = "0.1.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        plan = SceneAssemblyPlan.model_validate_json(
            context.canonical_path("assembly/assembly_plan.json").read_text(encoding="utf-8")
        )
        specs = [
            InputSpec("assembly/assembly_plan.json", "scene_assembly_plan"),
            InputSpec("assembly/research_visual_bundle.json", "scene_assembly_bundle"),
            InputSpec(
                "assembly/deployment_eligible_visual_bundle.json",
                "scene_assembly_bundle",
            ),
            InputSpec(
                "assembly/overlap_diagnostics.json",
                "scene_assembly_overlap_diagnostics",
            ),
        ]
        for item in plan.assets:
            specs.append(
                InputSpec(
                    item.asset.asset_path,
                    "scene_assembly_visual_asset",
                    expected_sha256=item.asset.asset_sha256,
                    include_producer_signature=False,
                )
            )
        return specs

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return completion_healthcheck(
            context,
            AssemblyPreviewConfig,
            worker_name=self.name,
        )

    def prepare(self, context: StageContext) -> None:
        context.path("assembly/previews").mkdir(parents=True, exist_ok=True)
        context.path("assembly/preview_assets").mkdir(parents=True, exist_ok=True)
        context.path("assembly/raw/logs").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        outputs = [
            OutputSpec(
                "assembly/preview_manifest.json",
                "scene_assembly_preview_manifest",
                "application/json",
                self.name,
                validation="json",
                model=SceneAssemblyPreviewManifest,
            ),
            OutputSpec(
                "assembly/preview_assets/research_scene.glb",
                "scene_assembly_preview_glb",
                "model/gltf-binary",
                self.name,
            ),
            OutputSpec(
                "assembly/preview_assets/deployment_scene.glb",
                "scene_assembly_preview_glb",
                "model/gltf-binary",
                self.name,
            ),
        ]
        outputs.extend(
            OutputSpec(
                f"assembly/previews/{name}.png",
                "scene_assembly_preview",
                "image/png",
                self.name,
                validation="png",
            )
            for name in (
                "global_context",
                "measured_anchors",
                "research_assembly",
                "deployment_assembly",
                "object_decision_grid",
                "overlap_heatmap",
                "articulated_snapshot",
            )
        )
        return outputs

    def run(self, context: StageContext) -> StageResult:
        config = AssemblyPreviewConfig.model_validate(context.config.adapter.config)
        request_path = context.path("assembly/preview_request.json")
        plan_path = context.path("assembly/assembly_plan.json")
        research_path = context.path("assembly/research_visual_bundle.json")
        deployment_path = context.path("assembly/deployment_eligible_visual_bundle.json")
        overlap_path = context.path("assembly/overlap_diagnostics.json")
        atomic_write_json(
            request_path,
            {
                "schema_version": "0.1.0",
                "assembly_plan_path": "assembly/assembly_plan.json",
                "assembly_plan_sha256": sha256_file(plan_path),
                "research_bundle_path": "assembly/research_visual_bundle.json",
                "research_bundle_sha256": sha256_file(research_path),
                "deployment_bundle_path": "assembly/deployment_eligible_visual_bundle.json",
                "deployment_bundle_sha256": sha256_file(deployment_path),
                "overlap_diagnostics_path": "assembly/overlap_diagnostics.json",
                "overlap_diagnostics_sha256": sha256_file(overlap_path),
                "preview_configuration": {
                    "image_width": config.image_width,
                    "image_height": config.image_height,
                    "background_rgb": config.background_rgb,
                    "articulated_snapshot_policy": config.articulated_snapshot_policy,
                },
                "output_directory": "assembly",
                "diagnostic_only": True,
                "fake_mode": config.fake_mode,
                "seed": context.seed,
            },
        )
        command = worker_command(
            context,
            config,
            "assemble",
            "assembly/preview_request.json",
            "assembly",
        )
        run_process(
            command,
            context=context,
            name="scene_assembly",
            log_directory="assembly/raw/logs",
        )
        for path, expected in (
            (plan_path, sha256_file(context.canonical_path("assembly/assembly_plan.json"))),
            (
                research_path,
                sha256_file(context.canonical_path("assembly/research_visual_bundle.json")),
            ),
            (
                deployment_path,
                sha256_file(
                    context.canonical_path("assembly/deployment_eligible_visual_bundle.json")
                ),
            ),
            (
                overlap_path,
                sha256_file(context.canonical_path("assembly/overlap_diagnostics.json")),
            ),
        ):
            if sha256_file(path) != expected:
                raise RuntimeError(
                    f"assembly preview worker modified immutable upstream input {path.name}"
                )
        manifest = SceneAssemblyPreviewManifest.model_validate_json(
            context.path("assembly/preview_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.material_count_after < manifest.material_count_before:
            manifest.representation_warnings.append("preview material count decreased")
            atomic_write_json(context.path("assembly/preview_manifest.json"), manifest)
        return StageResult(
            outputs=[
                OutputSpec(
                    "assembly/preview_request.json",
                    "scene_assembly_preview_request",
                    "application/json",
                    self.name,
                    validation="json",
                )
            ],
            metrics={
                "previews": len(manifest.preview_paths),
                "preview_assets": len(manifest.preview_asset_paths),
                "representation_warnings": len(manifest.representation_warnings),
            },
        )


__all__ = ["AssemblyPreviewAdapter", "AssemblyPreviewConfig"]
