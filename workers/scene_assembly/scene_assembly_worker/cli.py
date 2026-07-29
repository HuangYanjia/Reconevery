from __future__ import annotations

import argparse
from pathlib import Path

import trimesh

from scene_assembly_worker.articulated_snapshot import snapshot_summary
from scene_assembly_worker.asset_io import read_json, safe_path, sha256_file, write_json
from scene_assembly_worker.diagnostics import object_lines
from scene_assembly_worker.glb_scene import build_scene, export_glb, material_counts
from scene_assembly_worker.healthcheck import healthcheck
from scene_assembly_worker.lineage import lineage_summary
from scene_assembly_worker.overlap import overlap_summary
from scene_assembly_worker.previews import render_scene_preview, write_preview
from scene_assembly_worker.schema import PreviewRequest

PREVIEWS = (
    "global_context",
    "measured_anchors",
    "research_assembly",
    "deployment_assembly",
    "object_decision_grid",
    "overlap_heatmap",
    "articulated_snapshot",
)


def assemble(request_path: Path, input_root: Path, output_dir: Path) -> None:
    request = PreviewRequest.model_validate(read_json(request_path))
    references = (
        (request.assembly_plan_path, request.assembly_plan_sha256),
        (request.research_bundle_path, request.research_bundle_sha256),
        (request.deployment_bundle_path, request.deployment_bundle_sha256),
        (request.overlap_diagnostics_path, request.overlap_diagnostics_sha256),
    )
    for relative, expected in references:
        path = safe_path(input_root, relative)
        if sha256_file(path) != expected:
            raise ValueError(f"assembly preview input hash mismatch: {relative}")
    plan = read_json(safe_path(input_root, request.assembly_plan_path))
    research = read_json(safe_path(input_root, request.research_bundle_path))
    deployment = read_json(safe_path(input_root, request.deployment_bundle_path))
    overlap = read_json(safe_path(input_root, request.overlap_diagnostics_path))
    assets = plan.get("assets", [])
    if not isinstance(assets, list):
        raise ValueError("assembly plan assets must be a list")
    research_ids = {str(item) for item in research.get("asset_ids", [])}
    deployment_ids = {str(item) for item in deployment.get("asset_ids", [])}
    research_scene, material_before, texture_before = build_scene(
        input_root=input_root,
        planned_assets=assets,
        included_asset_ids=research_ids,
    )
    deployment_scene, deployment_material_before, deployment_texture_before = build_scene(
        input_root=input_root,
        planned_assets=assets,
        included_asset_ids=deployment_ids,
    )
    export_glb(research_scene, output_dir / "preview_assets/research_scene.glb")
    export_glb(deployment_scene, output_dir / "preview_assets/deployment_scene.glb")
    reloaded_research = trimesh.load(
        output_dir / "preview_assets/research_scene.glb",
        force="scene",
        process=False,
    )
    reloaded_deployment = trimesh.load(
        output_dir / "preview_assets/deployment_scene.glb",
        force="scene",
        process=False,
    )
    material_after, texture_after = material_counts(reloaded_research)
    deployment_material_after, deployment_texture_after = material_counts(reloaded_deployment)
    material_before += deployment_material_before
    texture_before += deployment_texture_before
    material_after += deployment_material_after
    texture_after += deployment_texture_after
    configuration = request.preview_configuration
    width = int(configuration.get("image_width", 960))
    height = int(configuration.get("image_height", 640))
    lines = [
        *object_lines(plan),
        lineage_summary(plan),
        overlap_summary(overlap),
        snapshot_summary(plan),
    ]
    bundle_lines = {
        "research_assembly": [
            *object_lines(research),
            lineage_summary(plan),
            overlap_summary(overlap),
        ],
        "deployment_assembly": [
            *object_lines(deployment),
            lineage_summary(plan),
            overlap_summary(overlap),
        ],
    }
    role_asset_ids = {
        "global_context": {
            str(item["asset"]["asset_id"])
            for item in assets
            if isinstance(item, dict)
            and isinstance(item.get("asset"), dict)
            and item["asset"].get("role") == "global_context"
        },
        "measured_anchors": {
            str(item["asset"]["asset_id"])
            for item in assets
            if isinstance(item, dict)
            and isinstance(item.get("asset"), dict)
            and item["asset"].get("role") == "measured_anchor"
        },
        "articulated_snapshot": {
            str(item["asset"]["asset_id"])
            for item in assets
            if isinstance(item, dict)
            and isinstance(item.get("asset"), dict)
            and item["asset"].get("object_id") is not None
        },
    }
    specialized_scenes = {
        name: build_scene(
            input_root=input_root,
            planned_assets=assets,
            included_asset_ids=asset_ids,
        )[0]
        for name, asset_ids in role_asset_ids.items()
    }
    scene_previews = {
        "global_context": specialized_scenes["global_context"],
        "measured_anchors": specialized_scenes["measured_anchors"],
        "research_assembly": research_scene,
        "deployment_assembly": deployment_scene,
        "overlap_heatmap": research_scene,
        "articulated_snapshot": specialized_scenes["articulated_snapshot"],
    }
    for name in PREVIEWS:
        if name in scene_previews:
            render_scene_preview(
                output_dir / f"previews/{name}.png",
                title=name.replace("_", " ").title(),
                scene=scene_previews[name],
                width=width,
                height=height,
                lines=bundle_lines.get(name, lines),
            )
            continue
        write_preview(
            output_dir / f"previews/{name}.png",
            title=name.replace("_", " ").title(),
            width=width,
            height=height,
            lines=lines,
        )
    warnings = []
    if material_after < material_before:
        warnings.append("preview GLB export did not preserve every source material")
    write_json(
        output_dir / "preview_manifest.json",
        {
            "schema_version": "0.1.0",
            "preview_paths": {name: f"assembly/previews/{name}.png" for name in PREVIEWS},
            "preview_asset_paths": {
                "research": "assembly/preview_assets/research_scene.glb",
                "deployment": "assembly/preview_assets/deployment_scene.glb",
            },
            "material_count_before": material_before,
            "material_count_after": material_after,
            "texture_count_before": texture_before,
            "texture_count_after": texture_after,
            "representation_warnings": warnings,
            "diagnostic_only": True,
            "source_geometry_modified": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["assemble", "healthcheck"])
    parser.add_argument("--request", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.action == "healthcheck":
        healthcheck()
        return
    if args.request is None or args.input_root is None or args.output_dir is None:
        parser.error("assemble requires --request, --input-root, and --output-dir")
    assemble(args.request, args.input_root, args.output_dir)


if __name__ == "__main__":
    main()
