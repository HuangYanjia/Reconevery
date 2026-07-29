from __future__ import annotations

from pathlib import Path

import trimesh

from scene_assembly_worker.asset_io import safe_path
from scene_assembly_worker.transforms import matrix4


def build_scene(
    *,
    input_root: Path,
    planned_assets: list[dict[str, object]],
    included_asset_ids: set[str],
) -> tuple[trimesh.Scene, int, int]:
    output = trimesh.Scene()
    material_count = 0
    texture_count = 0
    for item in planned_assets:
        asset = item["asset"]
        if not isinstance(asset, dict) or str(asset["asset_id"]) not in included_asset_ids:
            continue
        source = safe_path(input_root, str(asset["asset_path"]))
        loaded = trimesh.load(source, force="scene", process=False)
        if not isinstance(loaded, trimesh.Scene):
            loaded = trimesh.Scene(loaded)
        root_transform = matrix4(list(item["asset_to_assembly_world"]))
        for node_name in loaded.graph.nodes_geometry:
            node_transform, geometry_name = loaded.graph[node_name]
            geometry = loaded.geometry[geometry_name].copy()
            visual = getattr(geometry, "visual", None)
            if visual is not None and getattr(visual, "material", None) is not None:
                material_count += 1
                material = visual.material
                if getattr(material, "baseColorTexture", None) is not None:
                    texture_count += 1
            output.add_geometry(
                geometry,
                node_name=f"{asset['asset_id']}::{node_name}",
                transform=root_transform @ node_transform,
            )
    return output, material_count, texture_count


def export_glb(scene: trimesh.Scene, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(scene.export(file_type="glb"))


def material_counts(scene: trimesh.Scene) -> tuple[int, int]:
    materials = 0
    textures = 0
    for geometry in scene.geometry.values():
        visual = getattr(geometry, "visual", None)
        material = getattr(visual, "material", None)
        if material is None:
            continue
        materials += 1
        if getattr(material, "baseColorTexture", None) is not None:
            textures += 1
    return materials, textures


__all__ = ["build_scene", "export_glb", "material_counts"]
