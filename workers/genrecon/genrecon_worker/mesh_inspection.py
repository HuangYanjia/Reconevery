from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def transform_outputs_to_colmap(output_dir: Path, working_to_colmap: np.ndarray) -> None:
    mesh_path = output_dir / "mesh.ply"
    scene_path = output_dir / "scene.glb"
    if not mesh_path.is_file() or not scene_path.is_file():
        raise RuntimeError("official GenRecon output is missing mesh.ply or scene.glb")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError("official GenRecon mesh is empty or unreadable")
    mesh.apply_transform(working_to_colmap)
    if not np.isfinite(mesh.vertices).all():
        raise RuntimeError("transformed GenRecon mesh contains non-finite coordinates")
    mesh.export(mesh_path)

    scene = trimesh.load(scene_path, force="scene", process=False)
    if not isinstance(scene, trimesh.Scene) or not scene.geometry:
        raise RuntimeError("official GenRecon GLB is empty or unreadable")
    scene.apply_transform(working_to_colmap)
    scene.export(scene_path)
