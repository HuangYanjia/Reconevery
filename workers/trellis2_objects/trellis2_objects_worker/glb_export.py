from __future__ import annotations

from pathlib import Path
from typing import Any


def export_official_glb(
    mesh: Any,
    output_path: Path,
    *,
    texture_size: int,
    decimation_target: int,
) -> None:
    import o_voxel

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(output_path), extension_webp=True)
