from __future__ import annotations

import math
import shutil
import struct
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw

from recon2sim.artifacts import CompactFaceIndexManifest, ObjectSurfaceHypothesis
from recon2sim.genrecon import read_ply_mesh, sha256_file
from recon2sim.ir import (
    AlignmentStatus,
    CameraAxes,
    CoordinateConvention,
    LinearUnits,
    ScaleStatus,
    TransformDirection,
    WorldFrame,
)

UINT32_MAX = 2**32 - 1


def coordinate_metadata_is_raw_colmap(convention: CoordinateConvention) -> bool:
    return (
        convention.world_frame is WorldFrame.COLMAP_ARBITRARY
        and convention.alignment_status is AlignmentStatus.UNORIENTED
        and convention.camera_axes is CameraAxes.X_RIGHT_Y_DOWN_Z_FORWARD
        and convention.linear_units is LinearUnits.ARBITRARY_UNITS
        and convention.scale_status is ScaleStatus.SCALE_AMBIGUOUS
        and convention.transform_direction is TransformDirection.WORLD_FROM_CAMERA
    )


def write_compact_face_ids(
    path: Path,
    face_ids: Iterable[int],
    *,
    global_mesh_sha256: str,
    relative_path: str | None = None,
) -> CompactFaceIndexManifest:
    values = list(face_ids)
    if values != sorted(set(values)):
        raise ValueError("face IDs must be sorted and unique")
    if values and values[0] < 0:
        raise ValueError("face IDs cannot be negative")
    dtype: Literal["uint32", "uint64"] = (
        "uint32" if not values or values[-1] <= UINT32_MAX else "uint64"
    )
    format_code = "I" if dtype == "uint32" else "Q"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        for value in values:
            file.write(struct.pack("<" + format_code, value))
    return CompactFaceIndexManifest(
        relative_path=relative_path or path.as_posix(),
        dtype=dtype,
        count=len(values),
        global_mesh_sha256=global_mesh_sha256,
        minimum_face_id=values[0] if values else None,
        maximum_face_id=values[-1] if values else None,
        content_sha256=sha256_file(path),
    )


def read_compact_face_ids(
    root: Path,
    manifest: CompactFaceIndexManifest,
    *,
    global_face_count: int | None = None,
) -> tuple[int, ...]:
    path = root / manifest.relative_path
    if not path.is_file():
        raise FileNotFoundError(f"face-index array is missing: {manifest.relative_path}")
    if sha256_file(path) != manifest.content_sha256:
        raise ValueError(f"face-index hash mismatch: {manifest.relative_path}")
    size = 4 if manifest.dtype == "uint32" else 8
    payload = path.read_bytes()
    if len(payload) != manifest.count * size:
        raise ValueError(f"face-index byte count mismatch: {manifest.relative_path}")
    if not payload:
        values: tuple[int, ...] = ()
    else:
        format_code = "I" if manifest.dtype == "uint32" else "Q"
        values = tuple(item[0] for item in struct.iter_unpack("<" + format_code, payload))
    if values != tuple(sorted(set(values))):
        raise ValueError(f"face-index array is not sorted and unique: {manifest.relative_path}")
    if values:
        if values[0] != manifest.minimum_face_id or values[-1] != manifest.maximum_face_id:
            raise ValueError(f"face-index range mismatch: {manifest.relative_path}")
        if global_face_count is not None and values[-1] >= global_face_count:
            raise ValueError(
                f"face ID {values[-1]} exceeds global mesh face count {global_face_count}"
            )
    return values


def validate_surface_mesh(
    root: Path,
    hypothesis: ObjectSurfaceHypothesis,
) -> None:
    accepted = read_compact_face_ids(
        root,
        hypothesis.accepted_global_face_ids,
        global_face_count=hypothesis.global_face_count,
    )
    read_compact_face_ids(
        root,
        hypothesis.ambiguous_global_face_ids,
        global_face_count=hypothesis.global_face_count,
    )
    if hypothesis.status == "unresolved":
        return
    if hypothesis.surface_mesh_path is None:
        raise ValueError(f"object {hypothesis.object_id!r} has no surface mesh")
    mesh = read_ply_mesh(root / hypothesis.surface_mesh_path)
    if len(mesh.faces) != len(accepted) or len(mesh.faces) != hypothesis.face_count:
        raise ValueError(
            f"object {hypothesis.object_id!r} surface mesh face count does not match face IDs"
        )
    if len(mesh.vertices) != hypothesis.vertex_count:
        raise ValueError(f"object {hypothesis.object_id!r} vertex count is inconsistent")
    if any(not math.isfinite(value) for vertex in mesh.vertices for value in vertex):
        raise ValueError(f"object {hypothesis.object_id!r} contains non-finite vertices")
    if not mesh.vertices or not mesh.faces:
        raise ValueError(f"object {hypothesis.object_id!r} surface mesh is empty")


def render_summary_previews(root: Path, evidence: object) -> None:
    from recon2sim.artifacts import ObjectSurfaceEvidenceArtifact

    artifact = ObjectSurfaceEvidenceArtifact.model_validate(evidence)
    preview_root = root / "reconstruction" / "object_surfaces" / "previews"
    preview_root.mkdir(parents=True, exist_ok=True)
    rows = max(len(artifact.hypotheses), 1)
    width = 920
    height = 90 + rows * 58
    titles = {
        "global_face_assignment.png": "Phase 4 global face assignment summary",
        "object_surface_contact_sheet.png": "Partial object surfaces",
        "reprojection_contact_sheet.png": "Reprojection quality",
        "conflict_heatmap.png": "Same-class conflicts and semantic overlaps",
    }
    for filename, title in titles.items():
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((20, 18), title, fill="#111111")
        draw.text(
            (20, 42),
            "Coordinates: COLMAP arbitrary; scale ambiguous; output is not sim-ready",
            fill="#7a1f1f",
        )
        for index, hypothesis in enumerate(artifact.hypotheses):
            y = 78 + index * 58
            draw.text(
                (20, y),
                (
                    f"{hypothesis.object_id} [{hypothesis.status}] "
                    f"accepted={hypothesis.accepted_global_face_ids.count} "
                    f"ambiguous={hypothesis.ambiguous_global_face_ids.count} "
                    f"IoU={hypothesis.mean_reprojection_iou:.4f}"
                ),
                fill="#20252b",
            )
            bar_width = int(
                500
                * min(
                    1.0,
                    hypothesis.accepted_global_face_ids.count
                    / max(artifact.partition.global_face_count, 1),
                )
            )
            draw.rectangle((20, y + 24, 520, y + 36), fill="#dddddd")
            if bar_width:
                draw.rectangle((20, y + 24, 20 + bar_width, y + 36), fill="#2b8cbe")
        image.save(
            preview_root / filename,
            format="PNG",
            compress_level=6,
            optimize=False,
        )


def export_object_surface(root: Path, hypothesis: ObjectSurfaceHypothesis, output: Path) -> None:
    if hypothesis.surface_mesh_path is None:
        raise ValueError(f"object {hypothesis.object_id!r} has no resolved surface mesh")
    source = root / hypothesis.surface_mesh_path
    if not source.is_file():
        raise FileNotFoundError(f"partial surface mesh is missing: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)


def export_object_face_ids(
    root: Path,
    hypothesis: ObjectSurfaceHypothesis,
    output: Path,
) -> None:
    read_compact_face_ids(
        root,
        hypothesis.accepted_global_face_ids,
        global_face_count=hypothesis.global_face_count,
    )
    source = root / hypothesis.accepted_global_face_ids.relative_path
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
