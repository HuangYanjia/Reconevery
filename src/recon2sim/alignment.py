from __future__ import annotations

import math
import struct
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

from recon2sim.artifacts import (
    AlignmentTransform,
    CameraMeshAlignmentDiagnostics,
    CameraMeshAlignmentPreviewManifest,
    CameraMeshAlignmentResult,
)
from recon2sim.genrecon import read_ply_mesh


def _determinant3(matrix: list[list[float]]) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _multiply4(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [sum(left[row][index] * right[index][column] for index in range(4)) for column in range(4)]
        for row in range(4)
    ]


def validate_similarity_transform(transform: AlignmentTransform) -> None:
    matrix = transform.matrix_original_mesh_to_aligned_colmap
    inverse = transform.inverse_matrix
    product = _multiply4(matrix, inverse)
    roundtrip = math.sqrt(
        sum(
            (product[row][column] - (1.0 if row == column else 0.0)) ** 2
            for row in range(4)
            for column in range(4)
        )
    )
    if roundtrip > 1e-6 or transform.roundtrip_error > 1e-6:
        raise ValueError("alignment transform is not invertible within tolerance")
    determinant = _determinant3([row[:3] for row in matrix[:3]])
    if not math.isfinite(determinant) or determinant <= 0:
        raise ValueError("alignment transform must have a finite positive determinant")
    expected = transform.scale**3
    if abs(determinant - expected) > max(1e-6, abs(expected) * 1e-6):
        raise ValueError("alignment transform determinant and scale disagree")
    if any(not math.isfinite(component) for row in matrix for component in row):
        raise ValueError("alignment transform contains non-finite values")


def transform_point(
    point: tuple[float, float, float],
    matrix: list[list[float]],
) -> tuple[float, float, float]:
    x, y, z = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    )


def export_aligned_ply(
    source: Path,
    destination: Path,
    transform: AlignmentTransform,
) -> None:
    """Write a derived binary geometry-only PLY without changing the source mesh."""
    validate_similarity_transform(transform)
    mesh = read_ply_mesh(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment derived alignment export; original mesh remains canonical evidence\n"
        f"element vertex {len(mesh.vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {len(mesh.faces)}\n"
        "property list uchar uint vertex_indices\n"
        "end_header\n"
    )
    with destination.open("wb") as file:
        file.write(header.encode("ascii"))
        for vertex in mesh.vertices:
            file.write(
                struct.pack(
                    "<fff",
                    *transform_point(
                        vertex,
                        transform.matrix_original_mesh_to_aligned_colmap,
                    ),
                )
            )
        for face in mesh.faces:
            file.write(struct.pack("<BIII", 3, *face))


def render_alignment_previews(
    run_dir: Path,
    alignment: CameraMeshAlignmentResult,
    diagnostics: CameraMeshAlignmentDiagnostics,
    previews: CameraMeshAlignmentPreviewManifest,
) -> None:
    """Regenerate deterministic diagnostic summaries without the GPU worker."""
    common = [
        f"status: {alignment.status}",
        f"accepted: {alignment.accepted}",
        f"scale: {alignment.transform.scale:.8g}",
        f"rotation degrees: {alignment.transform.rotation_degrees:.6f}",
        (
            "translation / scene diagonal: "
            f"{alignment.transform.translation_scene_diagonal_ratio:.6f}"
        ),
        (
            "validation median residual: "
            f"{alignment.baseline_validation_metrics.sparse_depth_residual_median} -> "
            f"{alignment.aligned_validation_metrics.sparse_depth_residual_median}"
        ),
        (
            "validation p90 residual: "
            f"{alignment.baseline_validation_metrics.sparse_depth_residual_p90} -> "
            f"{alignment.aligned_validation_metrics.sparse_depth_residual_p90}"
        ),
        f"diagnosis: {diagnostics.diagnosis}",
        "coordinates: arbitrary COLMAP frame; unoriented; scale ambiguous",
    ]
    for field, relative_path in previews.model_dump(mode="json").items():
        path = run_dir.joinpath(*Path(relative_path).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (1000, 600), (246, 247, 249))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 1000, 72), fill=(33, 41, 52))
        title = field.removesuffix("_path").replace("_", " ")
        draw.text((30, 25), title, fill=(255, 255, 255))
        y = 105
        for line in common:
            for wrapped in textwrap.wrap(line, width=105) or [""]:
                draw.text((40, y), wrapped, fill=(28, 35, 43))
                y += 28
            y += 8
        image.save(path, format="PNG", optimize=False)
