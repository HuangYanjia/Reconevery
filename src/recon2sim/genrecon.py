from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, cast

from PIL import Image, ImageDraw, ImageFont

from recon2sim.artifacts import (
    CameraReconstruction,
    GenReconCameraPackageManifest,
    GenReconRegisteredFrame,
    GlobalSceneMeshStatistics,
    IngestManifest,
    ObservationLineage,
)
from recon2sim.colmap import ColmapModel
from recon2sim.ir import CoordinateConvention

OFFICIAL_GENRECON_REPOSITORY: Literal["https://github.com/kasothaphie/GenRecon"] = (
    "https://github.com/kasothaphie/GenRecon"
)
OFFICIAL_GENRECON_COMMIT: Literal["eaf1468118d20469d17079a4a19737297d2ef87b"] = (
    "eaf1468118d20469d17079a4a19737297d2ef87b"
)
OFFICIAL_GENRECON_SUBMODULES = {
    "o-voxel/third_party/eigen": "21e4582d1739107337a03460c81412981130373e"
}
OFFICIAL_CHECKPOINT_URLS = {
    "sparse_structure": "https://kaldir.vc.cit.tum.de/genrecon/sparse_structure.pt",
    "shape_slat": "https://kaldir.vc.cit.tum.de/genrecon/shape_slat.pt",
    "texture_slat": "https://kaldir.vc.cit.tum.de/genrecon/texture_slat.pt",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_observation_lineage(
    manifest: IngestManifest,
    camera: CameraReconstruction,
    *,
    manifest_sha256: str,
    camera_reconstruction_sha256: str,
    segmentation_input_digest: str | None = None,
    genrecon_input_digest: str | None = None,
) -> ObservationLineage:
    if manifest.frame_sequence_digest is None:
        raise ValueError("ingest manifest does not contain a frame-sequence digest")
    if (
        camera.frame_sequence_digest is not None
        and camera.frame_sequence_digest != manifest.frame_sequence_digest
    ):
        raise ValueError("camera reconstruction frame-sequence digest does not match ingest")
    return ObservationLineage(
        manifest_sha256=manifest_sha256,
        frame_sequence_digest=manifest.frame_sequence_digest,
        frame_ids=[frame.frame_id for frame in manifest.frames],
        frame_paths=[frame.relative_path for frame in manifest.frames],
        frame_sha256_by_id={frame.frame_id: frame.sha256 for frame in manifest.frames},
        camera_reconstruction_sha256=camera_reconstruction_sha256,
        registered_frame_ids=camera.registered_frame_ids,
        unregistered_frame_ids=camera.unregistered_frame_ids,
        segmentation_input_digest=segmentation_input_digest,
        genrecon_input_digest=genrecon_input_digest,
    )


def _float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("COLMAP text serialization rejects non-finite values")
    return format(value, ".17g")


def _package_content_digest(paths: Iterable[Path]) -> str:
    entries = [(path.name, sha256_file(path)) for path in sorted(paths, key=lambda item: item.name)]
    return stable_digest(entries)


def export_colmap_text_package(
    *,
    model: ColmapModel,
    manifest: IngestManifest,
    camera: CameraReconstruction,
    output_dir: Path,
    selected_model_id: str,
    source_model_hashes: dict[str, str],
    manifest_sha256: str,
    camera_reconstruction_sha256: str,
) -> GenReconCameraPackageManifest:
    """Write one selected COLMAP model with deterministic camera/image identifiers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest.frame_sequence_digest is None:
        raise ValueError("ingest manifest does not contain a frame-sequence digest")
    by_name = {Path(frame.relative_path).name: frame for frame in manifest.frames}
    if len(by_name) != len(manifest.frames):
        raise ValueError("normalized frame basenames must be unique for GenRecon export")
    model_by_name = {image.name: image for image in model.images.values()}
    registered_names = {
        Path(frame.relative_path).name
        for frame in manifest.frames
        if frame.frame_id in set(camera.registered_frame_ids)
    }
    missing_model_images = registered_names - set(model_by_name)
    extra_model_images = set(model_by_name) - registered_names
    if missing_model_images or extra_model_images:
        raise ValueError(
            "selected COLMAP model and typed camera registration disagree: "
            f"missing={sorted(missing_model_images)}, extra={sorted(extra_model_images)}"
        )

    ordered_frames = [
        frame for frame in manifest.frames if frame.frame_id in set(camera.registered_frame_ids)
    ]
    ordered_images = [model_by_name[Path(frame.relative_path).name] for frame in ordered_frames]
    original_camera_ids = sorted({image.camera_id for image in ordered_images})
    camera_id_map = {
        original_id: package_id for package_id, original_id in enumerate(original_camera_ids, 1)
    }
    image_id_map = {
        image.image_id: package_id for package_id, image in enumerate(ordered_images, 1)
    }

    cameras_path = output_dir / "cameras.txt"
    camera_lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(original_camera_ids)}",
    ]
    for original_id in original_camera_ids:
        colmap_camera = model.cameras[original_id]
        camera_lines.append(
            " ".join(
                [
                    str(camera_id_map[original_id]),
                    colmap_camera.model_name,
                    str(colmap_camera.width),
                    str(colmap_camera.height),
                    *[_float(value) for value in colmap_camera.params],
                ]
            )
        )
    cameras_path.write_text("\n".join(camera_lines) + "\n", encoding="utf-8")

    images_path = output_dir / "images.txt"
    image_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(ordered_images)}",
    ]
    registered_records: list[GenReconRegisteredFrame] = []
    for frame, image in zip(ordered_frames, ordered_images, strict=True):
        package_image_id = image_id_map[image.image_id]
        image_lines.append(
            " ".join(
                [
                    str(package_image_id),
                    *[_float(value) for value in image.qvec_wxyz],
                    *[_float(value) for value in image.tvec],
                    str(camera_id_map[image.camera_id]),
                    Path(frame.relative_path).name,
                ]
            )
        )
        image_lines.append(
            " ".join(
                token
                for point in image.points2d
                for token in (_float(point.x), _float(point.y), str(point.point3d_id))
            )
        )
        registered_records.append(
            GenReconRegisteredFrame(
                frame_id=frame.frame_id,
                source_relative_path=frame.relative_path,
                package_image_name=Path(frame.relative_path).name,
                sha256=frame.sha256,
                original_colmap_image_id=image.image_id,
                package_image_id=package_image_id,
                original_colmap_camera_id=image.camera_id,
                package_camera_id=camera_id_map[image.camera_id],
            )
        )
    images_path.write_text("\n".join(image_lines) + "\n", encoding="utf-8")

    points_path = output_dir / "points3D.txt"
    point_lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        f"# Number of points: {len(model.points3d)}",
    ]
    for point_id in sorted(model.points3d):
        point = model.points3d[point_id]
        track_tokens: list[str] = []
        for element in point.track:
            track_image_id = image_id_map.get(element.image_id)
            if track_image_id is None:
                raise ValueError(f"point {point_id} track references image outside selected model")
            track_tokens.extend([str(track_image_id), str(element.point2d_index)])
        point_lines.append(
            " ".join(
                [
                    str(point.point3d_id),
                    *[_float(value) for value in point.xyz],
                    *[str(value) for value in point.rgb],
                    _float(point.error),
                    *track_tokens,
                ]
            )
        )
    points_path.write_text("\n".join(point_lines) + "\n", encoding="utf-8")

    registered_path = output_dir / "registered_frames.json"
    registered_payload = {
        "master_frame_ids": [frame.frame_id for frame in manifest.frames],
        "registered_frame_ids": camera.registered_frame_ids,
        "unregistered_frame_ids": camera.unregistered_frame_ids,
        "eligible_frame_ids": [frame.frame_id for frame in ordered_frames],
        "frames": [record.model_dump(mode="json") for record in registered_records],
    }
    registered_path.write_text(
        json.dumps(registered_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    content_digest = _package_content_digest(
        [cameras_path, images_path, points_path, registered_path]
    )
    return GenReconCameraPackageManifest(
        source_manifest_sha256=manifest_sha256,
        frame_sequence_digest=manifest.frame_sequence_digest,
        camera_reconstruction_sha256=camera_reconstruction_sha256,
        selected_model_id=selected_model_id,
        source_model_paths=source_model_hashes,
        master_frame_ids=[frame.frame_id for frame in manifest.frames],
        registered_frame_ids=camera.registered_frame_ids,
        unregistered_frame_ids=camera.unregistered_frame_ids,
        eligible_frame_ids=[frame.frame_id for frame in ordered_frames],
        registered_frames=registered_records,
        package_content_sha256=content_digest,
        coordinate_convention=camera.coordinate_convention,
    )


def validate_camera_package(root: Path, package: GenReconCameraPackageManifest) -> None:
    paths = [
        root / package.cameras_path,
        root / package.images_path,
        root / package.points3d_path,
        root / "camera/genrecon_package/registered_frames.json",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"GenRecon camera package is incomplete: {missing}")
    if _package_content_digest(paths) != package.package_content_sha256:
        raise ValueError("GenRecon camera package content hash does not match its manifest")


_PLY_SCALAR_TYPES: dict[str, tuple[str, int]] = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


@dataclass(frozen=True)
class MeshData:
    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]


def _read_binary_scalar(file: BinaryIO, type_name: str) -> int | float:
    definition = _PLY_SCALAR_TYPES.get(type_name)
    if definition is None:
        raise ValueError(f"unsupported PLY scalar type {type_name!r}")
    format_code, size = definition
    payload = file.read(size)
    if len(payload) != size:
        raise ValueError("truncated binary PLY")
    return cast(int | float, struct.unpack("<" + format_code, payload)[0])


def read_ply_mesh(path: Path) -> MeshData:
    with path.open("rb") as file:
        if file.readline() != b"ply\n":
            raise ValueError("mesh is not a PLY file")
        format_name = ""
        elements: list[tuple[str, int, list[tuple[str, ...]]]] = []
        current_properties: list[tuple[str, ...]] | None = None
        while True:
            raw = file.readline()
            if not raw:
                raise ValueError("PLY header has no end_header")
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError("PLY header is not ASCII") from exc
            parts = line.split()
            if not parts or parts[0] in {"comment", "obj_info"}:
                continue
            if parts[0] == "format":
                format_name = parts[1]
            elif parts[0] == "element":
                current_properties = []
                elements.append((parts[1], int(parts[2]), current_properties))
            elif parts[0] == "property":
                if current_properties is None:
                    raise ValueError("PLY property appears before an element")
                current_properties.append(tuple(parts[1:]))
            elif parts[0] == "end_header":
                break
        if format_name not in {"ascii", "binary_little_endian"}:
            raise ValueError(f"unsupported PLY format {format_name!r}")

        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        if format_name == "ascii":
            remaining = file.read().decode("ascii").splitlines()
            line_index = 0
            for element_name, count, properties in elements:
                for _ in range(count):
                    if line_index >= len(remaining):
                        raise ValueError("truncated ASCII PLY")
                    tokens = remaining[line_index].split()
                    line_index += 1
                    if element_name == "vertex":
                        scalar_names = [prop[-1] for prop in properties if prop[0] != "list"]
                        values = [float(token) for token in tokens[: len(scalar_names)]]
                        table = dict(zip(scalar_names, values, strict=True))
                        vertices.append((table["x"], table["y"], table["z"]))
                    elif element_name == "face":
                        face_count = int(tokens[0])
                        if face_count != 3:
                            raise ValueError("only triangular PLY faces are supported")
                        faces.append((int(tokens[1]), int(tokens[2]), int(tokens[3])))
            return MeshData(tuple(vertices), tuple(faces))

        for element_name, count, properties in elements:
            for _ in range(count):
                scalar_values: dict[str, int | float] = {}
                list_values: dict[str, list[int | float]] = {}
                for prop in properties:
                    if prop[0] == "list":
                        _, count_type, item_type, name = prop
                        item_count = int(_read_binary_scalar(file, count_type))
                        list_values[name] = [
                            _read_binary_scalar(file, item_type) for _ in range(item_count)
                        ]
                    else:
                        type_name, name = prop
                        scalar_values[name] = _read_binary_scalar(file, type_name)
                if element_name == "vertex":
                    vertices.append(
                        (
                            float(scalar_values["x"]),
                            float(scalar_values["y"]),
                            float(scalar_values["z"]),
                        )
                    )
                elif element_name == "face":
                    indices = list_values.get("vertex_indices")
                    if indices is None:
                        indices = next(iter(list_values.values()), [])
                    if len(indices) != 3:
                        raise ValueError("only triangular PLY faces are supported")
                    faces.append((int(indices[0]), int(indices[1]), int(indices[2])))
        return MeshData(tuple(vertices), tuple(faces))


def _mesh_topology(
    vertex_count: int, faces: tuple[tuple[int, int, int], ...]
) -> tuple[int, int, int]:
    parent = list(range(vertex_count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    edges: Counter[tuple[int, int]] = Counter()
    degenerate = 0
    used_vertices: set[int] = set()
    for face in faces:
        if len(set(face)) != 3:
            degenerate += 1
        for index in face:
            if index < 0 or index >= vertex_count:
                raise ValueError(f"mesh face references invalid vertex index {index}")
            used_vertices.add(index)
        union(face[0], face[1])
        union(face[1], face[2])
        for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (min(left, right), max(left, right))
            edges[edge] += 1
    components = len({find(index) for index in used_vertices})
    non_manifold = sum(count > 2 for count in edges.values())
    return max(components, 1), degenerate, non_manifold


def inspect_glb(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()
    if len(payload) < 20:
        raise ValueError("GLB is too short")
    magic, version, total_length = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or total_length != len(payload):
        raise ValueError("GLB header is invalid")
    offset = 12
    json_document: dict[str, object] | None = None
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise ValueError("GLB chunk header is truncated")
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        end = offset + chunk_length
        if end > len(payload):
            raise ValueError("GLB chunk escapes the file")
        if chunk_type == 0x4E4F534A:
            try:
                parsed = json.loads(payload[offset:end].rstrip(b" \t\r\n\0").decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("GLB JSON chunk is invalid") from exc
            if not isinstance(parsed, dict):
                raise ValueError("GLB JSON root is not an object")
            json_document = parsed
        offset = end
    if json_document is None or not isinstance(json_document.get("asset"), dict):
        raise ValueError("GLB has no JSON asset metadata")
    materials = json_document.get("materials", [])
    textures = json_document.get("textures", [])
    return (
        len(materials) if isinstance(materials, list) else 0,
        len(textures) if isinstance(textures, list) else 0,
    )


def inspect_global_mesh(mesh_path: Path, glb_path: Path) -> GlobalSceneMeshStatistics:
    mesh = read_ply_mesh(mesh_path)
    if not mesh.vertices or not mesh.faces:
        raise ValueError("global mesh must contain vertices and faces")
    if any(not math.isfinite(value) for vertex in mesh.vertices for value in vertex):
        raise ValueError("global mesh contains non-finite coordinates")
    minimum = (
        min(vertex[0] for vertex in mesh.vertices),
        min(vertex[1] for vertex in mesh.vertices),
        min(vertex[2] for vertex in mesh.vertices),
    )
    maximum = (
        max(vertex[0] for vertex in mesh.vertices),
        max(vertex[1] for vertex in mesh.vertices),
        max(vertex[2] for vertex in mesh.vertices),
    )
    extent = (
        maximum[0] - minimum[0],
        maximum[1] - minimum[1],
        maximum[2] - minimum[2],
    )
    if max(extent) <= 0:
        raise ValueError("global mesh bounding box is degenerate")
    components, degenerate, non_manifold = _mesh_topology(len(mesh.vertices), mesh.faces)
    materials, textures = inspect_glb(glb_path)
    return GlobalSceneMeshStatistics(
        vertex_count=len(mesh.vertices),
        face_count=len(mesh.faces),
        disconnected_components=components,
        degenerate_faces=degenerate,
        non_manifold_edge_count=non_manifold,
        finite_coordinates=True,
        bounding_box_min=minimum,
        bounding_box_max=maximum,
        bounding_box_extent=extent,
        material_count=materials,
        texture_count=textures,
        glb_parse_status="valid",
    )


def _normalized_points(
    points: Iterable[tuple[float, float]],
    width: int,
    height: int,
    padding: int = 30,
) -> list[tuple[int, int]]:
    values = list(points)
    if not values:
        return []
    minimum_x = min(point[0] for point in values)
    maximum_x = max(point[0] for point in values)
    minimum_y = min(point[1] for point in values)
    maximum_y = max(point[1] for point in values)
    span_x = maximum_x - minimum_x or 1.0
    span_y = maximum_y - minimum_y or 1.0
    return [
        (
            padding + round((x - minimum_x) / span_x * (width - 2 * padding)),
            height - padding - round((y - minimum_y) / span_y * (height - 2 * padding)),
        )
        for x, y in values
    ]


def render_camera_trajectory_preview(
    camera: CameraReconstruction,
    sparse_points: Iterable[tuple[float, float, float]],
    output_path: Path,
) -> None:
    font = ImageFont.load_default()
    trajectory = Image.new("RGB", (1000, 760), "white")
    trajectory_draw = ImageDraw.Draw(trajectory)
    sparse_points_list = list(sparse_points)
    point_xy = [(point[0], point[1]) for point in sparse_points_list]
    camera_xy = [
        (
            pose.transform_world_from_camera.translation[0],
            pose.transform_world_from_camera.translation[1],
        )
        for pose in camera.poses
    ]
    combined = _normalized_points([*point_xy, *camera_xy], 960, 650)
    sparse_count = len(point_xy)
    for x, y in combined[:sparse_count]:
        trajectory_draw.point((x + 20, y + 50), fill="#6c757d")
    camera_pixels = [(x + 20, y + 50) for x, y in combined[sparse_count:]]
    if len(camera_pixels) > 1:
        trajectory_draw.line(camera_pixels, fill="#005f73", width=3)
    for index, pixel in enumerate(camera_pixels):
        trajectory_draw.ellipse(
            (pixel[0] - 4, pixel[1] - 4, pixel[0] + 4, pixel[1] + 4),
            fill="#ee9b00",
        )
        trajectory_draw.text((pixel[0] + 5, pixel[1]), str(index), fill="black", font=font)
    trajectory_draw.text(
        (20, 15),
        (
            f"COLMAP sparse points: {len(point_xy)} | registered cameras: {len(camera_xy)} | "
            "arbitrary axes, unoriented, scale ambiguous"
        ),
        fill="#9b2226",
        font=font,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory.save(output_path, format="PNG", optimize=False)


def render_global_previews(
    *,
    root: Path,
    manifest: IngestManifest,
    camera: CameraReconstruction,
    sparse_points: Iterable[tuple[float, float, float]],
    mesh_path: Path,
) -> list[str]:
    preview_dir = root / "reconstruction/global/previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    mesh = read_ply_mesh(mesh_path)

    global_preview = Image.new("RGB", (1200, 430), "white")
    draw = ImageDraw.Draw(global_preview)
    projections = ((0, 1, "X / Y"), (0, 2, "X / Z"), (1, 2, "Y / Z"))
    colors = ("#006d77", "#9b2226", "#5f0f40")
    for panel, (first_axis, second_axis, label) in enumerate(projections):
        x_offset = panel * 400
        projected = _normalized_points(
            ((vertex[first_axis], vertex[second_axis]) for vertex in mesh.vertices),
            380,
            360,
        )
        for face in mesh.faces:
            polygon = [(projected[index][0] + x_offset, projected[index][1] + 35) for index in face]
            draw.line([*polygon, polygon[0]], fill=colors[panel], width=1)
        draw.text((x_offset + 12, 10), label, fill="black", font=font)
    draw.text(
        (12, 410),
        "COLMAP arbitrary axes - unoriented - scale ambiguous",
        fill="#9b2226",
        font=font,
    )
    global_path = preview_dir / "global_scene_preview.png"
    global_preview.save(global_path, format="PNG", optimize=False)

    trajectory_path = preview_dir / "camera_trajectory_and_sparse_points.png"
    render_camera_trajectory_preview(camera, sparse_points, trajectory_path)

    contact = Image.new("RGB", (1200, 700), "#f5f5f5")
    contact_draw = ImageDraw.Draw(contact)
    selected = manifest.frames[:4]
    tile_width = 300
    for index, frame in enumerate(selected):
        with Image.open(root / frame.relative_path) as source:
            image = source.convert("RGB")
            image.thumbnail((tile_width - 16, 260))
            x = index * tile_width + (tile_width - image.width) // 2
            y = 35 + (260 - image.height) // 2
            contact.paste(image, (x, y))
        contact_draw.text((index * tile_width + 8, 10), frame.frame_id, fill="black", font=font)
    with Image.open(global_path) as source_preview:
        preview = source_preview.convert("RGB")
        preview.thumbnail((1160, 350))
        contact.paste(preview, ((1200 - preview.width) // 2, 330))
    contact_draw.text(
        (20, 310),
        "Normalized inputs above; generated global visual geometry below",
        fill="black",
        font=font,
    )
    contact_path = preview_dir / "input_vs_geometry_contact_sheet.png"
    contact.save(contact_path, format="PNG", optimize=False)
    return [
        global_path.relative_to(root).as_posix(),
        trajectory_path.relative_to(root).as_posix(),
        contact_path.relative_to(root).as_posix(),
    ]


def read_colmap_text_points(path: Path) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8:
                raise ValueError(f"malformed COLMAP points3D text row in {path}")
            point = (float(parts[1]), float(parts[2]), float(parts[3]))
            if any(not math.isfinite(value) for value in point):
                raise ValueError("COLMAP points3D text contains non-finite coordinates")
            points.append(point)
    return points


def coordinate_metadata_is_raw_colmap(convention: CoordinateConvention) -> bool:
    payload = convention.model_dump(mode="json")
    return payload == {
        "world_frame": "colmap_arbitrary",
        "alignment_status": "unoriented",
        "camera_axes": "x_right_y_down_z_forward",
        "handedness": "right",
        "linear_units": "arbitrary_units",
        "scale_status": "scale_ambiguous",
        "quaternion_order": "xyzw",
        "transform_direction": "world_from_camera",
    }


__all__ = [
    "OFFICIAL_CHECKPOINT_URLS",
    "OFFICIAL_GENRECON_COMMIT",
    "OFFICIAL_GENRECON_REPOSITORY",
    "OFFICIAL_GENRECON_SUBMODULES",
    "MeshData",
    "build_observation_lineage",
    "coordinate_metadata_is_raw_colmap",
    "export_colmap_text_package",
    "inspect_glb",
    "inspect_global_mesh",
    "read_ply_mesh",
    "read_colmap_text_points",
    "render_camera_trajectory_preview",
    "render_global_previews",
    "sha256_file",
    "stable_digest",
    "validate_camera_package",
]
