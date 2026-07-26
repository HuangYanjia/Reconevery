from __future__ import annotations

import hashlib
import json
import math
import re
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from dense_mvs_worker.colmap_version import inspect_colmap
from dense_mvs_worker.dense_io import read_array, read_consistency_graph
from dense_mvs_worker.patchmatch import write_patch_match_config
from dense_mvs_worker.version import __version__


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gpu_memory_bytes(pid: int) -> int:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            try:
                return int(fields[1]) * 1024 * 1024
            except ValueError:
                return 0
    return 0


def run(command: list[str], log_root: Path, name: str) -> tuple[int, float, int]:
    started = time.monotonic()
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{name}.stdout.log"
    stderr_path = log_root / f"{name}.stderr.log"
    peak_gpu = 0
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True)
        while process.poll() is None:
            peak_gpu = max(peak_gpu, _gpu_memory_bytes(process.pid))
            time.sleep(5.0)
        return_code = process.returncode
    if return_code != 0:
        error = stderr_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"{name} failed with code {return_code}: {error[-4000:]}")
    return return_code, time.monotonic() - started, peak_gpu


def build_environment(executable: str) -> tuple[str | None, str | None]:
    cuda_version = None
    try:
        linked = subprocess.run(
            ["ldd", executable],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        ).stdout
        match = re.search(r"/cuda-([0-9]+\.[0-9]+)/", linked)
        cuda_version = match.group(1) if match else None
    except (OSError, subprocess.TimeoutExpired):
        pass
    compiler = None
    try:
        result = subprocess.run(
            ["c++", "--version"], capture_output=True, text=True, check=False, timeout=30
        )
        compiler = result.stdout.splitlines()[0] if result.stdout.splitlines() else None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return cuda_version, compiler


def parse_cameras(path: Path) -> dict[int, dict[str, object]]:
    cameras: dict[int, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        camera_id, model, width, height = (
            int(fields[0]),
            fields[1],
            int(fields[2]),
            int(fields[3]),
        )
        params = [float(value) for value in fields[4:]]
        if model == "PINHOLE":
            intrinsics = params[:4]
        elif model == "SIMPLE_PINHOLE":
            intrinsics = [params[0], params[0], params[1], params[2]]
        else:
            raise ValueError(f"unexpected undistorted COLMAP camera model {model}")
        cameras[camera_id] = {
            "model": "PINHOLE",
            "width": width,
            "height": height,
            "intrinsics": intrinsics,
        }
    return cameras


def parse_images(path: Path) -> list[tuple[int, int, str]]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    result: list[tuple[int, int, str]] = []
    for index in range(0, len(lines), 2):
        fields = lines[index].split()
        result.append((int(fields[0]), int(fields[8]), fields[9]))
    return result


def source_camera(camera: dict[str, Any]) -> tuple[str, list[float], list[float]]:
    intrinsics = camera["intrinsics"]
    return (
        camera["model"].upper(),
        [intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]],
        list(intrinsics.get("distortion", [])),
    )


def source_distortion(model: str, values: list[float]) -> np.ndarray:
    if model in {"PINHOLE", "SIMPLE_PINHOLE"}:
        return np.zeros(5, dtype=np.float64)
    if model == "SIMPLE_RADIAL":
        return np.array([values[0], 0, 0, 0, 0], dtype=np.float64)
    if model == "RADIAL":
        return np.array([values[0], values[1], 0, 0, 0], dtype=np.float64)
    if model == "OPENCV":
        padded = [*values, 0, 0, 0, 0]
        return np.array([padded[0], padded[1], padded[2], padded[3], 0], dtype=np.float64)
    raise ValueError(f"unsupported camera model {model}")


def remap_record(
    *,
    frame_id: str,
    source_path: Path,
    dense_path: Path,
    source_model_name: str,
    source_intrinsics: list[float],
    source_distortion_values: list[float],
    dense_camera: dict[str, object],
) -> dict[str, object]:
    source_image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    dense_image = cv2.imread(str(dense_path), cv2.IMREAD_COLOR)
    if source_image is None or dense_image is None:
        raise ValueError(f"could not read source or dense image for {frame_id}")
    source_height, source_width = source_image.shape[:2]
    dense_height, dense_width = dense_image.shape[:2]
    fx, fy, cx, cy = source_intrinsics
    dense_intrinsics = [float(value) for value in dense_camera["intrinsics"]]
    k_source = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    d_source = source_distortion(source_model_name, source_distortion_values)
    dfx, dfy, dcx, dcy = dense_intrinsics
    k_dense = np.array([[dfx, 0, dcx], [0, dfy, dcy], [0, 0, 1]], dtype=np.float64)
    map_x, map_y = cv2.initUndistortRectifyMap(
        k_source,
        d_source,
        None,
        k_dense,
        (dense_width, dense_height),
        cv2.CV_32FC1,
    )
    independently_remapped = cv2.remap(
        source_image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    error = float(
        np.mean(np.abs(independently_remapped.astype(np.float32) - dense_image.astype(np.float32)))
    )
    digest = hashlib.sha256(map_x.tobytes() + map_y.tobytes()).hexdigest()
    return {
        "frame_id": frame_id,
        "source_camera_model": source_model_name,
        "source_intrinsics": source_intrinsics,
        "source_distortion": source_distortion_values,
        "source_dimensions": [source_width, source_height],
        "dense_camera_model": "PINHOLE",
        "dense_intrinsics": dense_intrinsics,
        "dense_dimensions": [dense_width, dense_height],
        "roi_xywh": [0, 0, dense_width, dense_height],
        "map_hash": digest,
        "source_rgb_hash": sha256(source_path),
        "dense_rgb_hash": sha256(dense_path),
        "rgb_remap_mean_absolute_error": error,
        "mask_resampling": "nearest",
    }


def ply_statistics(path: Path) -> tuple[int, list[float], list[float], np.ndarray]:
    vertices: list[list[float]] = []
    with path.open("rb") as file:
        vertex_count = 0
        fmt = ""
        vertex_properties: list[tuple[str, str]] = []
        reading_vertices = False
        while True:
            line = file.readline().decode("ascii").strip()
            if line.startswith("format "):
                fmt = line.split()[1]
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[2])
                reading_vertices = True
            elif line.startswith("element "):
                reading_vertices = False
            elif reading_vertices and line.startswith("property "):
                fields = line.split()
                if fields[1] == "list":
                    raise ValueError("list properties are unsupported in fused PLY vertices")
                vertex_properties.append((fields[2], fields[1]))
            if line == "end_header":
                break
        if fmt == "ascii":
            for _ in range(vertex_count):
                fields = file.readline().split()
                vertices.append([float(fields[0]), float(fields[1]), float(fields[2])])
            points = np.asarray(vertices)
        elif fmt == "binary_little_endian":
            type_codes = {
                "char": "i1",
                "uchar": "u1",
                "short": "<i2",
                "ushort": "<u2",
                "int": "<i4",
                "uint": "<u4",
                "float": "<f4",
                "double": "<f8",
            }
            try:
                dtype = np.dtype(
                    [(name, type_codes[property_type]) for name, property_type in vertex_properties]
                )
            except KeyError as exc:
                raise ValueError(f"unsupported fused PLY property type {exc.args[0]}") from exc
            records = np.fromfile(file, dtype=dtype, count=vertex_count)
            if records.size != vertex_count or not {"x", "y", "z"}.issubset(
                records.dtype.names or ()
            ):
                raise ValueError("fused PLY has invalid binary vertex data")
            points = np.column_stack((records["x"], records["y"], records["z"]))
        else:
            raise ValueError(f"unsupported fused PLY format {fmt!r}")
    if points.size == 0 or not np.isfinite(points).all():
        raise ValueError("fused dense point cloud is empty or non-finite")
    return vertex_count, points.min(axis=0).tolist(), points.max(axis=0).tolist(), points


def _labeled_tile(image: np.ndarray, label: str) -> Image.Image:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    tile = Image.fromarray(image.astype(np.uint8), mode="RGB")
    tile.thumbnail((300, 190), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (300, 220), "white")
    canvas.paste(tile, ((300 - tile.width) // 2, 26))
    ImageDraw.Draw(canvas).text((8, 6), label, fill=(20, 25, 30))
    return canvas


def contact_sheet(path: Path, title: str, tiles: list[tuple[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = 3
    rows = max(1, math.ceil(len(tiles) / columns))
    image = Image.new("RGB", (columns * 300, 42 + rows * 220), (238, 240, 242))
    draw = ImageDraw.Draw(image)
    draw.text((14, 14), title, fill=(20, 25, 30))
    for index, (label, tile) in enumerate(tiles):
        image.paste(
            _labeled_tile(tile, label), ((index % columns) * 300, 42 + index // columns * 220)
        )
    image.save(path)


def depth_preview(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth[valid], [2, 98])
        if high > low:
            normalized[valid] = np.clip(255 * (depth[valid] - low) / (high - low), 0, 255).astype(
                np.uint8
            )
    colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def point_cloud_preview(points: np.ndarray) -> np.ndarray:
    sample = points[:: max(1, len(points) // 200_000)]
    centered = sample - np.median(sample, axis=0)
    _, eigenvectors = np.linalg.eigh(np.cov(centered.T))
    projected = centered @ eigenvectors[:, -2:]
    depth = centered @ eigenvectors[:, -3]
    low = np.percentile(projected, 1, axis=0)
    high = np.percentile(projected, 99, axis=0)
    extent = np.maximum(high - low, 1e-8)
    pixels = np.clip((projected - low) / extent * np.array([879, 499]), 0, [879, 499]).astype(int)
    colors = cv2.applyColorMap(
        np.clip(255 * (depth - depth.min()) / max(float(np.ptp(depth)), 1e-8), 0, 255).astype(
            np.uint8
        ),
        cv2.COLORMAP_TURBO,
    ).reshape(-1, 3)
    image = np.full((500, 880, 3), 245, dtype=np.uint8)
    rows = 499 - pixels[:, 1]
    columns = pixels[:, 0]
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            target_rows = np.clip(rows + row_offset, 0, 499)
            target_columns = np.clip(columns + column_offset, 0, 879)
            image[target_rows, target_columns] = colors[:, ::-1]
    return image


def infer(request_path: Path, input_root: Path, output_dir: Path) -> dict[str, object]:
    started = time.monotonic()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    version, _, commit_prefix = inspect_colmap(request["executable"])
    if version != request["official_colmap_version"]:
        raise RuntimeError(
            f"COLMAP {version} does not match pin {request['official_colmap_version']}"
        )
    if not request["official_colmap_commit"].startswith(commit_prefix):
        raise RuntimeError(
            f"COLMAP binary commit {commit_prefix} does not match "
            f"pin {request['official_colmap_commit']}"
        )
    manifest = json.loads((input_root / request["manifest_path"]).read_text(encoding="utf-8"))
    camera = json.loads(
        (input_root / request["camera_reconstruction_path"]).read_text(encoding="utf-8")
    )
    frame_by_id = {frame["frame_id"]: frame for frame in manifest["frames"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = output_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    selected = [
        input_root / item["relative_path"] for item in request["selected_sparse_model_files"]
    ]
    sparse_root = selected[0].parent
    registered_image_parents = {
        (input_root / request["normalized_frame_paths"][frame_id]).parent
        for frame_id in request["registered_frame_ids"]
    }
    if len(registered_image_parents) != 1:
        raise ValueError(
            "registered normalized images must share one directory for COLMAP dense MVS"
        )
    image_root = registered_image_parents.pop()
    log_root = output_dir / "raw" / "logs"
    executable = request["executable"]
    image_command = [
        executable,
        "image_undistorter",
        "--image_path",
        str(image_root),
        "--input_path",
        str(sparse_root),
        "--output_path",
        str(workspace),
        "--output_type",
        "COLMAP",
        "--max_image_size",
        str(request["undistortion_configuration"]["max_image_size"]),
    ]
    _, undistort_seconds, undistort_peak_gpu = run(image_command, log_root, "image_undistorter")
    sparse_txt = output_dir / "raw" / "sparse_txt"
    sparse_txt.mkdir(parents=True, exist_ok=True)
    converter = [
        executable,
        "model_converter",
        "--input_path",
        str(workspace / "sparse"),
        "--output_path",
        str(sparse_txt),
        "--output_type",
        "TXT",
    ]
    run(converter, log_root, "model_converter")
    cameras = parse_cameras(sparse_txt / "cameras.txt")
    images = parse_images(sparse_txt / "images.txt")
    registered = request["registered_frame_ids"]
    normalized_by_name = {
        Path(request["normalized_frame_paths"][frame_id]).name: frame_id for frame_id in registered
    }
    frame_records = []
    undistortion_records = []
    source_model_name, source_intrinsics, source_distortion_values = source_camera(camera)
    for image_id, camera_id, image_name in images:
        frame_id = normalized_by_name.get(Path(image_name).name)
        if frame_id is None:
            raise ValueError(f"dense workspace invented or renamed image {image_name!r}")
        frame = frame_by_id[frame_id]
        dense_name = Path(image_name).name
        dense_path = workspace / "images" / image_name
        if not dense_path.is_file():
            dense_path = workspace / "images" / dense_name
        dense_camera = cameras[camera_id]
        record = remap_record(
            frame_id=frame_id,
            source_path=input_root / frame["relative_path"],
            dense_path=dense_path,
            source_model_name=source_model_name,
            source_intrinsics=source_intrinsics,
            source_distortion_values=source_distortion_values,
            dense_camera=dense_camera,
        )
        tolerance = float(request["undistortion_configuration"]["rgb_remap_tolerance"])
        if record["rgb_remap_mean_absolute_error"] > tolerance:
            raise RuntimeError(
                f"independent undistortion differs from COLMAP for {frame_id}: "
                f"{record['rgb_remap_mean_absolute_error']:.3f} > {tolerance}"
            )
        undistortion_records.append(record)
        frame_records.append(
            {
                "frame_id": frame_id,
                "source_relative_path": frame["relative_path"],
                "source_sha256": frame["sha256"],
                "colmap_image_id": image_id,
                "workspace_filename": dense_path.relative_to(input_root).as_posix(),
                "source_dimensions": [frame["width"], frame["height"]],
                "dense_dimensions": record["dense_dimensions"],
                "dense_camera_id": camera_id,
                "dense_camera_model": "PINHOLE",
                "dense_intrinsics": dense_camera["intrinsics"],
            }
        )
    order = {frame_id: index for index, frame_id in enumerate(registered)}
    frame_records.sort(key=lambda item: order[item["frame_id"]])
    undistortion_records.sort(key=lambda item: order[item["frame_id"]])
    if [item["frame_id"] for item in frame_records] != registered:
        raise RuntimeError("COLMAP dense workspace frame set/order differs from request")
    patch_config = workspace / "stereo" / "patch-match.cfg"
    if not patch_config.is_file():
        raise RuntimeError("image_undistorter did not create stereo/patch-match.cfg")
    write_patch_match_config(
        patch_config,
        ordered_frame_ids=registered,
        filename_by_frame={
            item["frame_id"]: Path(item["workspace_filename"]).name for item in frame_records
        },
        mode=request["patchmatch_configuration"]["source_view_selection"],
        explicit_source_ids=request["patchmatch_configuration"]["source_view_ids"],
        neighbor_count=int(request["patchmatch_configuration"]["sequential_neighbor_count"]),
    )
    patch_command = [
        executable,
        "patch_match_stereo",
        "--workspace_path",
        str(workspace),
        "--workspace_format",
        "COLMAP",
        "--PatchMatchStereo.geom_consistency",
        str(request["patchmatch_configuration"]["geom_consistency"]).lower(),
        "--PatchMatchStereo.write_consistency_graph",
        "true",
        "--PatchMatchStereo.max_image_size",
        str(request["undistortion_configuration"]["max_image_size"]),
        "--PatchMatchStereo.cache_size",
        str(request["patchmatch_configuration"]["cache_size_gb"]),
        "--PatchMatchStereo.gpu_index",
        "0" if request["patchmatch_configuration"]["use_gpu"] else "-1",
    ]
    _, patch_seconds, patch_peak_gpu = run(patch_command, log_root, "patch_match_stereo")
    fused = output_dir / "fused.ply"
    fusion_cfg = request["fusion_configuration"]
    fusion_command = [
        executable,
        "stereo_fusion",
        "--workspace_path",
        str(workspace),
        "--workspace_format",
        "COLMAP",
        "--input_type",
        "geometric",
        "--output_path",
        str(fused),
        "--StereoFusion.min_num_pixels",
        str(fusion_cfg["min_num_pixels"]),
        "--StereoFusion.max_reproj_error",
        str(fusion_cfg["max_reproj_error"]),
        "--StereoFusion.max_depth_error",
        str(fusion_cfg["max_depth_error"]),
        "--StereoFusion.max_normal_error",
        str(fusion_cfg["max_normal_error"]),
        "--StereoFusion.check_num_images",
        str(fusion_cfg["check_num_images"]),
    ]
    _, fusion_seconds, fusion_peak_gpu = run(fusion_command, log_root, "stereo_fusion")
    records = []
    failed = []
    depth_tiles: list[tuple[str, np.ndarray]] = []
    normal_tiles: list[tuple[str, np.ndarray]] = []
    consistency_tiles: list[tuple[str, np.ndarray]] = []
    coverage_tiles: list[tuple[str, np.ndarray]] = []
    names_by_frame = {
        item["frame_id"]: Path(item["workspace_filename"]).name for item in frame_records
    }
    for frame_id in registered:
        name = names_by_frame[frame_id]
        depth_path = workspace / "stereo" / "depth_maps" / f"{name}.geometric.bin"
        normal_path = workspace / "stereo" / "normal_maps" / f"{name}.geometric.bin"
        graph_path = workspace / "stereo" / "consistency_graphs" / f"{name}.geometric.bin"
        if not depth_path.is_file() or not normal_path.is_file() or not graph_path.is_file():
            failed.append(frame_id)
            continue
        depth = read_array(depth_path, 1)
        normal = read_array(normal_path, 3)
        graph, source_lists = read_consistency_graph(graph_path, len(registered))
        if depth.shape != normal.shape[:2] or depth.shape != graph.shape:
            raise ValueError(f"dense map dimensions disagree for {frame_id}")
        valid_depth = np.isfinite(depth) & (depth > 0)
        finite_normal = np.isfinite(normal).all(axis=2)
        values = depth[valid_depth]
        if values.size:
            percentiles = {
                key: float(value)
                for key, value in zip(
                    ("p10", "p50", "p90"), np.percentile(values, [10, 50, 90]), strict=True
                )
            }
        else:
            percentiles = {"p10": 0.0, "p50": 0.0, "p90": 0.0}
        source_counts = graph[graph > 0]
        depth_tiles.append((frame_id, depth_preview(depth)))
        normal_rgb = np.clip((normal + 1.0) * 127.5, 0, 255).astype(np.uint8)
        normal_rgb[~finite_normal] = 0
        normal_tiles.append((frame_id, normal_rgb))
        consistency_color = cv2.applyColorMap(
            np.clip(graph * (255 / max(1, len(registered) - 1)), 0, 255).astype(np.uint8),
            cv2.COLORMAP_VIRIDIS,
        )
        consistency_tiles.append((frame_id, cv2.cvtColor(consistency_color, cv2.COLOR_BGR2RGB)))
        dense_rgb = cv2.imread(
            str(input_root / frame_records[order[frame_id]]["workspace_filename"]),
            cv2.IMREAD_COLOR,
        )
        if dense_rgb is not None:
            coverage_tiles.append(
                (
                    f"{frame_id} valid={float(valid_depth.mean()):.3f}",
                    cv2.cvtColor(dense_rgb, cv2.COLOR_BGR2RGB),
                )
            )
        records.append(
            {
                "frame_id": frame_id,
                "depth_path": depth_path.relative_to(input_root).as_posix(),
                "normal_path": normal_path.relative_to(input_root).as_posix(),
                "consistency_graph_path": graph_path.relative_to(input_root).as_posix(),
                "dimensions": [depth.shape[1], depth.shape[0]],
                "depth_channels": 1,
                "normal_channels": 3,
                "positive_finite_depth_count": int(valid_depth.sum()),
                "valid_depth_ratio": float(valid_depth.mean()),
                "depth_percentiles": percentiles,
                "finite_normal_ratio": float(finite_normal.mean()),
                "consistency_valid_pixel_count": int((graph > 0).sum()),
                "mean_consistency_source_count": (
                    float(source_counts.mean()) if source_counts.size else 0.0
                ),
                "median_consistency_source_count": (
                    float(np.median(source_counts)) if source_counts.size else 0.0
                ),
                "source_view_ids": sorted({source for values in source_lists for source in values}),
                "depth_sha256": sha256(depth_path),
                "normal_sha256": sha256(normal_path),
                "consistency_sha256": sha256(graph_path),
                "warnings": [],
            }
        )
    point_count, bounds_min, bounds_max, fused_points = ply_statistics(fused)
    diagonal = float(np.linalg.norm(np.asarray(bounds_max) - np.asarray(bounds_min)))
    if point_count <= 0 or not math.isfinite(diagonal) or diagonal <= 0:
        raise RuntimeError("stereo_fusion produced empty or degenerate geometry")
    selected_hashes = {
        Path(item["relative_path"]).name: item["sha256"]
        for item in request["selected_sparse_model_files"]
    }
    write_json(
        output_dir / "workspace_manifest.json",
        {
            "schema_version": "0.1.0",
            "manifest_sha256": request["manifest_sha256"],
            "frame_sequence_digest": request["frame_sequence_digest"],
            "camera_reconstruction_sha256": request["camera_reconstruction_sha256"],
            "selected_sparse_model_hashes": selected_hashes,
            "registered_frame_ids": registered,
            "frames": frame_records,
            "patch_match_config_path": patch_config.relative_to(input_root).as_posix(),
            "patch_match_config_sha256": sha256(patch_config),
            "workspace_path": "reconstruction/dense/workspace",
            "coordinate_convention": camera["coordinate_convention"],
        },
    )
    write_json(
        output_dir / "undistortion_manifest.json",
        {
            "schema_version": "0.1.0",
            "policy": "official_colmap_image_undistorter",
            "records": undistortion_records,
            "rgb_remap_tolerance": request["undistortion_configuration"]["rgb_remap_tolerance"],
        },
    )
    write_json(
        output_dir / "depth_manifest.json",
        {
            "schema_version": "0.1.0",
            "map_type": "geometric",
            "records": records,
            "failed_frame_ids": failed,
        },
    )
    write_json(
        output_dir / "fusion.json",
        {
            "schema_version": "0.1.0",
            "fused_point_cloud_path": "reconstruction/dense/fused.ply",
            "fused_point_cloud_sha256": sha256(fused),
            "point_count": point_count,
            "normal_count": point_count,
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
            "scene_diagonal_arbitrary_units": diagonal,
            "coordinate_convention": camera["coordinate_convention"],
            "scale_status": "scale_ambiguous",
        },
    )
    total = time.monotonic() - started
    peak_gpu = max(undistort_peak_gpu, patch_peak_gpu, fusion_peak_gpu)
    peak_host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    write_json(
        output_dir / "diagnostics.json",
        {
            "schema_version": "0.1.0",
            "registered_frame_count": len(registered),
            "successful_depth_map_count": len(records),
            "failed_depth_map_count": len(failed),
            "fused_point_count": point_count,
            "image_undistortion_seconds": undistort_seconds,
            "patchmatch_seconds": patch_seconds,
            "fusion_seconds": fusion_seconds,
            "total_runtime_seconds": total,
            "peak_gpu_memory_bytes": peak_gpu,
            "peak_host_memory_bytes": peak_host,
            "warnings": [],
        },
    )
    commands = {
        "image_undistorter": image_command,
        "patch_match_stereo": patch_command,
        "stereo_fusion": fusion_command,
    }
    cuda_version, compiler = build_environment(executable)
    write_json(
        output_dir / "worker_manifest.json",
        {
            "schema_version": "0.1.0",
            "worker_version": __version__,
            "official_colmap_repository": request["official_colmap_repository"],
            "official_colmap_version": request["official_colmap_version"],
            "official_colmap_commit": request["official_colmap_commit"],
            "colmap_license": "BSD-3-Clause",
            "build_configuration": {
                "source": "official_release",
                "verified_binary_commit_prefix": commit_prefix,
            },
            "cuda_version": cuda_version,
            "compiler": compiler,
            "request_sha256": sha256(request_path),
            "manifest_sha256": request["manifest_sha256"],
            "frame_sequence_digest": request["frame_sequence_digest"],
            "camera_reconstruction_sha256": request["camera_reconstruction_sha256"],
            "registered_frame_ids": registered,
            "command_arguments": commands,
            "return_codes": {name: 0 for name in commands},
            "runtime_seconds": total,
            "peak_gpu_memory_bytes": peak_gpu,
            "peak_host_memory_bytes": peak_host,
            "raw_output_paths": [
                path.relative_to(input_root).as_posix() for path in sorted(log_root.glob("*"))
            ],
            "warnings": [],
        },
    )
    contact_sheet(
        output_dir / "previews" / "depth_contact_sheet.png",
        "Geometric depth maps (arbitrary COLMAP units)",
        depth_tiles,
    )
    contact_sheet(
        output_dir / "previews" / "normal_contact_sheet.png",
        "Geometric normal maps",
        normal_tiles,
    )
    contact_sheet(
        output_dir / "previews" / "consistency_contact_sheet.png",
        "Geometric consistency source counts",
        consistency_tiles,
    )
    contact_sheet(
        output_dir / "previews" / "camera_dense_coverage.png",
        "Official COLMAP undistorted RGB and valid-depth ratios",
        coverage_tiles,
    )
    point_preview = Image.fromarray(point_cloud_preview(fused_points), mode="RGB")
    point_canvas = Image.new("RGB", (880, 540), (245, 245, 245))
    point_canvas.paste(point_preview, (0, 40))
    ImageDraw.Draw(point_canvas).text(
        (14, 14),
        "Fused measured dense points (PCA diagnostic view; arbitrary axes and scale)",
        fill=(20, 25, 30),
    )
    point_canvas.save(output_dir / "previews" / "fused_point_cloud.png")
    return {"registered_frames": len(registered), "depth_maps": len(records)}
