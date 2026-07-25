from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from PIL import Image


def resolve_worker_output_directory(root: Path, output_dir: Path) -> Path:
    root = root.resolve()
    candidate = output_dir if output_dir.is_absolute() else root / output_dir
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(f"worker output directory escapes the attempt workspace: {output_dir}")
    return resolved


@contextmanager
def prepared_video_frames(
    root: Path,
    frame_order: list[str],
    frame_paths: list[str],
    frame_dimensions: dict[str, tuple[int, int]],
) -> Iterator[Path]:
    if len(frame_order) != len(frame_paths):
        raise RuntimeError("frame_order and frame_paths have different lengths")
    root = root.resolve()
    with tempfile.TemporaryDirectory(prefix=".sam3-video-", dir=root) as directory:
        video_dir = Path(directory)
        for index, (frame_id, relative_path) in enumerate(
            zip(frame_order, frame_paths, strict=True)
        ):
            relative = Path(relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe normalized frame path: {relative_path}")
            source = (root / relative).resolve()
            if not source.is_relative_to(root):
                raise RuntimeError(
                    f"normalized frame escapes the attempt workspace: {relative_path}"
                )
            expected_dimensions = frame_dimensions.get(frame_id)
            if expected_dimensions is None:
                raise RuntimeError(f"missing frame dimensions for {frame_id}")
            with Image.open(source) as image:
                image.load()
                if image.format != "PNG":
                    raise RuntimeError(f"normalized frame {frame_id} is not a PNG")
                if image.size != expected_dimensions:
                    raise RuntimeError(
                        f"normalized frame {frame_id} has dimensions {image.size}, "
                        f"expected {expected_dimensions}"
                    )
            destination = video_dir / f"{index:06d}.png"
            shutil.copy2(source, destination)
            with Image.open(destination) as copied:
                copied.load()
                if copied.size != expected_dimensions:
                    raise RuntimeError(f"prepared frame dimensions changed for {frame_id}")
        yield video_dir
