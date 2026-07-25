from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from PIL import Image


@dataclass(frozen=True)
class FrameMetrics:
    blur_score: float
    mean_brightness: float
    grayscale_variance: float
    signature: tuple[int, ...]


def measure_frame(path: Path) -> FrameMetrics:
    with Image.open(path) as image:
        grayscale = image.convert("L")
        grayscale.thumbnail((256, 256), Image.Resampling.LANCZOS)
        width, height = grayscale.size
        pixels = list(cast(Sequence[int], grayscale.get_flattened_data()))

        mean = sum(pixels) / len(pixels)
        variance = sum((value - mean) ** 2 for value in pixels) / len(pixels)
        laplacian: list[float] = []
        if width >= 3 and height >= 3:
            for y in range(1, height - 1):
                row = y * width
                for x in range(1, width - 1):
                    index = row + x
                    laplacian.append(
                        float(
                            pixels[index - width]
                            + pixels[index + width]
                            + pixels[index - 1]
                            + pixels[index + 1]
                            - 4 * pixels[index]
                        )
                    )
        laplacian_mean = sum(laplacian) / len(laplacian) if laplacian else 0.0
        blur_score = (
            sum((value - laplacian_mean) ** 2 for value in laplacian) / len(laplacian)
            if laplacian
            else 0.0
        )
        signature_image = grayscale.resize((32, 32), Image.Resampling.BILINEAR)
        signature = tuple(cast(Sequence[int], signature_image.get_flattened_data()))
    return FrameMetrics(
        blur_score=blur_score,
        mean_brightness=mean,
        grayscale_variance=variance,
        signature=signature,
    )


def normalized_signature_difference(
    first: tuple[int, ...],
    second: tuple[int, ...],
) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("frame signatures must have the same non-zero length")
    return sum(abs(left - right) for left, right in zip(first, second, strict=True)) / (
        255.0 * len(first)
    )


__all__ = ["FrameMetrics", "measure_frame", "normalized_signature_difference"]
