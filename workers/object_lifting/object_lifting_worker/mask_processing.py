from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MaskRegions:
    mask: Any
    core: Any
    boundary: Any
    exterior: Any


def preprocess_mask(
    binary_mask: Any,
    *,
    core_erosion_pixels: int,
    boundary_width_pixels: int,
    exclusion_dilation_pixels: int,
) -> MaskRegions:
    import cv2
    import numpy as np

    mask = binary_mask > 0

    def morphology(source: Any, radius: int, operation: int) -> Any:
        if radius <= 0:
            return source.copy()
        kernel_size = radius * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.morphologyEx(
            source.astype(np.uint8),
            operation,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)

    core = morphology(mask, core_erosion_pixels, cv2.MORPH_ERODE)
    inner = morphology(mask, boundary_width_pixels, cv2.MORPH_ERODE)
    outer = morphology(mask, boundary_width_pixels, cv2.MORPH_DILATE)
    boundary = outer & ~inner
    exclusion = morphology(
        mask,
        boundary_width_pixels + exclusion_dilation_pixels,
        cv2.MORPH_DILATE,
    )
    return MaskRegions(mask=mask, core=core, boundary=boundary & mask, exterior=~exclusion)
