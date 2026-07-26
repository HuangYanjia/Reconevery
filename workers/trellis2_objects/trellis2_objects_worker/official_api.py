from __future__ import annotations

from pathlib import Path


def load_rgba_pipeline(snapshot: Path) -> object:
    """Load the official pipeline without an unused background-removal model."""
    from trellis2.pipelines import Trellis2ImageTo3DPipeline, rembg

    original = rembg.BiRefNet
    rembg.BiRefNet = lambda **_: None
    try:
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(snapshot))
    finally:
        rembg.BiRefNet = original
    if pipeline.rembg_model is not None:
        raise RuntimeError("RGBA TRELLIS.2 pipeline unexpectedly initialized background removal")
    return pipeline
