from pathlib import Path

from recon2sim.artifacts import (
    Sam3InferenceRequest,
    SegmentationPromptManifest,
    SegmentationTrackingArtifact,
)
from recon2sim.ir import SceneIR
from recon2sim.storage import atomic_write_json

atomic_write_json(Path("schemas/scene_ir.schema.json"), SceneIR.model_json_schema())
atomic_write_json(
    Path("schemas/segmentation_prompts.schema.json"),
    SegmentationPromptManifest.model_json_schema(),
)
atomic_write_json(
    Path("schemas/sam3_inference_request.schema.json"),
    Sam3InferenceRequest.model_json_schema(),
)
atomic_write_json(
    Path("schemas/segmentation_tracking.schema.json"),
    SegmentationTrackingArtifact.model_json_schema(),
)
