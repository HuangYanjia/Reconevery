from pathlib import Path

from recon2sim.artifacts import (
    EndToEndConsistencyReport,
    GenReconCameraPackageManifest,
    GenReconCheckpointManifest,
    GenReconInferenceRequest,
    GenReconWorkerManifest,
    GlobalSceneDiagnostics,
    GlobalSceneReconstructionArtifact,
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
for filename, model in {
    "genrecon_camera_package.schema.json": GenReconCameraPackageManifest,
    "genrecon_checkpoints.schema.json": GenReconCheckpointManifest,
    "genrecon_inference_request.schema.json": GenReconInferenceRequest,
    "genrecon_worker_manifest.schema.json": GenReconWorkerManifest,
    "global_scene_reconstruction.schema.json": GlobalSceneReconstructionArtifact,
    "global_scene_diagnostics.schema.json": GlobalSceneDiagnostics,
    "phase3_e2e_consistency.schema.json": EndToEndConsistencyReport,
}.items():
    atomic_write_json(Path("schemas") / filename, model.model_json_schema())
