from pathlib import Path

from recon2sim.artifacts import (
    AlignmentCandidateManifest,
    AlignmentDatasetSplit,
    AlignmentIterationManifest,
    CameraMeshAlignmentArtifact,
    CameraMeshAlignmentDiagnostics,
    CameraMeshAlignmentPreviewManifest,
    CameraMeshAlignmentRequest,
    CameraMeshAlignmentResult,
    CameraMeshAlignmentWorkerManifest,
    EndToEndConsistencyReport,
    GenReconCameraPackageManifest,
    GenReconCheckpointManifest,
    GenReconInferenceRequest,
    GenReconWorkerManifest,
    GlobalSceneDiagnostics,
    GlobalSceneReconstructionArtifact,
    ObjectLiftingAlignmentComparison,
    ObjectSurfaceDiagnostics,
    ObjectSurfaceEvidenceArtifact,
    ObjectSurfaceLiftingRequest,
    ObjectSurfaceMethodComparison,
    ObjectSurfaceWorkerManifest,
    Phase4_2ConsistencyReport,
    Phase4ConsistencyReport,
    Sam3InferenceRequest,
    SegmentationPromptManifest,
    SegmentationTrackingArtifact,
    SparseDepthObservationManifest,
    TransformChainAudit,
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
    "object_surface_lifting_request.schema.json": ObjectSurfaceLiftingRequest,
    "object_surface_worker_manifest.schema.json": ObjectSurfaceWorkerManifest,
    "object_surface_evidence.schema.json": ObjectSurfaceEvidenceArtifact,
    "object_surface_diagnostics.schema.json": ObjectSurfaceDiagnostics,
    "object_surface_method_comparison.schema.json": ObjectSurfaceMethodComparison,
    "camera_mesh_alignment.schema.json": CameraMeshAlignmentArtifact,
    "phase4_object_surface_consistency.schema.json": Phase4ConsistencyReport,
    "camera_mesh_alignment_request.schema.json": CameraMeshAlignmentRequest,
    "camera_mesh_alignment_worker_manifest.schema.json": (CameraMeshAlignmentWorkerManifest),
    "camera_mesh_alignment_result.schema.json": CameraMeshAlignmentResult,
    "camera_mesh_alignment_diagnostics.schema.json": CameraMeshAlignmentDiagnostics,
    "camera_mesh_alignment_previews.schema.json": CameraMeshAlignmentPreviewManifest,
    "transform_chain_audit.schema.json": TransformChainAudit,
    "sparse_depth_observations.schema.json": SparseDepthObservationManifest,
    "alignment_dataset_split.schema.json": AlignmentDatasetSplit,
    "alignment_candidates.schema.json": AlignmentCandidateManifest,
    "alignment_iterations.schema.json": AlignmentIterationManifest,
    "object_lifting_alignment_comparison.schema.json": (ObjectLiftingAlignmentComparison),
    "phase4_2_camera_mesh_alignment.schema.json": Phase4_2ConsistencyReport,
}.items():
    atomic_write_json(Path("schemas") / filename, model.model_json_schema())
