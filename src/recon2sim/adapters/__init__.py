from recon2sim.adapters.alignment import (
    CameraMeshAlignmentAdapter,
    Phase4_2ConsistencyValidationAdapter,
)
from recon2sim.adapters.base import (
    Adapter,
    ArtifactRecord,
    HealthcheckResult,
    InputSpec,
    OutputSpec,
    StageContext,
    StageResult,
)
from recon2sim.adapters.colmap import ColmapCameraRecoveryAdapter
from recon2sim.adapters.command import CommandAdapter, DockerCommandAdapter
from recon2sim.adapters.completion_candidates import MeasuredOnlyCandidateAdapter
from recon2sim.adapters.completion_evaluation import CompletionCandidateEvaluationAdapter
from recon2sim.adapters.completion_inputs import CompletionEvidencePackageAdapter
from recon2sim.adapters.completion_registration import (
    CompletionCandidateRegistrationAdapter,
)
from recon2sim.adapters.completion_selection import (
    CompletionSelectionAdapter,
    Phase5BConsistencyValidationAdapter,
)
from recon2sim.adapters.dense_mvs import DenseMVSAdapter
from recon2sim.adapters.genrecon import (
    GenReconCameraPackageAdapter,
    GenReconGlobalReconstructionAdapter,
    Phase3EndToEndConsistencyAdapter,
)
from recon2sim.adapters.ingest import FFmpegIngestAdapter
from recon2sim.adapters.measured_geometry import (
    MeasuredGeneratedComparisonAdapter,
    MeasuredObjectGeometryAdapter,
    Phase5AConsistencyValidationAdapter,
)
from recon2sim.adapters.mock import (
    MockCameraRecoveryAdapter,
    MockExportAdapter,
    MockGlobalReconstructionAdapter,
    MockIngestAdapter,
    MockObjectReconstructionAdapter,
    MockPhysicsValidatorAdapter,
    MockSceneCompilerAdapter,
    MockSceneIRAssemblyAdapter,
    MockSegmentationTrackingAdapter,
)
from recon2sim.adapters.object_lifting import (
    ObjectSurfaceLiftingAdapter,
    Phase4ConsistencyValidationAdapter,
)
from recon2sim.adapters.sam3 import Sam3SegmentationTrackingAdapter
from recon2sim.adapters.sam3d_objects import Sam3DObjectsCandidateAdapter
from recon2sim.adapters.trellis2_objects import Trellis2ObjectCandidateAdapter

REGISTRY: dict[str, type[Adapter]] = {
    "mock_ingest": MockIngestAdapter,
    "mock_camera_recovery": MockCameraRecoveryAdapter,
    "mock_segmentation_tracking": MockSegmentationTrackingAdapter,
    "mock_global_reconstruction": MockGlobalReconstructionAdapter,
    "mock_object_reconstruction": MockObjectReconstructionAdapter,
    "mock_scene_ir_assembly": MockSceneIRAssemblyAdapter,
    "mock_scene_compiler": MockSceneCompilerAdapter,
    "mock_physics_validator": MockPhysicsValidatorAdapter,
    "mock_export": MockExportAdapter,
    "command": CommandAdapter,
    "docker_command": DockerCommandAdapter,
    "ffmpeg_ingest": FFmpegIngestAdapter,
    "colmap_camera_recovery": ColmapCameraRecoveryAdapter,
    "sam3_segmentation_tracking": Sam3SegmentationTrackingAdapter,
    "genrecon_camera_package": GenReconCameraPackageAdapter,
    "genrecon_global_reconstruction": GenReconGlobalReconstructionAdapter,
    "phase3_e2e_consistency": Phase3EndToEndConsistencyAdapter,
    "object_surface_lifting": ObjectSurfaceLiftingAdapter,
    "phase4_consistency_validation": Phase4ConsistencyValidationAdapter,
    "camera_mesh_alignment": CameraMeshAlignmentAdapter,
    "phase4_2_consistency_validation": Phase4_2ConsistencyValidationAdapter,
    "dense_mvs": DenseMVSAdapter,
    "measured_object_geometry": MeasuredObjectGeometryAdapter,
    "measured_generated_comparison": MeasuredGeneratedComparisonAdapter,
    "phase5a_consistency_validation": Phase5AConsistencyValidationAdapter,
    "completion_evidence_package": CompletionEvidencePackageAdapter,
    "sam3d_object_candidates": Sam3DObjectsCandidateAdapter,
    "trellis2_object_candidates": Trellis2ObjectCandidateAdapter,
    "measured_only_candidates": MeasuredOnlyCandidateAdapter,
    "completion_candidate_registration": CompletionCandidateRegistrationAdapter,
    "completion_candidate_evaluation": CompletionCandidateEvaluationAdapter,
    "completion_candidate_selection": CompletionSelectionAdapter,
    "phase5b_consistency_validation": Phase5BConsistencyValidationAdapter,
}

__all__ = [
    "Adapter",
    "ArtifactRecord",
    "CommandAdapter",
    "CameraMeshAlignmentAdapter",
    "ColmapCameraRecoveryAdapter",
    "CompletionCandidateEvaluationAdapter",
    "CompletionCandidateRegistrationAdapter",
    "CompletionEvidencePackageAdapter",
    "CompletionSelectionAdapter",
    "DockerCommandAdapter",
    "DenseMVSAdapter",
    "FFmpegIngestAdapter",
    "GenReconCameraPackageAdapter",
    "GenReconGlobalReconstructionAdapter",
    "HealthcheckResult",
    "InputSpec",
    "OutputSpec",
    "ObjectSurfaceLiftingAdapter",
    "MeasuredObjectGeometryAdapter",
    "MeasuredGeneratedComparisonAdapter",
    "MeasuredOnlyCandidateAdapter",
    "REGISTRY",
    "Phase3EndToEndConsistencyAdapter",
    "Phase4ConsistencyValidationAdapter",
    "Phase4_2ConsistencyValidationAdapter",
    "Phase5AConsistencyValidationAdapter",
    "Phase5BConsistencyValidationAdapter",
    "Sam3SegmentationTrackingAdapter",
    "Sam3DObjectsCandidateAdapter",
    "StageContext",
    "StageResult",
    "Trellis2ObjectCandidateAdapter",
]
