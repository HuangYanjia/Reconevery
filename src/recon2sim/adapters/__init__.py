from recon2sim.adapters.alignment import (
    CameraMeshAlignmentAdapter,
    Phase4_2ConsistencyValidationAdapter,
)
from recon2sim.adapters.articulated_retrieval import (
    ArtVIPRetrievalAdapter,
    PartNetRetrievalAdapter,
)
from recon2sim.adapters.articulation_capture import ArticulationCaptureAdapter
from recon2sim.adapters.articulation_evaluation import ArticulationEvaluationAdapter
from recon2sim.adapters.articulation_fitting import ArticulationFittingAdapter
from recon2sim.adapters.articulation_motion import ArticulationMotionAdapter
from recon2sim.adapters.articulation_selection import (
    ArticulationSelectionAdapter,
    Phase5CConsistencyValidationAdapter,
)
from recon2sim.adapters.articulation_state_alignment import (
    ArticulationStateAlignmentAdapter,
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
from recon2sim.adapters.particulate import ParticulateAdapter
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
    "articulation_capture": ArticulationCaptureAdapter,
    "articulation_state_alignment": ArticulationStateAlignmentAdapter,
    "articulation_motion": ArticulationMotionAdapter,
    "artvip_retrieval": ArtVIPRetrievalAdapter,
    "partnet_retrieval": PartNetRetrievalAdapter,
    "particulate_candidates": ParticulateAdapter,
    "articulation_fitting": ArticulationFittingAdapter,
    "articulation_evaluation": ArticulationEvaluationAdapter,
    "articulation_selection": ArticulationSelectionAdapter,
    "phase5c_consistency_validation": Phase5CConsistencyValidationAdapter,
}

__all__ = [
    "Adapter",
    "ArtVIPRetrievalAdapter",
    "ArticulationCaptureAdapter",
    "ArticulationEvaluationAdapter",
    "ArticulationFittingAdapter",
    "ArticulationMotionAdapter",
    "ArticulationSelectionAdapter",
    "ArticulationStateAlignmentAdapter",
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
    "ParticulateAdapter",
    "MeasuredObjectGeometryAdapter",
    "MeasuredGeneratedComparisonAdapter",
    "MeasuredOnlyCandidateAdapter",
    "REGISTRY",
    "Phase3EndToEndConsistencyAdapter",
    "Phase4ConsistencyValidationAdapter",
    "Phase4_2ConsistencyValidationAdapter",
    "Phase5AConsistencyValidationAdapter",
    "Phase5BConsistencyValidationAdapter",
    "Phase5CConsistencyValidationAdapter",
    "PartNetRetrievalAdapter",
    "Sam3SegmentationTrackingAdapter",
    "Sam3DObjectsCandidateAdapter",
    "StageContext",
    "StageResult",
    "Trellis2ObjectCandidateAdapter",
]
