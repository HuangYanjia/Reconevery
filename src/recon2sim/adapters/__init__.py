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
}

__all__ = [
    "Adapter",
    "ArtifactRecord",
    "CommandAdapter",
    "CameraMeshAlignmentAdapter",
    "ColmapCameraRecoveryAdapter",
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
    "REGISTRY",
    "Phase3EndToEndConsistencyAdapter",
    "Phase4ConsistencyValidationAdapter",
    "Phase4_2ConsistencyValidationAdapter",
    "Phase5AConsistencyValidationAdapter",
    "Sam3SegmentationTrackingAdapter",
    "StageContext",
    "StageResult",
]
