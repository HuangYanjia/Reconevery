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
from recon2sim.adapters.genrecon import (
    GenReconCameraPackageAdapter,
    GenReconGlobalReconstructionAdapter,
    Phase3EndToEndConsistencyAdapter,
)
from recon2sim.adapters.ingest import FFmpegIngestAdapter
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
}

__all__ = [
    "Adapter",
    "ArtifactRecord",
    "CommandAdapter",
    "ColmapCameraRecoveryAdapter",
    "DockerCommandAdapter",
    "FFmpegIngestAdapter",
    "GenReconCameraPackageAdapter",
    "GenReconGlobalReconstructionAdapter",
    "HealthcheckResult",
    "InputSpec",
    "OutputSpec",
    "REGISTRY",
    "Phase3EndToEndConsistencyAdapter",
    "Sam3SegmentationTrackingAdapter",
    "StageContext",
    "StageResult",
]
