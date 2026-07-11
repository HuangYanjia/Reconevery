from recon2sim.adapters.base import (
    Adapter,
    ArtifactRecord,
    HealthcheckResult,
    OutputSpec,
    StageContext,
    StageResult,
)
from recon2sim.adapters.colmap import ColmapCameraRecoveryAdapter
from recon2sim.adapters.command import CommandAdapter, DockerCommandAdapter
from recon2sim.adapters.ingest import FfmpegIngestAdapter
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
    "colmap_camera_recovery": ColmapCameraRecoveryAdapter,
    "docker_command": DockerCommandAdapter,
    "ffmpeg_ingest": FfmpegIngestAdapter,
}

__all__ = [
    "Adapter",
    "ArtifactRecord",
    "CommandAdapter",
    "ColmapCameraRecoveryAdapter",
    "DockerCommandAdapter",
    "FfmpegIngestAdapter",
    "HealthcheckResult",
    "OutputSpec",
    "REGISTRY",
    "StageContext",
    "StageResult",
]
