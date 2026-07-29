from __future__ import annotations

from recon2sim.adapters.base import (
    HealthcheckResult,
    InputSpec,
    OutputSpec,
    StageContext,
    StageResult,
)
from recon2sim.adapters.completion_common import (
    CompletionWorkerConfig,
    completion_healthcheck,
    worker_command,
)
from recon2sim.adapters.ingest import run_process
from recon2sim.artifacts import (
    WorldCalibrationArtifact,
    WorldCalibrationDatasetSplit,
    WorldCalibrationDiagnostics,
    WorldCalibrationManifest,
    WorldCalibrationRequest,
)
from recon2sim.calibration import sha256_file
from recon2sim.storage import atomic_write_json


class WorldCalibrationConfig(CompletionWorkerConfig):
    worker_module: str = "world_calibration_worker"
    docker_image: str = "reconevery/world-calibration:phase6a"
    minimum_metric_evidence_records: int = 1
    minimum_gravity_evidence_records: int = 1
    minimum_forward_evidence_records: int = 1
    minimum_heldout_tag_detections: int = 3
    maximum_heldout_tag_translation_error_m: float = 0.02
    maximum_heldout_tag_rotation_error_degrees: float = 3.0
    minimum_known_distance_anchors: int = 1
    maximum_known_distance_relative_error: float = 0.02
    maximum_gravity_heldout_error_degrees: float = 3.0
    minimum_floor_point_count: int = 1000
    minimum_floor_spatial_extent_colmap: float = 0.25
    maximum_forward_uncertainty_degrees: float = 5.0
    maximum_sim3_roundtrip_error: float = 1.0e-8


def _dataset_split(manifest: WorldCalibrationManifest) -> WorldCalibrationDatasetSplit:
    fitting_frames: list[str] = []
    heldout_frames: list[str] = []
    if manifest.apriltag is not None:
        fitting_frames.extend(
            item.frame_id for item in manifest.apriltag.image_sources if item.split == "fitting"
        )
        heldout_frames.extend(
            item.frame_id for item in manifest.apriltag.image_sources if item.split == "heldout"
        )
        fitting_frames.extend(
            item.frame_id for item in manifest.apriltag.detections if item.split == "fitting"
        )
        heldout_frames.extend(
            item.frame_id for item in manifest.apriltag.detections if item.split == "heldout"
        )
    if manifest.known_distance is not None:
        all_frames = sorted({item.frame_id for item in manifest.known_distance.observations})
        if not fitting_frames and not heldout_frames:
            split_index = max(2, (2 * len(all_frames) + 2) // 3)
            fitting_frames.extend(all_frames[:split_index])
            heldout_frames.extend(all_frames[split_index:])
    if not fitting_frames and not heldout_frames:
        supporting = sorted(
            {
                item
                for gravity in manifest.gravity
                for item in gravity.supporting_ids
                if item.startswith("frame_")
            }
        )
        fitting_frames.extend(supporting[::2])
        heldout_frames.extend(supporting[1::2])
    fitting_ids = [
        record.evidence_id
        for record in manifest.evidence
        if "heldout" not in record.evidence_id.lower()
    ]
    heldout_ids = [
        record.evidence_id
        for record in manifest.evidence
        if "heldout" in record.evidence_id.lower()
    ]
    if manifest.known_distance is not None:
        fitting_ids.append("known_distance:fitting")
        heldout_ids.append("known_distance:heldout")
    return WorldCalibrationDatasetSplit(
        fitting_evidence_ids=sorted(set(fitting_ids)),
        heldout_evidence_ids=sorted(set(heldout_ids)),
        fitting_frame_ids=sorted(set(fitting_frames)),
        heldout_frame_ids=sorted(set(heldout_frames)),
        split_policy="declared_tag_split_else_deterministic_frame_order_v1",
    )


class WorldCalibrationAdapter:
    name = "world_calibration"
    version = "0.1.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        manifest = WorldCalibrationManifest.model_validate_json(
            context.canonical_path("calibration", "evidence_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        specs = [
            InputSpec(
                "calibration/evidence_manifest.json",
                "world_calibration_manifest",
            ),
            InputSpec(
                manifest.camera_reconstruction_path,
                "calibration_source_camera",
                expected_sha256=manifest.camera_reconstruction_sha256,
                include_producer_signature=False,
            ),
            InputSpec(
                manifest.source_scene_ir_path,
                "calibration_source_scene_ir",
                expected_sha256=manifest.source_scene_ir_sha256,
                include_producer_signature=False,
            ),
        ]
        seen = {item.relative_path for item in specs}
        for record in manifest.evidence:
            for source in record.source_files:
                if source.relative_path in seen:
                    continue
                seen.add(source.relative_path)
                specs.append(
                    InputSpec(
                        source.relative_path,
                        "calibration_evidence_source",
                        expected_sha256=source.sha256,
                        include_producer_signature=False,
                    )
                )
        return specs

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return completion_healthcheck(
            context,
            WorldCalibrationConfig,
            worker_name=self.name,
        )

    def prepare(self, context: StageContext) -> None:
        context.path("calibration", "raw", "logs").mkdir(parents=True, exist_ok=True)
        context.path("calibration", "previews").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        outputs = [
            OutputSpec(
                "calibration/request.json",
                "world_calibration_request",
                "application/json",
                self.name,
                validation="json",
                model=WorldCalibrationRequest,
            ),
            OutputSpec(
                "calibration/world_calibration.json",
                "world_calibration_artifact",
                "application/json",
                self.name,
                validation="json",
                model=WorldCalibrationArtifact,
            ),
            OutputSpec(
                "calibration/diagnostics.json",
                "world_calibration_diagnostics",
                "application/json",
                self.name,
                validation="json",
                model=WorldCalibrationDiagnostics,
            ),
            OutputSpec(
                "calibration/triangulated_landmarks.json",
                "triangulated_calibration_landmarks",
                "application/json",
                self.name,
                validation="json",
            ),
            OutputSpec(
                "calibration/apriltag_detections.json",
                "apriltag_detections",
                "application/json",
                self.name,
                validation="json",
            ),
        ]
        outputs.extend(
            OutputSpec(
                f"calibration/previews/{name}.png",
                "world_calibration_preview",
                "image/png",
                self.name,
                validation="png",
            )
            for name in (
                "metric_evidence",
                "tag_detections",
                "landmark_reprojection",
                "floor_plane",
                "gravity_evidence",
                "canonical_axes",
                "camera_trajectory_before_after",
                "scene_bounds_before_after",
                "heldout_validation",
            )
        )
        return outputs

    def run(self, context: StageContext) -> StageResult:
        config = WorldCalibrationConfig.model_validate(context.config.adapter.config)
        root = context.path("calibration")
        manifest_path = root / "evidence_manifest.json"
        manifest = WorldCalibrationManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        split = _dataset_split(manifest)
        request = WorldCalibrationRequest(
            manifest_path="calibration/evidence_manifest.json",
            manifest_sha256=sha256_file(manifest_path),
            frame_sequence_digest=manifest.frame_sequence_digest,
            camera_reconstruction_path=manifest.camera_reconstruction_path,
            camera_reconstruction_sha256=manifest.camera_reconstruction_sha256,
            source_scene_ir_path=manifest.source_scene_ir_path,
            source_scene_ir_sha256=manifest.source_scene_ir_sha256,
            dataset_split=split,
            solver_configuration={
                "candidate_selection_evidence": "fitting_only",
                "heldout_usage": "acceptance_only",
                "proper_positive_sim3": True,
                "seed": context.seed,
            },
            acceptance_gates={
                "minimum_metric_evidence_records": config.minimum_metric_evidence_records,
                "minimum_gravity_evidence_records": config.minimum_gravity_evidence_records,
                "minimum_forward_evidence_records": config.minimum_forward_evidence_records,
                "minimum_heldout_tag_detections": config.minimum_heldout_tag_detections,
                "maximum_heldout_tag_translation_error_m": (
                    config.maximum_heldout_tag_translation_error_m
                ),
                "maximum_heldout_tag_rotation_error_degrees": (
                    config.maximum_heldout_tag_rotation_error_degrees
                ),
                "minimum_known_distance_anchors": config.minimum_known_distance_anchors,
                "maximum_known_distance_relative_error": (
                    config.maximum_known_distance_relative_error
                ),
                "maximum_gravity_heldout_error_degrees": (
                    config.maximum_gravity_heldout_error_degrees
                ),
                "minimum_floor_point_count": config.minimum_floor_point_count,
                "minimum_floor_spatial_extent_colmap": (config.minimum_floor_spatial_extent_colmap),
                "maximum_forward_uncertainty_degrees": (config.maximum_forward_uncertainty_degrees),
                "maximum_sim3_roundtrip_error": config.maximum_sim3_roundtrip_error,
            },
            output_directory="calibration",
            seed=context.seed,
            fake_mode=config.fake_mode if config.execution_mode == "fake_worker" else None,
        )
        request_path = root / "request.json"
        atomic_write_json(request_path, request)
        command = worker_command(
            context,
            config,
            "solve",
            "calibration/request.json",
            "calibration",
        )
        run_process(
            command,
            context=context,
            name="world_calibration",
            log_directory="calibration/raw/logs",
        )
        artifact = WorldCalibrationArtifact.model_validate_json(
            (root / "world_calibration.json").read_text(encoding="utf-8")
        )
        if artifact.manifest_sha256 != sha256_file(manifest_path):
            raise RuntimeError("world calibration returned the wrong evidence-manifest hash")
        if artifact.dataset_split != split:
            raise RuntimeError("world calibration changed the fitting/held-out split")
        if sha256_file(context.path(*manifest.camera_reconstruction_path.split("/"))) != (
            manifest.camera_reconstruction_sha256
        ):
            raise RuntimeError("world calibration modified its source camera reconstruction")
        return StageResult(
            metrics={
                "calibration_status": artifact.status.value,
                "full_canonical_world": artifact.full_canonical_world_available,
                "heldout_evidence": len(split.heldout_evidence_ids),
            }
        )


__all__ = ["WorldCalibrationAdapter", "WorldCalibrationConfig"]
