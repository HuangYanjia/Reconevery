from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from PIL import Image

from recon2sim.adapters.base import (
    HealthcheckResult,
    InputSpec,
    OutputSpec,
    StageContext,
    StageResult,
)
from recon2sim.artifacts import CameraReconstruction, WorldCalibrationManifest
from recon2sim.calibration import sha256_file, stable_digest
from recon2sim.ir import SceneIR, StrictModel
from recon2sim.storage import atomic_write_json


class CalibrationEvidenceConfig(StrictModel):
    manifest_path: str | None = None
    reference_run: str | None = None
    camera_reconstruction_path: str = "camera/reconstruction.json"
    source_scene_ir_path: str = "scene_ir/phase5c_scene.json"
    fake_mode: Literal["perfect_full_canonical", "insufficient_evidence"] | None = None


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ValueError(f"calibration source path must be relative: {value!r}")
    return value


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"calibration manifest must be a mapping: {path}")
    return value


def _source_manifest(config: CalibrationEvidenceConfig) -> tuple[Path, dict[str, Any]]:
    if config.manifest_path is None:
        raise ValueError("real calibration evidence requires manifest_path")
    path = Path(config.manifest_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"calibration manifest does not exist: {path}")
    return path, _load_mapping(path)


def _dynamic_source_files(raw: dict[str, Any]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for evidence in raw.get("evidence", []):
        if not isinstance(evidence, dict):
            continue
        for source in evidence.get("source_files", []):
            if not isinstance(source, dict):
                continue
            relative = _safe_relative(str(source["relative_path"]))
            local_value = source.get("local_path")
            if local_value is None:
                continue
            local = Path(str(local_value)).expanduser().resolve()
            result.append((relative, local))
    return sorted(result)


def _source_geometry_files(
    config: CalibrationEvidenceConfig,
) -> list[tuple[str, Path]]:
    if config.reference_run is None:
        return []
    reference = Path(config.reference_run).expanduser().resolve()
    scene_path = reference / _safe_relative(config.source_scene_ir_path)
    scene = SceneIR.model_validate_json(scene_path.read_text(encoding="utf-8"))
    return [(asset.uri, reference / _safe_relative(asset.uri)) for asset in scene.geometry_assets]


def _strip_local_paths(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _strip_local_paths(item) for key, item in value.items() if key != "local_path"
        }
    if isinstance(value, list):
        return [_strip_local_paths(item) for item in value]
    return value


class CalibrationEvidenceAdapter:
    name = "calibration_evidence"
    version = "0.3.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        config = CalibrationEvidenceConfig.model_validate(context.config.adapter.config)
        if config.fake_mode is not None:
            return []
        manifest_path, raw = _source_manifest(config)
        if config.reference_run is None:
            raise ValueError("real calibration evidence requires reference_run")
        reference = Path(config.reference_run).expanduser().resolve()
        specs = [
            InputSpec(
                "calibration/source/manifest.yaml",
                "world_calibration_source_manifest",
                source_path=manifest_path,
            ),
            InputSpec(
                "calibration/source/camera_reconstruction.json",
                "camera_reconstruction",
                source_path=reference / _safe_relative(config.camera_reconstruction_path),
            ),
            InputSpec(
                "calibration/source/scene_ir.json",
                "source_scene_ir",
                source_path=reference / _safe_relative(config.source_scene_ir_path),
            ),
        ]
        for relative, local in _dynamic_source_files(raw):
            specs.append(
                InputSpec(
                    relative,
                    "calibration_evidence_source",
                    source_path=local,
                    materialization_mode="reflink_or_copy",
                )
            )
        for relative, local in _source_geometry_files(config):
            specs.append(
                InputSpec(
                    relative,
                    "calibration_source_geometry",
                    source_path=local,
                    materialization_mode="reflink_or_copy",
                )
            )
        return specs

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        if context is None:
            return HealthcheckResult(False, "calibration evidence healthcheck requires --config")
        try:
            config = CalibrationEvidenceConfig.model_validate(context.config.adapter.config)
            if config.fake_mode is None:
                manifest_path, raw = _source_manifest(config)
                if config.reference_run is None:
                    raise ValueError("reference_run is required")
                if not Path(config.reference_run).expanduser().resolve().is_dir():
                    raise ValueError("reference_run does not exist")
                for _, source in _dynamic_source_files(raw):
                    if not source.is_file():
                        raise ValueError(f"calibration evidence source does not exist: {source}")
                for _, source in _source_geometry_files(config):
                    if not source.is_file():
                        raise ValueError(f"source Scene IR geometry does not exist: {source}")
                if not manifest_path.is_file():
                    raise ValueError("calibration manifest does not exist")
        except ValueError as exc:
            return HealthcheckResult(False, str(exc))
        return HealthcheckResult(True, "typed calibration evidence preparation available")

    def prepare(self, context: StageContext) -> None:
        context.path("calibration", "source").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        outputs = [
            OutputSpec(
                "calibration/evidence_manifest.json",
                "world_calibration_manifest",
                "application/json",
                self.name,
                validation="json",
                model=WorldCalibrationManifest,
            ),
            OutputSpec(
                "calibration/source/camera_reconstruction.json",
                "calibration_source_camera",
                "application/json",
                self.name,
                validation="json",
                model=CameraReconstruction,
            ),
            OutputSpec(
                "calibration/source/scene_ir.json",
                "calibration_source_scene_ir",
                "application/json",
                self.name,
                validation="scene_ir",
                model=SceneIR,
            ),
        ]
        config = CalibrationEvidenceConfig.model_validate(context.config.adapter.config)
        if config.fake_mode == "perfect_full_canonical":
            outputs.extend(
                OutputSpec(
                    f"calibration/source/frames/frame_{index:06d}.png",
                    "calibration_evidence_source",
                    "image/png",
                    self.name,
                    validation="png",
                )
                for index in range(6)
            )
        elif config.fake_mode is None:
            _, raw = _source_manifest(config)
            outputs.extend(
                OutputSpec(
                    relative,
                    "calibration_evidence_source",
                    "application/octet-stream",
                    self.name,
                )
                for relative, _ in _dynamic_source_files(raw)
            )
            outputs.extend(
                OutputSpec(
                    relative,
                    "calibration_source_geometry",
                    "application/octet-stream",
                    self.name,
                )
                for relative, _ in _source_geometry_files(config)
            )
        return outputs

    def _fake_sources(self, context: StageContext) -> tuple[CameraReconstruction, SceneIR]:
        confidence = {"score": 1.0, "method": "phase6a_fake"}
        provenance = {
            "adapter_name": self.name,
            "adapter_version": self.version,
            "configuration": {"fake": True},
            "input_artifact_paths": [],
            "output_artifact_paths": [
                "calibration/source/camera_reconstruction.json",
                "calibration/source/scene_ir.json",
            ],
            "confidence": confidence,
            "source": "mock",
        }
        convention = {
            "world_frame": "colmap_arbitrary",
            "alignment_status": "unoriented",
            "camera_axes": "x_right_y_down_z_forward",
            "linear_units": "arbitrary_units",
            "scale_status": "scale_ambiguous",
            "transform_direction": "world_from_camera",
        }
        poses = []
        for index in range(6):
            poses.append(
                {
                    "frame_id": f"frame_{index:06d}",
                    "transform_world_from_camera": {
                        "translation": [float(index), float(index % 2), 1.0],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "scale": [1.0, 1.0, 1.0],
                    },
                    "confidence": confidence,
                }
            )
        camera = CameraReconstruction.model_validate(
            {
                "camera_id": "phase6a_fake_camera",
                "model": "PINHOLE",
                "intrinsics": {
                    "width": 640,
                    "height": 480,
                    "fx": 500.0,
                    "fy": 500.0,
                    "cx": 320.0,
                    "cy": 240.0,
                    "distortion": [],
                },
                "poses": poses,
                "registered_frame_ids": [item["frame_id"] for item in poses],
                "unregistered_frame_ids": [],
                "sparse_point_count": 100,
                "average_reprojection_error": 0.2,
                "confidence": confidence,
                "coordinate_convention": convention,
                "scale_status": "scale_ambiguous",
                "frame_sequence_digest": stable_digest([item["frame_id"] for item in poses]),
                "provenance": provenance,
            }
        )
        scene = SceneIR.model_validate(
            {
                "schema_version": "0.1.6",
                "metadata": {
                    "scene_id": "phase6a_fake_scene",
                    "name": "Phase 6A fake source scene",
                    "coordinate_convention": convention,
                    "source": "mock",
                    "provenance": [provenance],
                },
                "cameras": [],
                "frames": [],
                "objects": [],
                "geometry_assets": [],
                "material_assets": [],
                "collision_assets": [],
                "relations": [],
            }
        )
        atomic_write_json(context.path("calibration/source/camera_reconstruction.json"), camera)
        atomic_write_json(context.path("calibration/source/scene_ir.json"), scene)
        return camera, scene

    def _fake_manifest(
        self,
        context: StageContext,
        camera: CameraReconstruction,
        scene: SceneIR,
        *,
        full: bool,
    ) -> WorldCalibrationManifest:
        evidence: list[dict[str, object]] = []
        tag_record: dict[str, object] | None = None
        gravity: list[dict[str, object]] = []
        forward: dict[str, object] | None = None
        origin: dict[str, object] | None = None
        if full:
            source_files = []
            detections = []
            for index, pose in enumerate(camera.poses):
                relative = f"calibration/source/frames/frame_{index:06d}.png"
                image = Image.new("RGB", (640, 480), (230, 232, 235))
                image_path = context.path(*relative.split("/"))
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(image_path, format="PNG")
                image_hash = sha256_file(context.path(*relative.split("/")))
                source_files.append(
                    {
                        "relative_path": relative,
                        "sha256": image_hash,
                        "media_type": "image/png",
                    }
                )
                x, y, z = pose.transform_world_from_camera.translation
                detections.append(
                    {
                        "frame_id": pose.frame_id,
                        "image_path": relative,
                        "image_sha256": image_hash,
                        "tag_id": 0,
                        "corners_xy": [
                            [200.0, 160.0],
                            [300.0, 160.0],
                            [300.0, 260.0],
                            [200.0, 260.0],
                        ],
                        "decision_margin": 80.0,
                        "hamming": 0,
                        "camera_center_tag_m": [2.0 * x + 1.0, 2.0 * y - 2.0, 2.0 * z + 0.5],
                        "rotation_tag_from_camera": [
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                        ],
                        "pose_error": 0.2,
                        "split": "fitting" if index % 2 == 0 else "heldout",
                    }
                )
            evidence = [
                {
                    "evidence_id": "apriltag_fitting",
                    "evidence_type": "apriltag",
                    "trust": "metric_fiducial",
                    "role": "fitting",
                    "source_files": source_files[::2],
                    "supports_metric_scale": True,
                    "supports_gravity": False,
                    "supports_forward": False,
                    "supports_origin": False,
                    "measurement_uncertainty": 0.001,
                    "configuration": {},
                },
                {
                    "evidence_id": "apriltag_heldout",
                    "evidence_type": "apriltag",
                    "trust": "metric_fiducial",
                    "role": "heldout",
                    "source_files": source_files[1::2],
                    "supports_metric_scale": True,
                    "supports_gravity": False,
                    "supports_forward": False,
                    "supports_origin": False,
                    "measurement_uncertainty": 0.001,
                    "configuration": {},
                },
            ]
            tag_record = {
                "official_repository": "https://github.com/AprilRobotics/apriltag",
                "official_commit": "0e16a12dd380fd607e4afd54712ee9b1ffb9ec8f",
                "code_license": "BSD-2-Clause",
                "tag_family": "tagStandard41h12",
                "tag_id": 0,
                "detection_edge_size_m": 0.1,
                "detector_source_path": "apriltag_pose.h::estimate_tag_pose",
                "world_contract": {
                    "tag_origin_policy": "tag_center",
                    "canonical_up_from_tag_axis": "+Z_tag",
                    "canonical_forward_from_tag_axis": "+X_tag",
                    "mounting_description": "surveyed fixed tag board",
                    "mounting_uncertainty_degrees": 0.1,
                    "origin_uncertainty_m": 0.001,
                },
                "detections": detections,
            }
        return WorldCalibrationManifest.model_validate(
            {
                "schema_version": "0.2.0",
                "run_id": "phase6a_fake",
                "frame_sequence_digest": camera.frame_sequence_digest,
                "camera_reconstruction_path": "calibration/source/camera_reconstruction.json",
                "camera_reconstruction_sha256": sha256_file(
                    context.path("calibration/source/camera_reconstruction.json")
                ),
                "source_scene_ir_path": "calibration/source/scene_ir.json",
                "source_scene_ir_sha256": sha256_file(
                    context.path("calibration/source/scene_ir.json")
                ),
                "evidence": evidence,
                "apriltag": tag_record,
                "known_distance": None,
                "external_metric": [],
                "gravity": gravity,
                "floor_planes": [],
                "forward": forward,
                "origin": origin,
                "evidence_tier": "full_canonical" if full else "none",
            }
        )

    def run(self, context: StageContext) -> StageResult:
        config = CalibrationEvidenceConfig.model_validate(context.config.adapter.config)
        if config.fake_mode is not None:
            camera, scene = self._fake_sources(context)
            manifest = self._fake_manifest(
                context,
                camera,
                scene,
                full=config.fake_mode == "perfect_full_canonical",
            )
        else:
            raw = _load_mapping(context.path("calibration/source/manifest.yaml"))
            normalized = _strip_local_paths(raw)
            assert isinstance(normalized, dict)
            normalized["camera_reconstruction_path"] = (
                "calibration/source/camera_reconstruction.json"
            )
            normalized["camera_reconstruction_sha256"] = sha256_file(
                context.path("calibration/source/camera_reconstruction.json")
            )
            normalized["source_scene_ir_path"] = "calibration/source/scene_ir.json"
            normalized["source_scene_ir_sha256"] = sha256_file(
                context.path("calibration/source/scene_ir.json")
            )
            manifest = WorldCalibrationManifest.model_validate(normalized)
            CameraReconstruction.model_validate_json(
                context.path("calibration/source/camera_reconstruction.json").read_text(
                    encoding="utf-8"
                )
            )
            SceneIR.model_validate_json(
                context.path("calibration/source/scene_ir.json").read_text(encoding="utf-8")
            )
            for record in manifest.evidence:
                for source in record.source_files:
                    if sha256_file(context.path(*source.relative_path.split("/"))) != source.sha256:
                        raise ValueError(
                            f"calibration evidence hash mismatch: {source.relative_path}"
                        )
        atomic_write_json(context.path("calibration/evidence_manifest.json"), manifest)
        return StageResult(
            metrics={
                "evidence_records": len(manifest.evidence),
                "evidence_tier": manifest.evidence_tier.value,
            }
        )


__all__ = ["CalibrationEvidenceAdapter", "CalibrationEvidenceConfig"]
