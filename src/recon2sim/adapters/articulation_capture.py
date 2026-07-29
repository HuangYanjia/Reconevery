from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal

import yaml
from PIL import Image
from pydantic import Field

from recon2sim.adapters.base import (
    HealthcheckResult,
    InputSpec,
    OutputSpec,
    StageContext,
    StageResult,
)
from recon2sim.articulation import (
    capture_evidence_tier,
    sha256_file,
    split_articulation_evidence,
    stable_digest,
)
from recon2sim.artifacts import (
    ArticulatedEligibilityArtifact,
    ArticulatedEligibilityRecord,
    ArticulatedEligibilityStatus,
    ArticulatedJointType,
    ArticulatedPartStateGeometry,
    ArticulatedPartStateGeometryManifest,
    ArticulationBasePrompt,
    ArticulationCaptureManifest,
    ArticulationEvidenceSplit,
    ArticulationMovablePartPrompt,
    ArticulationObjectPrompt,
    ArticulationPartPromptManifest,
    ArticulationStateRecord,
    DenseDepthManifest,
)
from recon2sim.ir import (
    AlignmentStatus,
    AssetType,
    CameraAxes,
    CoordinateConvention,
    GeometrySourceType,
    LinearUnits,
    ScaleStatus,
    SceneIR,
    SceneMetadata,
    StrictModel,
    TransformDirection,
    WorldFrame,
)
from recon2sim.storage import atomic_write_json


class ArticulationCaptureConfig(StrictModel):
    capture_manifest: str | None = None
    part_prompt_manifest: str | None = None
    phase5b_selection: str | None = None
    fake_mode: Literal["success", "single_state", "two_state"] | None = None
    fake_state_count: int = Field(default=3, ge=1, le=8)
    explicit_override: bool = False


def _raw_colmap_convention() -> CoordinateConvention:
    return CoordinateConvention(
        world_frame=WorldFrame.COLMAP_ARBITRARY,
        alignment_status=AlignmentStatus.UNORIENTED,
        camera_axes=CameraAxes.X_RIGHT_Y_DOWN_Z_FORWARD,
        linear_units=LinearUnits.ARBITRARY_UNITS,
        scale_status=ScaleStatus.SCALE_AMBIGUOUS,
        transform_direction=TransformDirection.WORLD_FROM_CAMERA,
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_ascii_ply(path: Path, points: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
        *(f"{x:.9g} {y:.9g} {z:.9g}" for x, y, z in points),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _source_destination(state_id: str, relative_path: str) -> str:
    return f"reconstruction/articulation/source_states/{state_id}/{relative_path}"


class ArticulationCaptureAdapter:
    name = "articulation_capture"
    version = "0.1.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        config = ArticulationCaptureConfig.model_validate(context.config.adapter.config)
        if config.fake_mode is not None:
            return []
        capture_path, prompt_path = self._configured_paths(config)
        capture = _read_yaml(capture_path)
        prompt = ArticulationPartPromptManifest.model_validate(_read_yaml(prompt_path))
        object_id = str(capture["articulated_object_id"])
        prompt_object = next(
            item for item in prompt.objects if item.articulated_object_id == object_id
        )
        part_ids = [
            prompt_object.base.part_id,
            *(part.part_id for part in prompt_object.movable_parts if part.include),
        ]
        specs = [
            InputSpec(
                "reconstruction/articulation/source/capture.yaml",
                "articulation_capture_source",
                source_path=capture_path,
            ),
            InputSpec(
                "reconstruction/articulation/source/parts.yaml",
                "articulation_part_prompt_source",
                source_path=prompt_path,
            ),
        ]
        if config.phase5b_selection is not None:
            specs.append(
                InputSpec(
                    "reconstruction/articulation/source/phase5b_selection.json",
                    "phase5b_selection",
                    source_path=Path(config.phase5b_selection).expanduser().resolve(),
                )
            )
        elif not config.explicit_override:
            raise ValueError(
                "real articulation capture requires phase5b_selection unless explicit_override=true"
            )
        fixed_inputs = (
            ("inputs/manifest.json", "ingest_manifest"),
            ("camera/reconstruction.json", "camera_reconstruction"),
            ("observations/object_tracks.json", "segmentation_tracking"),
            ("reconstruction/dense/depth_manifest.json", "dense_depth_manifest"),
            (
                "reconstruction/dense/undistortion_manifest.json",
                "dense_undistortion_manifest",
            ),
            ("reconstruction/dense/fused.ply", "dense_fused_point_cloud"),
            (
                "reconstruction/measured_objects/geometry_manifest.json",
                "measured_object_geometry",
            ),
            ("validation/phase5a_measured_geometry.json", "phase5a_consistency"),
            ("scene_ir/phase5a_scene.json", "scene_ir"),
        )
        for state in capture.get("states", []):
            if not isinstance(state, dict):
                raise ValueError("capture states must be mappings")
            state_id = str(state["state_id"])
            raw_mapping = state.get("part_track_ids")
            if not isinstance(raw_mapping, dict):
                raise ValueError(
                    f"state {state_id} requires explicit part_track_ids for real capture"
                )
            part_track_ids = {str(key): str(value) for key, value in raw_mapping.items()}
            if set(part_track_ids) != set(part_ids):
                raise ValueError(
                    f"state {state_id} part_track_ids must map exactly {sorted(part_ids)}"
                )
            if len(part_track_ids) != len(set(part_track_ids.values())):
                raise ValueError(f"state {state_id} assigns one SAM track to multiple stable parts")
            run_root = Path(str(state["run_dir"])).expanduser().resolve()
            for relative_path, artifact_type in fixed_inputs:
                specs.append(
                    InputSpec(
                        _source_destination(state_id, relative_path),
                        artifact_type,
                        source_path=run_root / relative_path,
                        materialization_mode="reflink_or_copy",
                    )
                )
            depth = DenseDepthManifest.model_validate_json(
                (run_root / "reconstruction/dense/depth_manifest.json").read_text(encoding="utf-8")
            )
            for record in depth.records:
                specs.append(
                    InputSpec(
                        _source_destination(state_id, record.depth_path),
                        "dense_depth_map",
                        source_path=run_root / record.depth_path,
                        materialization_mode="reflink_or_copy",
                    )
                )
            tracks = _read_json(run_root / "observations/object_tracks.json")
            measured = _read_json(
                run_root / "reconstruction/measured_objects/geometry_manifest.json"
            )
            track_by_id = {item["object_id"]: item for item in tracks["tracks"]}
            measured_by_id = {item["object_id"]: item for item in measured["hypotheses"]}
            for part_id in part_ids:
                track_id = part_track_ids[part_id]
                hypothesis = measured_by_id.get(track_id)
                track = track_by_id.get(track_id)
                if hypothesis is None or track is None or hypothesis.get("point_cloud") is None:
                    raise ValueError(
                        f"state {state_id} track {track_id!r} has no Phase 5A measured "
                        f"geometry for stable part {part_id!r}"
                    )
                point_path = str(hypothesis["point_cloud"]["relative_path"])
                specs.append(
                    InputSpec(
                        _source_destination(state_id, point_path),
                        "measured_articulated_part_point_cloud",
                        source_path=run_root / point_path,
                        materialization_mode="reflink_or_copy",
                    )
                )
                for observation in track["observations"]:
                    mask_path = str(observation["mask_path"])
                    specs.append(
                        InputSpec(
                            _source_destination(state_id, mask_path),
                            "articulated_part_mask",
                            source_path=run_root / mask_path,
                            materialization_mode="reflink_or_copy",
                        )
                    )
        return specs

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        if context is None:
            return HealthcheckResult(False, "articulation capture healthcheck requires --config")
        try:
            config = ArticulationCaptureConfig.model_validate(context.config.adapter.config)
            if config.fake_mode is None:
                for path in self._configured_paths(config):
                    if not path.is_file():
                        raise ValueError(f"capture input does not exist: {path}")
        except (ValueError, StopIteration) as exc:
            return HealthcheckResult(False, str(exc))
        return HealthcheckResult(True, "typed multi-state capture validation available")

    def prepare(self, context: StageContext) -> None:
        context.path("reconstruction", "articulation", "measured_states").mkdir(
            parents=True, exist_ok=True
        )

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        root = "reconstruction/articulation"
        return [
            OutputSpec(
                f"{root}/eligibility.json",
                "articulated_eligibility",
                "application/json",
                self.name,
                validation="json",
                model=ArticulatedEligibilityArtifact,
            ),
            OutputSpec(
                f"{root}/part_prompt_manifest.json",
                "articulation_part_prompt_manifest",
                "application/json",
                self.name,
                validation="json",
                model=ArticulationPartPromptManifest,
            ),
            OutputSpec(
                f"{root}/capture_manifest.json",
                "articulation_capture_manifest",
                "application/json",
                self.name,
                validation="json",
                model=ArticulationCaptureManifest,
            ),
            OutputSpec(
                f"{root}/evidence_split.json",
                "articulation_evidence_split",
                "application/json",
                self.name,
                validation="json",
                model=ArticulationEvidenceSplit,
            ),
            OutputSpec(
                f"{root}/measured_states/manifest.json",
                "articulated_part_state_geometry_manifest",
                "application/json",
                self.name,
                validation="json",
                model=ArticulatedPartStateGeometryManifest,
            ),
            OutputSpec(
                f"{root}/measured_states/fitting_manifest.json",
                "articulated_fitting_part_state_geometry_manifest",
                "application/json",
                self.name,
                validation="json",
                model=ArticulatedPartStateGeometryManifest,
            ),
            OutputSpec(
                f"{root}/measured_states/heldout_manifest.json",
                "articulated_heldout_part_state_geometry_manifest",
                "application/json",
                self.name,
                validation="json",
                model=ArticulatedPartStateGeometryManifest,
            ),
        ]

    def run(self, context: StageContext) -> StageResult:
        config = ArticulationCaptureConfig.model_validate(context.config.adapter.config)
        if config.fake_mode is not None:
            return self._run_fake(context, config)
        return self._run_materialized(context, config)

    @staticmethod
    def _configured_paths(config: ArticulationCaptureConfig) -> tuple[Path, Path]:
        if config.capture_manifest is None or config.part_prompt_manifest is None:
            raise ValueError("real capture requires capture_manifest and part_prompt_manifest")
        return (
            Path(config.capture_manifest).expanduser().resolve(),
            Path(config.part_prompt_manifest).expanduser().resolve(),
        )

    def _run_fake(
        self,
        context: StageContext,
        config: ArticulationCaptureConfig,
    ) -> StageResult:
        state_count = (
            1
            if config.fake_mode == "single_state"
            else 2
            if config.fake_mode == "two_state"
            else config.fake_state_count
        )
        prompt = ArticulationPartPromptManifest(
            objects=[
                ArticulationObjectPrompt(
                    articulated_object_id="cabinet_0001",
                    semantic_label="cabinet",
                    base=ArticulationBasePrompt(
                        part_id="cabinet_body",
                        prompt_id="cabinet_body",
                        label="cabinet body",
                    ),
                    movable_parts=[
                        ArticulationMovablePartPrompt(
                            part_id="drawer",
                            prompt_id="drawer",
                            label="drawer",
                            expected_joint_hint=ArticulatedJointType.PRISMATIC,
                        )
                    ],
                )
            ]
        )
        root = context.path("reconstruction", "articulation")
        atomic_write_json(root / "part_prompt_manifest.json", prompt)
        atomic_write_json(
            root / "reference_phase5a_scene.json",
            SceneIR(
                schema_version="0.1.5",
                metadata=SceneMetadata(
                    scene_id="cabinet_multistate",
                    name="Synthetic articulated cabinet",
                    coordinate_convention=_raw_colmap_convention(),
                    source=GeometrySourceType.MOCK,
                ),
            ),
        )
        states: list[ArticulationStateRecord] = []
        geometries: list[ArticulatedPartStateGeometry] = []
        outputs: list[OutputSpec] = []
        registered_by_state: dict[str, list[str]] = {}
        for index in range(state_count):
            state_id = f"state_{index:03d}"
            frames = [f"{state_id}_frame_{frame:03d}" for frame in range(6)]
            registered_by_state[state_id] = frames
            evidence_root = f"reconstruction/articulation/measured_states/{state_id}/evidence"
            dense_records = []
            mask_paths_by_part: dict[str, list[str]] = {
                "cabinet_body": [],
                "drawer": [],
            }
            for frame in frames:
                dense_paths = {
                    "depth_path": f"{evidence_root}/dense/{frame}.depth.bin",
                    "normal_path": f"{evidence_root}/dense/{frame}.normal.bin",
                    "consistency_graph_path": (f"{evidence_root}/dense/{frame}.consistency.bin"),
                }
                dense_hashes = {}
                for field, relative_path in dense_paths.items():
                    dense_path = context.path(*Path(relative_path).parts)
                    dense_path.parent.mkdir(parents=True, exist_ok=True)
                    dense_path.write_bytes(f"fake {field} {frame}\n".encode())
                    dense_hashes[field] = sha256_file(dense_path)
                    outputs.append(
                        OutputSpec(
                            relative_path,
                            (
                                "articulation_dense_depth_map"
                                if field == "depth_path"
                                else "dense_mvs_binary"
                            ),
                            "application/octet-stream",
                            self.name,
                        )
                    )
                dense_records.append(
                    {
                        "frame_id": frame,
                        **dense_paths,
                        "dimensions": [16, 16],
                        "depth_channels": 1,
                        "normal_channels": 3,
                        "positive_finite_depth_count": 256,
                        "valid_depth_ratio": 1.0,
                        "depth_percentiles": {"p50": 1.0},
                        "finite_normal_ratio": 1.0,
                        "consistency_valid_pixel_count": 256,
                        "mean_consistency_source_count": 2.0,
                        "median_consistency_source_count": 2.0,
                        "source_view_ids": [0, 1],
                        "depth_sha256": dense_hashes["depth_path"],
                        "normal_sha256": dense_hashes["normal_path"],
                        "consistency_sha256": dense_hashes["consistency_graph_path"],
                        "warnings": [],
                    }
                )
                for part_id in mask_paths_by_part:
                    relative_mask = f"{evidence_root}/masks/{part_id}/{frame}.png"
                    mask_path = context.path(*Path(relative_mask).parts)
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("L", (16, 16), 255).save(mask_path)
                    mask_paths_by_part[part_id].append(relative_mask)
                    outputs.append(
                        OutputSpec(
                            relative_mask,
                            "articulated_part_mask",
                            "image/png",
                            self.name,
                            validation="binary_png",
                        )
                    )
            evidence_payloads = {
                f"{evidence_root}/camera_reconstruction.json": (
                    {"registered_frame_ids": frames, "poses": []},
                    "camera_reconstruction",
                ),
                f"{evidence_root}/object_tracks.json": (
                    {"tracks": []},
                    "segmentation_tracking",
                ),
                f"{evidence_root}/undistortion_manifest.json": (
                    {
                        "policy": "official_colmap_image_undistorter",
                        "records": [],
                    },
                    "dense_undistortion_manifest",
                ),
                f"{evidence_root}/depth_manifest.json": (
                    {
                        "map_type": "geometric",
                        "records": dense_records,
                        "failed_frame_ids": [],
                    },
                    "dense_depth_manifest",
                ),
            }
            for relative_path, (payload, artifact_type) in evidence_payloads.items():
                atomic_write_json(context.path(*Path(relative_path).parts), payload)
                outputs.append(
                    OutputSpec(
                        relative_path,
                        artifact_type,
                        "application/json",
                        self.name,
                        validation="json",
                    )
                )
            base = [(float(x), float(y), float(z)) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
            offset = 0.35 * index
            drawer = [
                (offset + 0.2 * x, 0.2 + 0.2 * y, 0.2 + 0.2 * z)
                for x in (0, 1)
                for y in (0, 1)
                for z in (0, 1)
            ]
            hashes: dict[str, str] = {}
            for part_id, label, points in (
                ("cabinet_body", "cabinet body", base),
                ("drawer", "drawer", drawer),
            ):
                relative_path = (
                    "reconstruction/articulation/measured_states/"
                    f"{state_id}/{part_id}/measured_points.ply"
                )
                path = context.path(*Path(relative_path).parts)
                _write_ascii_ply(path, points)
                content_hash = sha256_file(path)
                hashes[part_id] = content_hash
                outputs.append(
                    OutputSpec(
                        relative_path,
                        "measured_articulated_part_point_cloud",
                        "model/ply",
                        self.name,
                    )
                )
                geometries.append(
                    ArticulatedPartStateGeometry(
                        state_id=state_id,
                        articulated_object_id="cabinet_0001",
                        part_id=part_id,
                        source_track_id=f"{part_id}_{index:04d}",
                        prompt_id=part_id,
                        semantic_label=label,
                        measured_point_cloud_path=relative_path,
                        measured_point_cloud_sha256=content_hash,
                        point_count=len(points),
                        normal_count=0,
                        supporting_frame_ids=frames,
                        mask_paths=mask_paths_by_part[part_id],
                        state_alignment_sha256="0" * 64,
                        transformed_to_reference_frame=index == 0,
                        coordinate_convention=_raw_colmap_convention(),
                        scale_status=ScaleStatus.SCALE_AMBIGUOUS,
                    )
                )
            states.append(
                ArticulationStateRecord(
                    state_id=state_id,
                    run_dir=f"synthetic:{state_id}",
                    semantic_state_label=("closed", "half_open", "open")[min(index, 2)],
                    part_track_ids={
                        "cabinet_body": f"cabinet_body_{index:04d}",
                        "drawer": f"drawer_{index:04d}",
                    },
                    phase5a_consistency_passed=True,
                    ingest_manifest_sha256=stable_digest({"state": state_id, "ingest": True}),
                    frame_sequence_digest=stable_digest(frames),
                    camera_reconstruction_sha256=stable_digest({"state": state_id, "camera": True}),
                    segmentation_tracking_sha256=stable_digest(
                        {"state": state_id, "segmentation": True}
                    ),
                    dense_depth_manifest_sha256=stable_digest({"state": state_id, "dense": True}),
                    measured_geometry_sha256=stable_digest({"state": state_id, "measured": True}),
                    part_mask_hashes={
                        part_id: stable_digest({"state": state_id, "mask": part_id})
                        for part_id in hashes
                    },
                    measured_part_cloud_hashes=hashes,
                    registered_frame_ids=frames,
                    camera_evidence_path=(f"{evidence_root}/camera_reconstruction.json"),
                    segmentation_evidence_path=(f"{evidence_root}/object_tracks.json"),
                    undistortion_evidence_path=(f"{evidence_root}/undistortion_manifest.json"),
                    depth_evidence_path=(f"{evidence_root}/depth_manifest.json"),
                    dense_map_hashes={},
                )
            )
        return self._write_manifests(
            context,
            prompt,
            states,
            geometries,
            registered_by_state,
            phase5b_selection_sha256=stable_digest("fake_phase5b_selection"),
            explicitly_overridden=config.explicit_override,
            outputs=outputs,
        )

    def _run_materialized(
        self,
        context: StageContext,
        config: ArticulationCaptureConfig,
    ) -> StageResult:
        root = context.path("reconstruction", "articulation")
        source = root / "source"
        prompt = ArticulationPartPromptManifest.model_validate(_read_yaml(source / "parts.yaml"))
        capture_source = _read_yaml(source / "capture.yaml")
        object_id = str(capture_source["articulated_object_id"])
        prompt_object = next(
            item for item in prompt.objects if item.articulated_object_id == object_id
        )
        states: list[ArticulationStateRecord] = []
        geometries: list[ArticulatedPartStateGeometry] = []
        registered_by_state: dict[str, list[str]] = {}
        outputs: list[OutputSpec] = []
        routing_path = source / "phase5b_selection.json"
        phase5b_hash = (
            sha256_file(routing_path)
            if routing_path.is_file()
            else stable_digest(
                {
                    "explicit_override": True,
                    "articulated_object_id": object_id,
                }
            )
        )
        reference_scene_source: Path | None = None
        for raw_state in capture_source["states"]:
            state_id = str(raw_state["state_id"])
            state_root = root / "source_states" / state_id
            phase5a = _read_json(state_root / "validation/phase5a_measured_geometry.json")
            if not phase5a.get("passed", False):
                raise ValueError(f"articulation state {state_id} did not pass Phase 5A")
            ingest_path = state_root / "inputs/manifest.json"
            camera_path = state_root / "camera/reconstruction.json"
            tracks_path = state_root / "observations/object_tracks.json"
            depth_path = state_root / "reconstruction/dense/depth_manifest.json"
            measured_path = state_root / "reconstruction/measured_objects/geometry_manifest.json"
            undistortion_path = state_root / "reconstruction/dense/undistortion_manifest.json"
            ingest = _read_json(ingest_path)
            camera = _read_json(camera_path)
            tracks = _read_json(tracks_path)
            measured = _read_json(measured_path)
            depth = DenseDepthManifest.model_validate_json(depth_path.read_text(encoding="utf-8"))
            track_by_id = {item["object_id"]: item for item in tracks["tracks"]}
            measured_by_id = {item["object_id"]: item for item in measured["hypotheses"]}
            point_hashes: dict[str, str] = {}
            mask_hashes: dict[str, str] = {}
            registered = list(camera["registered_frame_ids"])
            registered_by_state[state_id] = registered
            part_specs = [
                (
                    prompt_object.base.part_id,
                    prompt_object.base.prompt_id,
                    prompt_object.base.label,
                ),
                *(
                    (part.part_id, part.prompt_id, part.label)
                    for part in prompt_object.movable_parts
                    if part.include
                ),
            ]
            raw_mapping = raw_state.get("part_track_ids")
            if not isinstance(raw_mapping, dict):
                raise ValueError(
                    f"state {state_id} requires explicit part_track_ids for real capture"
                )
            part_track_ids = {str(key): str(value) for key, value in raw_mapping.items()}
            stable_part_ids = {part_id for part_id, _, _ in part_specs}
            if set(part_track_ids) != stable_part_ids:
                raise ValueError(
                    f"state {state_id} part_track_ids must map exactly {sorted(stable_part_ids)}"
                )
            if len(part_track_ids) != len(set(part_track_ids.values())):
                raise ValueError(f"state {state_id} assigns one SAM track to multiple stable parts")
            for part_id, prompt_id, label in part_specs:
                track_id = part_track_ids[part_id]
                if track_id not in measured_by_id or track_id not in track_by_id:
                    raise ValueError(
                        f"state {state_id} mapping {part_id!r}->{track_id!r} does not "
                        "reference both a canonical SAM track and Phase 5A hypothesis"
                    )
                hypothesis = measured_by_id[track_id]
                track = track_by_id[track_id]
                if hypothesis.get("point_cloud") is None:
                    raise ValueError(
                        f"state {state_id} mapped track {track_id!r} has no measured point cloud"
                    )
                original_point_path = str(hypothesis["point_cloud"]["relative_path"])
                source_point = state_root / original_point_path
                output_relative = (
                    "reconstruction/articulation/measured_states/"
                    f"{state_id}/{part_id}/measured_points.ply"
                )
                output_point = context.path(*Path(output_relative).parts)
                output_point.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_point, output_point)
                point_hashes[part_id] = sha256_file(output_point)
                outputs.append(
                    OutputSpec(
                        output_relative,
                        "measured_articulated_part_point_cloud",
                        "model/ply",
                        self.name,
                    )
                )
                output_masks: list[str] = []
                for observation in track["observations"]:
                    original_mask = str(observation["mask_path"])
                    source_mask = state_root / original_mask
                    mask_relative = (
                        "reconstruction/articulation/measured_states/"
                        f"{state_id}/{part_id}/masks/{observation['frame_id']}.png"
                    )
                    destination = context.path(*Path(mask_relative).parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_mask, destination)
                    output_masks.append(mask_relative)
                    mask_hashes[f"{part_id}:{observation['frame_id']}"] = sha256_file(destination)
                    outputs.append(
                        OutputSpec(
                            mask_relative,
                            "articulated_part_mask",
                            "image/png",
                            self.name,
                            validation="binary_png",
                        )
                    )
                geometries.append(
                    ArticulatedPartStateGeometry(
                        state_id=state_id,
                        articulated_object_id=object_id,
                        part_id=part_id,
                        source_track_id=track_id,
                        prompt_id=prompt_id,
                        semantic_label=label,
                        measured_point_cloud_path=output_relative,
                        measured_point_cloud_sha256=point_hashes[part_id],
                        point_count=int(hypothesis["point_cloud"]["point_count"]),
                        normal_count=int(hypothesis.get("validated_sample_count", 0)),
                        supporting_frame_ids=sorted(
                            {
                                *(
                                    str(frame_id)
                                    for frame_id in hypothesis.get("supporting_frame_ids", [])
                                ),
                                *(
                                    str(observation["frame_id"])
                                    for observation in hypothesis.get("observations", [])
                                    if isinstance(observation, dict)
                                    and observation.get("registered") is True
                                    and int(observation.get("validated_sample_count", 0)) > 0
                                    and observation.get("frame_id")
                                ),
                            }
                            & set(registered)
                        ),
                        mask_paths=output_masks,
                        state_alignment_sha256="0" * 64,
                        transformed_to_reference_frame=(
                            state_id == capture_source["reference_state_id"]
                        ),
                        coordinate_convention=_raw_colmap_convention(),
                        scale_status=ScaleStatus.SCALE_AMBIGUOUS,
                    )
                )
            evidence_root_relative = (
                f"reconstruction/articulation/measured_states/{state_id}/evidence"
            )
            evidence_root = context.path(*Path(evidence_root_relative).parts)
            evidence_root.mkdir(parents=True, exist_ok=True)
            evidence_paths = {
                "camera": f"{evidence_root_relative}/camera_reconstruction.json",
                "segmentation": f"{evidence_root_relative}/object_tracks.json",
                "undistortion": f"{evidence_root_relative}/undistortion_manifest.json",
                "depth": f"{evidence_root_relative}/depth_manifest.json",
            }
            shutil.copy2(camera_path, context.path(*Path(evidence_paths["camera"]).parts))
            shutil.copy2(tracks_path, context.path(*Path(evidence_paths["segmentation"]).parts))
            shutil.copy2(
                undistortion_path,
                context.path(*Path(evidence_paths["undistortion"]).parts),
            )
            dense_hashes: dict[str, str] = {}
            rewritten_records = []
            for record in depth.records:
                source_depth = state_root / record.depth_path
                relative_depth = (
                    f"{evidence_root_relative}/depth_maps/{record.frame_id}.geometric.bin"
                )
                destination_depth = context.path(*Path(relative_depth).parts)
                destination_depth.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_depth, destination_depth)
                content_hash = sha256_file(destination_depth)
                if content_hash != record.depth_sha256:
                    raise ValueError(
                        f"state {state_id} dense depth hash changed for {record.frame_id}"
                    )
                dense_hashes[record.frame_id] = content_hash
                rewritten_records.append(record.model_copy(update={"depth_path": relative_depth}))
                outputs.append(
                    OutputSpec(
                        relative_depth,
                        "articulation_dense_depth_map",
                        "application/octet-stream",
                        self.name,
                    )
                )
            atomic_write_json(
                context.path(*Path(evidence_paths["depth"]).parts),
                depth.model_copy(update={"records": rewritten_records}),
            )
            for path, artifact_type in (
                (evidence_paths["camera"], "camera_reconstruction"),
                (evidence_paths["segmentation"], "segmentation_tracking"),
                (evidence_paths["undistortion"], "dense_undistortion_manifest"),
                (evidence_paths["depth"], "dense_depth_manifest"),
            ):
                outputs.append(
                    OutputSpec(
                        path,
                        artifact_type,
                        "application/json",
                        self.name,
                        validation="json",
                    )
                )
            if state_id == capture_source["reference_state_id"]:
                reference_scene_source = state_root / "scene_ir/phase5a_scene.json"
            states.append(
                ArticulationStateRecord(
                    state_id=state_id,
                    run_dir=str(raw_state["run_dir"]),
                    semantic_state_label=str(raw_state["semantic_state_label"]),
                    part_track_ids=part_track_ids,
                    phase5a_consistency_passed=True,
                    ingest_manifest_sha256=sha256_file(ingest_path),
                    frame_sequence_digest=str(ingest["frame_sequence_digest"]),
                    camera_reconstruction_sha256=sha256_file(camera_path),
                    segmentation_tracking_sha256=sha256_file(tracks_path),
                    dense_depth_manifest_sha256=sha256_file(depth_path),
                    measured_geometry_sha256=sha256_file(measured_path),
                    part_mask_hashes=mask_hashes,
                    measured_part_cloud_hashes=point_hashes,
                    registered_frame_ids=registered,
                    camera_evidence_path=evidence_paths["camera"],
                    segmentation_evidence_path=evidence_paths["segmentation"],
                    undistortion_evidence_path=evidence_paths["undistortion"],
                    depth_evidence_path=evidence_paths["depth"],
                    dense_map_hashes=dense_hashes,
                )
            )
        if reference_scene_source is None:
            raise ValueError("reference state Scene IR was not materialized")
        shutil.copy2(reference_scene_source, root / "reference_phase5a_scene.json")
        return self._write_manifests(
            context,
            prompt,
            states,
            geometries,
            registered_by_state,
            phase5b_selection_sha256=phase5b_hash,
            explicitly_overridden=config.explicit_override,
            outputs=outputs,
            reference_state_id=str(capture_source["reference_state_id"]),
        )

    def _write_manifests(
        self,
        context: StageContext,
        prompt: ArticulationPartPromptManifest,
        states: list[ArticulationStateRecord],
        geometries: list[ArticulatedPartStateGeometry],
        registered_by_state: dict[str, list[str]],
        *,
        phase5b_selection_sha256: str,
        explicitly_overridden: bool,
        outputs: list[OutputSpec],
        reference_state_id: str | None = None,
    ) -> StageResult:
        root = context.path("reconstruction", "articulation")
        prompt_object = prompt.objects[0]
        atomic_write_json(root / "part_prompt_manifest.json", prompt)
        capture = ArticulationCaptureManifest(
            articulated_object_id=prompt_object.articulated_object_id,
            reference_state_id=reference_state_id or states[0].state_id,
            states=states,
            prompt_manifest_sha256=sha256_file(root / "part_prompt_manifest.json"),
            capture_state_count=len(states),
            capture_evidence_tier=capture_evidence_tier(len(states)),
        )
        atomic_write_json(root / "capture_manifest.json", capture)
        atomic_write_json(
            root / "eligibility.json",
            ArticulatedEligibilityArtifact(
                phase5b_selection_sha256=phase5b_selection_sha256,
                records=[
                    ArticulatedEligibilityRecord(
                        articulated_object_id=prompt_object.articulated_object_id,
                        semantic_label=prompt_object.semantic_label,
                        asset_type_hint=AssetType.ARTICULATED,
                        state_count=len(states),
                        movable_part_count=len(prompt_object.movable_parts),
                        status=(
                            ArticulatedEligibilityStatus.ELIGIBLE_MULTI_STATE
                            if len(states) >= 2
                            else ArticulatedEligibilityStatus.ELIGIBLE_PRIOR_ONLY
                        ),
                        explicitly_overridden=explicitly_overridden,
                        reason="explicit typed static-state articulation capture",
                    )
                ],
            ),
        )
        atomic_write_json(
            root / "evidence_split.json",
            split_articulation_evidence(
                prompt_object.articulated_object_id,
                [state.state_id for state in states],
                registered_by_state,
                seed=context.seed,
            ),
        )
        geometry_manifest = ArticulatedPartStateGeometryManifest(
            capture_manifest_sha256=sha256_file(root / "capture_manifest.json"),
            geometries=geometries,
        )
        atomic_write_json(
            root / "measured_states" / "manifest.json",
            geometry_manifest,
        )
        split = ArticulationEvidenceSplit.model_validate_json(
            (root / "evidence_split.json").read_text(encoding="utf-8")
        )
        fitting_states = {
            *split.candidate_generation_states,
            *split.kinematic_fitting_states,
        }
        atomic_write_json(
            root / "measured_states" / "fitting_manifest.json",
            geometry_manifest.model_copy(
                update={
                    "geometries": [item for item in geometries if item.state_id in fitting_states]
                }
            ),
        )
        atomic_write_json(
            root / "measured_states" / "heldout_manifest.json",
            geometry_manifest.model_copy(
                update={
                    "geometries": [
                        item
                        for item in geometries
                        if item.state_id in set(split.heldout_validation_states)
                    ]
                }
            ),
        )
        return StageResult(
            outputs=[
                *outputs,
                OutputSpec(
                    "reconstruction/articulation/reference_phase5a_scene.json",
                    "scene_ir",
                    "application/json",
                    self.name,
                    validation="scene_ir",
                    model=SceneIR,
                ),
            ],
            metrics={"state_count": len(states), "part_geometry_count": len(geometries)},
        )


__all__ = ["ArticulationCaptureAdapter", "ArticulationCaptureConfig"]
