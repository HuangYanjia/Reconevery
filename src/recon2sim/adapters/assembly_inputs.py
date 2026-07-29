from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml

from recon2sim.adapters.base import (
    HealthcheckResult,
    InputSpec,
    OutputSpec,
    StageContext,
    StageResult,
)
from recon2sim.artifacts import SceneAssemblyInputManifest
from recon2sim.assembly import IDENTITY_MATRIX4, stable_digest
from recon2sim.calibration import sha256_file
from recon2sim.ir import SceneIR, StrictModel
from recon2sim.storage import atomic_write_json

FakeAssemblyMode = Literal[
    "source_arbitrary_measured_only",
    "full_canonical_scene",
    "metric_only_scene",
    "gravity_only_scene",
    "global_context_plus_measured",
    "accepted_rigid_candidate",
    "license_blocked_rigid_candidate",
    "accepted_articulated_candidate",
    "rejected_articulated_candidate",
    "unresolved_object",
    "cross_lineage_asset_rejection",
    "accepted_state_alignment_lineage",
    "missing_asset_hash",
    "double_world_transform",
    "double_articulated_transform",
    "preview_material_loss",
    "deployment_bundle_excluding_research_asset",
    "empty_global_context",
    "no_selected_candidates",
    "path_escape",
    "timeout",
    "worker_modifying_upstream_assets",
]


class AssemblyInputsConfig(StrictModel):
    manifest_path: str | None = None
    fake_mode: FakeAssemblyMode | None = None


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ValueError(f"assembly source path must be relative: {value!r}")
    return value


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"assembly input manifest must be a mapping: {path}")
    return value


def _manifest_path(config: AssemblyInputsConfig) -> Path:
    if config.manifest_path is None:
        raise ValueError("real assembly inputs require manifest_path")
    path = Path(config.manifest_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"assembly input manifest does not exist: {path}")
    return path


def _local_source_records(value: object) -> list[tuple[str, Path, str | None]]:
    result: list[tuple[str, Path, str | None]] = []

    def visit(item: object) -> None:
        if isinstance(item, dict):
            local_value = item.get("local_path")
            if local_value is not None:
                destination_value = item.get("path", item.get("asset_path"))
                if destination_value is None:
                    raise ValueError("assembly local_path requires path or asset_path")
                destination = _safe_relative(str(destination_value))
                source = Path(str(local_value)).expanduser().resolve()
                expected = item.get("sha256", item.get("asset_sha256"))
                result.append(
                    (destination, source, str(expected) if expected is not None else None)
                )
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    unique: dict[str, tuple[Path, str | None]] = {}
    for destination, source, expected in result:
        previous = unique.get(destination)
        if previous is not None and previous != (source, expected):
            raise ValueError(f"assembly destination {destination!r} has conflicting sources")
        unique[destination] = (source, expected)
    return [
        (destination, source, expected)
        for destination, (source, expected) in sorted(unique.items())
    ]


def _normalized_manifest(value: object, root: Path) -> object:
    if isinstance(value, dict):
        result = {
            str(key): _normalized_manifest(child, root)
            for key, child in value.items()
            if key != "local_path"
        }
        if "local_path" in value:
            relative = _safe_relative(str(value.get("path", value.get("asset_path"))))
            digest = sha256_file(root / relative)
            if "asset_path" in value:
                result["asset_sha256"] = digest
            else:
                result["sha256"] = digest
        return result
    if isinstance(value, list):
        return [_normalized_manifest(child, root) for child in value]
    return value


def _write_ascii_ply(path: Path, *, offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                f"{offset} 0 0",
                f"{offset + 1} 0 0",
                f"{offset} 1 0",
                f"{offset} 0 1",
                "",
            ]
        ),
        encoding="ascii",
    )


def _fake_scene(context: StageContext) -> tuple[Path, Path]:
    camera_path = context.path("assembly/source/camera_reconstruction.json")
    camera_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        camera_path,
        {
            "schema_version": "0.1.0",
            "frame_sequence_digest": "a" * 64,
            "registered_frame_ids": ["frame_000000", "frame_000001"],
        },
    )
    scene_path = context.path("assembly/source/scene_ir.json")
    confidence = {"score": 1.0, "method": "phase6b_fake"}
    provenance = {
        "adapter_name": "scene_assembly_inputs",
        "adapter_version": "0.1.0",
        "configuration": {"fake": True},
        "input_artifact_paths": [],
        "output_artifact_paths": ["assembly/source/scene_ir.json"],
        "confidence": confidence,
        "source": "mock",
        "timestamp": "2026-01-01T00:00:00Z",
    }
    convention = {
        "world_frame": "colmap_arbitrary",
        "alignment_status": "unoriented",
        "camera_axes": "x_right_y_down_z_forward",
        "linear_units": "arbitrary_units",
        "scale_status": "scale_ambiguous",
        "transform_direction": "world_from_camera",
    }
    scene = SceneIR.model_validate(
        {
            "schema_version": "0.1.9",
            "metadata": {
                "scene_id": "phase6b_fake_scene",
                "name": "Phase 6B fake source scene",
                "coordinate_convention": convention,
                "created_at": "2026-01-01T00:00:00Z",
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
    atomic_write_json(scene_path, scene)
    return camera_path, scene_path


def _license(*, production: bool, research: bool = True) -> dict[str, object]:
    return {
        "license_id": "fake-production" if production else "fake-research",
        "license_name": "CC0-1.0" if production else "Research-Evaluation-Only",
        "research_evaluation_allowed": research,
        "production_selectable": production,
        "commercial_review_status": "approved" if production else "research_only",
        "restrictions": [] if production else ["research evaluation only"],
    }


def _fake_manifest(context: StageContext, mode: FakeAssemblyMode) -> SceneAssemblyInputManifest:
    if mode == "path_escape":
        _safe_relative("../escape.ply")
    camera_path, scene_path = _fake_scene(context)
    measured_path = context.path("assembly/source/assets/cup_measured.ply")
    _write_ascii_ply(measured_path, offset=0.0)
    global_path = context.path("assembly/source/assets/global_context.ply")
    _write_ascii_ply(global_path, offset=-0.25)
    candidate_path = context.path("assembly/source/assets/cup_candidate.ply")
    _write_ascii_ply(candidate_path, offset=0.05)
    evaluation_path = context.path("assembly/source/evaluations/cup_candidate.json")
    atomic_write_json(
        evaluation_path,
        {
            "candidate_id": "cup_candidate",
            "passed_hard_gates": mode
            not in {"rejected_articulated_candidate", "no_selected_candidates"},
        },
    )
    measured_motion_path = context.path("assembly/source/articulation/measured_motion.json")
    atomic_write_json(
        measured_motion_path,
        {
            "joint_type": "prismatic",
            "axis": [1.0, 0.0, 0.0],
            "observed_positions": {"closed": 0.0, "open": 1.0},
        },
    )
    kinematic_path = context.path("assembly/source/articulation/kinematic_bundle.json")
    atomic_write_json(
        kinematic_path,
        {
            "candidate_id": "cup_candidate",
            "links": ["base", "drawer"],
            "joints": [{"joint_id": "drawer_joint", "joint_type": "prismatic"}],
            "prismatic_position_scale_to_m": 2.0,
        },
    )
    reference = {
        "path": "assembly/source/scene_ir.json",
        "sha256": sha256_file(scene_path),
    }
    lineage: dict[str, object] = {
        "lineage_id": "fake_lineage",
        "frame_sequence_digest": "a" * 64,
        "camera_reconstruction": {
            "path": "assembly/source/camera_reconstruction.json",
            "sha256": sha256_file(camera_path),
        },
        "source_scene_ir": reference,
        "world_frame": "colmap_arbitrary",
    }
    lineages: list[dict[str, object]] = [lineage]
    object_lineage = "fake_lineage"
    if mode == "cross_lineage_asset_rejection":
        lineages.append(
            {
                **lineage,
                "lineage_id": "foreign_lineage",
                "frame_sequence_digest": "b" * 64,
            }
        )
        object_lineage = "foreign_lineage"
    elif mode == "accepted_state_alignment_lineage":
        alignment_path = context.path("assembly/source/state_alignment.json")
        atomic_write_json(alignment_path, {"accepted": True, "state": "state_001"})
        lineages.append(
            {
                **lineage,
                "lineage_id": "aligned_state_lineage",
                "frame_sequence_digest": "b" * 64,
                "connected_to_lineage_id": "fake_lineage",
                "accepted_alignment": {
                    "path": "assembly/source/state_alignment.json",
                    "sha256": sha256_file(alignment_path),
                },
                "transform_connected_from_lineage": list(IDENTITY_MATRIX4),
            }
        )
        object_lineage = "aligned_state_lineage"
    identity = list(IDENTITY_MATRIX4)
    measured_object_transform = identity
    if mode == "double_world_transform":
        measured_object_transform = [
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    assets: list[dict[str, object]] = [
        {
            "asset_id": "cup_measured",
            "object_id": "cup_0001",
            "lineage_id": object_lineage,
            "role": "measured_anchor",
            "source": "measured",
            "asset_path": "assembly/source/assets/cup_measured.ply",
            "asset_sha256": sha256_file(measured_path),
            "format": "ply",
            "asset_native_space": "reference_world",
            "asset_to_object": identity,
            "object_to_source_world": measured_object_transform,
            "bounds_native": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "license": _license(production=True),
        }
    ]
    global_modes = {
        "global_context_plus_measured",
        "full_canonical_scene",
        "metric_only_scene",
        "gravity_only_scene",
        "accepted_rigid_candidate",
        "license_blocked_rigid_candidate",
        "deployment_bundle_excluding_research_asset",
        "preview_material_loss",
    }
    if mode in global_modes:
        assets.append(
            {
                "asset_id": "global_context",
                "object_id": None,
                "lineage_id": "fake_lineage",
                "role": "global_context",
                "source": "generated",
                "asset_path": "assembly/source/assets/global_context.ply",
                "asset_sha256": sha256_file(global_path),
                "format": "ply",
                "asset_native_space": "global_context",
                "asset_to_object": identity,
                "object_to_source_world": identity,
                "bounds_native": [-0.25, 0.0, 0.0, 0.75, 1.0, 1.0],
                "license": _license(
                    production=mode != "deployment_bundle_excluding_research_asset"
                ),
            }
        )
    candidate_modes = {
        "accepted_rigid_candidate",
        "license_blocked_rigid_candidate",
        "accepted_articulated_candidate",
        "rejected_articulated_candidate",
        "double_articulated_transform",
        "deployment_bundle_excluding_research_asset",
        "no_selected_candidates",
        "preview_material_loss",
    }
    if mode in candidate_modes:
        candidate_asset_to_object = identity
        candidate_object_to_source_world = identity
        if mode == "accepted_rigid_candidate":
            candidate_object_to_source_world = [
                1.0,
                0.0,
                0.0,
                2.0,
                0.0,
                1.0,
                0.0,
                3.0,
                0.0,
                0.0,
                1.0,
                4.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        elif mode == "accepted_articulated_candidate":
            candidate_asset_to_object = [
                1.0,
                0.0,
                0.0,
                0.25,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
            candidate_object_to_source_world = [
                0.0,
                -1.0,
                0.0,
                2.0,
                1.0,
                0.0,
                0.0,
                3.0,
                0.0,
                0.0,
                1.0,
                4.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        candidate_role = (
            "articulated_visual"
            if mode
            in {
                "accepted_articulated_candidate",
                "rejected_articulated_candidate",
                "double_articulated_transform",
            }
            else "visual_completion"
        )
        production = mode == "accepted_rigid_candidate"
        selected = mode not in {"rejected_articulated_candidate", "no_selected_candidates"}
        validated = mode not in {"rejected_articulated_candidate", "no_selected_candidates"}
        assets.append(
            {
                "asset_id": "cup_candidate_visual",
                "object_id": "cup_0001",
                "lineage_id": object_lineage,
                "role": candidate_role,
                "source": "generated",
                "asset_path": "assembly/source/assets/cup_candidate.ply",
                "asset_sha256": sha256_file(candidate_path),
                "format": "ply",
                "asset_native_space": (
                    "reference_world"
                    if mode == "double_articulated_transform"
                    else "link_local"
                    if candidate_role == "articulated_visual"
                    else "candidate_base"
                ),
                "asset_to_object": candidate_asset_to_object,
                "object_to_source_world": candidate_object_to_source_world,
                "bounds_native": [0.05, 0.0, 0.0, 1.05, 1.0, 1.0],
                "selected_upstream": selected,
                "observation_validation_passed": validated,
                "candidate_id": "cup_candidate",
                "candidate_evaluation": {
                    "path": "assembly/source/evaluations/cup_candidate.json",
                    "sha256": sha256_file(evaluation_path),
                },
                "representation_id": "visual_ply",
                "articulation_id": (
                    "cup_articulation" if candidate_role == "articulated_visual" else None
                ),
                "link_id": "drawer" if candidate_role == "articulated_visual" else None,
                "kinematic_bundle": (
                    {
                        "path": "assembly/source/articulation/kinematic_bundle.json",
                        "sha256": sha256_file(kinematic_path),
                    }
                    if candidate_role == "articulated_visual"
                    else None
                ),
                "license": _license(
                    production=production,
                    research=mode != "license_blocked_rigid_candidate",
                ),
            }
        )
    calibration_status: str | None = None
    world_transform: list[float] | None = None
    if mode == "full_canonical_scene":
        calibration_status = "accepted_full_canonical"
        world_transform = [
            2.0,
            0.0,
            0.0,
            1.0,
            0.0,
            2.0,
            0.0,
            -2.0,
            0.0,
            0.0,
            2.0,
            0.5,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    elif mode == "metric_only_scene":
        calibration_status = "accepted_metric_only"
        world_transform = [
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    elif mode == "gravity_only_scene":
        calibration_status = "accepted_gravity_only"
        world_transform = identity
    candidate_ids = [
        item["asset_id"]
        for item in assets
        if item["role"] in {"visual_completion", "articulated_visual"}
    ]
    measured_motion = {
        "path": "assembly/source/articulation/measured_motion.json",
        "sha256": sha256_file(measured_motion_path),
    }
    object_record: dict[str, object] = {
        "object_id": "cup_0001",
        "lineage_id": object_lineage,
        "asset_type": (
            "articulated"
            if mode in {"accepted_articulated_candidate", "rejected_articulated_candidate"}
            else "rigid"
        ),
        "measured_anchor_asset_ids": ["cup_measured"],
        "global_context_asset_ids": (
            ["global_context"]
            if any(item["asset_id"] == "global_context" for item in assets)
            else []
        ),
        "candidate_asset_ids": candidate_ids,
        "preferred_research_candidate_id": (
            "cup_candidate"
            if candidate_ids
            and mode not in {"rejected_articulated_candidate", "no_selected_candidates"}
            else None
        ),
        "preferred_deployment_candidate_id": (
            "cup_candidate" if mode == "accepted_rigid_candidate" else None
        ),
        "upstream_status": (
            "rejected_heldout_state"
            if mode == "rejected_articulated_candidate"
            else "unresolved_no_candidate"
            if mode in {"unresolved_object", "no_selected_candidates"}
            else "accepted"
        ),
        "measured_motion": (
            measured_motion
            if mode in {"accepted_articulated_candidate", "rejected_articulated_candidate"}
            else None
        ),
        "kinematic_bundle": (
            {
                "path": "assembly/source/articulation/kinematic_bundle.json",
                "sha256": sha256_file(kinematic_path),
            }
            if mode == "accepted_articulated_candidate"
            else None
        ),
    }
    raw = {
        "schema_version": "0.1.0",
        "assembly_id": f"phase6b_fake_{mode}",
        "calibration_policy": "use_full_canonical_if_available",
        "primary_lineage_id": "fake_lineage",
        "lineages": lineages,
        "source_scene_ir": reference,
        "calibration_status": calibration_status,
        "source_world_to_assembly_world": world_transform,
        "assets": assets,
        "objects": [object_record],
        "global_scene_policy": "layered_no_carve_v1",
    }
    if mode == "missing_asset_hash":
        raw["assets"][0]["asset_sha256"] = "0" * 64  # type: ignore[index]
    return SceneAssemblyInputManifest.model_validate(raw)


class AssemblyInputsAdapter:
    name = "scene_assembly_inputs"
    version = "0.1.0"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        config = AssemblyInputsConfig.model_validate(context.config.adapter.config)
        if config.fake_mode is not None:
            return []
        path = _manifest_path(config)
        raw = _load_mapping(path)
        specs = [
            InputSpec(
                "assembly/source/input_manifest.yaml",
                "scene_assembly_source_manifest",
                source_path=path,
            )
        ]
        for relative, source, expected in _local_source_records(raw):
            specs.append(
                InputSpec(
                    relative,
                    "scene_assembly_source",
                    source_path=source,
                    expected_sha256=expected,
                    materialization_mode="reflink_or_copy",
                )
            )
        return specs

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        if context is None:
            return HealthcheckResult(False, "assembly inputs healthcheck requires --config")
        try:
            config = AssemblyInputsConfig.model_validate(context.config.adapter.config)
            if config.fake_mode is None:
                raw = _load_mapping(_manifest_path(config))
                sources = _local_source_records(raw)
                if not sources:
                    raise ValueError("real assembly manifest has no selectively materialized files")
                for _, source, expected in sources:
                    if not source.is_file():
                        raise ValueError(f"assembly source does not exist: {source}")
                    if source.is_symlink():
                        raise ValueError(f"assembly source must not be a symlink: {source}")
                    if expected is not None and sha256_file(source) != expected:
                        raise ValueError(f"assembly source hash mismatch: {source}")
        except ValueError as exc:
            return HealthcheckResult(False, str(exc))
        return HealthcheckResult(True, "typed selective scene assembly inputs available")

    def prepare(self, context: StageContext) -> None:
        context.path("assembly/source").mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                "assembly/input_manifest.json",
                "scene_assembly_input_manifest",
                "application/json",
                self.name,
                validation="json",
                model=SceneAssemblyInputManifest,
            )
        ]

    def run(self, context: StageContext) -> StageResult:
        config = AssemblyInputsConfig.model_validate(context.config.adapter.config)
        if config.fake_mode is not None:
            manifest = _fake_manifest(context, config.fake_mode)
        else:
            raw = _load_mapping(context.path("assembly/source/input_manifest.yaml"))
            normalized = _normalized_manifest(raw, context.run_dir)
            assert isinstance(normalized, dict)
            manifest = SceneAssemblyInputManifest.model_validate(normalized)
        outputs: list[OutputSpec] = []
        for reference in [
            manifest.source_scene_ir,
            *(item.camera_reconstruction for item in manifest.lineages),
            *(item.source_scene_ir for item in manifest.lineages),
            *(item.accepted_alignment for item in manifest.lineages if item.accepted_alignment),
            manifest.calibration_artifact,
            manifest.canonical_wrapper,
            *(item.candidate_evaluation for item in manifest.assets if item.candidate_evaluation),
            *(item.kinematic_bundle for item in manifest.assets if item.kinematic_bundle),
            *(item.license.source_record for item in manifest.assets if item.license.source_record),
            *(item.measured_motion for item in manifest.objects if item.measured_motion),
            *(item.kinematic_bundle for item in manifest.objects if item.kinematic_bundle),
        ]:
            if reference is None:
                continue
            path = context.path(*reference.path.split("/"))
            if not path.is_file() or sha256_file(path) != reference.sha256:
                raise ValueError(f"assembly source reference hash mismatch: {reference.path}")
            outputs.append(
                OutputSpec(
                    reference.path,
                    "scene_assembly_source",
                    "application/json",
                    self.name,
                    validation="json",
                )
            )
        for asset in manifest.assets:
            path = context.path(*asset.asset_path.split("/"))
            if not path.is_file() or sha256_file(path) != asset.asset_sha256:
                raise ValueError(f"assembly asset hash mismatch: {asset.asset_path}")
            outputs.append(
                OutputSpec(
                    asset.asset_path,
                    "scene_assembly_visual_asset",
                    "application/octet-stream",
                    self.name,
                )
            )
        atomic_write_json(context.path("assembly/input_manifest.json"), manifest)
        return StageResult(
            outputs=outputs,
            metrics={
                "lineages": len(manifest.lineages),
                "objects": len(manifest.objects),
                "assets": len(manifest.assets),
                "input_digest": stable_digest(manifest.model_dump(mode="json")),
            },
        )


__all__ = ["AssemblyInputsAdapter", "AssemblyInputsConfig", "FakeAssemblyMode"]
