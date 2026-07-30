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
from recon2sim.assembly_sources import normalize_assembly_manifest
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
    "different_bundle_candidates",
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
    atomic_write_json(
        camera_path,
        {
            "camera_id": "phase6b_fake_camera",
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
            "poses": [
                {
                    "frame_id": frame_id,
                    "transform_world_from_camera": {
                        "translation": [float(index), 0.25, 1.0],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "scale": [1.0, 1.0, 1.0],
                    },
                    "confidence": confidence,
                }
                for index, frame_id in enumerate(("frame_000000", "frame_000001"))
            ],
            "registered_frame_ids": ["frame_000000", "frame_000001"],
            "unregistered_frame_ids": [],
            "sparse_point_count": 20,
            "average_reprojection_error": 0.1,
            "confidence": confidence,
            "coordinate_convention": convention,
            "scale_status": "scale_ambiguous",
            "frame_sequence_digest": "a" * 64,
            "provenance": provenance,
        },
    )
    scene_path = context.path("assembly/source/scene_ir.json")
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
            "cameras": [
                {
                    "camera_id": "source_camera",
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
                    "poses": [
                        {
                            "frame_id": "frame_000000",
                            "transform_world_from_camera": {
                                "translation": [1.0, 2.0, 3.0],
                                "rotation_xyzw": [0.0, 0.0, 0.3826834324, 0.9238795325],
                                "scale": [1.0, 1.0, 1.0],
                            },
                            "confidence": confidence,
                        }
                    ],
                    "coordinate_convention": convention,
                    "scale_status": "scale_ambiguous",
                    "provenance": provenance,
                }
            ],
            "frames": [
                {
                    "frame_id": "frame_000000",
                    "frame_path": "frames/frame_000000.png",
                    "timestamp_s": 0.0,
                    "camera_id": "source_camera",
                    "observations": [],
                }
            ],
            "objects": [
                {
                    "object_id": "source_rigid",
                    "name": "Source rigid object",
                    "asset_type": "rigid",
                    "transform": {
                        "translation": [4.0, 5.0, 6.0],
                        "rotation_xyzw": [0.0, 0.2588190451, 0.0, 0.9659258263],
                        "scale": [1.2, 1.2, 1.2],
                    },
                    "geometry_asset_ids": ["source_reference_mesh"],
                    "physics": {"is_static": True},
                    "provenance": [provenance],
                    "confidence": confidence,
                },
                {
                    "object_id": "source_articulated",
                    "name": "Source articulated object",
                    "asset_type": "articulated",
                    "transform": {
                        "translation": [-2.0, 1.0, 0.5],
                        "rotation_xyzw": [0.0, 0.0, -0.2588190451, 0.9659258263],
                        "scale": [1.7, 1.7, 1.7],
                    },
                    "physics": {"is_static": True},
                    "articulation": {
                        "articulation_id": "source_articulation",
                        "links": [{"link_id": "source_base", "name": "Source base"}],
                        "joints": [],
                    },
                    "provenance": [provenance],
                    "confidence": confidence,
                },
            ],
            "geometry_assets": [
                {
                    "asset_id": "source_reference_mesh",
                    "asset_type": "rigid",
                    "uri": "assembly/source/assets/cup_measured.ply",
                    "format": "ply",
                    "source": "measured",
                    "coordinate_convention": convention,
                    "scale_status": "scale_ambiguous",
                    "source_space_geometry": True,
                    "provenance": provenance,
                },
                {
                    "asset_id": "global_scene_pbr",
                    "asset_type": "static_structure",
                    "uri": "reconstruction/global/scene.glb",
                    "format": "glb",
                    "source": "generated",
                    "coordinate_convention": convention,
                    "scale_status": "scale_ambiguous",
                    "source_space_geometry": True,
                    "provenance": provenance,
                },
                {
                    "asset_id": "global_scene_mesh",
                    "asset_type": "static_structure",
                    "uri": "reconstruction/global/mesh.ply",
                    "format": "ply",
                    "source": "generated",
                    "coordinate_convention": convention,
                    "scale_status": "scale_ambiguous",
                    "source_space_geometry": True,
                    "provenance": provenance,
                },
            ],
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
    measured_geometry_path = context.path("assembly/source/measured_geometry.json")
    atomic_write_json(
        measured_geometry_path,
        {"object_id": "cup_0001", "path": "assembly/source/assets/cup_measured.ply"},
    )
    global_path = context.path("reconstruction/global/mesh.ply")
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
        "artifact_type": "source_scene_ir",
    }
    lineage: dict[str, object] = {
        "lineage_id": "fake_lineage",
        "frame_sequence_digest": "a" * 64,
        "camera_reconstruction": {
            "path": "assembly/source/camera_reconstruction.json",
            "sha256": sha256_file(camera_path),
            "artifact_type": "camera_reconstruction",
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
        capture_path = context.path("assembly/source/articulation_capture_manifest.json")
        child_camera_path = context.path("assembly/source/aligned_camera_reconstruction.json")
        child_camera = json.loads(camera_path.read_text(encoding="utf-8"))
        child_camera["frame_sequence_digest"] = "b" * 64
        atomic_write_json(child_camera_path, child_camera)
        lineage["source_state_id"] = "state_000"
        state_common = {
            "run_dir": "states/fake",
            "part_track_ids": {"cabinet_body": "body_track", "drawer": "drawer_track"},
            "phase5a_consistency_passed": True,
            "ingest_manifest_sha256": "1" * 64,
            "segmentation_tracking_sha256": "2" * 64,
            "dense_depth_manifest_sha256": "3" * 64,
            "measured_geometry_sha256": "4" * 64,
            "part_mask_hashes": {"cabinet_body": "5" * 64, "drawer": "6" * 64},
            "measured_part_cloud_hashes": {
                "cabinet_body": "7" * 64,
                "drawer": "8" * 64,
            },
            "registered_frame_ids": ["frame_000000", "frame_000001"],
            "camera_evidence_path": "camera/reconstruction.json",
            "segmentation_evidence_path": "observations/object_tracks.json",
            "undistortion_evidence_path": "reconstruction/dense/undistortion_manifest.json",
            "depth_evidence_path": "reconstruction/dense/depth_manifest.json",
            "dense_map_hashes": {"frame_000000": "9" * 64},
        }
        atomic_write_json(
            capture_path,
            {
                "schema_version": "0.2.0",
                "articulated_object_id": "cabinet_0001",
                "reference_state_id": "state_000",
                "states": [
                    {
                        **state_common,
                        "state_id": "state_000",
                        "semantic_state_label": "closed",
                        "frame_sequence_digest": "a" * 64,
                        "camera_reconstruction_sha256": sha256_file(camera_path),
                    },
                    {
                        **state_common,
                        "state_id": "state_001",
                        "semantic_state_label": "open",
                        "frame_sequence_digest": "b" * 64,
                        "camera_reconstruction_sha256": sha256_file(child_camera_path),
                    },
                ],
                "prompt_manifest_sha256": "0" * 64,
                "capture_state_count": 2,
                "capture_evidence_tier": "two_state_motion_supported",
            },
        )
        transform = {
            "matrix_reference_from_state": list(IDENTITY_MATRIX4),
            "inverse_matrix": list(IDENTITY_MATRIX4),
            "scale": 1.0,
            "rotation_determinant": 1.0,
            "translation": [0.0, 0.0, 0.0],
            "fitting_median_residual_scene_diagonal": 0.0,
            "fitting_p90_residual_scene_diagonal": 0.0,
            "heldout_static_depth_inlier_fraction": 1.0,
            "static_correspondence_count": 500,
            "excluded_movable_part_ids": ["drawer"],
            "accepted": True,
        }
        atomic_write_json(
            alignment_path,
            {
                "schema_version": "0.2.0",
                "capture_manifest_sha256": sha256_file(capture_path),
                "reference_state_id": "state_000",
                "transforms": [
                    {**transform, "state_id": "state_000"},
                    {**transform, "state_id": "state_001"},
                ],
                "capture_state_count": 2,
                "accepted_alignment_state_ids": ["state_000", "state_001"],
                "aligned_state_count": 2,
                "static_evidence_only": True,
                "source_states_unchanged": True,
                "runtime_seconds": 0.0,
            },
        )
        lineages.append(
            {
                "lineage_id": "aligned_state_lineage",
                "frame_sequence_digest": "b" * 64,
                "camera_reconstruction": {
                    "path": "assembly/source/aligned_camera_reconstruction.json",
                    "sha256": sha256_file(child_camera_path),
                    "artifact_type": "camera_reconstruction",
                },
                "source_scene_ir": reference,
                "world_frame": "colmap_arbitrary",
                "source_state_id": "state_001",
                "connected_to_lineage_id": "fake_lineage",
                "accepted_alignment": {
                    "path": "assembly/source/state_alignment.json",
                    "sha256": sha256_file(alignment_path),
                    "artifact_type": "state_alignment",
                },
                "alignment_capture_manifest": {
                    "path": "assembly/source/articulation_capture_manifest.json",
                    "sha256": sha256_file(capture_path),
                    "artifact_type": "articulation_capture_manifest",
                },
                "alignment_state_id": "state_001",
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
            "source_native_asset_path": "assembly/source/assets/cup_measured.ply",
            "format": "ply",
            "asset_native_space": "reference_world",
            "asset_to_object": identity,
            "object_to_source_world": measured_object_transform,
            "bounds_native": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "license": _license(production=True),
            "measured_geometry": {
                "path": "assembly/source/measured_geometry.json",
                "sha256": sha256_file(measured_geometry_path),
                "artifact_type": "measured_geometry",
            },
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
        "different_bundle_candidates",
        "preview_material_loss",
    }
    if mode in global_modes:
        phase3_path = context.path("assembly/source/global_scene_reconstruction.json")
        worker_path = context.path("assembly/source/genrecon_worker_manifest.json")
        global_source_path = context.path("assembly/source/global_context_source.json")
        phase3_mesh_path = context.path("reconstruction/global/mesh.ply")
        phase3_scene_path = context.path("reconstruction/global/scene.glb")
        _write_ascii_ply(phase3_mesh_path, offset=-0.25)
        phase3_scene_path.parent.mkdir(parents=True, exist_ok=True)
        phase3_scene_path.write_bytes(b"glTF-phase6b-diagnostic-source\n")
        phase3_repository = "https://github.com/kasothaphie/GenRecon"
        phase3_commit = "eaf1468118d20469d17079a4a19737297d2ef87b"
        runtime_revision = "1" * 40
        runtime_revisions = {
            "facebook/dinov3-vitl16-pretrain-lvd1689m": runtime_revision,
            "microsoft/TRELLIS-image-large": "2" * 40,
            "microsoft/TRELLIS.2-4B": "3" * 40,
        }
        working_transform = {
            "strategy": "identity",
            "matrix_colmap_to_working": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "matrix_working_to_colmap": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "determinant": 1.0,
            "roundtrip_max_error": 0.0,
            "semantic_status": "internal_unoriented_preprocessing",
        }
        atomic_write_json(
            phase3_path,
            {
                "schema_version": "0.1.0",
                "scene_asset_path": "reconstruction/global/scene.glb",
                "mesh_asset_path": "reconstruction/global/mesh.ply",
                "scene_ir_path": "scene_ir/scene.json",
                "format": "glb",
                "coordinate_convention": {
                    "world_frame": "colmap_arbitrary",
                    "alignment_status": "unoriented",
                    "camera_axes": "x_right_y_down_z_forward",
                    "linear_units": "arbitrary_units",
                    "scale_status": "scale_ambiguous",
                    "transform_direction": "world_from_camera",
                },
                "scale_status": "scale_ambiguous",
                "manifest_sha256": "0" * 64,
                "frame_sequence_digest": "a" * 64,
                "camera_reconstruction_sha256": sha256_file(camera_path),
                "camera_package_sha256": "4" * 64,
                "input_frame_count": 2,
                "registered_frame_count": 2,
                "unregistered_frame_count": 0,
                "eligible_frame_ids": ["frame_000000", "frame_000001"],
                "actual_selected_frame_ids": ["frame_000000", "frame_000001"],
                "mesh": {
                    "vertex_count": 4,
                    "face_count": 2,
                    "disconnected_components": 1,
                    "degenerate_faces": 0,
                    "non_manifold_edge_count": 0,
                    "finite_coordinates": True,
                    "bounding_box_min": [-0.25, 0.0, 0.0],
                    "bounding_box_max": [0.75, 1.0, 1.0],
                    "bounding_box_extent": [1.0, 1.0, 1.0],
                    "material_count": 1,
                    "texture_count": 0,
                    "glb_parse_status": "valid",
                },
                "chunk_count": 1,
                "checkpoints": [],
                "official_repository": phase3_repository,
                "official_code_commit": phase3_commit,
                "runtime_model_repository": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "runtime_model_revision": runtime_revision,
                "runtime_repository_revisions": runtime_revisions,
                "runtime_seconds": 0.01,
                "peak_gpu_memory_bytes": 0,
                "seed": 42,
                "provenance": {
                    "adapter_name": "scene_assembly_inputs",
                    "adapter_version": "0.1.0",
                    "configuration": {"fake": True},
                    "input_artifact_paths": [],
                    "output_artifact_paths": [
                        "reconstruction/global/scene.glb",
                        "reconstruction/global/mesh.ply",
                    ],
                    "confidence": {"score": 1.0, "method": "phase6b_fake"},
                    "source": "generated",
                    "timestamp": "2026-01-01T00:00:00Z",
                },
            },
        )
        atomic_write_json(
            worker_path,
            {
                "schema_version": "0.1.0",
                "official_repository": phase3_repository,
                "official_code_commit": phase3_commit,
                "submodule_commits": {},
                "official_license": "MIT",
                "checkpoint_records": [],
                "runtime_model_repository": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "runtime_model_revision": runtime_revision,
                "runtime_repository_revisions": runtime_revisions,
                "worker_version": "phase6b-fixture",
                "python_version": "3.12",
                "torch_version": None,
                "torchvision_version": None,
                "cuda_version": None,
                "device_name": "deterministic fixture",
                "device": "fake",
                "precision": "float32",
                "seed": 42,
                "request_sha256": "5" * 64,
                "frame_sequence_digest": "a" * 64,
                "camera_package_sha256": "4" * 64,
                "registered_frame_ids": ["frame_000000", "frame_000001"],
                "selected_frame_ids": ["frame_000000", "frame_000001"],
                "working_transform": working_transform,
                "reconstruct_return_code": 0,
                "glb_conversion_return_code": 0,
                "runtime_seconds": 0.01,
                "peak_gpu_memory_bytes": 0,
                "raw_output_paths": [],
                "warnings": [],
            },
        )
        atomic_write_json(
            global_source_path,
            {
                "schema_version": "0.1.0",
                "lineage_id": "fake_lineage",
                "frame_sequence_digest": "a" * 64,
                "camera_reconstruction_sha256": sha256_file(camera_path),
                "coordinate_convention": {
                    "world_frame": "colmap_arbitrary",
                    "alignment_status": "unoriented",
                    "camera_axes": "x_right_y_down_z_forward",
                    "linear_units": "arbitrary_units",
                    "scale_status": "scale_ambiguous",
                    "transform_direction": "world_from_camera",
                },
                "phase3_reconstruction": {
                    "path": "assembly/source/global_scene_reconstruction.json",
                    "sha256": sha256_file(phase3_path),
                    "artifact_type": "phase3_global_reconstruction",
                },
                "genrecon_worker_manifest": {
                    "path": "assembly/source/genrecon_worker_manifest.json",
                    "sha256": sha256_file(worker_path),
                    "artifact_type": "global_context_manifest",
                },
                "source_scene_ir": reference,
                "assets": [
                    {
                        "assembly_asset_id": "global_context",
                        "source_geometry_asset_id": "global_scene_mesh",
                        "source_native_asset_path": "reconstruction/global/mesh.ply",
                        "sha256": sha256_file(global_path),
                        "format": "ply",
                        "source": "generated",
                    }
                ],
            },
        )
        assets.append(
            {
                "asset_id": "global_context",
                "object_id": None,
                "lineage_id": "fake_lineage",
                "role": "global_context",
                "source": "generated",
                "asset_path": "reconstruction/global/mesh.ply",
                "asset_sha256": sha256_file(global_path),
                "source_native_asset_path": "reconstruction/global/mesh.ply",
                "format": "ply",
                "asset_native_space": "global_context",
                "asset_to_object": identity,
                "object_to_source_world": identity,
                "bounds_native": [-0.25, 0.0, 0.0, 0.75, 1.0, 1.0],
                "global_scene_reconstruction": {
                    "path": "assembly/source/global_scene_reconstruction.json",
                    "sha256": sha256_file(phase3_path),
                    "artifact_type": "phase3_global_reconstruction",
                },
                "global_context_source": {
                    "path": "assembly/source/global_context_source.json",
                    "sha256": sha256_file(global_source_path),
                    "artifact_type": "global_context_source",
                },
                "license_source_record": {
                    "path": "assembly/source/genrecon_worker_manifest.json",
                    "sha256": sha256_file(worker_path),
                    "artifact_type": "global_context_manifest",
                },
                "license": {
                    **_license(production=mode != "deployment_bundle_excluding_research_asset"),
                    "source_record": {
                        "path": "assembly/source/genrecon_worker_manifest.json",
                        "sha256": sha256_file(worker_path),
                        "artifact_type": "global_context_manifest",
                    },
                },
            }
        )
    candidate_modes = {
        "accepted_rigid_candidate",
        "license_blocked_rigid_candidate",
        "accepted_articulated_candidate",
        "rejected_articulated_candidate",
        "double_articulated_transform",
        "deployment_bundle_excluding_research_asset",
        "different_bundle_candidates",
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
        production = mode in {"accepted_rigid_candidate", "different_bundle_candidates"}
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
                "source_native_asset_path": "assembly/source/assets/cup_candidate.ply",
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
                    "artifact_type": (
                        "articulated_evaluation"
                        if candidate_role == "articulated_visual"
                        else "rigid_evaluation"
                    ),
                },
                "candidate_selection": {
                    "path": "assembly/source/evaluations/cup_candidate.json",
                    "sha256": sha256_file(evaluation_path),
                    "artifact_type": (
                        "articulated_selection"
                        if candidate_role == "articulated_visual"
                        else "rigid_selection"
                    ),
                },
                "candidate_generation": {
                    "path": "assembly/source/evaluations/cup_candidate.json",
                    "sha256": sha256_file(evaluation_path),
                    "artifact_type": (
                        "articulated_candidate_manifest"
                        if candidate_role == "articulated_visual"
                        else "rigid_generation"
                    ),
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
                        "artifact_type": "kinematic_bundle",
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
        if mode == "different_bundle_candidates":
            assets[-1]["asset_id"] = "production_candidate_B_visual"
            assets[-1]["candidate_id"] = "production_candidate_B"
            research_path = context.path("assembly/source/assets/research_candidate_A.ply")
            _write_ascii_ply(research_path, offset=0.1)
            research_asset = dict(assets[-1])
            research_asset.update(
                {
                    "asset_id": "research_candidate_A_visual",
                    "candidate_id": "research_candidate_A",
                    "asset_path": "assembly/source/assets/research_candidate_A.ply",
                    "source_native_asset_path": "assembly/source/assets/research_candidate_A.ply",
                    "asset_sha256": sha256_file(research_path),
                    "license": _license(production=False),
                }
            )
            assets.append(research_asset)
            measured_part_path = context.path("assembly/source/assets/cup_measured_part2.ply")
            _write_ascii_ply(measured_part_path, offset=1.0)
            measured_part = dict(assets[0])
            measured_part.update(
                {
                    "asset_id": "cup_measured_part2",
                    "asset_path": "assembly/source/assets/cup_measured_part2.ply",
                    "source_native_asset_path": ("assembly/source/assets/cup_measured_part2.ply"),
                    "asset_sha256": sha256_file(measured_part_path),
                    "bounds_native": [1.0, 0.0, 0.0, 2.0, 1.0, 1.0],
                }
            )
            assets.append(measured_part)
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
    candidate_ids = [
        item["asset_id"]
        for item in assets
        if item["role"] in {"visual_completion", "articulated_visual"}
    ]
    measured_motion = {
        "path": "assembly/source/articulation/measured_motion.json",
        "sha256": sha256_file(measured_motion_path),
        "artifact_type": "measured_motion",
    }
    object_record: dict[str, object] = {
        "object_id": "cup_0001",
        "lineage_id": object_lineage,
        "asset_type": (
            "articulated"
            if mode in {"accepted_articulated_candidate", "rejected_articulated_candidate"}
            else "rigid"
        ),
        "measured_anchor_asset_ids": (
            ["cup_measured", "cup_measured_part2"]
            if mode == "different_bundle_candidates"
            else ["cup_measured"]
        ),
        "global_context_asset_ids": (
            ["global_context"]
            if any(item["asset_id"] == "global_context" for item in assets)
            else []
        ),
        "candidate_asset_ids": candidate_ids,
        "preferred_research_candidate_id": (
            ("research_candidate_A" if mode == "different_bundle_candidates" else "cup_candidate")
            if candidate_ids
            and mode not in {"rejected_articulated_candidate", "no_selected_candidates"}
            else None
        ),
        "preferred_deployment_candidate_id": (
            "production_candidate_B"
            if mode == "different_bundle_candidates"
            else "cup_candidate"
            if mode == "accepted_rigid_candidate"
            else None
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
                "artifact_type": "kinematic_bundle",
            }
            if mode == "accepted_articulated_candidate"
            else None
        ),
    }
    raw = {
        "schema_version": "0.3.0",
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
            manifest = normalize_assembly_manifest(normalized, context.run_dir)
        outputs: list[OutputSpec] = []
        for reference in [
            manifest.source_scene_ir,
            *(item.camera_reconstruction for item in manifest.lineages),
            *(item.source_scene_ir for item in manifest.lineages),
            *(item.accepted_alignment for item in manifest.lineages if item.accepted_alignment),
            *(
                item.alignment_capture_manifest
                for item in manifest.lineages
                if item.alignment_capture_manifest
            ),
            manifest.calibration_artifact,
            manifest.canonical_wrapper,
            *(item.candidate_selection for item in manifest.assets if item.candidate_selection),
            *(item.candidate_evaluation for item in manifest.assets if item.candidate_evaluation),
            *(item.candidate_generation for item in manifest.assets if item.candidate_generation),
            *(item.measured_geometry for item in manifest.assets if item.measured_geometry),
            *(item.kinematic_bundle for item in manifest.assets if item.kinematic_bundle),
            *(
                item.global_scene_reconstruction
                for item in manifest.assets
                if item.global_scene_reconstruction
            ),
            *(item.global_context_source for item in manifest.assets if item.global_context_source),
            *(item.license_source_record for item in manifest.assets if item.license_source_record),
            *(item.license.source_record for item in manifest.assets if item.license.source_record),
            *(
                item.rigid_selection_artifact
                for item in manifest.objects
                if item.rigid_selection_artifact
            ),
            *(
                item.rigid_evaluation_artifact
                for item in manifest.objects
                if item.rigid_evaluation_artifact
            ),
            *(
                item.rigid_registration_artifact
                for item in manifest.objects
                if item.rigid_registration_artifact
            ),
            *(
                reference
                for item in manifest.objects
                for reference in item.rigid_generation_artifacts
            ),
            *(
                reference
                for item in manifest.objects
                for reference in item.representation_parity_artifacts
            ),
            *(
                item.articulated_selection_artifact
                for item in manifest.objects
                if item.articulated_selection_artifact
            ),
            *(
                item.articulated_candidate_manifest
                for item in manifest.objects
                if item.articulated_candidate_manifest
            ),
            *(
                item.articulated_evaluation_artifact
                for item in manifest.objects
                if item.articulated_evaluation_artifact
            ),
            *(
                item.articulated_fitting_artifact
                for item in manifest.objects
                if item.articulated_fitting_artifact
            ),
            *(
                item.articulated_link_assignment_artifact
                for item in manifest.objects
                if item.articulated_link_assignment_artifact
            ),
            *(
                item.selected_identity_manifest
                for item in manifest.objects
                if item.selected_identity_manifest
            ),
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
