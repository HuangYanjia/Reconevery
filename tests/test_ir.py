from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from recon2sim.artifacts import (
    CameraReconstruction,
    IngestManifest,
    ObjectReconstructionArtifact,
    ObjectTracksArtifact,
)
from recon2sim.images import png_dimensions
from recon2sim.ir import (
    AssetType,
    ConfidenceRecord,
    CoordinateConvention,
    GeometrySourceType,
    PhysicsProperties,
    SceneIR,
    Transform,
)


def test_real_pydantic_constraints() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        ConfidenceRecord(score=1.01, method="test")
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        PhysicsProperties(mass_kg=-0.01)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ConfidenceRecord.model_validate({"score": 0.5, "method": "test", "unexpected": 1})


def test_legacy_coordinate_and_transform_fields_remain_readable() -> None:
    convention = CoordinateConvention.model_validate(
        {
            "world_axes": "x_forward_y_left_z_up",
            "handedness": "right",
            "units": "meters",
            "quaternion_order": "xyzw",
            "camera_transform_direction": "world_from_camera",
        }
    )
    transform = Transform.model_validate({"translation_m": [1, 2, 3]})

    assert convention.linear_units.value == "meters"
    assert "units" not in convention.model_dump()
    assert transform.translation == (1.0, 2.0, 3.0)
    assert "translation_m" not in transform.model_dump()


def test_malformed_enum_is_rejected(scene_payload: dict[str, Any]) -> None:
    payload = copy.deepcopy(scene_payload)
    payload["metadata"]["source"] = "imaginary_source"
    with pytest.raises(ValidationError, match="Input should be"):
        SceneIR.model_validate(payload)


def test_scene_ir_rejects_duplicate_object_ids(scene_payload: dict[str, Any]) -> None:
    payload = copy.deepcopy(scene_payload)
    payload["objects"].append(copy.deepcopy(payload["objects"][0]))
    with pytest.raises(ValidationError, match="duplicate object IDs"):
        SceneIR.model_validate(payload)


def test_scene_ir_rejects_duplicate_asset_ids(scene_payload: dict[str, Any]) -> None:
    payload = copy.deepcopy(scene_payload)
    payload["material_assets"][0]["asset_id"] = payload["geometry_assets"][0]["asset_id"]
    with pytest.raises(ValidationError, match="duplicate asset IDs"):
        SceneIR.model_validate(payload)


def test_scene_ir_rejects_broken_asset_reference(scene_payload: dict[str, Any]) -> None:
    payload = copy.deepcopy(scene_payload)
    payload["objects"][0]["geometry_asset_ids"] = ["missing_geometry"]
    with pytest.raises(ValidationError, match="unknown geometry assets"):
        SceneIR.model_validate(payload)


def test_scene_ir_rejects_broken_relation_reference(scene_payload: dict[str, Any]) -> None:
    payload = copy.deepcopy(scene_payload)
    payload["relations"][0]["subject_id"] = "missing_object"
    with pytest.raises(ValidationError, match="references unknown objects"):
        SceneIR.model_validate(payload)


def test_scene_ir_rejects_broken_articulation_link_reference(
    scene_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(scene_payload)
    cabinet = next(obj for obj in payload["objects"] if obj["object_id"] == "cabinet")
    cabinet["articulation"]["joints"][0]["child_link_id"] = "missing_link"
    with pytest.raises(ValidationError, match="references unknown links"):
        SceneIR.model_validate(payload)


def test_scene_ir_rejects_duplicate_links_and_joints(scene_payload: dict[str, Any]) -> None:
    duplicate_link = copy.deepcopy(scene_payload)
    cabinet = next(obj for obj in duplicate_link["objects"] if obj["object_id"] == "cabinet")
    cabinet["articulation"]["links"].append(copy.deepcopy(cabinet["articulation"]["links"][0]))
    with pytest.raises(ValidationError, match="duplicate articulation link IDs"):
        SceneIR.model_validate(duplicate_link)

    duplicate_joint = copy.deepcopy(scene_payload)
    cabinet = next(obj for obj in duplicate_joint["objects"] if obj["object_id"] == "cabinet")
    cabinet["articulation"]["joints"].append(copy.deepcopy(cabinet["articulation"]["joints"][0]))
    with pytest.raises(ValidationError, match="duplicate articulation joint IDs"):
        SceneIR.model_validate(duplicate_joint)


def test_articulation_must_match_asset_type(scene_payload: dict[str, Any]) -> None:
    articulated_without_metadata = copy.deepcopy(scene_payload)
    cabinet = next(
        obj for obj in articulated_without_metadata["objects"] if obj["object_id"] == "cabinet"
    )
    cabinet["articulation"] = None
    with pytest.raises(ValidationError, match="must contain an articulation"):
        SceneIR.model_validate(articulated_without_metadata)

    rigid_with_metadata = copy.deepcopy(scene_payload)
    cabinet = next(obj for obj in rigid_with_metadata["objects"] if obj["object_id"] == "cabinet")
    cabinet["asset_type"] = AssetType.RIGID.value
    with pytest.raises(ValidationError, match="must not contain an articulation"):
        SceneIR.model_validate(rigid_with_metadata)


def test_checked_in_json_schema_is_complete_and_current() -> None:
    generated = SceneIR.model_json_schema()
    checked_in = json.loads(Path("schemas/scene_ir.schema.json").read_text(encoding="utf-8"))
    assert checked_in == generated
    assert set(generated["properties"]) >= {
        "metadata",
        "cameras",
        "frames",
        "objects",
        "geometry_assets",
        "material_assets",
        "collision_assets",
        "relations",
    }
    assert "metadata" in generated["required"]
    assert generated["additionalProperties"] is False
    assert "$defs" in generated
    assert set(generated["$defs"]["AssetType"]["enum"]) == {item.value for item in AssetType}
    assert set(generated["$defs"]["GeometrySourceType"]["enum"]) == {
        item.value for item in GeometrySourceType
    }
    score_schema = generated["$defs"]["ConfidenceRecord"]["properties"]["score"]
    assert score_schema["minimum"] == 0
    assert score_schema["maximum"] == 1
    assert generated["$defs"]["CameraIntrinsics"]["additionalProperties"] is False


def test_mock_artifacts_form_one_connected_data_flow(completed_run: Path) -> None:
    manifest = IngestManifest.model_validate_json(
        (completed_run / "inputs" / "manifest.json").read_text(encoding="utf-8")
    )
    camera = CameraReconstruction.model_validate_json(
        (completed_run / "camera" / "reconstruction.json").read_text(encoding="utf-8")
    )
    tracks = ObjectTracksArtifact.model_validate_json(
        (completed_run / "observations" / "object_tracks.json").read_text(encoding="utf-8")
    )
    reconstructions = ObjectReconstructionArtifact.model_validate_json(
        (completed_run / "reconstruction" / "objects" / "results.json").read_text(encoding="utf-8")
    )
    scene = SceneIR.model_validate_json(
        (completed_run / "scene_ir" / "scene.json").read_text(encoding="utf-8")
    )

    assert {pose.frame_id for pose in camera.poses} == {frame.frame_id for frame in manifest.frames}
    assert {result.object_id for result in reconstructions.results} == {
        track.object_id for track in tracks.tracks
    }
    assert {frame.frame_id for frame in scene.frames} == {
        frame.frame_id for frame in manifest.frames
    }
    assert scene.cameras[0].poses == camera.poses
    assert all(frame.observations for frame in scene.frames)

    for frame in manifest.frames:
        path = completed_run / frame.relative_path
        assert png_dimensions(path) == (frame.width, frame.height)
        assert len(frame.sha256) == 64
    for track in tracks.tracks:
        for observation in track.observations:
            png_dimensions(completed_run / observation.mask_path)

    object_ids = {obj.object_id for obj in scene.objects}
    assert object_ids == {"floor", "table", "cup", "cabinet"}
    assert "drawer" not in object_ids
    cabinet = next(obj for obj in scene.objects if obj.object_id == "cabinet")
    assert cabinet.articulation is not None
    assert {link.link_id for link in cabinet.articulation.links} == {
        "cabinet_body",
        "cabinet_drawer",
    }
    assert scene.geometry_assets and scene.material_assets and scene.collision_assets
