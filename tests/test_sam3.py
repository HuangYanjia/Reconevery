from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pydantic import ValidationError
from typer.testing import CliRunner

from recon2sim.adapters import (
    REGISTRY,
    HealthcheckResult,
    OutputSpec,
    StageContext,
    StageResult,
)
from recon2sim.adapters.mock import MockCameraRecoveryAdapter
from recon2sim.adapters.sam3 import (
    Sam3AdapterConfig,
    Sam3SegmentationTrackingAdapter,
)
from recon2sim.artifacts import (
    CameraReconstruction,
    FrameQualityReport,
    IngestManifest,
    Sam3InferenceRequest,
    SegmentationPrompt,
    SegmentationPromptManifest,
    SegmentationTrackingArtifact,
)
from recon2sim.cli import app
from recon2sim.config import AdapterConfig, PipelineConfig, StageConfig, load_config
from recon2sim.pipeline import PipelineRunner
from recon2sim.segmentation import (
    export_coco,
    load_prompt_manifest,
    normalize_semantic_label,
    render_previews,
    select_anchor_frames,
    validate_canonical_mask,
    validate_prompt_references,
)
from recon2sim.storage import atomic_write_json


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(
    *,
    mode: str = "success_multi",
    prompt_path: Path | None = None,
) -> PipelineConfig:
    config = load_config(Path("configs/sam3_fake.yaml"))
    sam = config.stages["segmentation_tracking"].adapter
    sam.config["fake_mode"] = mode
    if prompt_path is not None:
        sam.config["prompt_manifest"] = str(prompt_path)
    return config


def _run(
    tmp_path: Path,
    *,
    mode: str = "success_multi",
    config: PipelineConfig | None = None,
) -> tuple[Path, dict[str, Any]]:
    run_dir = tmp_path / "run"
    manifest = PipelineRunner(
        config or _config(mode=mode),
        Path("examples/tabletop"),
        run_dir,
    ).run()
    return run_dir, manifest


def _artifact(run_dir: Path) -> SegmentationTrackingArtifact:
    return SegmentationTrackingArtifact.model_validate_json(
        (run_dir / "observations" / "object_tracks.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def fake_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir, _ = _run(tmp_path_factory.mktemp("sam3-success"))
    return run_dir


@pytest.mark.parametrize(
    ("field", "value", "prompt_type"),
    [
        ("text", "cup", "text"),
        ("box_xyxy", [1, 2, 10, 12], "box"),
        ("points", [{"x": 2, "y": 3, "label": 1}], "point"),
        ("mask_path", "prompts/cup.png", "mask"),
    ],
)
def test_prompt_contracts(
    field: str,
    value: object,
    prompt_type: str,
) -> None:
    payload: dict[str, object] = {"prompt_id": "cup", "label": "cup", field: value}
    if prompt_type != "text":
        payload["frame_id"] = "frame_000000"
    prompt = SegmentationPrompt.model_validate(payload)
    assert prompt.prompt_type is not None
    assert prompt.prompt_type.value == prompt_type


def test_prompt_ids_are_unique_and_labels_are_nonempty() -> None:
    prompt = {"prompt_id": "cup", "label": "cup", "text": "cup"}
    with pytest.raises(ValidationError, match="prompt IDs must be unique"):
        SegmentationPromptManifest.model_validate(
            {"schema_version": "0.1.0", "prompts": [prompt, prompt]}
        )
    with pytest.raises(ValidationError, match="must not be blank"):
        SegmentationPrompt.model_validate({"prompt_id": "empty", "label": " ", "text": "cup"})


def test_unsupported_prompt_combinations_are_rejected() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        SegmentationPrompt.model_validate(
            {
                "prompt_id": "mixed",
                "label": "cup",
                "text": "cup",
                "frame_id": "frame_000000",
                "box_xyxy": [0, 0, 10, 10],
            }
        )


def test_prompt_reference_and_geometry_validation(
    fake_run: Path,
    tmp_path: Path,
) -> None:
    manifest = IngestManifest.model_validate_json(
        (fake_run / "inputs" / "manifest.json").read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="unknown frame"):
        validate_prompt_references(
            SegmentationPromptManifest(
                prompts=[
                    SegmentationPrompt(
                        prompt_id="box",
                        label="box",
                        frame_id="missing",
                        box_xyxy=(0, 0, 3, 3),
                    )
                ]
            ),
            manifest,
            prompt_root=tmp_path,
        )
    for payload, message in [
        (
            {
                "prompt_id": "box",
                "label": "box",
                "frame_id": manifest.frames[0].frame_id,
                "box_xyxy": [0, 0, manifest.frames[0].width + 1, 2],
            },
            "box is outside",
        ),
        (
            {
                "prompt_id": "point",
                "label": "point",
                "frame_id": manifest.frames[0].frame_id,
                "points": [{"x": manifest.frames[0].width, "y": 1, "label": 1}],
            },
            "point.*outside",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            validate_prompt_references(
                SegmentationPromptManifest(prompts=[SegmentationPrompt.model_validate(payload)]),
                manifest,
                prompt_root=tmp_path,
            )


def test_seed_mask_dimensions_and_binary_values(
    fake_run: Path,
    tmp_path: Path,
) -> None:
    manifest = IngestManifest.model_validate_json(
        (fake_run / "inputs" / "manifest.json").read_text(encoding="utf-8")
    )
    frame = manifest.frames[0]
    prompt = SegmentationPrompt(
        prompt_id="mask",
        label="mask",
        frame_id=frame.frame_id,
        mask_path="mask.png",
    )
    prompt_manifest = SegmentationPromptManifest(prompts=[prompt])
    Image.new("L", (frame.width + 1, frame.height), 0).save(tmp_path / "mask.png")
    with pytest.raises(ValueError, match="dimensions"):
        validate_prompt_references(
            prompt_manifest,
            manifest,
            prompt_root=tmp_path,
        )
    Image.new("L", (frame.width, frame.height), 128).save(tmp_path / "mask.png")
    with pytest.raises(ValueError, match="only 0 and 255"):
        validate_prompt_references(
            prompt_manifest,
            manifest,
            prompt_root=tmp_path,
        )
    Image.new("L", (frame.width, frame.height), 255).save(tmp_path / "mask.png")
    validate_prompt_references(prompt_manifest, manifest, prompt_root=tmp_path)


def test_frame_order_and_best_registered_anchor(fake_run: Path) -> None:
    manifest = IngestManifest.model_validate_json(
        (fake_run / "inputs" / "manifest.json").read_text(encoding="utf-8")
    )
    request = Sam3InferenceRequest.model_validate_json(
        (fake_run / "observations" / "sam3_request.json").read_text(encoding="utf-8")
    )
    quality = FrameQualityReport.model_validate_json(
        (fake_run / "inputs" / "frame_qa.json").read_text(encoding="utf-8")
    )
    camera = CameraReconstruction.model_validate_json(
        (fake_run / "camera" / "reconstruction.json").read_text(encoding="utf-8")
    )
    anchors, _ = select_anchor_frames(
        manifest,
        quality,
        camera,
        strategy="best_quality_registered_frame",
    )
    assert request.frame_order == [frame.frame_id for frame in manifest.frames]
    assert request.frame_paths == [frame.relative_path for frame in manifest.frames]
    assert request.anchor_frames[0].frame_id == anchors[0].frame_id


def test_explicit_anchor_selection(fake_run: Path) -> None:
    manifest = IngestManifest.model_validate_json(
        (fake_run / "inputs" / "manifest.json").read_text(encoding="utf-8")
    )
    quality = FrameQualityReport.model_validate_json(
        (fake_run / "inputs" / "frame_qa.json").read_text(encoding="utf-8")
    )
    camera = CameraReconstruction.model_validate_json(
        (fake_run / "camera" / "reconstruction.json").read_text(encoding="utf-8")
    )
    selected = manifest.frames[-1].frame_id
    anchors, diagnostics = select_anchor_frames(
        manifest,
        quality,
        camera,
        strategy="explicit",
        explicit_frame_ids=[selected],
    )
    assert [anchor.frame_id for anchor in anchors] == [selected]
    assert diagnostics[0].selection_reason == "explicitly configured"


def test_semantic_label_normalization() -> None:
    assert normalize_semantic_label("  Cabinet Handle / Left  ") == "cabinet_handle_left"
    assert normalize_semantic_label("***") == "object"


def test_multiple_instances_receive_stable_ids(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path, mode="multiple_instances")
    tracks = _artifact(run_dir).tracks
    assert [track.object_id for track in tracks] == ["table_0001", "table_0002"]
    assert len({track.raw_model_object_id for track in tracks}) == 2


def test_prompt_instance_limit_is_enforced(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.yaml"
    prompt_path.write_text(
        "schema_version: '0.1.0'\n"
        "prompts:\n"
        "  - prompt_id: table\n"
        "    label: table\n"
        "    text: table\n"
        "    instance_limit: 1\n",
        encoding="utf-8",
    )
    run_dir, _ = _run(
        tmp_path,
        config=_config(mode="multiple_instances", prompt_path=prompt_path),
    )
    diagnostics = json.loads(
        (run_dir / "observations" / "diagnostics.json").read_text(encoding="utf-8")
    )
    assert [track.object_id for track in _artifact(run_dir).tracks] == ["table_0001"]
    assert diagnostics["dropped_tracks"][0]["reason_code"] == "instance_limit"


def test_canonical_ids_do_not_depend_on_raw_ids(tmp_path: Path) -> None:
    first_dir, _ = _run(tmp_path / "first", mode="raw_ids_a")
    second_dir, _ = _run(tmp_path / "second", mode="raw_ids_b")
    first = _artifact(first_dir)
    second = _artifact(second_dir)
    assert [track.object_id for track in first.tracks] == [
        track.object_id for track in second.tracks
    ]
    assert [track.raw_model_object_id for track in first.tracks] != [
        track.raw_model_object_id for track in second.tracks
    ]


def test_repeated_normalization_is_byte_identical(tmp_path: Path) -> None:
    config = _config(mode="success_multi")
    run_dir, _ = _run(tmp_path, config=config)
    path = run_dir / "observations" / "object_tracks.json"
    first = path.read_bytes()
    PipelineRunner(config, Path("examples/tabletop"), run_dir).run(
        from_stage="segmentation_tracking",
        until_stage="segmentation_tracking",
    )
    assert path.read_bytes() == first


def test_canonical_mask_contract_and_derived_boxes(fake_run: Path) -> None:
    manifest = IngestManifest.model_validate_json(
        (fake_run / "inputs" / "manifest.json").read_text(encoding="utf-8")
    )
    frames = {frame.frame_id: frame for frame in manifest.frames}
    for track in _artifact(fake_run).tracks:
        for observation in track.observations:
            frame = frames[observation.frame_id]
            area, bbox = validate_canonical_mask(
                fake_run / observation.mask_path,
                expected_size=(frame.width, frame.height),
                expected_area=observation.mask_area_pixels,
                expected_bbox=observation.bbox_xywh,
            )
            assert area == observation.mask_area_pixels
            assert bbox == observation.bbox_xywh


def test_canonical_mask_rejects_wrong_mode_dimensions_and_values(
    tmp_path: Path,
) -> None:
    rgb = tmp_path / "rgb.png"
    Image.new("RGB", (4, 4), (255, 255, 255)).save(rgb)
    with pytest.raises(ValueError, match="mode L"):
        validate_canonical_mask(rgb, expected_size=(4, 4))
    nonbinary = tmp_path / "nonbinary.png"
    Image.new("L", (4, 4), 128).save(nonbinary)
    with pytest.raises(ValueError, match="only 0 and 255"):
        validate_canonical_mask(nonbinary, expected_size=(4, 4))
    binary = tmp_path / "binary.png"
    Image.new("L", (5, 4), 255).save(binary)
    with pytest.raises(ValueError, match="dimensions"):
        validate_canonical_mask(binary, expected_size=(4, 4))


@pytest.mark.parametrize(
    "mode",
    [
        "invalid_dimensions",
        "empty_mask",
        "non_binary_mask",
        "invalid_box",
        "non_finite_score",
        "out_of_range_score",
        "unknown_frame",
        "duplicate_observation",
        "fragmented_track",
    ],
)
def test_invalid_or_short_tracks_are_dropped_not_worker_failures(
    tmp_path: Path,
    mode: str,
) -> None:
    run_dir, _ = _run(tmp_path, mode=mode)
    artifact = _artifact(run_dir)
    diagnostics = json.loads(
        (run_dir / "observations" / "diagnostics.json").read_text(encoding="utf-8")
    )
    assert artifact.tracks == []
    assert diagnostics["dropped_tracks"]


def test_coverage_filter_is_distinct_from_short_track_filter(tmp_path: Path) -> None:
    config = _config(mode="fragmented_track")
    sam = config.stages["segmentation_tracking"].adapter.config
    sam["min_track_observations"] = 1
    sam["min_track_coverage"] = 0.8
    run_dir, _ = _run(tmp_path, config=config)
    diagnostics = json.loads(
        (run_dir / "observations" / "diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["dropped_tracks"][0]["reason_code"] == "insufficient_coverage"


def test_duplicate_suppression_is_within_prompt_only(tmp_path: Path) -> None:
    duplicate_dir, _ = _run(tmp_path / "duplicate", mode="duplicate_tracks")
    duplicate_diagnostics = json.loads(
        (duplicate_dir / "observations" / "diagnostics.json").read_text(encoding="utf-8")
    )
    assert len(_artifact(duplicate_dir).tracks) == 1
    assert duplicate_diagnostics["dropped_tracks"][0]["reason_code"] == "duplicate_track"

    cross_label_dir, _ = _run(tmp_path / "cross-label", mode="success_multi")
    assert len(_artifact(cross_label_dir).tracks) == 3


def test_no_detection_is_a_successful_empty_result(tmp_path: Path) -> None:
    run_dir, manifest = _run(tmp_path, mode="no_detections")
    assert manifest["stages"]["segmentation_tracking"]["status"] == "succeeded"
    assert _artifact(run_dir).tracks == []
    diagnostics = json.loads(
        (run_dir / "observations" / "diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["no_matching_prompt_ids"] == ["cabinet", "cup", "table"]


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("missing_output", "completed without worker_result"),
        ("malformed_json", "output is malformed"),
        ("nonzero_exit", "failed with return code 17"),
        ("oom", "out of GPU memory"),
        ("unauthorized", "checkpoint access was denied"),
    ],
)
def test_worker_failures_are_actionable(
    tmp_path: Path,
    mode: str,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _run(tmp_path, mode=mode)


def test_timeout_is_terminated_and_recorded(tmp_path: Path) -> None:
    config = _config(mode="timeout")
    config.stages["segmentation_tracking"].adapter.timeout_s = 0.1
    with pytest.raises(RuntimeError, match="timed out"):
        _run(tmp_path, config=config)
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["segmentation_tracking"]["attempts"][0]["status"] == "failed"


def test_retries_use_fresh_attempts(tmp_path: Path) -> None:
    config = _config(mode="nonzero_exit")
    config.stages["segmentation_tracking"].adapter.retries = 1
    with pytest.raises(RuntimeError, match="return code 17"):
        _run(tmp_path, config=config)
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    attempts = manifest["stages"]["segmentation_tracking"]["attempts"]
    assert [attempt["status"] for attempt in attempts] == ["failed", "failed"]
    assert attempts[0]["workspace"] != attempts[1]["workspace"]


@pytest.mark.parametrize("failure_mode", ["nonzero_exit", "missing_output"])
def test_failed_worker_preserves_previous_canonical_outputs(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    run_dir, _ = _run(tmp_path, mode="success_multi")
    path = run_dir / "observations" / "object_tracks.json"
    previous = path.read_bytes()
    failed_config = _config(mode=failure_mode)
    with pytest.raises(RuntimeError):
        PipelineRunner(
            failed_config,
            Path("examples/tabletop"),
            run_dir,
        ).run(from_stage="segmentation_tracking")
    assert path.read_bytes() == previous


def test_token_values_are_redacted_from_all_retained_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "hf_phase2_secret_value"
    monkeypatch.setenv("HF_TOKEN", token)
    config = _config(mode="leak_token")
    config.stages["segmentation_tracking"].adapter.env.append("HF_TOKEN")
    run_dir, _ = _run(tmp_path, config=config)
    for path in run_dir.rglob("*"):
        if path.is_file():
            assert token.encode() not in path.read_bytes(), path
    logs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (run_dir / "observations" / "raw" / "logs").glob("*.log")
    )
    assert "[REDACTED]" in logs


class NoiseAncestorAdapter:
    name = "sam3_noise_ancestor_test"
    version = "0.1.0"

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return HealthcheckResult(True, "noise fixture ready")

    def prepare(self, context: StageContext) -> None:
        pass

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        return [
            OutputSpec(
                "camera/colmap/database.db",
                "colmap_database",
                "application/octet-stream",
                "test",
            ),
            OutputSpec(
                "camera/colmap/logs/mapper.log",
                "colmap_log",
                "text/plain",
                "test",
            ),
            OutputSpec(
                "camera/colmap/sparse/0/cameras.bin",
                "colmap_binary_model",
                "application/octet-stream",
                "test",
            ),
        ]

    def run(self, context: StageContext) -> StageResult:
        for spec in self.expected_outputs(context):
            path = context.path(spec.relative_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"raw-colmap-noise")
        return StageResult()


def test_selective_materialization_excludes_raw_colmap_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(REGISTRY, NoiseAncestorAdapter.name, NoiseAncestorAdapter)
    config = _config()
    stages = config.stages
    stages["colmap_noise"] = StageConfig(
        adapter=AdapterConfig(name=NoiseAncestorAdapter.name),
        depends_on=["camera_recovery"],
    )
    segmentation = stages.pop("segmentation_tracking")
    segmentation.depends_on.append("colmap_noise")
    stages["segmentation_tracking"] = segmentation
    run_dir, manifest = _run(tmp_path, config=config)
    attempt = manifest["stages"]["segmentation_tracking"]["attempts"][0]
    workspace = run_dir / attempt["workspace"]
    materialized = {item["relative_path"] for item in attempt["materialized_inputs"]}
    assert {
        "inputs/manifest.json",
        "inputs/frame_qa.json",
        "camera/reconstruction.json",
        "observations/prompt_inputs/prompts.yaml",
    } <= materialized
    assert all(path.startswith("frames/") for path in materialized if path.startswith("frames/"))
    assert "camera/colmap/database.db" not in materialized
    assert not (workspace / "camera" / "colmap").exists()


def test_canonical_upstream_hashes_remain_unchanged(fake_run: Path) -> None:
    request = Sam3InferenceRequest.model_validate_json(
        (fake_run / "observations" / "sam3_request.json").read_text(encoding="utf-8")
    )
    assert _sha256(fake_run / request.frame_manifest_path) == request.frame_manifest_sha256
    assert (
        _sha256(fake_run / request.camera_reconstruction_path)
        == request.camera_reconstruction_sha256
    )
    manifest = IngestManifest.model_validate_json(
        (fake_run / request.frame_manifest_path).read_text(encoding="utf-8")
    )
    assert all(_sha256(fake_run / frame.relative_path) == frame.sha256 for frame in manifest.frames)


def test_prompt_hash_invalidates_segmentation_only(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.yaml"
    shutil.copy2("configs/prompts/tabletop.yaml", prompt_path)
    config = _config(prompt_path=prompt_path)
    run_dir, _ = _run(tmp_path, config=config)
    prompt_path.write_text(
        prompt_path.read_text(encoding="utf-8") + "\n# hash-only change\n",
        encoding="utf-8",
    )
    resumed = PipelineRunner(config, Path("examples/tabletop"), run_dir).run(resume=True)
    assert resumed["stages"]["ingest"]["last_execution"] == "cache_hit"
    assert resumed["stages"]["camera_recovery"]["last_execution"] == "cache_hit"
    assert resumed["stages"]["segmentation_tracking"]["last_execution"] == "executed"


def test_seed_mask_hash_is_recorded_and_invalidates_segmentation(tmp_path: Path) -> None:
    seed_mask = tmp_path / "seed.png"
    Image.new("L", (32, 24), 0).save(seed_mask)
    prompt_path = tmp_path / "prompts.yaml"
    prompt_path.write_text(
        "schema_version: '0.1.0'\n"
        "prompts:\n"
        "  - prompt_id: cup_seed\n"
        "    label: cup\n"
        "    frame_id: frame_000000\n"
        "    mask_path: seed.png\n",
        encoding="utf-8",
    )
    config = _config(mode="one_object", prompt_path=prompt_path)
    run_dir, _ = _run(tmp_path, config=config)
    prompts = SegmentationPromptManifest.model_validate_json(
        (run_dir / "observations" / "prompts.json").read_text(encoding="utf-8")
    )
    normalized_seed = "observations/prompt_inputs/masks/cup_seed.png"
    assert prompts.input_hashes[normalized_seed] == _sha256(run_dir / normalized_seed)
    request = Sam3InferenceRequest.model_validate_json(
        (run_dir / "observations" / "sam3_request.json").read_text(encoding="utf-8")
    )
    assert request.prompt_manifest_sha256 == _sha256(run_dir / "observations" / "prompts.json")

    Image.new("L", (32, 24), 255).save(seed_mask)
    resumed = PipelineRunner(config, Path("examples/tabletop"), run_dir).run(resume=True)
    assert resumed["stages"]["ingest"]["last_execution"] == "cache_hit"
    assert resumed["stages"]["camera_recovery"]["last_execution"] == "cache_hit"
    assert resumed["stages"]["segmentation_tracking"]["last_execution"] == "executed"


class PartialCameraAdapter(MockCameraRecoveryAdapter):
    name = "partial_camera_for_sam3_test"
    version = "0.1.0"

    def run(self, context: StageContext) -> StageResult:
        result = super().run(context)
        path = context.path("camera", "reconstruction.json")
        camera = CameraReconstruction.model_validate_json(path.read_text(encoding="utf-8"))
        payload = camera.model_dump(mode="json")
        payload["poses"] = payload["poses"][:-1]
        payload["registered_frame_ids"] = payload["registered_frame_ids"][:-1]
        payload["unregistered_frame_ids"] = [camera.registered_frame_ids[-1]]
        atomic_write_json(path, CameraReconstruction.model_validate(payload))
        return result


def test_unregistered_frames_keep_2d_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(REGISTRY, PartialCameraAdapter.name, PartialCameraAdapter)
    config = _config()
    config.stages["camera_recovery"].adapter.name = PartialCameraAdapter.name
    run_dir, _ = _run(tmp_path, config=config)
    artifact = _artifact(run_dir)
    last_observations = [track.observations[-1] for track in artifact.tracks]
    assert last_observations
    assert all(not observation.camera_pose_available for observation in last_observations)


def test_previews_and_coco_export_are_deterministic(
    fake_run: Path,
    tmp_path: Path,
) -> None:
    manifest = IngestManifest.model_validate_json(
        (fake_run / "inputs" / "manifest.json").read_text(encoding="utf-8")
    )
    camera = CameraReconstruction.model_validate_json(
        (fake_run / "camera" / "reconstruction.json").read_text(encoding="utf-8")
    )
    artifact = _artifact(fake_run)
    preview_paths = render_previews(fake_run, manifest, artifact, camera)
    first_hashes = {path: _sha256(fake_run / path) for path in preview_paths}
    assert {path: _sha256(fake_run / path) for path in preview_paths} == first_hashes
    first_coco = tmp_path / "first.json"
    second_coco = tmp_path / "second.json"
    export_coco(fake_run, manifest, artifact, first_coco)
    export_coco(fake_run, manifest, artifact, second_coco)
    assert first_coco.read_bytes() == second_coco.read_bytes()
    payload = json.loads(first_coco.read_text(encoding="utf-8"))
    assert len(payload["annotations"]) == 9
    assert [category["id"] for category in payload["categories"]] == [1, 2, 3]


def test_segmentation_cli_commands(fake_run: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    inspect = runner.invoke(app, ["segmentation", "inspect", str(fake_run)])
    assert inspect.exit_code == 0
    assert '"kept_tracks": 3' in inspect.output
    preview = runner.invoke(
        app,
        ["segmentation", "render-preview", str(fake_run)],
    )
    assert preview.exit_code == 0
    output = tmp_path / "annotations.json"
    coco = runner.invoke(
        app,
        [
            "segmentation",
            "export-coco",
            str(fake_run),
            "--output",
            str(output),
        ],
    )
    assert coco.exit_code == 0
    assert output.is_file()


def _health_context(config: dict[str, Any], *, env: list[str] | None = None) -> StageContext:
    return StageContext(
        stage_name="segmentation_tracking",
        input_dir=Path("."),
        run_dir=Path("."),
        canonical_run_dir=Path("."),
        config=StageConfig(
            adapter=AdapterConfig(
                name="sam3_segmentation_tracking",
                config=config,
                env=env or ["PATH"],
            )
        ),
        seed=7,
    )


def test_fake_worker_healthcheck_is_configuration_aware() -> None:
    config = Sam3AdapterConfig.model_validate(
        _config().stages["segmentation_tracking"].adapter.config
    )
    result = Sam3SegmentationTrackingAdapter().healthcheck(
        _health_context(config.model_dump(mode="json"))
    )
    assert result.ok
    assert "fake_worker" in result.message


def _isolated_worker_python(tmp_path: Path, body: str = "") -> Path:
    environment = tmp_path / "sam3-env"
    executable = environment / "bin" / "python"
    executable.parent.mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    executable.write_text(f"#!/usr/bin/env python3\n{body}", encoding="utf-8")
    executable.chmod(0o755)
    return executable


def test_only_pinned_official_checkpoint_pairs_are_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    base = _config().stages["segmentation_tracking"].adapter.config.copy()
    base.update(
        {
            "execution_mode": "local_worker",
            "worker_python": str(_isolated_worker_python(tmp_path)),
            "device": "cuda",
            "precision": "bfloat16",
            "model_mode": "sam3",
            "checkpoint_repository": "facebook/sam3",
            "checkpoint_revision": "3c879f39826c281e95690f02c7821c4de09afae7",
        }
    )
    Sam3AdapterConfig.model_validate(base)
    base["checkpoint_revision"] = "main"
    with pytest.raises(ValidationError, match="pinned official checkpoint"):
        Sam3AdapterConfig.model_validate(base)


def test_local_worker_healthcheck_uses_configured_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    executable = _isolated_worker_python(
        tmp_path,
        "import json\n"
        "print(json.dumps({'available': True, 'official_code_commit': "
        "'46957e47805eaa273f4aa7bbbd25a88bca9108ce'}))\n",
    )
    config = {
        "execution_mode": "local_worker",
        "worker_python": str(executable),
        "worker_module": "sam3_worker",
        "prompt_manifest": "configs/prompts/tabletop.yaml",
        "device": "cuda",
        "precision": "bfloat16",
    }
    result = Sam3SegmentationTrackingAdapter().healthcheck(_health_context(config))
    assert result.ok
    assert "46957e47805eaa273f4aa7bbbd25a88bca9108ce" in result.message


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("anchor_count", 2, "anchor_count=1"),
        ("precision", "float16", "precision=bfloat16"),
        ("strategy", "full_video_text_prompt", "detect_then_track"),
    ],
)
def test_real_backend_rejects_unsupported_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    config: dict[str, object] = {
        "execution_mode": "local_worker",
        "worker_python": str(_isolated_worker_python(tmp_path)),
        "prompt_manifest": "configs/prompts/tabletop.yaml",
        "device": "cuda",
        "precision": "bfloat16",
        field: value,
    }
    with pytest.raises(ValidationError, match=message):
        Sam3AdapterConfig.model_validate(config)


def test_local_worker_requires_visible_gpu_and_isolated_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    config = {
        "execution_mode": "local_worker",
        "worker_python": shutil.which("python") or "python",
        "prompt_manifest": "configs/prompts/tabletop.yaml",
        "device": "cuda",
        "precision": "bfloat16",
    }
    with pytest.raises(ValidationError, match="CUDA_VISIBLE_DEVICES"):
        Sam3AdapterConfig.model_validate(config)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValidationError, match="core Python environment|isolated virtual"):
        Sam3AdapterConfig.model_validate(config)


def test_docker_healthcheck_checks_image_gpu_and_uid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "docker-args.log"
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "pathlib.Path(os.environ['FAKE_DOCKER_LOG']).open('a').write(' '.join(args) + '\\n')\n"
        "if args[:1] == ['version']:\n"
        "    print('27.0.0')\n"
        "elif args[:2] == ['image', 'inspect']:\n"
        "    print('sha256:phase2')\n"
        "else:\n"
        '    print(\'{"available": true, "device": "cuda"}\')\n',
        encoding="utf-8",
    )
    docker.chmod(0o755)
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(log_path))
    config = {
        "execution_mode": "docker",
        "docker_executable": str(docker),
        "docker_image": "reconevery/sam3:test",
        "prompt_manifest": "configs/prompts/tabletop.yaml",
        "device": "cuda",
        "precision": "bfloat16",
    }
    result = Sam3SegmentationTrackingAdapter().healthcheck(_health_context(config))
    assert result.ok
    commands = log_path.read_text(encoding="utf-8")
    assert "image inspect" in commands
    assert "--gpus all" in commands
    if hasattr(os, "getuid"):
        assert f"--user {os.getuid()}:{os.getgid()}" in commands


def test_checked_in_segmentation_schemas_are_current() -> None:
    models = {
        "segmentation_prompts.schema.json": SegmentationPromptManifest,
        "sam3_inference_request.schema.json": Sam3InferenceRequest,
        "segmentation_tracking.schema.json": SegmentationTrackingArtifact,
    }
    for filename, model in models.items():
        checked_in = json.loads((Path("schemas") / filename).read_text(encoding="utf-8"))
        assert checked_in == model.model_json_schema()


def test_prompt_manifest_file_is_typed() -> None:
    prompts = load_prompt_manifest(Path("configs/prompts/tabletop.yaml"))
    assert [prompt.prompt_id for prompt in prompts.prompts] == [
        "table",
        "cup",
        "cabinet",
    ]
