from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ValidationError

from recon2sim.adapters import REGISTRY
from recon2sim.adapters.base import StageContext
from recon2sim.artifacts import (
    CameraDiagnostics,
    CameraReconstruction,
    EndToEndConsistencyReport,
    FrameQualityReport,
    GenReconWorkerManifest,
    GlobalSceneDiagnostics,
    GlobalSceneReconstructionArtifact,
    IngestManifest,
    Sam3WorkerManifest,
    SegmentationDiagnostics,
    SegmentationPromptManifest,
    SegmentationTrackingArtifact,
)
from recon2sim.config import load_config
from recon2sim.genrecon import read_colmap_text_points, render_global_previews
from recon2sim.ir import SceneIR
from recon2sim.pipeline import PipelineConfigurationError, PipelineRunner
from recon2sim.segmentation import export_coco, render_previews
from recon2sim.storage import atomic_write_json

app = typer.Typer(
    help="Recon2Sim typed observation-to-simulation pipeline.",
    no_args_is_help=True,
)
adapters_app = typer.Typer(
    help="Inspect configured adapter implementations.",
    no_args_is_help=True,
)
ingest_app = typer.Typer(help="Inspect normalized ingest artifacts.", no_args_is_help=True)
camera_app = typer.Typer(help="Inspect and export camera recovery artifacts.", no_args_is_help=True)
segmentation_app = typer.Typer(
    help="Inspect and export canonical segmentation tracks.",
    no_args_is_help=True,
)
reconstruction_app = typer.Typer(
    help="Inspect and export global visual reconstruction artifacts.",
    no_args_is_help=True,
)
validation_app = typer.Typer(
    help="Inspect and verify cross-stage consistency reports.",
    no_args_is_help=True,
)
app.add_typer(adapters_app, name="adapters")
app.add_typer(ingest_app, name="ingest")
app.add_typer(camera_app, name="camera")
app.add_typer(segmentation_app, name="segmentation")
app.add_typer(reconstruction_app, name="reconstruction")
app.add_typer(validation_app, name="validation")


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Directory to initialize.")] = Path("."),
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Initialized Recon2Sim workspace at {path}")


@app.command()
def run(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input",
            help="Input video or observation directory.",
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
        ),
    ],
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Pipeline YAML config.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    run_dir: Annotated[Path, typer.Option("--run-dir", help="Run output directory.")],
    resume: Annotated[
        bool,
        typer.Option(help="Reuse successful stages whose signatures and outputs still match."),
    ] = False,
    from_stage: Annotated[
        str | None,
        typer.Option("--from-stage", help="First stage to execute."),
    ] = None,
    until_stage: Annotated[
        str | None,
        typer.Option("--until-stage", help="Last stage to execute."),
    ] = None,
) -> None:
    try:
        manifest = PipelineRunner(load_config(config_path), input_dir, run_dir).run(
            resume=resume,
            from_stage=from_stage,
            until_stage=until_stage,
        )
    except (
        FileNotFoundError,
        PipelineConfigurationError,
        RuntimeError,
        ValidationError,
        ValueError,
    ) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "stages": {
                    name: {
                        "status": entry["status"],
                        "last_execution": entry.get("last_execution"),
                    }
                    for name, entry in manifest["stages"].items()
                },
            },
            indent=2,
        )
    )


@app.command()
def status(
    run_dir: Annotated[
        Path,
        typer.Argument(
            help="Existing run directory.",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise typer.BadParameter(f"run manifest does not exist: {manifest_path}")
    typer.echo(manifest_path.read_text(encoding="utf-8"))


@app.command("validate-ir")
def validate_ir(
    path: Annotated[
        Path,
        typer.Argument(
            help="Scene IR JSON file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
) -> None:
    try:
        scene = SceneIR.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise typer.BadParameter(f"invalid Scene IR {path}: {exc}") from exc
    typer.echo(f"valid Scene IR: {scene.metadata.scene_id} ({len(scene.objects)} objects)")


@app.command()
def inspect(
    run_dir: Annotated[
        Path,
        typer.Argument(
            help="Existing run directory.",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
) -> None:
    scene_path = run_dir / "scene_ir" / "scene.json"
    if not scene_path.is_file():
        raise typer.BadParameter(f"Scene IR does not exist: {scene_path}")
    scene = SceneIR.model_validate_json(scene_path.read_text(encoding="utf-8"))
    typer.echo(
        json.dumps(
            {
                "scene_id": scene.metadata.scene_id,
                "objects": [obj.object_id for obj in scene.objects],
                "relations": len(scene.relations),
            },
            indent=2,
        )
    )


@app.command()
def clean(
    run_dir: Annotated[Path, typer.Argument(help="Run directory to delete.")],
    force: Annotated[
        bool,
        typer.Option("--force", help="Required to delete the run directory."),
    ] = False,
) -> None:
    if not force:
        raise typer.BadParameter("Pass --force to delete a run directory.")
    if not run_dir.is_dir():
        raise typer.BadParameter(f"run directory does not exist: {run_dir}")
    shutil.rmtree(run_dir)
    typer.echo(f"deleted {run_dir}")


@adapters_app.command("list")
def list_adapters() -> None:
    for name in sorted(REGISTRY):
        typer.echo(name)


@adapters_app.command("healthcheck")
def healthcheck(
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="Check the adapter settings in a pipeline YAML config.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
) -> None:
    if config_path is None:
        for name, adapter_class in sorted(REGISTRY.items()):
            result = adapter_class().healthcheck()
            typer.echo(f"{name}: {'ok' if result.ok else 'fail'} - {result.message}")
        return

    config = load_config(config_path)
    for stage_name, stage in config.stages.items():
        configured_adapter_class = REGISTRY.get(stage.adapter.name)
        if configured_adapter_class is None:
            typer.echo(f"{stage_name} ({stage.adapter.name}): fail - adapter is not registered")
            continue
        context = StageContext(
            stage_name=stage_name,
            input_dir=Path("."),
            run_dir=Path("."),
            canonical_run_dir=Path("."),
            config=stage,
            seed=config.seed,
        )
        result = configured_adapter_class().healthcheck(context)
        typer.echo(
            f"{stage_name} ({stage.adapter.name}): "
            f"{'ok' if result.ok else 'fail'} - {result.message}"
        )


def _artifact_model[ModelT: BaseModel](
    run_dir: Path,
    relative_path: str,
    model: type[ModelT],
) -> ModelT:
    path = run_dir / relative_path
    if not path.is_file():
        raise typer.BadParameter(f"required artifact does not exist: {path}")
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise typer.BadParameter(f"invalid artifact {path}: {exc}") from exc


def _segmentation_inputs(
    run_dir: Path,
) -> tuple[
    IngestManifest,
    CameraReconstruction,
    SegmentationPromptManifest,
    Sam3WorkerManifest,
    SegmentationTrackingArtifact,
    SegmentationDiagnostics,
]:
    return (
        _artifact_model(run_dir, "inputs/manifest.json", IngestManifest),
        _artifact_model(run_dir, "camera/reconstruction.json", CameraReconstruction),
        _artifact_model(
            run_dir,
            "observations/prompts.json",
            SegmentationPromptManifest,
        ),
        _artifact_model(
            run_dir,
            "observations/worker_manifest.json",
            Sam3WorkerManifest,
        ),
        _artifact_model(
            run_dir,
            "observations/object_tracks.json",
            SegmentationTrackingArtifact,
        ),
        _artifact_model(
            run_dir,
            "observations/diagnostics.json",
            SegmentationDiagnostics,
        ),
    )


@ingest_app.command("inspect")
def inspect_ingest(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    manifest = _artifact_model(run_dir, "inputs/manifest.json", IngestManifest)
    report_path = run_dir / "inputs" / "frame_qa.json"
    report = (
        FrameQualityReport.model_validate_json(report_path.read_text(encoding="utf-8"))
        if report_path.is_file()
        else None
    )
    typer.echo(
        json.dumps(
            {
                "source_type": manifest.source_type,
                "source_input": manifest.source_input_path,
                "decoded_frames": manifest.total_decoded_frames,
                "selected_frames": len(manifest.frames),
                "rejected_frames": sum(entry.status.value == "rejected" for entry in report.entries)
                if report is not None
                else 0,
                "frame_size": (
                    [manifest.frames[0].width, manifest.frames[0].height]
                    if manifest.frames
                    else None
                ),
            },
            indent=2,
            default=str,
        )
    )


@camera_app.command("inspect")
def inspect_camera(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    reconstruction = _artifact_model(
        run_dir,
        "camera/reconstruction.json",
        CameraReconstruction,
    )
    diagnostics_path = run_dir / "camera" / "diagnostics.json"
    diagnostics = (
        CameraDiagnostics.model_validate_json(diagnostics_path.read_text(encoding="utf-8"))
        if diagnostics_path.is_file()
        else None
    )
    selected = (
        next((model for model in diagnostics.models if model.selected), None)
        if diagnostics is not None
        else None
    )
    typer.echo(
        json.dumps(
            {
                "input_frames": (
                    diagnostics.input_frame_count
                    if diagnostics is not None
                    else len(reconstruction.poses)
                ),
                "selected_frames": (
                    diagnostics.selected_frame_count
                    if diagnostics is not None
                    else len(reconstruction.poses)
                ),
                "registered_frames": len(reconstruction.registered_frame_ids),
                "registration_ratio": selected.registration_ratio if selected else None,
                "camera_model": reconstruction.model,
                "intrinsics": reconstruction.intrinsics.model_dump(mode="json"),
                "sparse_points": reconstruction.sparse_point_count,
                "world_frame": reconstruction.coordinate_convention.world_frame,
                "alignment_status": reconstruction.coordinate_convention.alignment_status,
                "camera_axes": reconstruction.coordinate_convention.camera_axes,
                "linear_units": reconstruction.coordinate_convention.linear_units,
                "scale_status": reconstruction.scale_status,
                "warnings": diagnostics.warnings if diagnostics is not None else [],
            },
            indent=2,
            default=str,
        )
    )


@camera_app.command("export-trajectory")
def export_camera_trajectory(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    output: Annotated[Path, typer.Option("--output", help="Trajectory JSON output path.")],
) -> None:
    reconstruction = _artifact_model(
        run_dir,
        "camera/reconstruction.json",
        CameraReconstruction,
    )
    atomic_write_json(
        output,
        {
            "camera_id": reconstruction.camera_id,
            "coordinate_convention": reconstruction.coordinate_convention.model_dump(mode="json"),
            "scale_status": reconstruction.scale_status,
            "poses": [pose.model_dump(mode="json") for pose in reconstruction.poses],
        },
    )
    typer.echo(f"wrote {len(reconstruction.poses)} poses to {output}")


@camera_app.command("colmap-stats")
def colmap_stats(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    diagnostics = _artifact_model(run_dir, "camera/diagnostics.json", CameraDiagnostics)
    typer.echo(
        json.dumps(
            {
                "selected_model": diagnostics.selected_model,
                "models": [model.model_dump(mode="json") for model in diagnostics.models],
                "warnings": diagnostics.warnings,
            },
            indent=2,
        )
    )


@segmentation_app.command("inspect")
def inspect_segmentation(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    manifest, camera, prompts, worker, artifact, diagnostics = _segmentation_inputs(run_dir)
    typer.echo(
        json.dumps(
            {
                "backend_mode": diagnostics.backend_mode,
                "official_code_commit": worker.official_code_commit,
                "checkpoint": (f"{worker.checkpoint_repository}@{worker.checkpoint_revision}"),
                "prompt_count": diagnostics.prompt_count,
                "prompt_labels": [prompt.label for prompt in prompts.prompts if prompt.enabled],
                "input_frames": len(manifest.frames),
                "registered_frames": len(camera.registered_frame_ids),
                "unregistered_frames": len(camera.unregistered_frame_ids),
                "anchor_frames": [anchor.frame_id for anchor in diagnostics.anchor_frames],
                "raw_tracks": diagnostics.raw_track_count,
                "kept_tracks": diagnostics.kept_track_count,
                "dropped_tracks": len(diagnostics.dropped_tracks),
                "mask_count": diagnostics.mask_count,
                "mean_coverage": diagnostics.mean_coverage,
                "mean_confidence": diagnostics.mean_confidence,
                "runtime_seconds": diagnostics.runtime_seconds,
                "peak_gpu_memory_bytes": diagnostics.peak_gpu_memory_bytes,
                "object_ids": [track.object_id for track in artifact.tracks],
                "warnings": diagnostics.warnings,
            },
            indent=2,
            default=str,
        )
    )


@segmentation_app.command("render-preview")
def render_segmentation_preview(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    manifest, camera, _, _, artifact, _ = _segmentation_inputs(run_dir)
    try:
        outputs = render_previews(
            run_dir,
            manifest,
            artifact,
            camera,
            include_frame_previews=True,
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"preview rendering failed: {exc}") from exc
    typer.echo(f"wrote {len(outputs)} deterministic previews under observations/previews")


@segmentation_app.command("export-coco")
def export_segmentation_coco(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="COCO-style annotation JSON output path."),
    ],
) -> None:
    manifest, _, _, _, artifact, _ = _segmentation_inputs(run_dir)
    try:
        export_coco(run_dir, manifest, artifact, output)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"COCO export failed: {exc}") from exc
    annotation_count = sum(len(track.observations) for track in artifact.tracks)
    typer.echo(f"wrote {annotation_count} annotations to {output}")


@reconstruction_app.command("inspect-global")
def inspect_global_reconstruction(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    metadata = _artifact_model(
        run_dir,
        "reconstruction/global/metadata.json",
        GlobalSceneReconstructionArtifact,
    )
    diagnostics = _artifact_model(
        run_dir,
        "reconstruction/global/diagnostics.json",
        GlobalSceneDiagnostics,
    )
    worker = _artifact_model(
        run_dir,
        "reconstruction/global/worker_manifest.json",
        GenReconWorkerManifest,
    )
    typer.echo(
        json.dumps(
            {
                "official_code_commit": worker.official_code_commit,
                "runtime_model": (
                    f"{worker.runtime_model_repository}@{worker.runtime_model_revision}"
                ),
                "checkpoint_hashes": {
                    record.checkpoint_id: record.sha256 for record in worker.checkpoint_records
                },
                "input_frames": metadata.input_frame_count,
                "registered_frames": metadata.registered_frame_count,
                "selected_genrecon_views": len(metadata.actual_selected_frame_ids),
                "sparse_points": diagnostics.initial_sparse_points,
                "cleaned_points": diagnostics.cleaned_sparse_points,
                "chunks": metadata.chunk_count,
                "vertices": metadata.mesh.vertex_count,
                "faces": metadata.mesh.face_count,
                "materials": metadata.mesh.material_count,
                "bounding_box": {
                    "min": metadata.mesh.bounding_box_min,
                    "max": metadata.mesh.bounding_box_max,
                },
                "runtime_seconds": metadata.runtime_seconds,
                "peak_gpu_memory_bytes": metadata.peak_gpu_memory_bytes,
                "coordinate_convention": metadata.coordinate_convention.model_dump(mode="json"),
                "warnings": diagnostics.warnings,
            },
            indent=2,
            default=str,
        )
    )


@reconstruction_app.command("render-global-preview")
def render_global_reconstruction_preview(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    manifest = _artifact_model(run_dir, "inputs/manifest.json", IngestManifest)
    camera = _artifact_model(
        run_dir,
        "camera/reconstruction.json",
        CameraReconstruction,
    )
    try:
        outputs = render_global_previews(
            root=run_dir,
            manifest=manifest,
            camera=camera,
            sparse_points=read_colmap_text_points(run_dir / "camera/genrecon_package/points3D.txt"),
            mesh_path=run_dir / "reconstruction/global/mesh.ply",
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"global preview rendering failed: {exc}") from exc
    typer.echo(f"wrote {len(outputs)} deterministic global previews")


@reconstruction_app.command("export-global-mesh")
def export_global_mesh(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output path for the canonical global PLY mesh."),
    ],
) -> None:
    metadata = _artifact_model(
        run_dir,
        "reconstruction/global/metadata.json",
        GlobalSceneReconstructionArtifact,
    )
    source = run_dir / metadata.mesh_asset_path
    if not source.is_file():
        raise typer.BadParameter(f"canonical global mesh does not exist: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    typer.echo(f"wrote global mesh to {output}")


@validation_app.command("inspect-phase3-e2e")
def inspect_phase3_e2e(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    report = _artifact_model(
        run_dir,
        "validation/phase3_e2e_consistency.json",
        EndToEndConsistencyReport,
    )
    typer.echo(
        json.dumps(
            {
                "passed": report.passed,
                "checks": {
                    check.check_id: {
                        "passed": check.passed,
                        "message": check.message,
                    }
                    for check in report.checks
                },
                "capability_boundary": {
                    "object_level_2d_3d_fusion_implemented": (
                        report.object_level_2d_3d_fusion_implemented
                    ),
                    "sim_ready_scene_implemented": report.sim_ready_scene_implemented,
                    "metric_scale_known": report.metric_scale_known,
                    "canonical_gravity_alignment_known": (report.canonical_gravity_alignment_known),
                },
                "warnings": report.warnings,
            },
            indent=2,
        )
    )


@validation_app.command("verify-phase3-e2e")
def verify_phase3_e2e(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    report = _artifact_model(
        run_dir,
        "validation/phase3_e2e_consistency.json",
        EndToEndConsistencyReport,
    )
    if not report.passed:
        failed = [check.check_id for check in report.checks if not check.passed]
        raise typer.BadParameter(f"Phase 3 consistency verification failed: {failed}")
    typer.echo(f"Phase 3 end-to-end consistency passed ({len(report.checks)} checks)")
