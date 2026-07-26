from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ValidationError

from recon2sim.adapters import REGISTRY
from recon2sim.adapters.base import StageContext
from recon2sim.alignment import export_aligned_ply, render_alignment_previews
from recon2sim.artifacts import (
    CameraDiagnostics,
    CameraMeshAlignmentArtifact,
    CameraMeshAlignmentDiagnostics,
    CameraMeshAlignmentPreviewManifest,
    CameraMeshAlignmentResult,
    CameraReconstruction,
    CandidateEvaluationManifest,
    CandidateGenerationManifest,
    CandidateSelectionArtifact,
    CompletionDiagnostics,
    CompletionEligibilityArtifact,
    CompletionEvidenceSplit,
    DenseDepthManifest,
    DenseFusionArtifact,
    DenseMVSDiagnostics,
    DenseWorkspaceManifest,
    EndToEndConsistencyReport,
    FrameQualityReport,
    GenReconWorkerManifest,
    GlobalSceneDiagnostics,
    GlobalSceneReconstructionArtifact,
    IngestManifest,
    MeasuredGeneratedComparisonArtifact,
    MeasuredObjectDiagnostics,
    MeasuredObjectGeometryArtifact,
    MeasuredObjectHypothesis,
    ObjectLiftingAlignmentComparison,
    ObjectSurfaceDiagnostics,
    ObjectSurfaceEvidenceArtifact,
    ObjectSurfaceHypothesis,
    ObjectSurfaceMethodComparison,
    Phase4_2ConsistencyReport,
    Phase4ConsistencyReport,
    Phase5AConsistencyReport,
    Phase5BConsistencyReport,
    Sam3WorkerManifest,
    SegmentationDiagnostics,
    SegmentationPromptManifest,
    SegmentationTrackingArtifact,
    TransformChainAudit,
)
from recon2sim.config import load_config
from recon2sim.genrecon import read_colmap_text_points, render_global_previews
from recon2sim.ir import SceneIR
from recon2sim.object_lifting import (
    export_object_face_ids,
    export_object_surface,
    render_summary_previews,
)
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
objects_app = typer.Typer(
    help="Inspect and export observation-supported partial object surfaces.",
    no_args_is_help=True,
)
alignment_app = typer.Typer(
    help="Inspect and export camera-to-global-mesh alignment artifacts.",
    no_args_is_help=True,
)
dense_app = typer.Typer(
    help="Inspect and export official COLMAP dense MVS artifacts.",
    no_args_is_help=True,
)
completion_app = typer.Typer(
    help="Inspect rigid visual-completion candidates and held-out selection.",
    no_args_is_help=True,
)
app.add_typer(adapters_app, name="adapters")
app.add_typer(ingest_app, name="ingest")
app.add_typer(camera_app, name="camera")
app.add_typer(segmentation_app, name="segmentation")
app.add_typer(reconstruction_app, name="reconstruction")
app.add_typer(validation_app, name="validation")
app.add_typer(objects_app, name="objects")
app.add_typer(alignment_app, name="alignment")
app.add_typer(dense_app, name="dense")
app.add_typer(completion_app, name="completion")


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
                "runtime_repository_revisions": worker.runtime_repository_revisions,
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


def _surface_artifact(run_dir: Path) -> ObjectSurfaceEvidenceArtifact:
    return _artifact_model(
        run_dir,
        "reconstruction/object_surfaces/evidence_manifest.json",
        ObjectSurfaceEvidenceArtifact,
    )


def _surface_hypothesis(
    artifact: ObjectSurfaceEvidenceArtifact,
    object_id: str,
) -> ObjectSurfaceHypothesis:
    for hypothesis in artifact.hypotheses:
        if hypothesis.object_id == object_id:
            return hypothesis
    raise typer.BadParameter(f"unknown Phase 4 object ID: {object_id}")


@objects_app.command("inspect-surfaces")
def inspect_surfaces(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    artifact = _surface_artifact(run_dir)
    diagnostics = _artifact_model(
        run_dir,
        "reconstruction/object_surfaces/diagnostics.json",
        ObjectSurfaceDiagnostics,
    )
    comparison = _artifact_model(
        run_dir,
        "reconstruction/object_surfaces/method_comparison.json",
        ObjectSurfaceMethodComparison,
    )
    alignment = _artifact_model(
        run_dir,
        "reconstruction/object_surfaces/camera_mesh_alignment.json",
        CameraMeshAlignmentArtifact,
    )
    typer.echo(
        json.dumps(
            {
                "track_count": diagnostics.track_count,
                "accepted_objects": diagnostics.accepted_object_count,
                "partial_objects": diagnostics.partial_object_count,
                "ambiguous_objects": diagnostics.ambiguous_object_count,
                "unresolved_objects": diagnostics.unresolved_object_count,
                "global_face_count": diagnostics.global_face_count,
                "assigned_face_count": diagnostics.accepted_face_count,
                "unassigned_face_ratio": diagnostics.unassigned_face_ratio,
                "multi_label_overlap_count": diagnostics.different_label_overlap_count,
                "same_class_conflict_count": diagnostics.same_class_conflict_count,
                "runtime_seconds": diagnostics.runtime_seconds,
                "peak_gpu_memory_bytes": diagnostics.peak_gpu_memory_bytes,
                "lifting_method": comparison.selected_method,
                "method_comparison": comparison.conclusion,
                "alignment_sufficient_for_lifting": (alignment.alignment_sufficient_for_lifting),
                "mesh_pixel_coverage_mean": alignment.mesh_pixel_coverage_mean,
                "sparse_depth_residual_median": alignment.sparse_depth_residual_median,
                "sparse_depth_inlier_fraction": alignment.sparse_depth_inlier_fraction,
                "diagnosed_bottleneck": diagnostics.diagnosed_bottleneck,
                "coordinate_convention": artifact.coordinate_convention.model_dump(mode="json"),
                "geometry_status": artifact.geometry_status,
                "hidden_surface_completion": artifact.hidden_surface_completion,
                "sim_ready": artifact.sim_ready,
                "warnings": [*artifact.warnings, *diagnostics.warnings],
            },
            indent=2,
        )
    )


@objects_app.command("inspect-surface")
def inspect_surface(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    object_id: Annotated[str, typer.Argument(help="Canonical SAM object ID.")],
) -> None:
    hypothesis = _surface_hypothesis(_surface_artifact(run_dir), object_id)
    typer.echo(
        json.dumps(
            {
                "object_id": hypothesis.object_id,
                "semantic_label": hypothesis.semantic_label,
                "supporting_frames": hypothesis.supporting_registered_frame_ids,
                "accepted_faces": hypothesis.accepted_global_face_ids.count,
                "ambiguous_faces": hypothesis.ambiguous_global_face_ids.count,
                "components": hypothesis.component_count,
                "exact_components": hypothesis.exact_component_count,
                "seam_aware_components": hypothesis.seam_aware_component_count,
                "bbox_min": hypothesis.bbox_min,
                "bbox_max": hypothesis.bbox_max,
                "mean_support_score": hypothesis.mean_face_support_score,
                "mean_reprojection_iou": hypothesis.mean_reprojection_iou,
                "association_precision": hypothesis.association_precision,
                "mask_recall": hypothesis.mask_recall,
                "multiview_support": hypothesis.multiview_support,
                "observed_surface_coverage": hypothesis.observed_surface_coverage,
                "association_confidence": hypothesis.association_confidence,
                "completeness_confidence": hypothesis.completeness_confidence,
                "status": hypothesis.status,
                "geometry_status": hypothesis.geometry_status,
                "sim_ready": hypothesis.sim_ready,
                "warnings": hypothesis.warnings,
            },
            indent=2,
        )
    )


@objects_app.command("render-surface-previews")
def render_surface_previews(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    artifact = _surface_artifact(run_dir)
    render_summary_previews(run_dir, artifact)
    typer.echo("regenerated deterministic Phase 4 summary previews")


@objects_app.command("export-surface")
def export_surface(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    object_id: Annotated[str, typer.Argument(help="Canonical SAM object ID.")],
    output: Annotated[Path, typer.Option("--output", help="Destination PLY path.")],
) -> None:
    hypothesis = _surface_hypothesis(_surface_artifact(run_dir), object_id)
    export_object_surface(run_dir, hypothesis, output)
    typer.echo(f"exported {object_id} partial surface to {output}")


@objects_app.command("export-face-ids")
def export_face_ids(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    object_id: Annotated[str, typer.Argument(help="Canonical SAM object ID.")],
    output: Annotated[Path, typer.Option("--output", help="Destination binary path.")],
) -> None:
    hypothesis = _surface_hypothesis(_surface_artifact(run_dir), object_id)
    export_object_face_ids(run_dir, hypothesis, output)
    typer.echo(f"exported {object_id} original global face IDs to {output}")


@validation_app.command("inspect-phase4")
def inspect_phase4(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    report = _artifact_model(
        run_dir,
        "validation/phase4_object_surface_consistency.json",
        Phase4ConsistencyReport,
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
                    "real_2d_tracks_lifted_to_global_3d": (
                        report.real_2d_tracks_lifted_to_global_3d
                    ),
                    "hidden_surface_completion_implemented": (
                        report.hidden_surface_completion_implemented
                    ),
                    "object_replacement_implemented": report.object_replacement_implemented,
                    "sim_ready_scene_implemented": report.sim_ready_scene_implemented,
                    "metric_scale_known": report.metric_scale_known,
                    "canonical_gravity_alignment_known": (report.canonical_gravity_alignment_known),
                },
                "warnings": report.warnings,
            },
            indent=2,
        )
    )


@validation_app.command("verify-phase4")
def verify_phase4(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    report = _artifact_model(
        run_dir,
        "validation/phase4_object_surface_consistency.json",
        Phase4ConsistencyReport,
    )
    if not report.passed:
        failed = [check.check_id for check in report.checks if not check.passed]
        raise typer.BadParameter(f"Phase 4 consistency verification failed: {failed}")
    typer.echo(f"Phase 4 object-surface consistency passed ({len(report.checks)} checks)")


def _alignment_result(run_dir: Path) -> CameraMeshAlignmentResult:
    return _artifact_model(
        run_dir,
        "reconstruction/alignment/alignment.json",
        CameraMeshAlignmentResult,
    )


@alignment_app.command("inspect")
def inspect_alignment(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    result = _alignment_result(run_dir)
    diagnostics = _artifact_model(
        run_dir,
        "reconstruction/alignment/diagnostics.json",
        CameraMeshAlignmentDiagnostics,
    )
    baseline = result.baseline_validation_metrics
    aligned = result.aligned_validation_metrics
    typer.echo(
        json.dumps(
            {
                "status": result.status,
                "accepted": result.accepted,
                "scale": result.transform.scale,
                "rotation_degrees": result.transform.rotation_degrees,
                "translation_scene_diagonal_ratio": (
                    result.transform.translation_scene_diagonal_ratio
                ),
                "baseline_validation_residual": (baseline.sparse_depth_residual_median),
                "aligned_validation_residual": aligned.sparse_depth_residual_median,
                "baseline_p90_residual": baseline.sparse_depth_residual_p90,
                "aligned_p90_residual": aligned.sparse_depth_residual_p90,
                "baseline_inlier_fraction": baseline.inlier_fractions,
                "aligned_inlier_fraction": aligned.inlier_fractions,
                "mesh_coverage_before": baseline.mesh_pixel_coverage,
                "mesh_coverage_after": aligned.mesh_pixel_coverage,
                "bad_camera_fraction": aligned.bad_frame_fraction,
                "residual_is_locally_structured": (diagnostics.residual_is_locally_structured),
                "diagnosis": diagnostics.diagnosis,
                "coordinate_convention": result.coordinate_convention.model_dump(mode="json"),
                "warnings": [*result.warnings, *diagnostics.warnings],
            },
            indent=2,
        )
    )


@alignment_app.command("inspect-transform-chain")
def inspect_alignment_transform_chain(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    audit = _artifact_model(
        run_dir,
        "reconstruction/alignment/transform_chain_audit.json",
        TransformChainAudit,
    )
    typer.echo(json.dumps(audit.model_dump(mode="json"), indent=2))


@alignment_app.command("inspect-camera")
def inspect_alignment_camera(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    frame_id: Annotated[str, typer.Argument(help="Registered canonical frame ID.")],
) -> None:
    diagnostics = _artifact_model(
        run_dir,
        "reconstruction/alignment/diagnostics.json",
        CameraMeshAlignmentDiagnostics,
    )
    for camera in diagnostics.camera_metrics:
        if camera.frame_id == frame_id:
            typer.echo(json.dumps(camera.model_dump(mode="json"), indent=2))
            return
    raise typer.BadParameter(f"alignment diagnostics do not contain frame {frame_id!r}")


@alignment_app.command("render-previews")
def render_camera_mesh_alignment_previews(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    result = _alignment_result(run_dir)
    diagnostics = _artifact_model(
        run_dir,
        "reconstruction/alignment/diagnostics.json",
        CameraMeshAlignmentDiagnostics,
    )
    previews = _artifact_model(
        run_dir,
        "reconstruction/alignment/preview_manifest.json",
        CameraMeshAlignmentPreviewManifest,
    )
    render_alignment_previews(run_dir, result, diagnostics, previews)
    typer.echo("regenerated deterministic camera/mesh alignment previews")


@alignment_app.command("export-transform")
def export_camera_mesh_alignment_transform(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    output: Annotated[Path, typer.Option("--output", help="Destination JSON path.")],
) -> None:
    result = _alignment_result(run_dir)
    atomic_write_json(output, result.transform)
    typer.echo(f"exported camera/mesh alignment transform to {output}")


@alignment_app.command("export-aligned-mesh")
def export_camera_aligned_mesh(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    output: Annotated[Path, typer.Option("--output", help="Destination PLY path.")],
) -> None:
    result = _alignment_result(run_dir)
    if not result.accepted:
        raise typer.BadParameter(
            f"alignment is not accepted ({result.status}); aligned mesh export refused"
        )
    export_aligned_ply(
        run_dir / "reconstruction/global/mesh.ply",
        output,
        result.transform,
    )
    typer.echo(f"exported derived aligned mesh to {output}")


@alignment_app.command("compare-object-lifting")
def compare_aligned_object_lifting(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    comparison = _artifact_model(
        run_dir,
        "reconstruction/alignment/object_lifting_comparison.json",
        ObjectLiftingAlignmentComparison,
    )
    typer.echo(json.dumps(comparison.model_dump(mode="json"), indent=2))


@validation_app.command("inspect-phase4-2")
def inspect_phase4_2(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    report = _artifact_model(
        run_dir,
        "validation/phase4_2_camera_mesh_alignment.json",
        Phase4_2ConsistencyReport,
    )
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))


@validation_app.command("verify-phase4-2")
def verify_phase4_2(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    report = _artifact_model(
        run_dir,
        "validation/phase4_2_camera_mesh_alignment.json",
        Phase4_2ConsistencyReport,
    )
    if not report.passed:
        failed = [check.check_id for check in report.checks if not check.passed]
        raise typer.BadParameter(f"Phase 4.2 consistency verification failed: {failed}")
    typer.echo(f"Phase 4.2 camera/mesh consistency passed ({len(report.checks)} checks)")


@dense_app.command("inspect")
def inspect_dense_mvs(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    workspace = _artifact_model(
        run_dir,
        "reconstruction/dense/workspace_manifest.json",
        DenseWorkspaceManifest,
    )
    diagnostics = _artifact_model(
        run_dir,
        "reconstruction/dense/diagnostics.json",
        DenseMVSDiagnostics,
    )
    fusion = _artifact_model(
        run_dir,
        "reconstruction/dense/fusion.json",
        DenseFusionArtifact,
    )
    typer.echo(
        json.dumps(
            {
                "registered_frames": diagnostics.registered_frame_count,
                "successful_depth_maps": diagnostics.successful_depth_map_count,
                "failed_depth_maps": diagnostics.failed_depth_map_count,
                "fused_points": fusion.point_count,
                "scene_bounds": [fusion.bounds_min, fusion.bounds_max],
                "patchmatch_runtime_seconds": diagnostics.patchmatch_seconds,
                "fusion_runtime_seconds": diagnostics.fusion_seconds,
                "peak_gpu_memory_bytes": diagnostics.peak_gpu_memory_bytes,
                "coordinate_convention": fusion.coordinate_convention.model_dump(mode="json"),
                "frame_ids": workspace.registered_frame_ids,
                "warnings": diagnostics.warnings,
            },
            indent=2,
        )
    )


@dense_app.command("inspect-frame")
def inspect_dense_frame(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    frame_id: Annotated[str, typer.Argument(help="Registered canonical frame ID.")],
) -> None:
    depth = _artifact_model(
        run_dir,
        "reconstruction/dense/depth_manifest.json",
        DenseDepthManifest,
    )
    for record in depth.records:
        if record.frame_id == frame_id:
            typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))
            return
    raise typer.BadParameter(f"dense MVS has no successful depth map for {frame_id!r}")


@dense_app.command("render-previews")
def render_dense_previews(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    root = run_dir / "reconstruction" / "dense" / "previews"
    expected = [
        root / f"{name}.png"
        for name in (
            "depth_contact_sheet",
            "normal_contact_sheet",
            "consistency_contact_sheet",
            "fused_point_cloud",
            "camera_dense_coverage",
        )
    ]
    missing = [path.name for path in expected if not path.is_file()]
    if missing:
        raise typer.BadParameter(f"dense preview artifacts are missing: {missing}")
    typer.echo("dense MVS previews are present and deterministic")


@dense_app.command("export-fused")
def export_dense_fused(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    output: Annotated[Path, typer.Option("--output", help="Destination PLY path.")],
) -> None:
    source = run_dir / "reconstruction" / "dense" / "fused.ply"
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    typer.echo(f"exported dense fused point cloud to {output}")


def _measured_artifact(run_dir: Path) -> MeasuredObjectGeometryArtifact:
    return _artifact_model(
        run_dir,
        "reconstruction/measured_objects/geometry_manifest.json",
        MeasuredObjectGeometryArtifact,
    )


def _measured_hypothesis(
    artifact: MeasuredObjectGeometryArtifact, object_id: str
) -> MeasuredObjectHypothesis:
    for hypothesis in artifact.hypotheses:
        if hypothesis.object_id == object_id:
            return hypothesis
    raise typer.BadParameter(f"measured geometry has no object {object_id!r}")


@objects_app.command("inspect-measured")
def inspect_measured_objects(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    artifact = _measured_artifact(run_dir)
    diagnostics = _artifact_model(
        run_dir,
        "reconstruction/measured_objects/diagnostics.json",
        MeasuredObjectDiagnostics,
    )
    typer.echo(
        json.dumps(
            {
                "tracks": diagnostics.track_count,
                "accepted": diagnostics.accepted_object_count,
                "partial": diagnostics.partial_object_count,
                "unresolved": diagnostics.unresolved_object_count,
                "raw_samples": diagnostics.raw_sample_count,
                "validated_samples": diagnostics.validated_sample_count,
                "surfels": diagnostics.fused_surfel_count,
                "objects": [
                    {
                        "object_id": item.object_id,
                        "status": item.status,
                        "surfels": item.fused_surfel_count,
                        "supporting_views": item.supporting_view_count,
                        "reprojection_iou": item.reprojection_iou,
                    }
                    for item in artifact.hypotheses
                ],
                "warnings": diagnostics.warnings,
            },
            indent=2,
        )
    )


@objects_app.command("inspect-measured-object")
def inspect_measured_object(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    object_id: Annotated[str, typer.Argument(help="Canonical SAM object ID.")],
) -> None:
    typer.echo(
        json.dumps(
            _measured_hypothesis(_measured_artifact(run_dir), object_id).model_dump(mode="json"),
            indent=2,
        )
    )


def _export_measured_path(run_dir: Path, object_id: str, output: Path, *, surface: bool) -> None:
    hypothesis = _measured_hypothesis(_measured_artifact(run_dir), object_id)
    record = hypothesis.observed_surface if surface else hypothesis.point_cloud
    if record is None:
        kind = "observed surface" if surface else "measured points"
        raise typer.BadParameter(f"{object_id!r} has no {kind}")
    source = run_dir / record.relative_path
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)


@objects_app.command("export-measured-points")
def export_measured_points(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    object_id: Annotated[str, typer.Argument(help="Canonical SAM object ID.")],
    output: Annotated[Path, typer.Option("--output", help="Destination PLY path.")],
) -> None:
    _export_measured_path(run_dir, object_id, output, surface=False)
    typer.echo(f"exported measured points for {object_id} to {output}")


@objects_app.command("export-observed-surface")
def export_observed_surface(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    object_id: Annotated[str, typer.Argument(help="Canonical SAM object ID.")],
    output: Annotated[Path, typer.Option("--output", help="Destination PLY path.")],
) -> None:
    _export_measured_path(run_dir, object_id, output, surface=True)
    typer.echo(f"exported observed-only surface for {object_id} to {output}")


@objects_app.command("compare-measured-generated")
def compare_measured_generated(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    object_id: Annotated[str, typer.Argument(help="Canonical SAM object ID.")],
) -> None:
    artifact = _artifact_model(
        run_dir,
        "reconstruction/measured_generated/comparison.json",
        MeasuredGeneratedComparisonArtifact,
    )
    for item in artifact.objects:
        if item.object_id == object_id:
            typer.echo(json.dumps(item.model_dump(mode="json"), indent=2))
            return
    raise typer.BadParameter(f"comparison has no object {object_id!r}")


@validation_app.command("inspect-phase5a")
def inspect_phase5a(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    report = _artifact_model(
        run_dir,
        "validation/phase5a_measured_geometry.json",
        Phase5AConsistencyReport,
    )
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))


@validation_app.command("verify-phase5a")
def verify_phase5a(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    report = _artifact_model(
        run_dir,
        "validation/phase5a_measured_geometry.json",
        Phase5AConsistencyReport,
    )
    if not report.passed:
        failed = [check.check_id for check in report.checks if not check.passed]
        raise typer.BadParameter(f"Phase 5A consistency verification failed: {failed}")
    typer.echo(f"Phase 5A measured-geometry consistency passed ({len(report.checks)} checks)")


def _completion_selection(run_dir: Path) -> CandidateSelectionArtifact:
    return _artifact_model(
        run_dir,
        "reconstruction/completion/selection.json",
        CandidateSelectionArtifact,
    )


def _completion_generations(run_dir: Path) -> list[CandidateGenerationManifest]:
    return [
        _artifact_model(run_dir, path, CandidateGenerationManifest)
        for path in (
            "reconstruction/completion/sam3d_generation_manifest.json",
            "reconstruction/completion/trellis2_generation_manifest.json",
            "reconstruction/completion/measured_generation_manifest.json",
        )
    ]


@completion_app.command("inspect")
def inspect_completion(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    diagnostics = _artifact_model(
        run_dir,
        "reconstruction/completion/diagnostics.json",
        CompletionDiagnostics,
    )
    selection = _completion_selection(run_dir)
    typer.echo(
        json.dumps(
            {
                **diagnostics.model_dump(mode="json"),
                "objects": [item.model_dump(mode="json") for item in selection.objects],
            },
            indent=2,
        )
    )


@completion_app.command("inspect-object")
def inspect_completion_object(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    object_id: Annotated[str, typer.Argument(help="Canonical object ID.")],
) -> None:
    eligibility = _artifact_model(
        run_dir,
        "reconstruction/completion/eligibility.json",
        CompletionEligibilityArtifact,
    )
    split = _artifact_model(
        run_dir,
        "reconstruction/completion/evidence_split.json",
        CompletionEvidenceSplit,
    )
    selection = _completion_selection(run_dir)
    payload = {
        "eligibility": next(
            (
                item.model_dump(mode="json")
                for item in eligibility.records
                if item.object_id == object_id
            ),
            None,
        ),
        "evidence_split": next(
            (item.model_dump(mode="json") for item in split.objects if item.object_id == object_id),
            None,
        ),
        "selection": next(
            (
                item.model_dump(mode="json")
                for item in selection.objects
                if item.object_id == object_id
            ),
            None,
        ),
    }
    if payload["eligibility"] is None:
        raise typer.BadParameter(f"completion has no object {object_id!r}")
    typer.echo(json.dumps(payload, indent=2))


@completion_app.command("inspect-candidate")
def inspect_completion_candidate(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    candidate_id: Annotated[str, typer.Argument(help="Deterministic candidate ID.")],
) -> None:
    for generation in _completion_generations(run_dir):
        for candidate in generation.candidates:
            if candidate.candidate_id == candidate_id:
                typer.echo(json.dumps(candidate.model_dump(mode="json"), indent=2))
                return
    raise typer.BadParameter(f"completion has no candidate {candidate_id!r}")


@completion_app.command("render-previews")
def render_completion_previews(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    from recon2sim.adapters.completion_selection import CompletionSelectionAdapter

    selection = _completion_selection(run_dir)
    evaluation = _artifact_model(
        run_dir,
        "reconstruction/completion/evaluation_manifest.json",
        CandidateEvaluationManifest,
    )
    CompletionSelectionAdapter._write_previews(
        run_dir / "reconstruction/completion/previews",
        selection,
        evaluation,
        run_dir,
    )
    typer.echo("regenerated deterministic Phase 5B completion previews")


@completion_app.command("compare-candidates")
def compare_completion_candidates(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    object_id: Annotated[str, typer.Argument(help="Canonical object ID.")],
) -> None:
    evaluation = _artifact_model(
        run_dir,
        "reconstruction/completion/evaluation_manifest.json",
        CandidateEvaluationManifest,
    )
    items = [
        item.model_dump(mode="json")
        for item in evaluation.evaluations
        if item.object_id == object_id
    ]
    if not items:
        raise typer.BadParameter(f"completion has no evaluations for {object_id!r}")
    typer.echo(json.dumps(items, indent=2))


def _export_completion_candidate(run_dir: Path, candidate_id: str, output: Path) -> None:
    for generation in _completion_generations(run_dir):
        for candidate in generation.candidates:
            if candidate.candidate_id != candidate_id:
                continue
            source = run_dir / candidate.selection_asset_path
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output)
            return
    raise typer.BadParameter(f"completion has no candidate {candidate_id!r}")


@completion_app.command("export-candidate")
def export_completion_candidate(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    candidate_id: Annotated[str, typer.Argument(help="Deterministic candidate ID.")],
    output: Annotated[Path, typer.Option("--output", help="Destination asset path.")],
) -> None:
    _export_completion_candidate(run_dir, candidate_id, output)
    typer.echo(f"exported {candidate_id} to {output}")


@completion_app.command("export-selected")
def export_selected_completion(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    object_id: Annotated[str, typer.Argument(help="Canonical object ID.")],
    output: Annotated[Path, typer.Option("--output", help="Destination asset path.")],
) -> None:
    selected = next(
        (item for item in _completion_selection(run_dir).objects if item.object_id == object_id),
        None,
    )
    if selected is None or selected.selected_candidate is None:
        raise typer.BadParameter(f"{object_id!r} has no selected completion")
    _export_completion_candidate(run_dir, selected.selected_candidate, output)
    typer.echo(f"exported selected completion for {object_id} to {output}")


@completion_app.command("explain-selection")
def explain_completion_selection(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    object_id: Annotated[str, typer.Argument(help="Canonical object ID.")],
) -> None:
    evaluation = _artifact_model(
        run_dir,
        "reconstruction/completion/evaluation_manifest.json",
        CandidateEvaluationManifest,
    )
    selected = next(
        (item for item in _completion_selection(run_dir).objects if item.object_id == object_id),
        None,
    )
    if selected is None:
        raise typer.BadParameter(f"completion has no object {object_id!r}")
    typer.echo(
        json.dumps(
            {
                "selection": selected.model_dump(mode="json"),
                "candidate_gates": [
                    {
                        "candidate_id": item.candidate_id,
                        "passed": item.passed_hard_gates,
                        "failed_gates": item.failed_gates,
                        "metrics": item.metrics.model_dump(mode="json"),
                    }
                    for item in evaluation.evaluations
                    if item.object_id == object_id
                ],
            },
            indent=2,
        )
    )


@validation_app.command("inspect-phase5b")
def inspect_phase5b(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    report = _artifact_model(
        run_dir,
        "validation/phase5b_rigid_completion.json",
        Phase5BConsistencyReport,
    )
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))


@validation_app.command("verify-phase5b")
def verify_phase5b(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    report = _artifact_model(
        run_dir,
        "validation/phase5b_rigid_completion.json",
        Phase5BConsistencyReport,
    )
    if not report.passed:
        failed = [check.check_id for check in report.checks if not check.passed]
        raise typer.BadParameter(f"Phase 5B consistency verification failed: {failed}")
    typer.echo(f"Phase 5B rigid-completion consistency passed ({len(report.checks)} checks)")
