from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import BaseModel, ValidationError

from recon2sim.adapters import REGISTRY
from recon2sim.adapters.base import StageContext
from recon2sim.alignment import export_aligned_ply, render_alignment_previews
from recon2sim.artifacts import (
    ArticulatedCandidateManifest,
    ArticulatedCandidateSelection,
    ArticulatedEvaluationManifest,
    ArticulationCaptureManifest,
    ArticulationFittingManifest,
    ArticulationPartPromptManifest,
    ArticulationStateAlignmentArtifact,
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
    KnownDistanceLandmarkManifest,
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
    Phase5CConsistencyReport,
    Phase6AConsistencyReport,
    Phase6BConsistencyReport,
    Sam3WorkerManifest,
    SceneAssemblyBundle,
    SceneAssemblyLineageReport,
    SceneAssemblyPlan,
    SegmentationDiagnostics,
    SegmentationPromptManifest,
    SegmentationTrackingArtifact,
    TransformChainAudit,
    WorldCalibrationArtifact,
    WorldCalibrationManifest,
)
from recon2sim.calibration import rotate_vector, transform_point
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
articulation_app = typer.Typer(
    help="Inspect articulated visual hypotheses and held-out-state validation.",
    no_args_is_help=True,
)
calibration_app = typer.Typer(
    help="Inspect and export evidence-grounded metric world calibration.",
    no_args_is_help=True,
)
assembly_app = typer.Typer(
    help="Inspect and export calibration-optional visual scene assemblies.",
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
app.add_typer(articulation_app, name="articulation")
app.add_typer(calibration_app, name="calibration")
app.add_typer(assembly_app, name="assembly")


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


def _articulation_selection(run_dir: Path) -> ArticulatedCandidateSelection:
    return _artifact_model(
        run_dir,
        "reconstruction/articulation/selection.json",
        ArticulatedCandidateSelection,
    )


def _articulation_candidates(run_dir: Path) -> ArticulatedCandidateManifest:
    return _artifact_model(
        run_dir,
        "reconstruction/articulation/candidate_manifest.json",
        ArticulatedCandidateManifest,
    )


@articulation_app.command("capture-template")
def articulation_capture_template(
    object_id: Annotated[str, typer.Option("--object-id", help="Stable articulated object ID.")],
    states: Annotated[
        str,
        typer.Option("--states", help="Comma-separated semantic state labels."),
    ],
) -> None:
    labels = [value.strip() for value in states.split(",") if value.strip()]
    if not labels:
        raise typer.BadParameter("--states must contain at least one state label")
    payload = {
        "schema_version": "0.2.0",
        "articulated_object_id": object_id,
        "reference_state_id": f"state_000_{labels[0]}",
        "states": [
            {
                "state_id": f"state_{index:03d}_{label}",
                "run_dir": f"/absolute/path/to/{label}",
                "semantic_state_label": label,
                "part_track_ids": {
                    "cabinet_body": f"<canonical-track-id-for-cabinet-body-in-{label}>",
                    "drawer": f"<canonical-track-id-for-drawer-in-{label}>",
                },
            }
            for index, label in enumerate(labels)
        ],
    }
    typer.echo(yaml.safe_dump(payload, sort_keys=False))


@articulation_app.command("preflight-capture")
def preflight_articulation_capture(
    capture_manifest: Annotated[
        Path,
        typer.Option(
            "--capture-manifest",
            exists=True,
            dir_okay=False,
            help="Multi-state capture YAML.",
        ),
    ],
    part_manifest: Annotated[
        Path,
        typer.Option(
            "--part-manifest",
            exists=True,
            dir_okay=False,
            help="Stable part/prompt YAML.",
        ),
    ],
) -> None:
    capture_raw = yaml.safe_load(capture_manifest.read_text(encoding="utf-8"))
    prompt_raw = yaml.safe_load(part_manifest.read_text(encoding="utf-8"))
    if not isinstance(capture_raw, dict) or not isinstance(prompt_raw, dict):
        raise typer.BadParameter("capture and part manifests must contain YAML mappings")
    prompts = ArticulationPartPromptManifest.model_validate(prompt_raw)
    object_id = str(capture_raw.get("articulated_object_id", ""))
    prompt_object = next(
        (item for item in prompts.objects if item.articulated_object_id == object_id),
        None,
    )
    if prompt_object is None:
        raise typer.BadParameter(f"part manifest has no articulated object {object_id!r}")
    stable_parts = {
        prompt_object.base.part_id,
        *(part.part_id for part in prompt_object.movable_parts if part.include),
    }
    raw_states = capture_raw.get("states")
    if not isinstance(raw_states, list) or not raw_states:
        raise typer.BadParameter("capture manifest requires non-empty states")
    errors: list[str] = []
    rows: list[dict[str, object]] = []
    state_ids: list[str] = []
    for raw_state in raw_states:
        if not isinstance(raw_state, dict):
            errors.append("capture state is not a mapping")
            continue
        state_id = str(raw_state.get("state_id", ""))
        state_ids.append(state_id)
        run_dir = Path(str(raw_state.get("run_dir", ""))).expanduser()
        mapping_raw = raw_state.get("part_track_ids")
        mapping = (
            {str(key): str(value) for key, value in mapping_raw.items()}
            if isinstance(mapping_raw, dict)
            else {}
        )
        state_errors: list[str] = []
        if not run_dir.is_dir():
            state_errors.append("run_dir_missing")
        if set(mapping) != stable_parts:
            state_errors.append("part_mapping_incomplete")
        if len(mapping) != len(set(mapping.values())):
            state_errors.append("duplicate_track_mapping")
        tracks_by_id: dict[str, dict[str, object]] = {}
        measured_by_id: dict[str, dict[str, object]] = {}
        registered: set[str] = set()
        depth_count = 0
        if run_dir.is_dir():
            required_json = {
                "phase5a": run_dir / "validation/phase5a_measured_geometry.json",
                "camera": run_dir / "camera/reconstruction.json",
                "tracks": run_dir / "observations/object_tracks.json",
                "depth": run_dir / "reconstruction/dense/depth_manifest.json",
                "measured": (run_dir / "reconstruction/measured_objects/geometry_manifest.json"),
            }
            missing = [name for name, path in required_json.items() if not path.is_file()]
            state_errors.extend(f"{name}_missing" for name in missing)
            if not missing:
                payloads = {
                    name: json.loads(path.read_text(encoding="utf-8"))
                    for name, path in required_json.items()
                }
                if not payloads["phase5a"].get("passed", False):
                    state_errors.append("phase5a_failed")
                registered = set(payloads["camera"].get("registered_frame_ids", []))
                tracks_by_id = {
                    str(item["object_id"]): item for item in payloads["tracks"].get("tracks", [])
                }
                measured_by_id = {
                    str(item["object_id"]): item
                    for item in payloads["measured"].get("hypotheses", [])
                }
                depth_records = payloads["depth"].get("records", [])
                depth_count = len(depth_records)
                for record in depth_records:
                    path = run_dir / str(record.get("depth_path", ""))
                    if not path.is_file():
                        state_errors.append(f"depth_missing:{record.get('frame_id', 'unknown')}")
        for part_id in sorted(stable_parts):
            track_id = mapping.get(part_id)
            if not track_id:
                continue
            track = tracks_by_id.get(track_id)
            hypothesis = measured_by_id.get(track_id)
            if track is None:
                state_errors.append(f"track_missing:{part_id}->{track_id}")
                continue
            if hypothesis is None or hypothesis.get("point_cloud") is None:
                state_errors.append(f"measured_geometry_missing:{part_id}->{track_id}")
                continue
            point_cloud = hypothesis["point_cloud"]
            if not isinstance(point_cloud, dict):
                state_errors.append(f"point_cloud_invalid:{part_id}->{track_id}")
                continue
            point_path = run_dir / str(point_cloud.get("relative_path", ""))
            if not point_path.is_file():
                state_errors.append(f"point_cloud_missing:{part_id}->{track_id}")
            observations_raw = track.get("observations", [])
            observations = observations_raw if isinstance(observations_raw, list) else []
            masks = [
                run_dir / str(observation.get("mask_path", "")) for observation in observations
            ]
            if not masks or any(not path.is_file() for path in masks):
                state_errors.append(f"mask_missing:{part_id}->{track_id}")
            supporting_raw = hypothesis.get("supporting_frame_ids", [])
            supporting: set[str] = set()
            if isinstance(supporting_raw, list):
                supporting.update(str(value) for value in supporting_raw)
            measured_observations = hypothesis.get("observations", [])
            if isinstance(measured_observations, list):
                supporting.update(
                    str(observation["frame_id"])
                    for observation in measured_observations
                    if isinstance(observation, dict)
                    and observation.get("registered") is True
                    and int(observation.get("validated_sample_count", 0)) > 0
                    and observation.get("frame_id")
                )
            supporting &= registered
            if len(supporting) < 2:
                state_errors.append(f"insufficient_registered_views:{part_id}->{track_id}")
        errors.extend(f"{state_id}: {message}" for message in state_errors)
        rows.append(
            {
                "state": state_id,
                "phase5a": "pass" if "phase5a_failed" not in state_errors else "fail",
                "registered": len(registered),
                "depth_maps": depth_count,
                "mapped_parts": len(mapping),
                "status": "pass" if not state_errors else "fail",
            }
        )
    if len(state_ids) != len(set(state_ids)):
        errors.append("state IDs are not unique")
    reference_state = str(capture_raw.get("reference_state_id", ""))
    if reference_state not in state_ids:
        errors.append("reference state is missing")
    if len(state_ids) < 3:
        errors.append("a disjoint held-out articulation state is not feasible")
    typer.echo("state\tphase5a\tregistered\tdepth_maps\tmapped_parts\tstatus")
    for row in rows:
        typer.echo(
            f"{row['state']}\t{row['phase5a']}\t{row['registered']}\t"
            f"{row['depth_maps']}\t{row['mapped_parts']}\t{row['status']}"
        )
    if errors:
        for error in errors:
            typer.echo(f"ERROR: {error}", err=True)
        raise typer.Exit(code=1)
    typer.echo(
        f"capture preflight passed: {len(state_ids)} states, "
        f"reference={reference_state}, stable_parts={len(stable_parts)}"
    )


@articulation_app.command("inspect")
def inspect_articulation(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    capture = _artifact_model(
        run_dir,
        "reconstruction/articulation/capture_manifest.json",
        ArticulationCaptureManifest,
    )
    alignment = _artifact_model(
        run_dir,
        "reconstruction/articulation/state_alignment.json",
        ArticulationStateAlignmentArtifact,
    )
    candidates = _articulation_candidates(run_dir)
    selection = _articulation_selection(run_dir)
    typer.echo(
        json.dumps(
            {
                "object_id": capture.articulated_object_id,
                "capture_evidence_tier": capture.capture_evidence_tier,
                "states": [state.state_id for state in capture.states],
                "accepted_state_alignments": sum(
                    transform.accepted for transform in alignment.transforms
                ),
                "candidate_count": len(candidates.candidates),
                "selection": selection.model_dump(mode="json"),
            },
            indent=2,
        )
    )


@articulation_app.command("inspect-state")
def inspect_articulation_state(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    state_id: Annotated[str, typer.Argument(help="Articulation state ID.")],
) -> None:
    capture = _artifact_model(
        run_dir,
        "reconstruction/articulation/capture_manifest.json",
        ArticulationCaptureManifest,
    )
    alignment = _artifact_model(
        run_dir,
        "reconstruction/articulation/state_alignment.json",
        ArticulationStateAlignmentArtifact,
    )
    state = next((item for item in capture.states if item.state_id == state_id), None)
    transform = next(
        (item for item in alignment.transforms if item.state_id == state_id),
        None,
    )
    if state is None:
        raise typer.BadParameter(f"articulation has no state {state_id!r}")
    typer.echo(
        json.dumps(
            {
                "state": state.model_dump(mode="json"),
                "alignment": transform.model_dump(mode="json") if transform else None,
            },
            indent=2,
        )
    )


@articulation_app.command("inspect-part")
def inspect_articulation_part(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    part_id: Annotated[str, typer.Argument(help="Stable observed part ID.")],
) -> None:
    prompts = _artifact_model(
        run_dir,
        "reconstruction/articulation/part_prompt_manifest.json",
        ArticulationPartPromptManifest,
    )
    motion = json.loads(
        (run_dir / "reconstruction/articulation/measured_motion.json").read_text(encoding="utf-8")
    )
    prompt = next(
        (
            part
            for item in prompts.objects
            for part in item.movable_parts
            if part.part_id == part_id
        ),
        None,
    )
    joints = [
        joint for joint in motion.get("joint_hypotheses", []) if joint["child_part_id"] == part_id
    ]
    geometries = [
        geometry for geometry in motion.get("part_geometries", []) if geometry["part_id"] == part_id
    ]
    if prompt is None:
        raise typer.BadParameter(f"articulation has no part {part_id!r}")
    typer.echo(
        json.dumps(
            {
                "prompt": prompt.model_dump(mode="json"),
                "measured_geometries": geometries,
                "joint_hypotheses": joints,
            },
            indent=2,
        )
    )


@articulation_app.command("inspect-candidate")
def inspect_articulation_candidate(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    candidate_id: Annotated[str, typer.Argument(help="Articulated candidate ID.")],
) -> None:
    candidate = next(
        (
            item
            for item in _articulation_candidates(run_dir).candidates
            if item.candidate_id == candidate_id
        ),
        None,
    )
    if candidate is None:
        raise typer.BadParameter(f"articulation has no candidate {candidate_id!r}")
    typer.echo(json.dumps(candidate.model_dump(mode="json"), indent=2))


@articulation_app.command("inspect-joint")
def inspect_articulation_joint(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    joint_id: Annotated[str, typer.Argument(help="Candidate or measured joint ID.")],
) -> None:
    candidate_joints = [
        {"candidate_id": candidate.candidate_id, **joint.model_dump(mode="json")}
        for candidate in _articulation_candidates(run_dir).candidates
        for joint in candidate.joints
        if joint.joint_id == joint_id
    ]
    motion = json.loads(
        (run_dir / "reconstruction/articulation/measured_motion.json").read_text(encoding="utf-8")
    )
    measured = [
        joint for joint in motion.get("joint_hypotheses", []) if joint["joint_id"] == joint_id
    ]
    fitting = _artifact_model(
        run_dir,
        "reconstruction/articulation/fitting_manifest.json",
        ArticulationFittingManifest,
    )
    fitted = [
        {
            "candidate_id": item.candidate_id,
            **joint.model_dump(mode="json"),
        }
        for item in fitting.fittings
        if item.fitted_model is not None
        for joint in item.fitted_model.fitted_joints
        if joint.candidate_joint_id == joint_id or joint.measured_joint_id == joint_id
    ]
    evaluation = _artifact_model(
        run_dir,
        "reconstruction/articulation/evaluation_manifest.json",
        ArticulatedEvaluationManifest,
    )
    heldout = [
        {
            "candidate_id": item.candidate_id,
            "state_id": state.state_id,
            "inferred_q": state.inferred_joint_positions.get(joint_id),
            "joint_q_residual": state.joint_q_residual,
            "axis_error_degrees": state.axis_error_degrees,
            "pivot_residual_part_diagonals": state.pivot_residual_part_diagonals,
            "usable_views": state.usable_heldout_view_count,
            "view_provenance": [view.model_dump(mode="json") for view in state.view_evaluations],
        }
        for item in evaluation.evaluations
        for state in item.state_evaluations
        if joint_id in state.inferred_joint_positions
    ]
    if not candidate_joints and not measured and not fitted:
        raise typer.BadParameter(f"articulation has no joint {joint_id!r}")
    typer.echo(
        json.dumps(
            {
                "measured": measured,
                "candidates": candidate_joints,
                "fitted": fitted,
                "heldout": heldout,
            },
            indent=2,
        )
    )


@articulation_app.command("compare-candidates")
def compare_articulation_candidates(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    object_id: Annotated[str, typer.Argument(help="Articulated object ID.")],
) -> None:
    candidates = {
        item.candidate_id: item
        for item in _articulation_candidates(run_dir).candidates
        if item.articulated_object_id == object_id
    }
    evaluation = _artifact_model(
        run_dir,
        "reconstruction/articulation/evaluation_manifest.json",
        ArticulatedEvaluationManifest,
    )
    rows = [
        {
            "source_family": candidates[item.candidate_id].source_family,
            **item.model_dump(mode="json"),
        }
        for item in evaluation.evaluations
        if item.candidate_id in candidates
    ]
    if not rows:
        raise typer.BadParameter(f"articulation has no candidates for {object_id!r}")
    typer.echo(json.dumps(rows, indent=2))


@articulation_app.command("render-previews")
def render_articulation_previews(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    from recon2sim.adapters.articulation_selection import ArticulationSelectionAdapter

    ArticulationSelectionAdapter.render_previews(run_dir)
    typer.echo("regenerated deterministic Phase 5C articulation previews")


@articulation_app.command("export-kinematic-bundle")
def export_articulation_bundle(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    object_id: Annotated[str, typer.Argument(help="Articulated object ID.")],
    output: Annotated[Path, typer.Option("--output", help="Destination JSON path.")],
) -> None:
    source = run_dir / "reconstruction/articulation/selected" / object_id / "kinematic_bundle.json"
    if not source.is_file():
        raise typer.BadParameter(f"{object_id!r} has no selected kinematic bundle")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    typer.echo(f"exported kinematic bundle for {object_id} to {output}")


@articulation_app.command("export-preview-urdf")
def export_articulation_preview_urdf(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    object_id: Annotated[str, typer.Argument(help="Articulated object ID.")],
    output: Annotated[Path, typer.Option("--output", help="Destination URDF path.")],
) -> None:
    source = run_dir / "reconstruction/articulation/selected" / object_id / "preview_only.urdf"
    if not source.is_file():
        raise typer.BadParameter(f"{object_id!r} has no visual-only preview URDF")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    typer.echo(f"exported non-simulation-ready preview URDF to {output}")


@articulation_app.command("explain-selection")
def explain_articulation_selection(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    object_id: Annotated[str, typer.Argument(help="Articulated object ID.")],
) -> None:
    selected = next(
        (
            item
            for item in _articulation_selection(run_dir).objects
            if item.articulated_object_id == object_id
        ),
        None,
    )
    if selected is None:
        raise typer.BadParameter(f"articulation has no object {object_id!r}")
    evaluation = _artifact_model(
        run_dir,
        "reconstruction/articulation/evaluation_manifest.json",
        ArticulatedEvaluationManifest,
    )
    typer.echo(
        json.dumps(
            {
                "selection": selected.model_dump(mode="json"),
                "candidate_gates": [
                    item.model_dump(mode="json") for item in evaluation.evaluations
                ],
            },
            indent=2,
        )
    )


@validation_app.command("inspect-phase5c")
def inspect_phase5c(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    report = _artifact_model(
        run_dir,
        "validation/phase5c_articulated_reconstruction.json",
        Phase5CConsistencyReport,
    )
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))


@validation_app.command("verify-phase5c")
def verify_phase5c(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    report = _artifact_model(
        run_dir,
        "validation/phase5c_articulated_reconstruction.json",
        Phase5CConsistencyReport,
    )
    if not report.passed:
        failed = [check.check_id for check in report.checks if not check.passed]
        raise typer.BadParameter(f"Phase 5C consistency verification failed: {failed}")
    typer.echo(
        f"Phase 5C articulated-reconstruction consistency passed ({len(report.checks)} checks)"
    )


def _world_calibration(run_dir: Path) -> WorldCalibrationArtifact:
    return _artifact_model(
        run_dir,
        "calibration/world_calibration.json",
        WorldCalibrationArtifact,
    )


def _calibration_manifest(run_dir: Path) -> WorldCalibrationManifest:
    return _artifact_model(
        run_dir,
        "calibration/evidence_manifest.json",
        WorldCalibrationManifest,
    )


@calibration_app.command("inspect")
def inspect_calibration(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    typer.echo(json.dumps(_world_calibration(run_dir).model_dump(mode="json"), indent=2))


@calibration_app.command("inspect-evidence")
def inspect_calibration_evidence(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    typer.echo(json.dumps(_calibration_manifest(run_dir).model_dump(mode="json"), indent=2))


@calibration_app.command("inspect-tag")
def inspect_calibration_tag(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    tag_id: Annotated[int, typer.Argument(min=0)],
) -> None:
    record = _calibration_manifest(run_dir).apriltag
    if record is None or record.tag_id != tag_id:
        raise typer.BadParameter(f"calibration has no AprilTag {tag_id}")
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@calibration_app.command("inspect-landmark")
def inspect_calibration_landmark(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    landmark_id: Annotated[str, typer.Argument()],
) -> None:
    record = _calibration_manifest(run_dir).known_distance
    if record is None:
        raise typer.BadParameter("calibration has no known-distance landmarks")
    landmark = next(
        (item for item in record.landmarks if item.landmark_id == landmark_id),
        None,
    )
    if landmark is None:
        raise typer.BadParameter(f"calibration has no landmark {landmark_id!r}")
    typer.echo(json.dumps(landmark.model_dump(mode="json"), indent=2))


@calibration_app.command("render-previews")
def render_calibration_previews(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    names = (
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
    missing = [
        name for name in names if not (run_dir / "calibration/previews" / f"{name}.png").is_file()
    ]
    if missing:
        raise typer.BadParameter(f"calibration previews are missing: {missing}")
    typer.echo(f"validated {len(names)} deterministic calibration previews")


@calibration_app.command("export-transform")
def export_calibration_transform(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    artifact = _world_calibration(run_dir)
    if artifact.accepted_transform is None:
        raise typer.BadParameter(f"calibration status {artifact.status.value!r} has no transform")
    atomic_write_json(output, artifact.accepted_transform)
    typer.echo(f"exported accepted world transform to {output}")


@calibration_app.command("export-canonical-camera-trajectory")
def export_canonical_camera_trajectory(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    artifact = _world_calibration(run_dir)
    if not artifact.full_canonical_world_available:
        raise typer.BadParameter("full canonical calibration has not been accepted")
    scene = _artifact_model(
        run_dir,
        "scene_ir/phase6a_canonical_scene.json",
        SceneIR,
    )
    atomic_write_json(
        output,
        {
            "schema_version": "0.1.0",
            "coordinate_convention": scene.metadata.coordinate_convention.model_dump(mode="json"),
            "cameras": [item.model_dump(mode="json") for item in scene.cameras],
        },
    )
    typer.echo(f"exported canonical camera trajectory to {output}")


def _export_calibrated_ascii_ply(
    source: Path,
    output: Path,
    matrix: tuple[float, ...],
    rotation: tuple[float, ...],
) -> None:
    lines = source.read_text(encoding="ascii").splitlines()
    if not lines or lines[0] != "ply" or "format ascii 1.0" not in lines[:4]:
        raise typer.BadParameter("canonical mesh export currently supports ASCII PLY")
    header_end = lines.index("end_header")
    vertex_line = next(
        (line for line in lines[:header_end] if line.startswith("element vertex ")),
        None,
    )
    if vertex_line is None:
        raise typer.BadParameter("PLY has no vertex count")
    vertex_count = int(vertex_line.split()[2])
    properties = [
        line.split()[-1]
        for line in lines[:header_end]
        if line.startswith("property ") and not line.startswith("property list ")
    ]
    x_index, y_index, z_index = (
        properties.index("x"),
        properties.index("y"),
        properties.index("z"),
    )
    normal_indices = (
        (properties.index("nx"), properties.index("ny"), properties.index("nz"))
        if {"nx", "ny", "nz"} <= set(properties)
        else None
    )
    output_lines = lines[: header_end + 1]
    for line in lines[header_end + 1 : header_end + 1 + vertex_count]:
        values = line.split()
        point = transform_point(
            matrix,
            (float(values[x_index]), float(values[y_index]), float(values[z_index])),
        )
        for index, value in zip((x_index, y_index, z_index), point, strict=True):
            values[index] = f"{value:.12g}"
        if normal_indices is not None:
            normal = rotate_vector(
                rotation,
                tuple(float(values[index]) for index in normal_indices),
            )
            for index, value in zip(normal_indices, normal, strict=True):
                values[index] = f"{value:.12g}"
        output_lines.append(" ".join(values))
    output_lines.extend(lines[header_end + 1 + vertex_count :])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(output_lines) + "\n", encoding="ascii")


@calibration_app.command("export-canonical-mesh")
def export_canonical_mesh(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    asset_id: Annotated[str, typer.Option("--asset-id")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    artifact = _world_calibration(run_dir)
    transform = artifact.accepted_transform
    if not artifact.full_canonical_world_available or transform is None:
        raise typer.BadParameter("full canonical calibration has not been accepted")
    source_scene = _artifact_model(
        run_dir,
        _calibration_manifest(run_dir).source_scene_ir_path,
        SceneIR,
    )
    asset = next((item for item in source_scene.geometry_assets if item.asset_id == asset_id), None)
    if asset is None:
        raise typer.BadParameter(f"source scene has no geometry asset {asset_id!r}")
    source = run_dir / asset.uri
    if not source.is_file():
        raise typer.BadParameter(f"source geometry is not materialized: {asset.uri}")
    _export_calibrated_ascii_ply(
        source,
        output,
        transform.matrix_canonical_from_colmap,
        transform.rotation_canonical_from_colmap,
    )
    typer.echo(f"exported canonical metric geometry to {output}")


@calibration_app.command("create-apriltag-manifest")
def create_apriltag_manifest(
    output: Annotated[Path, typer.Option("--output")],
    tag_family: Annotated[str, typer.Option("--tag-family")] = "tagStandard41h12",
    tag_id: Annotated[int, typer.Option("--tag-id", min=0)] = 0,
    tag_size_m: Annotated[float, typer.Option("--tag-size-m", min=1e-9)] = 0.1,
) -> None:
    payload: dict[str, object] = {
        "schema_version": "0.2.0",
        "run_id": "replace_with_run_id",
        "frame_sequence_digest": "0" * 64,
        "camera_reconstruction_path": "calibration/source/camera_reconstruction.json",
        "camera_reconstruction_sha256": "0" * 64,
        "source_scene_ir_path": "calibration/source/scene_ir.json",
        "source_scene_ir_sha256": "0" * 64,
        "evidence": [
            {
                "evidence_id": "apriltag_fitting",
                "evidence_type": "apriltag",
                "trust": "metric_fiducial",
                "role": "fitting",
                "source_files": [
                    {
                        "relative_path": "calibration/source/tag_frame_fitting.png",
                        "sha256": "0" * 64,
                        "media_type": "image/png",
                    }
                ],
                "supports_metric_scale": True,
                "measurement_uncertainty": 0.001,
            }
        ],
        "apriltag": {
            "official_repository": "https://github.com/AprilRobotics/apriltag",
            "official_commit": "0e16a12dd380fd607e4afd54712ee9b1ffb9ec8f",
            "code_license": "BSD-2-Clause",
            "tag_family": tag_family,
            "tag_id": tag_id,
            "detection_edge_size_m": tag_size_m,
            "detector_source_path": "apriltag_pose.h::estimate_tag_pose",
            "image_sources": [
                {
                    "frame_id": "replace_with_registered_frame_id",
                    "image_path": "calibration/source/tag_frame_fitting.png",
                    "image_sha256": "0" * 64,
                    "width": 1920,
                    "height": 1080,
                    "intrinsics_fx_fy_cx_cy": [1000.0, 1000.0, 960.0, 540.0],
                    "image_coordinate_space": "registered_undistorted",
                    "split": "fitting",
                }
            ],
            "detections": [],
        },
        "known_distance": None,
        "external_metric": [],
        "gravity": [],
        "floor_planes": [],
        "forward": None,
        "origin": None,
        "evidence_tier": "scale_only",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    typer.echo(f"created AprilTag calibration manifest template at {output}")


@calibration_app.command("create-landmark-template")
def create_landmark_template(
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    payload = {
        "schema_version": "0.2.0",
        "landmarks": [
            {
                "landmark_id": "known_distance_0001",
                "point_a_id": "point_a",
                "point_b_id": "point_b",
                "known_distance_m": 1.0,
                "measurement_uncertainty_m": 0.001,
                "role": "fitting",
            }
        ],
        "observations": [
            {
                "frame_id": "frame_000000",
                "point_id": "point_a",
                "pixel_xy": [0.0, 0.0],
                "role": "fitting",
            },
            {
                "frame_id": "frame_000001",
                "point_id": "point_a",
                "pixel_xy": [0.0, 0.0],
                "role": "fitting",
            },
            {
                "frame_id": "frame_000002",
                "point_id": "point_a",
                "pixel_xy": [0.0, 0.0],
                "role": "heldout",
            },
            {
                "frame_id": "frame_000000",
                "point_id": "point_b",
                "pixel_xy": [0.0, 0.0],
                "role": "fitting",
            },
            {
                "frame_id": "frame_000001",
                "point_id": "point_b",
                "pixel_xy": [0.0, 0.0],
                "role": "fitting",
            },
            {
                "frame_id": "frame_000002",
                "point_id": "point_b",
                "pixel_xy": [0.0, 0.0],
                "role": "heldout",
            },
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    typer.echo(f"created known-distance landmark template at {output}")


@calibration_app.command("validate-landmark-manifest")
def validate_landmark_manifest(
    manifest: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False),
    ],
) -> None:
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    record = KnownDistanceLandmarkManifest.model_validate(payload)
    typer.echo(
        f"valid known-distance manifest: {len(record.landmarks)} distance anchors, "
        f"{len(record.observations)} observations"
    )


@validation_app.command("inspect-phase6a")
def inspect_phase6a(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    report = _artifact_model(
        run_dir,
        "validation/phase6a_world_calibration.json",
        Phase6AConsistencyReport,
    )
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))


@validation_app.command("verify-phase6a")
def verify_phase6a(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    report = _artifact_model(
        run_dir,
        "validation/phase6a_world_calibration.json",
        Phase6AConsistencyReport,
    )
    if not report.passed:
        failed = [item.check_id for item in report.checks if not item.passed]
        raise typer.BadParameter(f"Phase 6A consistency verification failed: {failed}")
    typer.echo(f"Phase 6A world-calibration consistency passed ({len(report.checks)} checks)")


def _assembly_plan(run_dir: Path) -> SceneAssemblyPlan:
    return _artifact_model(
        run_dir,
        "assembly/assembly_plan.json",
        SceneAssemblyPlan,
    )


def _assembly_bundle(run_dir: Path, *, deployment: bool) -> SceneAssemblyBundle:
    name = "deployment_eligible_visual_bundle.json" if deployment else "research_visual_bundle.json"
    return _artifact_model(run_dir, f"assembly/{name}", SceneAssemblyBundle)


@assembly_app.command("inspect")
def inspect_assembly(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    plan = _assembly_plan(run_dir)
    typer.echo(
        json.dumps(
            {
                "world": plan.world.model_dump(mode="json"),
                "decisions": [item.model_dump(mode="json") for item in plan.decisions],
                "layers": [item.model_dump(mode="json") for item in plan.layers],
                "global_scene_policy": plan.global_scene_policy,
            },
            indent=2,
        )
    )


@assembly_app.command("inspect-object")
def inspect_assembly_object(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    object_id: Annotated[str, typer.Argument(help="Assembly object ID.")],
) -> None:
    plan = _assembly_plan(run_dir)
    decision = next((item for item in plan.decisions if item.object_id == object_id), None)
    if decision is None:
        raise typer.BadParameter(f"assembly has no object {object_id!r}")
    asset_ids = set(decision.measured_anchor_asset_ids + decision.selected_visual_asset_ids)
    assets = [
        item.model_dump(mode="json") for item in plan.assets if item.asset.asset_id in asset_ids
    ]
    typer.echo(
        json.dumps(
            {"decision": decision.model_dump(mode="json"), "assets": assets},
            indent=2,
        )
    )


@assembly_app.command("explain-object")
def explain_assembly_object(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    object_id: Annotated[str, typer.Argument(help="Assembly object ID.")],
) -> None:
    decision = next(
        (item for item in _assembly_plan(run_dir).decisions if item.object_id == object_id),
        None,
    )
    if decision is None:
        raise typer.BadParameter(f"assembly has no object {object_id!r}")
    typer.echo(
        json.dumps(
            {
                "object_id": decision.object_id,
                "status": decision.status,
                "rationale": decision.rationale,
                "selected_candidate_id": decision.selected_candidate_id,
                "measured_anchor_asset_ids": decision.measured_anchor_asset_ids,
            },
            indent=2,
        )
    )


@assembly_app.command("inspect-lineage")
def inspect_assembly_lineage(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    report = _artifact_model(
        run_dir,
        "assembly/lineage_report.json",
        SceneAssemblyLineageReport,
    )
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))


@assembly_app.command("inspect-license")
def inspect_assembly_license(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    typer.echo(
        json.dumps(
            {
                "research": _assembly_bundle(
                    run_dir,
                    deployment=False,
                ).license_summary.model_dump(mode="json"),
                "deployment": _assembly_bundle(
                    run_dir,
                    deployment=True,
                ).license_summary.model_dump(mode="json"),
            },
            indent=2,
        )
    )


@assembly_app.command("render-previews")
def render_assembly_previews(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    names = (
        "global_context",
        "measured_anchors",
        "research_assembly",
        "deployment_assembly",
        "object_decision_grid",
        "overlap_heatmap",
        "articulated_snapshot",
    )
    missing = [
        name for name in names if not (run_dir / "assembly/previews" / f"{name}.png").is_file()
    ]
    if missing:
        raise typer.BadParameter(f"assembly previews are missing: {missing}")
    typer.echo(f"validated {len(names)} visual-only assembly previews")


def _export_assembly_file(run_dir: Path, relative: str, output: Path) -> None:
    source = run_dir / relative
    if not source.is_file():
        raise typer.BadParameter(f"assembly artifact is missing: {relative}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    typer.echo(f"exported {relative} to {output}")


@assembly_app.command("export-research-bundle")
def export_research_assembly_bundle(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    _export_assembly_file(run_dir, "assembly/research_visual_bundle.json", output)


@assembly_app.command("export-deployment-bundle")
def export_deployment_assembly_bundle(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    _export_assembly_file(
        run_dir,
        "assembly/deployment_eligible_visual_bundle.json",
        output,
    )


@assembly_app.command("export-compiler-manifest")
def export_assembly_compiler_manifest(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    _export_assembly_file(run_dir, "assembly/compiler_input_manifest.json", output)


@validation_app.command("inspect-phase6b")
def inspect_phase6b(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    report = _artifact_model(
        run_dir,
        "validation/phase6b_layered_scene_assembly.json",
        Phase6BConsistencyReport,
    )
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))


@validation_app.command("verify-phase6b")
def verify_phase6b(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    report = _artifact_model(
        run_dir,
        "validation/phase6b_layered_scene_assembly.json",
        Phase6BConsistencyReport,
    )
    if not report.passed:
        failed = [item.check_id for item in report.checks if not item.passed]
        raise typer.BadParameter(f"Phase 6B consistency verification failed: {failed}")
    typer.echo(f"Phase 6B layered-scene consistency passed ({len(report.checks)} checks)")
