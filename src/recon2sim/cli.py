from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ValidationError

from recon2sim.adapters import REGISTRY
from recon2sim.artifacts import (
    CameraDiagnostics,
    CameraReconstruction,
    FrameQualityReport,
    IngestManifest,
)
from recon2sim.config import load_config
from recon2sim.ir import SceneIR
from recon2sim.pipeline import PipelineConfigurationError, PipelineRunner
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
app.add_typer(adapters_app, name="adapters")
app.add_typer(ingest_app, name="ingest")
app.add_typer(camera_app, name="camera")


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
def healthcheck() -> None:
    for name, adapter_class in sorted(REGISTRY.items()):
        result = adapter_class().healthcheck()
        typer.echo(f"{name}: {'ok' if result.ok else 'fail'} - {result.message}")


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
