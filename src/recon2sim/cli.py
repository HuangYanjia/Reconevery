from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ValidationError

from recon2sim.adapters import REGISTRY, Adapter, StageContext
from recon2sim.artifacts import (
    CameraDiagnostics,
    CameraReconstruction,
    ColmapWorkspaceManifest,
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


def _read_run_model[ModelT: BaseModel](
    run_dir: Path, relative_path: str, model: type[ModelT]
) -> ModelT:
    path = run_dir / relative_path
    if not path.is_file():
        raise typer.BadParameter(f"required run artifact does not exist: {path}")
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise typer.BadParameter(f"invalid run artifact {path}: {exc}") from exc


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
            help="Input video file or observation/image directory.",
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
            help="Optionally healthcheck adapters with a pipeline configuration.",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    input_path: Annotated[
        Path | None,
        typer.Option(
            "--input",
            help="Optional input path for mode-aware ingest healthchecks.",
            exists=True,
            readable=True,
        ),
    ] = None,
) -> None:
    checks: list[tuple[str, type[Adapter], StageContext | None]]
    if config_path is None:
        checks = [(name, adapter_class, None) for name, adapter_class in sorted(REGISTRY.items())]
    else:
        config = load_config(config_path)
        checks = []
        for stage_name, stage in config.stages.items():
            adapter_class = REGISTRY.get(stage.adapter.name)
            if adapter_class is None:
                typer.echo(f"{stage_name}/{stage.adapter.name}: unavailable - unknown adapter")
                continue
            stage_context = StageContext(
                stage_name=stage_name,
                input_dir=input_path or Path("."),
                run_dir=Path("."),
                config=stage,
                seed=config.seed,
            )
            pass_context: StageContext | None = (
                stage_context if stage.adapter.name == "colmap_camera_recovery" else None
            )
            if stage.adapter.name == "ffmpeg_ingest" and input_path is not None:
                pass_context = stage_context
            checks.append((f"{stage_name}/{stage.adapter.name}", adapter_class, pass_context))
    for name, adapter_class, check_context in checks:
        try:
            result = adapter_class().healthcheck(check_context)
        except Exception as exc:
            typer.echo(f"{name}: unavailable - healthcheck failed: {exc}")
            continue
        state = "available" if result.ok else "unavailable"
        typer.echo(f"{name}: {state} - {result.message}")


@ingest_app.command("inspect")
def inspect_ingest(
    run_dir: Annotated[
        Path,
        typer.Argument(help="Existing run directory.", exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    manifest = _read_run_model(run_dir, "inputs/manifest.json", IngestManifest)
    qa = _read_run_model(run_dir, "inputs/frame_qa.json", FrameQualityReport)
    typer.echo(
        json.dumps(
            {
                "source_type": manifest.source_type,
                "source": manifest.source_input_reference,
                "source_sha256": manifest.source_sha256,
                "decoded_frames": manifest.total_decoded_frames,
                "selected_frames": qa.selected_count,
                "dropped_frames": qa.dropped_count,
                "ffmpeg_version": manifest.ffmpeg_version,
            },
            indent=2,
        )
    )


@camera_app.command("inspect")
def inspect_camera(
    run_dir: Annotated[
        Path,
        typer.Argument(help="Existing run directory.", exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    reconstruction = _read_run_model(run_dir, "camera/reconstruction.json", CameraReconstruction)
    diagnostics = _read_run_model(run_dir, "camera/diagnostics.json", CameraDiagnostics)
    typer.echo(
        json.dumps(
            {
                "input_frames": diagnostics.input_frame_count,
                "selected_frames": diagnostics.selected_frame_count,
                "registered_frames": diagnostics.registered_frames,
                "registration_ratio": diagnostics.registration_ratio,
                "camera_model": reconstruction.model,
                "intrinsics": reconstruction.intrinsics.model_dump(mode="json"),
                "sparse_points": diagnostics.sparse_points,
                "scale_status": reconstruction.scale_status,
                "world_frame_status": reconstruction.world_frame_status,
                "warnings": diagnostics.warnings,
            },
            indent=2,
        )
    )


@camera_app.command("export-trajectory")
def export_camera_trajectory(
    run_dir: Annotated[
        Path,
        typer.Argument(help="Existing run directory.", exists=True, file_okay=False, dir_okay=True),
    ],
    output: Annotated[Path, typer.Option("--output", help="Trajectory JSON destination.")],
) -> None:
    reconstruction = _read_run_model(run_dir, "camera/reconstruction.json", CameraReconstruction)
    atomic_write_json(
        output,
        {
            "camera_id": reconstruction.camera_id,
            "coordinate_convention": reconstruction.coordinate_convention.model_dump(mode="json"),
            "scale_status": reconstruction.scale_status,
            "world_frame_status": reconstruction.world_frame_status,
            "poses": [
                {
                    "frame_id": pose.frame_id,
                    "transform_world_from_camera": pose.transform_world_from_camera.model_dump(
                        mode="json"
                    ),
                }
                for pose in reconstruction.poses
            ],
        },
    )
    typer.echo(f"wrote {len(reconstruction.poses)} poses to {output}")


@camera_app.command("colmap-stats")
def camera_colmap_stats(
    run_dir: Annotated[
        Path,
        typer.Argument(help="Existing run directory.", exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    diagnostics = _read_run_model(run_dir, "camera/diagnostics.json", CameraDiagnostics)
    workspace = _read_run_model(
        run_dir,
        "camera/colmap/workspace_manifest.json",
        ColmapWorkspaceManifest,
    )
    typer.echo(
        json.dumps(
            {
                "execution_mode": workspace.execution_mode,
                "colmap_version": workspace.colmap_version,
                "selected_model": workspace.selected_model,
                "models": [model.model_dump(mode="json") for model in diagnostics.models],
                "commands": [command.name for command in workspace.commands],
            },
            indent=2,
        )
    )
