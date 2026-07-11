from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

import typer
from recon2sim.adapters import REGISTRY
from recon2sim.config import load_config
from recon2sim.ir import SceneIR
from recon2sim.pipeline import PipelineRunner

app = typer.Typer(help="Recon2Sim Phase 0 observation-to-simulation CLI.")
adapters_app = typer.Typer(help="Inspect configured adapter implementations.")
app.add_typer(adapters_app, name="adapters")


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Directory to initialize.")] = Path("."),
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Initialized Recon2Sim workspace at {path}")


@app.command()
def run(
    input: Annotated[Path, typer.Option("--input", help="Input observation directory.")],
    config: Annotated[Path, typer.Option("--config", help="Pipeline YAML config.")],
    run_dir: Annotated[Path, typer.Option("--run-dir", help="Run output directory.")],
    resume: Annotated[
        bool, typer.Option(help="Skip successful stages with matching signatures.")
    ] = False,
    from_stage: Annotated[str | None, typer.Option(help="First stage to run.")] = None,
    until_stage: Annotated[str | None, typer.Option(help="Last stage to run.")] = None,
) -> None:
    manifest = PipelineRunner(load_config(config), input, run_dir).run(
        resume=resume, from_stage=from_stage, until_stage=until_stage
    )
    typer.echo(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "stages": {k: v["status"] for k, v in manifest["stages"].items()},
            },
            indent=2,
        )
    )


@app.command()
def status(run_dir: Path) -> None:
    typer.echo((run_dir / "manifest.json").read_text())


@app.command("validate-ir")
def validate_ir(path: Path) -> None:
    scene = SceneIR.model_validate_json(path.read_text())
    typer.echo(f"valid Scene IR: {scene.metadata.scene_id} ({len(scene.objects)} objects)")


@app.command()
def inspect(run_dir: Path) -> None:
    scene_path = run_dir / "scene_ir" / "scene.json"
    scene = SceneIR.model_validate_json(scene_path.read_text())
    typer.echo(
        json.dumps(
            {
                "scene_id": scene.metadata.scene_id,
                "objects": [o.object_id for o in scene.objects],
                "relations": len(scene.relations),
            },
            indent=2,
        )
    )


@app.command()
def clean(
    run_dir: Path, force: bool = typer.Option(False, help="Required to delete the run directory.")
) -> None:
    if not force:
        raise typer.BadParameter("Pass --force to delete a run directory.")
    shutil.rmtree(run_dir, ignore_errors=True)
    typer.echo(f"deleted {run_dir}")


@adapters_app.command("list")
def list_adapters() -> None:
    for name in sorted(REGISTRY):
        typer.echo(name)


@adapters_app.command("healthcheck")
def healthcheck() -> None:
    for name, cls in sorted(REGISTRY.items()):
        result = cls().healthcheck()
        typer.echo(f"{name}: {'ok' if result.ok else 'fail'} - {result.message}")


def main() -> None:
    import sys
    from pathlib import Path as _Path

    args = sys.argv[1:]
    if not args:
        typer.echo("Recon2Sim CLI")
        return
    if args[0] == "adapters":
        group = app.groups["adapters"]
        fn = group.commands[args[1]]
        fn()
        return
    cmd = args[0]
    rest = args[1:]
    kwargs = {}
    pos = []
    i = 0
    while i < len(rest):
        if rest[i].startswith("--"):
            key = rest[i][2:].replace("-", "_")
            if i + 1 < len(rest) and not rest[i + 1].startswith("--"):
                val = rest[i + 1]
                kwargs[key] = (
                    _Path(val) if key in {"input", "config", "run_dir"} or "/" in val else val
                )
                i += 2
            else:
                kwargs[key] = True
                i += 1
        else:
            pos.append(_Path(rest[i]))
            i += 1
    app.commands[cmd](*pos, **kwargs)


if __name__ == "__main__":
    main()
