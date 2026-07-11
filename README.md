# Recon2Sim

Recon2Sim Phase 0 is a lightweight, CPU-only Observation-to-Simulation scaffold. Heavyweight robotics and reconstruction projects are isolated behind adapters and communicate through filesystem artifacts and canonical Scene IR JSON.

## Quickstart

```bash
uv run recon2sim run --input examples/tabletop --config configs/mock.yaml --run-dir runs/tabletop_demo
uv run recon2sim validate-ir runs/tabletop_demo/scene_ir/scene.json
uv run recon2sim status runs/tabletop_demo
```

Expected outputs include `manifest.json`, `resolved_config.yaml`, copied `frames/`, `camera/reconstruction.json`, `observations/object_tracks.json`, mock OBJ assets under `reconstruction/objects`, `scene_ir/scene.json`, `compiled/scene_package`, `validation/report.json`, and `logs/run.jsonl`.

## Checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

## Architecture Constraints

The core package never imports COLMAP, MapAnything, SAM 3, GenRecon, SceneSmith, Blender, Drake, MuJoCo, or Isaac Sim. Integrations must use mock, subprocess, Docker, or service adapters and explicit artifacts.
