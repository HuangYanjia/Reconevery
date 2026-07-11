# Recon2Sim Agent Guide

## Architecture Map
- `src/recon2sim/ir`: canonical Pydantic Scene IR and validation models.
- `src/recon2sim/adapters`: mock and process/container adapter boundaries.
- `src/recon2sim/pipeline`: deterministic filesystem DAG runner and manifests.
- `src/recon2sim/storage`: atomic artifact writes.
- `configs`, `docs`, `schemas`, `examples`, `tests`: runnable Phase 0 assets.

## Commands
- `uv run pytest`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy src`
- `uv run recon2sim run --input examples/tabletop --config configs/mock.yaml --run-dir runs/tabletop_demo`

## Standards
Use explicit types, deterministic mocks, atomic writes, small functions, and JSON/YAML filesystem artifacts. Never import heavyweight backends (COLMAP, MapAnything, SAM 3, GenRecon, SceneSmith, Blender, Drake, MuJoCo, Isaac Sim) into core code.

## Adapter Rules
Adapters isolate external environments through mock, subprocess, Docker, or service calls. New adapters must document inputs, outputs, health checks, timeouts, GPU metadata, provenance, and tests.

## Testing and Docs
Behavior changes require updated tests and documentation. Validate adapter outputs before admitting them into Scene IR.
