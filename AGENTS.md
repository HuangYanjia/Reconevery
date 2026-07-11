# Recon2Sim Agent Guide

## Current scope

Phase 0.1 is a deterministic CPU-only foundation. Do not import, install, download, or invoke
COLMAP, SAM 3, GenRecon, SceneSmith, Blender, Drake, MuJoCo, Isaac Sim, model checkpoints, or GPU
runtimes in core code. Future integrations belong behind filesystem adapters.

## Architecture map

- `src/recon2sim/ir`: canonical strict Pydantic Scene IR.
- `src/recon2sim/artifacts.py`: typed intermediate stage contracts.
- `src/recon2sim/adapters`: deterministic mocks and isolated command adapters.
- `src/recon2sim/pipeline`: DAG validation, signatures, cache/resume, retries, and manifests.
- `src/recon2sim/images.py`: dependency-free test PNG generation and validation.
- `src/recon2sim/storage`: atomic JSON, YAML, and text writes.
- `configs`, `schemas`, `examples`, `tests`, `docs`: reproducible Phase 0.1 assets.

## Required commands

```bash
uv sync --all-groups
uv run recon2sim --help
uv run recon2sim run --input examples/tabletop --config configs/mock.yaml --run-dir runs/tabletop_demo
uv run recon2sim validate-ir runs/tabletop_demo/scene_ir/scene.json
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

## Invariants

- Use real Pydantic v2, Typer, PyYAML, pytest, Ruff, and mypy; never add in-tree substitutes.
- Use explicit imports and explicit `__all__`; wildcard imports are prohibited.
- Every stage declares required outputs and validates them before success.
- Store run artifact paths relative to the run directory and record hashes and producer metadata.
- Cache signatures include config, adapter name/version, seed, input bytes, upstream artifacts,
  and upstream execution signatures. Cache hits keep `status=succeeded`.
- Scene IR is canonical. Exported mesh files and simulator/compiler outputs are derived artifacts.
- The articulated cabinet is one object with `cabinet_body` and `cabinet_drawer` links. Do not add
  a second top-level drawer without designing and validating an explicit cross-reference model.
- World coordinates are right-handed, +X forward, +Y left, +Z up, meters, quaternion `xyzw`, and
  poses transform camera coordinates into world coordinates.

## Change discipline

Behavior changes require tests and documentation. Scene IR changes also require regenerating
`schemas/scene_ir.schema.json`. Adapter changes must document inputs, outputs, schema identifiers,
allowed environment variables, timeout, retry behavior, healthcheck, provenance, and tests.

The recommended Phase 1 task is a narrowly scoped COLMAP command adapter that consumes the
existing ingest manifest/PNG contract and emits the existing typed camera reconstruction JSON.
It must not change the canonical Scene IR or pull COLMAP into the core environment.
