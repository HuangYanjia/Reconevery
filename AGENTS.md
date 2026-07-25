# Recon2Sim Agent Guide

## Current scope

Phase 1 provides real FFmpeg ingest and out-of-process COLMAP camera recovery. Do not import
COLMAP into Python or add SAM 3, GenRecon, SceneSmith, Blender, MVS/NeRF, Drake, MuJoCo, Isaac Sim,
model checkpoints, or GPU runtimes to core dependencies. External integrations belong behind
filesystem adapters.

## Architecture map

- `src/recon2sim/ir`: canonical strict Pydantic Scene IR.
- `src/recon2sim/artifacts.py`: typed intermediate stage contracts.
- `src/recon2sim/adapters`: deterministic mocks plus isolated FFmpeg and COLMAP adapters.
- `src/recon2sim/colmap`: strict binary model parsing and coordinate conversion.
- `src/recon2sim/frame_qa.py`: deterministic CPU frame-quality metrics.
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
uv run recon2sim adapters healthcheck
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

## Invariants

- Use real Pydantic v2, Typer, PyYAML, pytest, Ruff, and mypy; never add in-tree substitutes.
- Use explicit imports and explicit `__all__`; wildcard imports are prohibited.
- Every stage declares required outputs and validates them before success.
- Every attempt writes to `work/<stage>/attempt_<N>`; only validated outputs are promoted.
- Failed attempts retain their workspace and never replace previous canonical outputs.
- Store run artifact paths relative to the run directory and record hashes and producer metadata.
- Cache signatures include config, adapter name/version, seed, input bytes, upstream artifacts,
  and upstream execution signatures. Cache hits keep `status=succeeded`.
- Scene IR is canonical. Exported mesh files and simulator/compiler outputs are derived artifacts.
- The articulated cabinet is one object with `cabinet_body` and `cabinet_drawer` links. Do not add
  a second top-level drawer without designing and validating an explicit cross-reference model.
- World coordinates are right-handed, +X forward, +Y left, +Z up, quaternion `xyzw`, and poses
  transform camera coordinates into world coordinates. Monocular COLMAP output must remain
  explicitly `scale_ambiguous`.

## Change discipline

Behavior changes require tests and documentation. Scene IR changes also require regenerating
`schemas/scene_ir.schema.json`. Adapter changes must document inputs, outputs, schema identifiers,
allowed environment variables, timeout, retry behavior, healthcheck, provenance, and tests.

The recommended next task is a narrowly scoped Phase 2 SAM 3 segmentation/tracking adapter that
consumes the stable ingest and camera contracts. Do not add global/object reconstruction,
SceneSmith, physics simulation, or simulator export in the same change.
