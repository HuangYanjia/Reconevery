# Recon2Sim Agent Guide

## Current scope

Phase 1 contains two real boundaries: `ffmpeg_ingest` and `colmap_camera_recovery`. FFmpeg and
COLMAP are subprocesses, never Python imports. The downstream segmentation/tracking,
reconstruction, compiler, validator, and exporter remain mocks.

Do not add SAM 3, GenRecon, SceneSmith, Blender, NeRF, MVS/dense meshing, simulator SDKs,
checkpoint downloads, or a pipeline framework. Do not move COLMAP or FFmpeg into the core Python
dependency environment.

## Architecture map

- `src/recon2sim/ir`: canonical strict Pydantic Scene IR.
- `src/recon2sim/artifacts.py`: typed ingest, QA, camera, reconstruction, and export contracts.
- `src/recon2sim/adapters/ingest.py`: video/image detection, FFmpeg extraction, Pillow
  normalization, and deterministic QA.
- `src/recon2sim/colmap`: dependency-free binary parser and pose conversion.
- `src/recon2sim/adapters/colmap.py`: local/Docker COLMAP orchestration and normalization.
- `src/recon2sim/adapters/process.py`: allowlisted subprocess environments, timeouts, process
  group termination, and log preservation.
- `src/recon2sim/pipeline`: DAG validation, signatures, retries, attempt isolation, promotion,
  cache/resume, and manifests.
- `configs`, `docker`, `schemas`, `examples`, `tests`, `docs`: reproducible project assets.

## Required commands

```bash
uv sync --all-groups --locked
uv run recon2sim --help
uv run recon2sim run --input examples/tabletop --config configs/mock.yaml --run-dir runs/tabletop_demo
uv run recon2sim validate-ir runs/tabletop_demo/scene_ir/scene.json
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

Installed-tool checks are explicit and honest:

```bash
uv run recon2sim adapters healthcheck
uv run recon2sim adapters healthcheck --config configs/colmap_cpu.yaml --input <input>
```

## Invariants

- Use real Pydantic v2, Typer, PyYAML, Pillow, pytest, Ruff, and mypy. Never add in-tree
  replacements for third-party packages.
- Use explicit imports and `__all__`; wildcard imports are prohibited.
- Core code must not import COLMAP, PyCOLMAP, OpenCV, CUDA, Docker SDKs, or model libraries.
- External commands are argument lists with `shell=False` behavior and an explicit environment
  allowlist. Preserve stdout, stderr, return code, duration, timeout/interruption state, and the
  failed workspace.
- Every attempt writes only under `work/<stage>/attempt_<N>`. Validate there and promote only the
  full declared output set. Never credit or overwrite a stale canonical output.
- Store canonical run artifact paths relative to the run directory. Source references are
  explicitly relative to the configured input root; recorded exact external commands may contain
  host/container paths needed for reproduction.
- Cache signatures include config, adapter name/version, seed, source bytes, upstream artifact
  hashes, and upstream execution signatures. Cache hits keep `status=succeeded`.
- Scene IR remains canonical. Raw COLMAP files, meshes, compiler packages, trajectories, and
  simulator exports are referenced or derived artifacts.
- The cabinet is one articulated object with body/drawer links; do not duplicate its drawer as a
  top-level object without an explicit cross-reference design.
- Mock coordinates are right-handed +X forward, +Y left, +Z up, meters, `xyzw`,
  world-from-camera. Monocular COLMAP output must remain `scale_ambiguous`, `arbitrary_scale`, and
  `colmap_unaligned` until a future explicit alignment stage.
- Reject Phase 1 multi-camera models and unsupported camera models clearly; preserve raw output.

## Change discipline

Behavior changes require tests and documentation. Scene IR changes require
`uv run python scripts/generate_schema.py`. Adapter changes must document inputs, outputs,
schema IDs, command arguments, environment allowlist, timeout/retry behavior, healthcheck,
provenance, coordinate/scale semantics, and failure artifacts.

Mandatory CI must remain CPU-only and use fake executable tests. Real FFmpeg/COLMAP tests are
optional integration tests. The recommended next phase is SAM 3 segmentation/tracking behind the
existing typed `ObjectTracksArtifact`; it must not destabilize ingest, camera recovery, Scene IR,
or the mock quality gate.
