# Roadmap

## Completed foundation

- Phase 0: lightweight architecture, filesystem DAG, canonical Scene IR, and CPU mocks.
- Phase 0.1: real dependencies and packaging, strict schemas and references, artifact-driven mock
  stages, valid PNG/OBJ outputs, DAG validation, byte-sensitive caching, retries, command isolation,
  output validation, expanded artifact records, real Typer CLI, CI, tests, and documentation.

## Recommended Phase 1: COLMAP camera recovery

Keep the task narrow:

1. Add an out-of-process COLMAP command/container adapter; do not import COLMAP into core Python.
2. Consume the existing `inputs/manifest.json` and copied PNG frames.
3. Convert COLMAP intrinsics and poses into the existing `CameraReconstruction` contract and the
   documented right-handed, meters, `xyzw`, world-from-camera convention.
4. Record COLMAP version, command/config, input hashes, confidence method, logs, timing, and
   provenance.
5. Add small fixture-based tests for conversion, missing images, command failure, invalid output,
   and cache invalidation. Keep the default CI path CPU-only and mock-only.

This task should stop at `camera/reconstruction.json`; it should not add segmentation,
reconstruction models, scene compilation, Blender, or simulators.

## Later phases

- Phase 2: SAM 3 segmentation and tracking adapter.
- Phase 3: GenRecon global reconstruction adapter.
- Phase 4: rigid and articulated object reconstruction adapters.
- Phase 5: SceneSmith or equivalent scene compiler adapter.
- Phase 6: physics repair and explicit simulator exports.

Each phase must preserve the canonical Scene IR boundary and add real-adapter tests without making
the CPU-only mock quality gate depend on GPUs, checkpoints, Docker, or network model downloads.
