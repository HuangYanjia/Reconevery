# Roadmap

## Completed foundation

- Phase 0: lightweight architecture, filesystem DAG, canonical Scene IR, and CPU mocks.
- Phase 0.1: real dependencies and packaging, strict schemas and references, artifact-driven mock
  stages, valid PNG/OBJ outputs, DAG validation, byte-sensitive caching, retries, command isolation,
  output validation, expanded artifact records, real Typer CLI, CI, tests, and documentation.

## Completed Phase 1

- real video and image-directory ingest with normalized PNGs;
- FFmpeg/FFprobe subprocess execution and health checks;
- deterministic frame QA and typed reports;
- local/Docker COLMAP feature extraction, matching, and sparse mapping;
- strict binary parsing, pose inversion, camera mapping, and scale ambiguity;
- deterministic multi-model selection and typed diagnostics;
- attempt workspaces that reject stale output and protect previous results;
- fake-executable tests that keep mandatory CI independent of system tools.

## Phase 1.2 alignment and scale recovery

Add a typed alignment stage that consumes raw `colmap_arbitrary` camera reconstruction, estimates
gravity/canonical orientation and an external metric scale, transforms poses into the right-handed
+X-forward, +Y-left, +Z-up robot world, and emits new provenance. It must not relabel raw COLMAP
translations as meters without an observable scale reference.

## Recommended Phase 2

Add a SAM 3 segmentation and tracking adapter that consumes selected frames and registered camera
poses, emits typed masks/tracks, and preserves model provenance. Keep it independent of alignment
when masks only require image coordinates, and require aligned poses for any metric 3D behavior.
Do not combine it with global reconstruction, object reconstruction, SceneSmith, or simulators.

## Later phases

- Phase 3: GenRecon global reconstruction adapter.
- Phase 4: rigid and articulated object reconstruction adapters.
- Phase 5: SceneSmith or equivalent scene compiler adapter.
- Phase 6: physics repair and explicit simulator exports.

Each phase must preserve the canonical Scene IR boundary and add real-adapter tests without making
the CPU-only mock quality gate depend on GPUs, checkpoints, Docker, or network model downloads.
