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

## Phase 2 implementation

- typed text, box, point, and mask prompt manifests;
- selective attempt inputs that exclude raw COLMAP workspaces;
- isolated official SAM 3.1 Object Multiplex local/Docker worker;
- deterministic fake worker for the CPU-only gate;
- registered/unregistered-aware anchor selection;
- canonical binary masks, stable object IDs, track QA, diagnostics, and no-object results;
- deterministic contact sheet, timeline, frame previews, and COCO export;
- exact official code/checkpoint provenance and credential-safe healthchecks.

A real official SAM 3.1 checkpoint smoke has run on H100 and records the pinned code/checkpoint
identity, valid canonical masks, previews, COCO output, and cache hit.

## Phase 3 implementation

- deterministic observation-lineage digest across COLMAP, SAM, and GenRecon;
- minimal selected-model COLMAP text camera package;
- isolated official GenRecon local/Docker worker and CPU fake protocol;
- three official checkpoint hashes and exact Git/submodule identity;
- reversible internal chunking transform with outputs returned to raw COLMAP coordinates;
- typed global PBR scene, mesh diagnostics, Scene IR visual assets, and previews;
- cross-stage consistency and selective-materialization validation;
- real module-level and full COLMAP -> SAM -> GenRecon acceptance gates.

## Recommended Phase 2.5

Design optional VLM-assisted scene inventory as a separate prompt-generation stage. It must
produce a reviewable prompt manifest rather than bypass Phase 2 prompt contracts, and must not
claim exhaustive vocabulary or physical classification without validation.

## Later phases

- Phase 4: SAM-to-global-scene object association and rigid/articulated object reconstruction.
- Phase 5: SceneSmith or equivalent scene compiler adapter.
- Phase 6: physics repair and explicit simulator exports.

Each phase must preserve the canonical Scene IR boundary and add real-adapter tests without making
the CPU-only mock quality gate depend on GPUs, checkpoints, Docker, or network model downloads.
