# Roadmap

## Completed

- Phase 0: lightweight filesystem architecture, canonical Scene IR, and CPU mocks.
- Phase 0.1: real packaging/dependencies, strict schemas/references, artifact-driven mocks, DAG
  validation, byte-sensitive cache/resume, retries, command isolation, output validation, Typer
  CLI, GitHub Actions, and documentation.
- Phase 1: real FFmpeg/image ingest, deterministic frame QA, stale-output-safe attempt promotion,
  local/Docker COLMAP execution, binary sparse-model parsing, supported camera/distortion mapping,
  pose inversion, explicit scale ambiguity/world alignment state, model ranking/diagnostics,
  camera inspection/export CLI, fake executable integration tests, and optional native-tool paths.

Phase 1 stops after typed sparse camera recovery. Downstream adapters remain mocks by design.

## Recommended Phase 2: SAM 3 segmentation and tracking

Keep Phase 2 bounded to the existing `ObjectTracksArtifact` boundary:

1. Add an out-of-process SAM 3 segmentation/tracking adapter; do not import model runtimes into
   the core package.
2. Consume selected `inputs/manifest.json`, normalized PNGs, and registered camera information
   where useful; define behavior for unregistered frames explicitly.
3. Emit one stable track ID per object, per-frame boxes, valid mask PNGs, confidence, and full
   provenance through the existing typed artifact.
4. Isolate checkpoints/cache behind adapter configuration, pin model/license metadata, and never
   download weights implicitly during the mandatory test path.
5. Reuse attempt isolation, environment allowlists, timeout/interruption handling, required output
   validation, hashing, and cache invalidation.
6. Add fake-process and tiny fixture tests for missing masks, bad dimensions, track-ID instability,
   invalid JSON, retries, timeout, stale outputs, and failure preserving the previous result.
7. Keep mandatory CI CPU-only; real model/GPU tests remain separately marked integration tests.

Do not combine Phase 2 with global reconstruction, object mesh generation, SceneSmith, Blender,
or simulator export.

## Later phases

- Phase 3: real global scene reconstruction adapter (evaluate GenRecon or another bounded backend).
- Phase 4: rigid and articulated object reconstruction adapters.
- Phase 5: SceneSmith or equivalent scene compiler adapter.
- Phase 6: physics repair and explicit simulator exports.

Every phase must preserve canonical Scene IR, typed normalized files between tools, honest
scale/coordinate semantics, and a deterministic no-GPU quality gate.
