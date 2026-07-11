# Phase 1 Real Ingest and COLMAP Plan

Phase 1 adds the first real reconstruction boundary while preserving the Phase 0.1 runner,
canonical Scene IR, downstream mock contracts, and mandatory CPU-only CI.

1. Make every stage attempt write into a unique `work/<stage>/attempt_<N>` directory. Validate
   outputs there and promote files to canonical run paths only after the complete contract passes.
   Failed attempts must leave the previous successful canonical artifacts untouched.
2. Extend typed artifacts backward-compatibly for video/image ingest metadata, per-frame QA,
   COLMAP diagnostics and workspace provenance, registered/unregistered frames, scale ambiguity,
   and world-frame alignment status.
3. Implement an `ffmpeg_ingest` adapter that auto-detects video or image-directory inputs,
   extracts or normalizes deterministic RGB PNGs, computes lightweight QA, and records exact
   commands, tool versions, source hashes, selections, rejections, and logs.
4. Implement an internal COLMAP binary reader, supported camera-model mapping, rigid pose
   inversion, quaternion conversion, deterministic sparse-model ranking, confidence scoring, and
   explicit monocular scale ambiguity.
5. Implement `colmap_camera_recovery` for local or Docker execution using argument lists and an
   environment allowlist. Preserve the database, sparse models, per-command logs, diagnostics,
   and workspace manifest while rejecting missing, malformed, stale, or low-registration output.
6. Exercise real adapter code in mandatory CI with fake FFmpeg/COLMAP executables. Keep optional
   tests for installed tools separately marked and never require a GPU, Docker, or model download.
7. Add ingest/camera inspection CLI commands, local/CPU/Docker configs, a documented COLMAP image,
   capture and troubleshooting guidance, then run the mock pipeline, resume/invalidation tests,
   full quality gate, and any genuinely available real-tool smoke test.

Out of scope: segmentation/tracking models, dense or global reconstruction, object reconstruction,
SceneSmith, Blender, NeRF, simulator export, physics integration, and GPU model checkpoints.
