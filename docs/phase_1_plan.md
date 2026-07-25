# Phase 1 Implementation Plan

Phase 1 adds real observation ingest and sparse camera recovery while preserving the typed
filesystem contracts and existing DAG runner.

1. Run every stage attempt in `work/<stage>/attempt_<N>`, validate outputs there, and atomically
   promote successful files. Retain failed workspaces and keep previous canonical outputs intact.
2. Extend the ingest and camera artifact models backward-compatibly with source metadata, frame
   QA, registered/unregistered frames, diagnostics, and explicit scale status.
3. Add deterministic image-directory normalization and FFmpeg/FFprobe video extraction behind a
   dedicated adapter. Compute CPU-only blur, brightness, intensity variance, and duplicate QA.
4. Parse COLMAP `cameras.bin`, `images.bin`, and `points3D.bin`; map supported camera models and
   invert COLMAP world-to-camera poses into Recon2Sim world-from-camera transforms.
5. Add a dedicated COLMAP adapter with local and Docker execution, sequential/exhaustive
   matching, CPU/GPU flags, deterministic model ranking, diagnostics, preserved commands/logs,
   and typed outputs.
6. Add fake-executable integration coverage, inspection/export CLI commands, health checks,
   example configurations, a COLMAP Dockerfile, and Phase 1 documentation.
7. Run Ruff, formatting, mypy, pytest, schema validation, the mock pipeline twice with resume,
   stale-output and failed-promotion checks, and real-tool smoke tests only when tools are present.

Phase 1.1 corrects the raw COLMAP gauge metadata, applies EXIF orientation, makes interruption
non-retriable, removes audit counts from cache identity, adds transactional promotion rollback,
checks configured local/Docker tools, and makes production COLMAP configs stop after camera
recovery.
