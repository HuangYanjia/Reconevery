# Phase 3 Implementation Plan

Phase 3 adds official GenRecon as an isolated global visual-reconstruction backend while
keeping the lightweight Reconevery core free of model and CUDA dependencies.

## Decisions

1. Pin the official GenRecon repository at
   `eaf1468118d20469d17079a4a19737297d2ef87b` and verify its recursive submodule
   checkout before health or inference succeeds.
2. Keep SAM and GenRecon as parallel consumers of one observation lineage. GenRecon
   depends on real ingest, real camera recovery, and a minimal camera-package stage; it
   does not consume SAM masks or prompts.
3. Define a frame-sequence digest over ordered
   `(frame_id, normalized_path, normalized_frame_sha256)` tuples. Persist this digest in
   ingest, camera, SAM, GenRecon, and end-to-end validation artifacts.
4. Export only the selected COLMAP model as deterministic text plus registered normalized
   frames. Never materialize the COLMAP database, command logs, or rejected models into a
   GenRecon attempt.
5. Preserve the raw COLMAP coordinate contract:
   `colmap_arbitrary`, `unoriented`, OpenCV camera axes, arbitrary linear units,
   scale-ambiguous, and `world_from_camera`.
6. GenRecon may use a deterministic reversible PCA working transform for chunking. This
   is internal preprocessing, not gravity alignment. Canonical GLB and PLY outputs are
   transformed back to the original COLMAP frame.
7. Validate official code identity, all three checkpoint hashes, request and package
   hashes, selected view order, required raw intermediates, GLB structure, finite mesh
   coordinates, and non-degenerate bounds before promotion.
8. Use the existing attempt workspaces, process-group termination, retries, transactional
   promotion, content signatures, and selective `InputSpec` materialization.
9. Exercise the complete filesystem protocol in mandatory CPU tests with a deterministic
   fake worker. Keep real GenRecon and Docker validation manual and GPU-only.
10. Treat global geometry as a generated visual asset only. Phase 3 does not create
    object-level 3D fusion, collision geometry, physical properties, metric scale, or a
    simulation-ready scene.

## Delivery Order

1. Add the ordered lineage digest and minimal COLMAP text camera-package exporter.
2. Add typed request, checkpoint, worker, mesh, diagnostics, preview, and consistency
   artifacts.
3. Add the fake worker and lightweight GenRecon adapter with strict raw-output validation.
4. Add deterministic mesh inspection, COLMAP/global previews, and Scene IR visual-asset
   integration.
5. Add the end-to-end consistency validator and reconstruction/validation CLI commands.
6. Add the isolated official worker, commit/checkpoint verification, official inference
   and GLB conversion wrappers, Docker image, and manual workflow.
7. Add production, Docker, fake, and full end-to-end configurations and documentation.
8. Run Ruff, format, strict mypy, pytest, schema generation, the fake Phase 3 DAG twice,
   and verify the second run is entirely cache-hit.
9. Build the official H100 environment, hash the official checkpoints, run a real
   GenRecon module smoke, then run one real ingest -> COLMAP -> SAM 3.1 -> GenRecon
   consistency smoke and its resume pass.
10. Publish a draft pull request. It may leave draft state only after both real smoke
    acceptance levels pass.
