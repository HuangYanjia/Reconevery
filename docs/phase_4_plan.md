# Phase 4 Implementation Plan

Phase 4 associates canonical SAM tracks with visible faces of the canonical GenRecon global
mesh. It produces partial observation-supported surface hypotheses, not completed objects or
simulation assets.

## Boundaries

1. Keep `recon2sim` free of NumPy, trimesh, OpenCV, torch, CUDA, and rasterizer imports.
2. Materialize only typed ingest/camera/segmentation/global metadata, referenced canonical masks,
   and a read-only reference to the canonical global mesh.
3. Run projection, undistortion, nvdiffrast rasterization, evidence accumulation, connected
   components, reprojection, and mesh export in `workers/object_lifting`.
4. Preserve original GenRecon face IDs and raw COLMAP arbitrary/unoriented/scale-ambiguous
   coordinate semantics.
5. Treat unresolved tracks as valid outputs. Never generate hidden surfaces, collisions, physics,
   metric scale, or gravity alignment.

## Implementation Order

1. Add typed request, worker, face-index, object hypothesis, evidence, diagnostics, preview, and
   Phase 4 consistency models.
2. Add a standard-library compact little-endian face-ID reader/writer/validator and lightweight
   core preview/export helpers.
3. Add the core adapter with selective `InputSpec` declarations, reference-only mesh access,
   worker invocation, independent output validation, Scene IR partial fused assets, and
   transactional outputs.
4. Add a deterministic fake worker and full protocol failure modes for mandatory CPU tests.
5. Add synthetic worker tests for camera inversion/projection, distortion, visibility,
   occlusion, support scoring, overlap policy, connected components, and reprojection metrics.
6. Implement the isolated real worker with bounded per-camera nvdiffrast processing, OpenCV mask
   undistortion, original-face evidence, surface extraction, compact arrays, and deterministic
   previews.
7. Add the Phase 4 consistency validator, object/validation CLI commands, configurations, Docker
   image, schemas, and documentation.
8. Run all CPU gates and run the fake Phase 4 DAG twice.
9. Install the worker in the existing isolated H100 environment and run one real Phase 3 lineage
   through lifting, preview inspection, consistency validation, and an identical resume pass.
10. Publish a draft PR. Mark it ready only after real acceptance and CI pass.

## Real Acceptance Evidence

The real smoke must record the source mesh identity, processed registered frames, raster
resolution, processed face counts, accepted/ambiguous/unresolved objects, face counts and
components per object, reprojection metrics, conflicts/overlaps, unassigned ratio, runtime,
memory, preview paths, consistency checks, and cache results. Large meshes, masks, run outputs,
and credentials remain local.
