# Phase 3 End-to-End Acceptance

Phase 3 has two independent real acceptance levels.

## Module-Level GenRecon

A real module smoke passes only when the exact official Git/submodule checkout and all three
official checkpoint hashes verify, CUDA extensions import, `reconstruct_scene.py` produces
`to_glb_inputs.pt` and `chunk_inputs.pt`, `chunked_to_glb.py` produces a valid `scene.glb`, the
PLY is finite/non-empty, typed artifacts validate, previews exist, and a resume run cache-hits
global reconstruction.

## Shared-Lineage End to End

The required full smoke is:

```text
real input -> real ingest -> real COLMAP -> real SAM 3.1
                                  \------> real GenRecon
                         -> real consistency validator
```

The Phase 2 real SAM smoke used deterministic camera metadata and cannot satisfy this level.
The Phase 3 smoke must use a real COLMAP selected sparse model from the same normalized frames.

`validation/phase3_e2e_consistency.json` checks:

1. identical ingest manifest SHA;
2. identical ordered frame IDs, paths, and frame hashes;
3. identical frame-sequence digest;
4. real COLMAP registered/unregistered sets;
5. exact master order for SAM;
6. registered subset in master order for GenRecon;
7. GenRecon selected views remain an eligible ordered subset;
8. identical camera reconstruction SHA;
9. SAM `camera_pose_available` matches real registration;
10. raw arbitrary/unoriented/scale-ambiguous coordinate semantics;
11. reversible internal GenRecon working transform;
12. SAM attempt excludes raw COLMAP workspace;
13. GenRecon attempt excludes raw COLMAP and SAM artifacts.

The report always states:

```text
object_level_2d_3d_fusion_implemented=false
sim_ready_scene_implemented=false
metric_scale_known=false
canonical_gravity_alignment_known=false
```

The identical command must then run with `--resume`, and the run manifest must show cache hits
for ingest, camera recovery, SAM, camera package, GenRecon, and consistency validation.

Required visual evidence:

- `reconstruction/global/previews/camera_trajectory_and_sparse_points.png`;
- `observations/previews/contact_sheet.png` and `track_timeline.png`;
- `reconstruction/global/previews/global_scene_preview.png`;
- `reconstruction/global/previews/input_vs_geometry_contact_sheet.png`.

Do not mark the pull request ready based on fake output, a mock camera, or module-level GenRecon
alone.

## Validated H100 Smoke

The Phase 3 implementation passed both acceptance levels on 2026-07-25 with one NVIDIA H100
NVL. The shared image-directory lineage contained 16 normalized frames. Real COLMAP registered
12 frames (75%) and reconstructed 2,454 sparse points. Official SAM 3.1 processed the same
ordered frame sequence with `table`, `cup`, and `cabinet` prompts, retaining 4 tracks and 50
canonical masks.

Official GenRecon used all 12 registered frames and produced 9 chunks, a valid 661 MiB
`scene.glb`, and a finite global mesh with 6,385,868 vertices and 13,437,638 faces. Its runtime
was 1,142.94 seconds and recorded peak GPU memory was 14,170,456,064 bytes. All 13 consistency
checks passed, and the identical resumed command reported cache hits for all six stages.

The three official checkpoint SHA-256 prefixes were `e18c1caddb2357`, `d9e13be151a213`, and
`28f99217a4fbcd`. Large run outputs and checkpoint files remain local and are not committed.
This evidence confirms consistent module coexistence only: object-level 2D/3D fusion,
simulation-ready geometry, metric scale, and gravity alignment remain unimplemented.
