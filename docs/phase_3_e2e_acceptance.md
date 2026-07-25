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
