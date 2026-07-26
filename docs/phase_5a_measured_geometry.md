# Phase 5A Measured Object Geometry

## Contract

The worker associates canonical SAM instances with official COLMAP geometric
depth, not GenRecon triangles. For every registered object observation it:

1. maps the binary mask into the exact dense undistorted camera;
2. selects finite positive geometric depth inside the mask core;
3. requires a configurable number of consistency-graph source views;
4. backprojects through the dense PINHOLE camera;
5. applies the recorded `transform_world_from_camera`;
6. projects candidates into other observations of the same canonical track;
7. checks mask membership and relative depth agreement;
8. fuses retained samples into deterministic scene-relative voxels.

Distance thresholds are relative to depth, sample spacing, or scene diagonal.
They are never interpreted as meters.

## Outputs

Every non-empty hypothesis has `measured_points.ply`, `surfels.ply`,
`surfels.npz`, and per-view support. The Phase 5A production worker emits the
surfel result and leaves `observed_surface` unset. A future optional observed
triangulation may only connect adjacent valid depth pixels; it may not close
holes, cap mask boundaries, bridge depth discontinuities, infer a backside, or
claim watertightness.

An object without reliable depth is a successful `unresolved` result. All
hypotheses state:

```text
geometry_source = measured
geometry_status = partial_measured
hidden_surface_completion = not_implemented
watertight = false
sim_ready = false
metric_scale_known = false
canonical_gravity_alignment_known = false
completeness_confidence = 0
```

Measurement confidence describes only visible, multi-view-supported samples.
Reprojection precision, recall, and IoU are computed from depth-visible fused
surfel splats against the undistorted canonical mask. They are not copied from
the sample-retention ratio and do not measure hidden-shape completeness.

## CLI

```bash
uv run recon2sim objects inspect-measured runs/phase5a
uv run recon2sim objects inspect-measured-object runs/phase5a table_0001
uv run recon2sim objects export-measured-points \
  runs/phase5a table_0001 --output table_points.ply
uv run recon2sim objects export-observed-surface \
  runs/phase5a table_0001 --output table_observed.ply
uv run recon2sim validation verify-phase5a runs/phase5a
```

The optional measured/generated comparison is diagnostic. GenRecon is never an
input to canonical measured object geometry and does not alter its bytes.

## Tuning

Conservative defaults require two consistency-graph sources, two supporting
object views, and relative depth residual at most 0.03. Textureless, reflective,
transparent, moving, or thin objects may remain unresolved. Lowering thresholds
does not create measured evidence and should be reported as a dataset-specific
tradeoff.
