# Phase 6A Known-Distance Landmarks

Known-distance calibration requires explicit named 3D endpoints and a physical
measurement in meters. Anchors and image observations carry typed
`fitting`, `heldout`, or `diagnostic` roles. Each endpoint needs at least two
fitting observations and at least one held-out image observation.
Schema `0.2.0` makes these roles explicit; legacy manifests must add roles
instead of encoding them in evidence IDs.

The worker:

1. splits observations into fitting and held-out frames;
2. triangulates endpoints in the original COLMAP frame;
3. estimates scale from fitting-role distance anchors only;
4. freezes that scale before evaluating held-out-role distance anchors;
5. reports fitting and held-out reprojection separately;
6. reports fitting and independent held-out length residuals separately.

An unreferenced `scene_scale` setting is invalid. Bounding-box endpoints inferred
by software are not manual measured landmarks.

With one fitting distance anchor, metric calibration may be configured as
acceptable when held-out image reprojection passes. The artifact then records
`single_metric_anchor_no_independent_length_holdout`, leaves
`heldout_metric_relative_error` unavailable, and does not duplicate the fitting
residual. Multiple anchors must agree within their predeclared uncertainty and
acceptance gate. The solver never fits a different scale for each object.
