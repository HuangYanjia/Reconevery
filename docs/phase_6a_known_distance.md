# Phase 6A Known-Distance Landmarks

Known-distance calibration requires explicit named 3D endpoints and a physical
measurement in meters. Anchors and image observations carry typed
`fitting`, `heldout`, or `diagnostic` roles. Each endpoint needs at least two
fitting observations and at least one held-out image observation.
Schema `0.2.0` makes these roles explicit; legacy manifests must add roles
instead of encoding them in evidence IDs.

Real measurements record the tool, date, measurement definition, endpoint
descriptions, uncertainty, and meter units beside the distance anchor. Real
pixel observations record the annotation method and confidence as a pair.
These fields are optional only for backward-compatible synthetic fixtures;
acceptance manifests must populate them.

Observation pixels use `registered_source_image_pixels`. For a COLMAP `OPENCV`
camera, these are distorted source-image coordinates. The isolated calibration
worker applies the exact camera distortion model before DLT triangulation and
reprojection; callers must not pre-undistort the same pixels. `PINHOLE` inputs
pass through unchanged. Unsupported distortion layouts fail closed.

The worker:

1. splits observations into fitting and held-out frames;
2. converts registered source pixels into the camera projection space;
3. triangulates endpoints in the original COLMAP frame;
4. estimates scale from fitting-role distance anchors only;
5. freezes that scale before evaluating held-out-role distance anchors;
6. reports fitting and held-out reprojection separately;
7. reports fitting and independent held-out length residuals separately.

An unreferenced `scene_scale` setting is invalid. Bounding-box endpoints inferred
by software are not manual measured landmarks.

With one fitting distance anchor, metric calibration may be configured as
acceptable when held-out image reprojection passes. The artifact then records
`single_metric_anchor_no_independent_length_holdout`, leaves
`heldout_metric_relative_error` unavailable, and does not duplicate the fitting
residual. Multiple anchors must agree within their predeclared uncertainty and
acceptance gate. The solver never fits a different scale for each object.

For the cabinet O/U/R protocol, height `O-U` is fitting metric evidence and
width `O-R` is independent held-out metric evidence. The landmark world
derivation is a separate hashed artifact. It records the exact camera,
annotation manifest, metric-only triangulation, Phase 5C measured-motion, and
source Scene IR hashes. It also records bootstrap subsets and uncertainties.
The full-canonical solve rejects a derivation if any dependency hash, O/U/R
coordinate, typed up vector, typed forward vector, or typed origin differs from
the current evidence.
