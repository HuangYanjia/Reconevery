# Phase 6A Known-Distance Landmarks

Known-distance calibration requires explicit named 3D endpoints and a physical
measurement in meters. Each endpoint must be observed in at least two
registered frames; three or more are preferred.

The worker:

1. splits observations into fitting and held-out frames;
2. triangulates endpoints in the original COLMAP frame;
3. measures fitting and held-out reprojection error;
4. computes reconstructed arbitrary-unit distances;
5. robustly estimates one scene-wide meters-per-unit scale;
6. reports every anchor's relative residual.

An unreferenced `scene_scale` setting is invalid. Bounding-box endpoints inferred
by software are not manual measured landmarks.

Multiple anchors must agree within their predeclared uncertainty and acceptance
gate. The solver never fits a different scale for each object.
