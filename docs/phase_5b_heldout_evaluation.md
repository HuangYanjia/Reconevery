# Held-out completion evaluation

Candidate transforms freeze before held-out data is materialized. Candidates and the
measured-only baseline are rendered into the same real COLMAP cameras.

Dense depth distinguishes visible pixels, occluded candidate surfaces,
negative-space violations, and front-of-scene violations. Metrics include mask
precision/recall/IoU, depth residual/inliers, negative space, measured-surface
distance, normal agreement, and visible coverage.

Generated candidates must pass fixed gates and improve recall over the measured
baseline without catastrophic precision loss. RGB appearance is only an optional
low-weight tie-breaker.
