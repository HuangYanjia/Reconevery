# Phase 4.2 global Sim(3)

The tested model is:

```text
p_aligned = s R p_original + t
```

Only seven global parameters change: positive uniform scale, proper SO(3) rotation, and
translation normalized by the original scene diagonal. Cameras and mesh topology stay fixed.
The scale is a gauge correction in arbitrary units, not metric recovery.

Filtered COLMAP observations use the same OpenCV `getOptimalNewCameraMatrix(alpha=0)` undistortion
and full-image ROI policy as Phase 4. Depth residuals are normalized by sparse camera depth;
point-to-surface residuals use scene diagonal. Identity, robust extent, centroid, and right-handed
PCA hypotheses are deterministic. Correspondences are updated iteratively with MAD rejection and
Cauchy loss. The transform is bounded by configured scale, rotation, and scene-relative
translation limits.

Alternating registered frames and disjoint sparse point IDs form training and validation sets.
Acceptance is based on validation median/p90 depth improvement, absolute inlier improvement,
coverage preservation, bad-camera fraction, point-to-surface non-regression, finite invertibility,
and plausibility. Gates are fixed before observing the real result.

Residuals are reported by camera and GenRecon chunk. Structured post-fit residuals produce
`residual_is_locally_structured=true`; Phase 4.2 diagnoses but does not implement per-chunk or
non-rigid correction.

Candidate ambiguity is evaluated separately. If two plausible candidates have objectives within
one percent but differ materially in rotation, log scale, or scene-normalized translation, the
diagnostics record `candidate_solution_ambiguous=true` and list the competing candidate IDs. Such
a solution is not accepted merely because deterministic sorting can choose one candidate.
