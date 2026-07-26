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

The evaluation worker uses the official COLMAP dense-array layout exactly:

```text
payload.reshape((width, height, channels), order="F").transpose(1, 0, 2)
```

Depth requires one channel and normals require three finite channels. This is the
same parser contract as Phase 5A; truncated payloads and mismatched channels fail.

Evaluation has three auditable groups:

1. Canonical anchor sanity uses the backend layout, the official point-map crop
   intrinsics, and the original RGBA crop. It identifies export, camera, or
   representation failures.
2. Registered anchor and fitting-view evaluation use the frozen registration
   transform in full COLMAP cameras and distinguish registration failure from a
   valid native render.
3. Held-out evaluation uses the same frozen representation and transform.

Each frame records raw, visible, occluded, negative-space, and front-of-scene pixel
counts; candidate and scene depth ranges; projected and target boxes; and mask
metrics. Zero overlap is classified as `backend_export_invalid`,
`empty_candidate_render`, `registration_failed`, `fitting_view_inconsistent`, or
`fitting_overfit_heldout_failure` according to the first stage that fails.

The measured renderer control is an open mesh built only from fitting evidence. It
uses the same mesh loader, transform handling, target-camera renderer, occlusion
classification, and held-out metrics as generated candidates. It is a diagnostic
control, not a hidden-surface completion.
