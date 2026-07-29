# Phase 5C.2 correctness contract

Phase 5C.2 hardens candidate fitting and held-out evaluation before a real
closed/half-open/open capture is accepted. It does not change the Phase 5C
capability boundary.

## Joint sign convention

`fitted_axis` is normalized and oriented toward the measured motion axis.
`axis_sign` records only whether the native candidate axis was flipped. It is
provenance and is never applied to `q` a second time.

Candidate joint positions use:

```text
prismatic q_candidate = q_offset + q_measured / global_sim3_scale
revolute q_candidate  = q_offset + q_measured
```

`q_offset` is fitted only when at least two structure states provide measured
geometry. The fitted model records those evidence state IDs. Held-out states
never contribute to the offset, axis, pivot, base transform, or link assignment.

## Visual asset spaces

Every articulated link visual declares:

```text
visual_asset_space = candidate_base | link_local
visual_asset_transform_candidate_base
content SHA-256
```

Candidate-base assets require the identity transform. Registration, fitting
renders, held-out renders, selection, and Scene IR consume the same declared
representation. Workers re-hash each visual before loading it.

Particulate link meshes are exported in candidate-base space. Offline ArtVIP
and PartNet bundles must declare their own spaces; an old catalog without these
records is rejected and must be re-indexed.

## Held-out provenance

Each requested held-out frame records the camera reconstruction hash, dense
depth path and hash, every target mask path and hash, required links, rendered
links, missing links, rendered pixel counts, output path, and output hash.

A view is usable only when all mapped parts have target masks, dense depth is
valid, every mapped candidate link has a visual asset, every required link
renders non-empty pixels, and the combined render is non-empty.

Revolute diagnostics come from the held-out reference-to-state rigid
registration. Axis error, signed angular `q`, and pivot fixed-point residual are
held-out measurements rather than reused fitting metrics.

## Artifact identity

The evaluation manifest binds the exact request, candidate manifest, candidate,
fitted model, link assignment, evidence split, measured-state manifest,
state-alignment artifact, measured-motion artifact, and held-out evidence
digest. Selection fails closed when any identity differs. Scene IR retains the
per-asset coordinate-space transform and the selected fitted base Sim(3).

Official Particulate generation is not rerun when neither its request nor its
source mesh changed.

The result remains in arbitrary, unoriented COLMAP coordinates. Collision
generation, dynamics identification, physical validation, metric scale,
gravity alignment, and simulation readiness remain unimplemented.
