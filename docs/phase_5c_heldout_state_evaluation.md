# Phase 5C held-out-state evaluation

Candidate generation and structural fitting cannot access held-out state geometry.
After structure is frozen, evaluation may infer only the held-out scalar joint
position from measured geometry or a documented interpolation.

The default measured-geometry policy solves a bounded one-dimensional objective:
held-out measured points are compared asymmetrically against the candidate child
link while the base Sim(3), joint type, axis, pivot, graph, and link geometry remain
unchanged. Candidate-prior limits are respected. An observed-range-only bound may
expand by one observed span so an unseen open state is not clipped to the fitting
range.

Evaluation reports base, movable-part, and whole-object masks; per-link depth;
negative space; front-of-scene violations; base stability; joint residual; axis and
pivot consistency; and observed motion coverage. Missing render evidence fails
closed. Gates are configured before a real smoke and are not lowered after results.

For every held-out view, the evaluator composes:

```text
camera_from_state
* state_from_reference
* reference_from_candidate_base
* constrained_joint_transform(q)
```

It renders each assigned link using the same exact OpenCV-to-homogeneous-clip
contract as Phase 5B.1. Canonical masks are mapped into the official COLMAP dense
PINHOLE space with nearest-neighbor remapping, and candidate pixels behind reliable
dense depth are treated as occluded rather than false positives.

Exactly three accepted states are sufficient: generation, fitting, and actually
evaluated held-out states are counted as a distinct union. Passing requires usable
cameras, target masks, visible candidate pixels, dense depth, measured base motion,
and joint-type-specific residuals. Unavailable metrics fail their gates rather than
being serialized as zero.
