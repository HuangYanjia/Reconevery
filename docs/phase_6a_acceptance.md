# Phase 6A Acceptance

Initial research gates are fixed before real evaluation:

```yaml
minimum_metric_evidence_records: 1
minimum_gravity_evidence_records: 1
minimum_forward_evidence_records: 1
minimum_heldout_tag_detections: 3
maximum_heldout_tag_translation_error_m: 0.02
maximum_heldout_tag_rotation_error_degrees: 3.0
minimum_known_distance_anchors: 1
maximum_known_distance_relative_error: 0.02
maximum_heldout_landmark_reprojection_error_px: 2.0
allow_single_metric_anchor_without_length_holdout: true
maximum_gravity_heldout_error_degrees: 3.0
maximum_forward_uncertainty_degrees: 5.0
maximum_sim3_roundtrip_error: 1.0e-8
```

Real full-canonical acceptance requires a recorded metric measurement, disjoint
fitting/held-out evidence, accepted gravity, forward, origin, a finite proper
Sim(3), canonical camera validation, one derived metric geometry export, real
rigid/articulated propagation, 34/34 consistency checks, and an identical
resume with all Phase 6A stages cache hitting.

For landmark-derived full canonicalization, the calibration artifact,
canonical wrapper, and canonical Scene IR must all reference the exact copied
`landmark_world_derivation.json` bytes. Its dependency hashes and recomputed
O/U/R coordinates are checked before solving. Source-space geometry remains
unchanged and declares that the world wrapper is required.

Example real execution:

```bash
uv run recon2sim run \
  --input <calibration-frames-or-video> \
  --config <machine-local-phase6a.yaml> \
  --run-dir <calibration-run>
uv run recon2sim validation verify-phase6a <calibration-run>
uv run recon2sim run \
  --input <calibration-frames-or-video> \
  --config <machine-local-phase6a.yaml> \
  --run-dir <calibration-run> --resume
```

Metric-only and rejected results are useful outputs, but they do not satisfy
full-canonical readiness. Gates are never lowered after seeing a real result.

Phase 6A produces no collisions, dynamics, simulator exports, or
simulation-ready claims.
