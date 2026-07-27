# Phase 5C.3 asset and selection integrity

Phase 5C uses three explicit articulated asset spaces:

- `reference_world`: immutable measured Phase 5A points in the aligned reference
  state's arbitrary COLMAP frame.
- `candidate_base`: generated or retrieved geometry whose vertices are already in
  the candidate base frame.
- `link_local`: generated or retrieved link geometry with an explicit transform into
  the candidate base frame.

Reference-world measured anchors remain on `ObjectInstance.geometry_asset_ids`.
When a generated or retrieved candidate is selected, articulated links reference
only candidate-base or link-local visuals. This prevents the fitted object Sim(3)
from being applied a second time to measured evidence. No measured bytes are
rewritten in Phase 5C.3.

## Selected records

Selection writes one deterministic record per selected identity:

```text
reconstruction/articulation/selected/<object_id>/
├── selected_candidate.json
├── fitted_kinematic_model.json
├── selected_link_assignment.json
├── selected_evaluation.json
├── selected_identity_manifest.json
├── kinematic_bundle.json
└── preview_only.urdf
```

The identity manifest and kinematic bundle refer to the four source records by exact
relative path and file SHA-256. Scene IR repeats the exact selected paths and hashes.
The consistency validator independently hashes every file and compares its typed
contents with the canonical candidate, fitting, assignment, and evaluation
manifests.

## Preview transforms

The visual-only URDF records the fitted reference-world-from-candidate-base Sim(3)
as diagnostic metadata. Each `<visual>` contains an `<origin>` derived from its
declared asset transform, and its uniform scale is written on `<mesh>`. This preview
does not contain collision, inertial, dynamics, metric, or gravity claims and remains
`simulation_ready=false`.
