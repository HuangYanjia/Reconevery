# Candidate registration

Registration estimates one proper positive-scale Sim(3):

```text
p_world = scale * rotation * p_candidate + translation
```

Measured fitting points, candidate shape, and cameras remain fixed. Fitting evidence
is rebuilt from fitting frames only using the Phase 5A contract: mask-core erosion,
SAM score, COLMAP consistency-graph support, relative depth-discontinuity rejection,
and multi-view mask/depth agreement. The all-view Phase 5A cloud is diagnostic and
never enters registration when held-out views exist. Its rejection counts and point
and normal hashes are recorded in
`evidence/<object_id>/training_measured_geometry.json`.

Meshes use area-weighted surface samples; Gaussian candidates retain
opacity-filtered support semantics. Fitting is asymmetric from partial measured
points to candidate surface, with robust trimming. Initialization evaluates
identity, measured-bbox/centroid, deterministic right-handed PCA axis hypotheses,
and an explicitly converted official backend layout where available. Every
initialization is scored on fitting evidence only. The best training hypothesis is
refined by asymmetric ICP.

SAM 3D Objects layout values are local to the inferred crop camera and use
PyTorch3D camera axes. They are converted to OpenCV axes with
`diag(-1, -1, 1)`; PyTorch3D's row-vector rotation is transposed into
Reconevery's column-vector matrix before composition with the anchor COLMAP
`world_from_camera` transform. This is an auditable initialization, not a claim
that the backend layout already occupies the Reconevery world frame.

The second bounded refinement uses only registration fitting frames. It optimizes
the same seven similarity parameters against mask support, dense-depth agreement,
and front-of-scene negative-space violations. The refinement is retained only when
its fitting objective improves. Its input paths and before/after objective are
recorded in the typed request and transform artifact. Held-out frames are absent
from the registration workspace.

Reflections, non-uniform scale, camera updates, and vertex deformation are rejected.
Measured normals are transformed from the dense camera maps into the COLMAP world,
and unsigned candidate/measured normal agreement is recorded rather than a
placeholder value.

Equivalent symmetry hypotheses remain explicit rather than receiving false pose
certainty.

Candidates declare independent `registration_asset_id`, `evaluation_asset_id`, and
`selection_asset_id` values. The selected asset must be the representation evaluated
under the frozen transform. Cross-representation transform or metric transfer is
invalid unless an explicit `representation_parity.json` passes its configured
silhouette, bounding-box, and centroid gates.
