# Candidate registration

Registration estimates one proper positive-scale Sim(3):

```text
p_world = scale * rotation * p_candidate + translation
```

Measured fitting points, candidate shape, and cameras remain fixed. Meshes use
area-weighted surface samples; Gaussian candidates retain opacity-filtered support
semantics. Fitting is asymmetric from partial measured points to candidate surface,
with robust trimming. Initialization evaluates identity and deterministic
right-handed PCA axis hypotheses on training points only. The best training
hypothesis is refined by asymmetric ICP.

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
