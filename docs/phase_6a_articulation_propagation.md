# Phase 6A Articulation Propagation

Source Phase 5C artifacts remain immutable. Phase 5C defines
`ObjectInstance.transform` as candidate/object-local to COLMAP world, while joint
axes, pivots, and prismatic positions are object-local. Revolute positions are
radians.

Propagation rules:

```text
object root             T_canonical_from_colmap @ T_colmap_from_object
link-local geometry     unchanged
joint local axis        unchanged
joint local pivot       unchanged
prismatic local q       unchanged
revolute q              unchanged radians
```

The wrapper records a typed conversion rather than rewriting q:

```text
prismatic_position_space = object_local
prismatic_position_scale_to_m =
    world_scale_m_per_colmap
    * source_object_scale_colmap_per_local_unit
```

A compiler converts q with that product exactly once. The canonical hierarchy
therefore produces the same world-space child-link pose as applying the world
calibration to the complete source hierarchy, including non-unit candidate root
scale, negative q, non-identity rotations, and revolute pivots.

Reference-world measured assets retain their bytes and receive one wrapper.
Candidate-base and link-local assets retain their local conventions. No local
quantity is transformed after the root composition.
