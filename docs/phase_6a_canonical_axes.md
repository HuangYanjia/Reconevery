# Phase 6A Canonical Axes

The canonical world is:

```text
right handed
+X forward
+Y left
+Z up
meters
```

Up and forward evidence are explicit. Forward is projected onto the horizontal
plane, normalized, and re-orthogonalized. Left is:

```text
+Y = +Z cross +X
```

Nearly parallel up/forward evidence is rejected. Origin is separately declared
as a fiducial origin, reconstructed landmark, floor-projected reference camera,
or configured object anchor.

The transform is a uniform positive-scale proper Sim(3), includes an explicit
inverse, and must pass a `1e-8` round-trip gate. A reference-camera forward is
used only when explicitly configured.

## Cabinet O/U/R contract

For the real cabinet landmark protocol:

```text
O = fixed front-left-bottom cabinet point
U = fixed front-left-top cabinet point
R = fixed front-right-bottom cabinet point

up = normalize(U - O)
right = normalize((R - O) - dot(R - O, up) * up)
forward candidates = +/- normalize(cross(up, right))
origin = O
```

The forward sign is selected using the Phase 5C prismatic drawer axis derived
from closed and fitting-state evidence. The held-out open-state q is not used.
The drawer axis selects only between the two geometric forward candidates; it
does not alter O/U/R, the axis magnitude, or the fitted metric scale.

The derivation is accepted only when its typed evidence values and dependency
hashes match the current solve. Bootstrap or jackknife subsets of the fitting
annotations provide the angular and origin uncertainty. Semantic state labels
and ordinary-object size priors do not establish any canonical axis.
