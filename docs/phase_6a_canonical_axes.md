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
