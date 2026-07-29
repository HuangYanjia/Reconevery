# Phase 6A Articulation Propagation

Source Phase 5C artifacts remain immutable. The canonical wrapper composes the
accepted world transform with the articulated object base.

Propagation rules:

```text
base translation       rotate and scale
base rotation          rotate
prismatic axis         rotate only
prismatic q/range      multiply by metric scale once
revolute axis          rotate only
revolute pivot         rotate, scale, translate
revolute q/range       unchanged radians
```

Candidate-prior limits are converted only when their unit relationship is
verified. Reference-world measured assets retain their bytes and receive one
wrapper transform. Candidate-base and link-local assets retain their local
geometry conventions. The wrapper never double transforms measured geometry.
