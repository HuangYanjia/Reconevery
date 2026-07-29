# Phase 5C official Particulate integration

Official source:

```text
https://github.com/RuiningLi/particulate
commit dee37a75c449f324d9989993461ee09eaccc1686
```

Official model:

```text
rayli/Particulate
revision 096167e661feb92a443535d15916323ec8a01613
model.pt sha256 ad6f14067dadf85335119199b94e8249401376d5700c9b627c3608594ea99b5c
```

PartField runtime:

```text
mikaelaangel/partfield-ckpt
revision 90b9b1e08b6a12fdcb6ee26b4854a26235e1765f
model_objaverse.ckpt sha256 463efc8a3afd3913142aa025e0125c00f16ef452b8de6a132ebe32bbe7877ee4
```

The worker invokes official `infer.py` with argument arrays and `--eval`, preserves
official GLB/NPZ/OBJ outputs, and normalizes official face-part and joint predictions.
The official requirements leave Transformers unbounded. The reviewed environment
pins Transformers 4.46.3 because the current 5.x release imports APIs unavailable
in the official PyTorch 2.4.0 stack.
Particulate expects +Z up. The first real-capture implementation requires an explicit
`working_axis_hint`, records it as a prior, records the single evaluated hypothesis
and both reversible matrices, and transforms links, axes, and pivots back. It does
not silently choose the first item from a hypothesis list and does not claim gravity
alignment. Multi-hypothesis scoring remains future work.

The official model card declares Apache-2.0 code and CC-BY-4.0 model weights.
PartField's checkpoint is NVIDIA non-commercial research-only. Therefore the current
combined backend is never production-selectable.
