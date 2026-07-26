# Official Particulate worker

This worker verifies the official Particulate Git commit, official model hash, and
PartField runtime hash before invoking the unmodified official `infer.py` command.
It preserves official GLB and evaluation outputs, splits the source mesh by official
face-part predictions, and reverses the temporary +Z working transform for links,
axes, and pivots.

Pinned identities:

- code: `dee37a75c449f324d9989993461ee09eaccc1686`
- `rayli/Particulate` revision
  `096167e661feb92a443535d15916323ec8a01613`
- `model.pt` SHA-256
  `ad6f14067dadf85335119199b94e8249401376d5700c9b627c3608594ea99b5c`
- PartField `model_objaverse.ckpt` SHA-256
  `463efc8a3afd3913142aa025e0125c00f16ef452b8de6a132ebe32bbe7877ee4`
- Transformers `4.46.3`; the official unbounded requirement currently resolves
  to Transformers 5.x, which is incompatible with the official PyTorch 2.4 pin.

PartField is non-commercial research-only, so generated candidates are never
production-selectable without a future independently reviewed model stack.
