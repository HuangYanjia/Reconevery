# Scene Assembly Worker

This isolated Phase 6B worker creates diagnostic PNG and GLB previews from a frozen
typed assembly plan. It may import NumPy, Pillow, and trimesh; the Reconevery core
does not.

The worker receives an attempt-local selective workspace. It never receives the
canonical run root, never modifies source assets, never carves the global mesh, and
never creates collision or physics assets. Preview GLBs contain one visual snapshot
and are not canonical compiler inputs.

```bash
uv sync --project workers/scene_assembly
uv run --project workers/scene_assembly python -m scene_assembly_worker healthcheck
```
