# Recon2Sim

Recon2Sim Phase 0.1 is a typed, CPU-only observation-to-simulation foundation. Its mock
pipeline exercises the same filesystem contracts, validation, DAG, caching, retries, and
provenance that future real adapters must satisfy. It does not install or run COLMAP, SAM 3,
GenRecon, SceneSmith, Blender, simulator SDKs, model checkpoints, or GPU code.

## Quickstart

Python 3.12 is pinned in `.python-version`. Install `uv`, then run:

```bash
uv sync --all-groups
uv run recon2sim --help
uv run recon2sim run \
  --input examples/tabletop \
  --config configs/mock.yaml \
  --run-dir runs/tabletop_demo
uv run recon2sim validate-ir runs/tabletop_demo/scene_ir/scene.json
```

Resume without changing successful status:

```bash
uv run recon2sim run \
  --input examples/tabletop \
  --config configs/mock.yaml \
  --run-dir runs/tabletop_demo \
  --resume
```

A cache hit remains `"status": "succeeded"` and records
`"last_execution": "cache_hit"`. Changing input bytes, stage configuration, or an upstream
artifact invalidates the affected stage and all of its dependents.

## Mock data flow

The checked-in PNG frames are tiny generated test fixtures. The pipeline produces and consumes:

```text
examples/tabletop/frames/*.png
  -> inputs/manifest.json + frames/*.png
  -> camera/reconstruction.json
  -> observations/object_tracks.json + observations/masks/*.png
  -> reconstruction/global/{floor.obj,metadata.json}
     reconstruction/objects/{results.json,*.obj}
  -> scene_ir/scene.json
  -> compiled/scene_package/{package.json,mock_scene.obj}
  -> validation/report.json
  -> export_manifest.json
```

`scene_ir/scene.json` is the canonical semantic and physical scene. OBJ files are visual or
collision mesh artifacts referenced by the IR. `compiled/scene_package` is a mock compiler
output, and its package explicitly contains no simulator outputs. These are separate contracts.

The cabinet is one top-level articulated object. Its body and drawer are articulation links;
the drawer is not duplicated as an independent `ObjectInstance`.

## Coordinate convention

- right-handed world frame;
- +X forward, +Y left, +Z up;
- meters;
- quaternions ordered `(x, y, z, w)`;
- camera poses are `transform_world_from_camera`.

The convention is stored in typed camera and Scene IR metadata rather than left implicit.

## Quality gate

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

The generated schema at `schemas/scene_ir.schema.json` is checked against
`SceneIR.model_json_schema()` in tests. Regenerate it with:

```bash
uv run python scripts/generate_schema.py
```

## Adapter boundary

Core code imports only lightweight dependencies. Future heavyweight tools must run behind an
adapter boundary and exchange declared, typed, validated artifacts. See `docs/adapters.md` and
`docs/roadmap.md`; Phase 1 begins with camera recovery only.
