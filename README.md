# Recon2Sim

Recon2Sim Phase 1 is a typed observation-to-simulation pipeline with real video/image ingest,
deterministic frame QA, and out-of-process COLMAP sparse camera recovery. The downstream
segmentation, reconstruction, compilation, validation, and export stages remain deterministic
mocks. COLMAP, FFmpeg, and Docker are optional system tools rather than Python dependencies.

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

For a directory containing one video, or an `images/` directory of JPEG/PNG files:

```bash
uv run recon2sim adapters healthcheck --config configs/colmap.yaml
uv run recon2sim run \
  --input examples/real_video \
  --config configs/colmap.yaml \
  --run-dir runs/real_video_colmap
uv run recon2sim ingest inspect runs/real_video_colmap
uv run recon2sim camera inspect runs/real_video_colmap
uv run recon2sim camera export-trajectory \
  runs/real_video_colmap --output trajectory.json
```

`configs/colmap.yaml` and `configs/colmap_cpu.yaml` intentionally stop after camera recovery.
`configs/colmap_with_mock_downstream.yaml` is the explicit integration demo for real ingest/COLMAP
followed by Phase 0.1 mocks. The optional pinned COLMAP 3.11.1 image is:

```bash
docker build -t reconevery/colmap:phase1 docker/colmap
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

## Real ingest and camera recovery

`ffmpeg_ingest` accepts a video file, a directory containing one video, or a deterministic
JPEG/PNG image collection. Video decoding uses FFmpeg and FFprobe as subprocesses. Frame QA
records Laplacian-variance sharpness, mean grayscale brightness, intensity variance, and a
downsampled-pixel duplicate score in `inputs/frame_qa.json`. Defaults are conservative and
require dataset-specific tuning.

`colmap_camera_recovery` runs feature extraction, sequential or exhaustive matching, mapping,
binary-model parsing, and deterministic sparse-model selection. It preserves `database.db`,
every sparse model, commands, and logs under `camera/colmap/`. See
[`docs/phase_1_colmap.md`](docs/phase_1_colmap.md) for setup and capture guidance.

## Coordinate convention

Raw Phase 1 COLMAP output is explicitly `world_frame="colmap_arbitrary"`,
`alignment_status="unoriented"`, `camera_axes="x_right_y_down_z_forward"`,
`linear_units="arbitrary_units"`, and `scale_status="scale_ambiguous"`. Poses are
`transform_world_from_camera`, quaternions are `xyzw`, and the unit-neutral `translation` values
remain in COLMAP's arbitrary gauge.

The canonical robot scene frame is a separate, later contract: right-handed +X forward, +Y left,
+Z up, with meters after external alignment and scale recovery. Phase 1 does not perform or imply
that conversion.

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

Core code imports only lightweight dependencies. Heavyweight tools run behind an adapter boundary
and exchange declared, typed, validated artifacts. SAM 3, GenRecon, SceneSmith, Blender,
simulators, MVS, NeRF, and model checkpoints remain out of Phase 1.
