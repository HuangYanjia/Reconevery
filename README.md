# Recon2Sim

Recon2Sim Phase 2 is a typed observation pipeline with real video/image ingest, frame QA,
out-of-process COLMAP sparse camera recovery, and prompt-driven SAM 3.1 segmentation/tracking.
Heavy tools and model runtimes remain isolated from the lightweight core package. Global/object
3D reconstruction, compilation, validation, and export remain deterministic mocks.

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

Run the complete CPU-only fake SAM protocol, including canonical masks and previews:

```bash
uv run recon2sim adapters healthcheck --config configs/sam3_fake.yaml
uv run recon2sim run \
  --input examples/tabletop \
  --config configs/sam3_fake.yaml \
  --run-dir runs/tabletop_sam3_fake
uv run recon2sim segmentation inspect runs/tabletop_sam3_fake
uv run recon2sim segmentation render-preview runs/tabletop_sam3_fake
uv run recon2sim segmentation export-coco \
  runs/tabletop_sam3_fake --output runs/tabletop_sam3_fake/annotations.json
```

`configs/sam3.yaml` and `configs/sam3_docker.example.yaml` use the official gated
`facebook/sam3.1` checkpoint and stop after segmentation. Accept Meta's terms and provide
`HF_TOKEN`, an authorized cache, or a mounted local official checkpoint. See
[`docs/phase_2_sam3.md`](docs/phase_2_sam3.md).

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

## SAM 3 segmentation and tracking

`sam3_segmentation_tracking` consumes only the normalized manifest, frame QA, typed camera
reconstruction, selected frames, and a validated prompt manifest. It never copies the raw COLMAP
database, sparse models, or logs into the SAM attempt. Text, box, point, and mask prompt contracts
are typed; the pinned official public video API supports text, box, and point requests. It does
not expose mask-seed requests, so the real worker reports that limitation instead of using an
undocumented internal method.

The default real backend is official SAM 3.1 Object Multiplex at code commit
`46957e47805eaa273f4aa7bbbd25a88bca9108ce` and checkpoint revision
`daa63191845a41281374e725f4c9e51c7a824460`. Canonical tracks preserve raw model IDs separately,
assign stable semantic IDs such as `cup_0001`, and write one grayscale `0/255` PNG per visible
object/frame. Frames without a COLMAP pose remain valid 2D observations and carry
`camera_pose_available=false`.

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

Scene IR, prompt, request, and segmentation schemas under `schemas/` are checked against their
Pydantic models in tests. Regenerate them with:

```bash
uv run python scripts/generate_schema.py
```

## Adapter boundary

Core code imports only lightweight dependencies. Official SAM code, PyTorch, CUDA packages, and
checkpoints exist only in `workers/sam3` or `docker/sam3`. GenRecon, SceneSmith, Blender,
simulators, MVS, NeRF, and automatic VLM prompt inventory remain outside Phase 2.
