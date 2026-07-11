# Recon2Sim

Recon2Sim Phase 1 adds real video/image ingest and real sparse camera recovery through
out-of-process FFmpeg and COLMAP. The canonical Scene IR, resumable DAG, typed artifacts, and
CPU-only mock pipeline remain intact. Segmentation, object/global reconstruction, compilation,
validation, and export after the camera stage are still deterministic mocks.

The core Python environment contains no COLMAP bindings, OpenCV, model checkpoints, GPU runtime,
SAM 3, GenRecon, SceneSmith, Blender, NeRF, or simulator SDK.

## Install and verify

Python 3.12 is pinned in `.python-version`:

```bash
uv sync --all-groups --locked
uv run recon2sim --help
uv run recon2sim adapters healthcheck
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

The mandatory test suite uses tiny fake FFmpeg/COLMAP executables. It requires no GPU, Docker,
external reconstruction binary, model download, or network access.

## Deterministic mock demo

```bash
uv run recon2sim run \
  --input examples/tabletop \
  --config configs/mock.yaml \
  --run-dir runs/tabletop_demo
uv run recon2sim validate-ir runs/tabletop_demo/scene_ir/scene.json
```

Resume keeps successful state and records cache behavior separately:

```bash
uv run recon2sim run \
  --input examples/tabletop \
  --config configs/mock.yaml \
  --run-dir runs/tabletop_demo \
  --resume
```

A hit remains `"status": "succeeded"` with `"last_execution": "cache_hit"`. Input bytes,
adapter configuration, upstream output bytes, adapter identity/version, and seed participate in
stage signatures.

## Real video or image-directory run

Install FFmpeg (including `ffprobe`) and COLMAP, then put one video in
`examples/real_video/` or pass a directory of JPEG/PNG images:

```bash
uv run recon2sim adapters healthcheck \
  --config configs/colmap.yaml \
  --input examples/real_video

uv run recon2sim run \
  --input examples/real_video \
  --config configs/colmap.yaml \
  --run-dir runs/real_video_colmap
```

Use `configs/colmap_cpu.yaml` on a machine without CUDA. The real stages produce:

```text
inputs/manifest.json
inputs/frame_qa.json
frames/frame_*.png
camera/reconstruction.json
camera/diagnostics.json
camera/colmap/database.db
camera/colmap/sparse/<model>/{cameras.bin,images.bin,points3D.bin}
camera/colmap/logs/*
camera/colmap/workspace_manifest.json
```

Inspect or export the result:

```bash
uv run recon2sim ingest inspect runs/real_video_colmap
uv run recon2sim camera inspect runs/real_video_colmap
uv run recon2sim camera colmap-stats runs/real_video_colmap
uv run recon2sim camera export-trajectory \
  runs/real_video_colmap --output trajectory.json
```

`configs/colmap.yaml` uses sequential matching and GPU SIFT. CPU and Docker examples are in
`configs/colmap_cpu.yaml` and `configs/colmap_docker.example.yaml`. Build the optional pinned
image with:

```bash
docker build -t reconevery/colmap:phase1 docker/colmap
docker version
docker image inspect reconevery/colmap:phase1
```

## Ingest and frame QA

`ffmpeg_ingest` auto-detects a single video or an image directory. Video is decoded by FFmpeg;
images are EXIF-oriented and normalized with Pillow. Outputs are deterministic six-digit RGB PNG
names. QA computes grayscale mean brightness, intensity variance, discrete-Laplacian variance,
and a 16×16 near-duplicate score. The defaults are deliberately conservative, not universal;
tune them for each capture. Rejected frames can remain under `diagnostics/rejected_frames/`, while
only selected frames appear in the normalized manifest.

## Camera coordinates and scale

COLMAP stores world-to-camera `qvec` (`wxyz`) and `tvec`. Recon2Sim normalizes the quaternion,
builds the world-to-camera rotation, inverts the rigid transform, and emits
`transform_world_from_camera` with quaternion order `xyzw`.

Monocular sparse reconstruction does not determine metric scale or gravity/world alignment.
Real COLMAP output is therefore explicitly marked:

```text
scale_status = scale_ambiguous
coordinate_convention.units = arbitrary_scale
world_frame_status = colmap_unaligned
coordinate_convention.world_axes = colmap_arbitrary
```

Do not interpret those translations as meters until an explicit later alignment/scaling step.
The mock path remains right-handed, +X forward, +Y left, +Z up, meters, and world-from-camera.

## Canonical versus derived artifacts

`scene_ir/scene.json` is canonical semantic/physical state. OBJ files are referenced visual or
collision assets. `compiled/scene_package` and future simulator outputs are derived products, not
replacements for Scene IR. The cabinet remains one articulated top-level object whose body and
drawer are links; the drawer is not duplicated as another object.

Every attempt writes to `work/<stage>/attempt_<N>/`. Only a fully validated attempt is promoted
to canonical paths. A failed or stale-output attempt keeps its debugging workspace and cannot
overwrite the last successful result.

See [Phase 1 guide](docs/phase_1_colmap.md), [architecture](docs/architecture.md),
[adapter contracts](docs/adapters.md), and [Scene IR](docs/scene_ir.md).

Regenerate the checked-in Pydantic v2 JSON Schema with:

```bash
uv run python scripts/generate_schema.py
```
