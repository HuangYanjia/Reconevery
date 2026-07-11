# Phase 1: Real Ingest and COLMAP Camera Recovery

Phase 1 accepts one video or a JPEG/PNG directory, creates selected normalized frames, runs
COLMAP sparse structure-from-motion, preserves its native workspace, and normalizes camera output
for the existing downstream mock pipeline.

It deliberately excludes segmentation models, object tracking models, dense/MVS reconstruction,
NeRF, object reconstruction, SceneSmith, Blender, and simulator export.

## Why tools run out of process

FFmpeg and COLMAP are native, independently versioned applications with optional GPU/runtime
requirements. Keeping them behind subprocess and file contracts prevents their dependencies from
contaminating the small core Python environment, preserves exact native output for diagnosis, and
lets local, Docker, fake-test, CPU, and GPU execution share one typed downstream interface.

## Capture recommendations

- Translate the camera through space; rotation from one fixed point provides weak parallax.
- Maintain substantial overlap between consecutive views.
- Move slowly enough to avoid motion blur and rolling-shutter artifacts.
- Capture multiple heights and viewing angles around the scene.
- Keep exposure/focus stable when possible.
- Avoid large moving people/objects and scenes dominated by transparent, reflective, or
  textureless surfaces.
- Do not assume monocular reconstruction scale is metric.

## Input and extraction

Accepted video extensions are MP4, MOV, MKV, AVI, M4V, and WebM. Image directories accept JPEG
and PNG recursively. Auto mode requires either exactly one video or images, not both.

Video mode uses `ffprobe` JSON metadata and FFmpeg. `target_fps` limits temporal density,
`max_frames` bounds work, `resize_max_edge` preserves aspect ratio, and an optional
`scene_change_threshold` adds deterministic scene filtering. Existing attempt output is never
overwritten. Image mode normalizes EXIF orientation and RGB PNG bytes with Pillow; it does not use
OpenCV.

Frame names are `frame_000000.png`, `frame_000001.png`, and so on based on extraction/source
order. Manifest frame entries include ID, run-relative normalized path, SHA-256, dimensions,
timestamp, source type, input-root-relative source reference, and original index.

QA uses simple CPU grayscale statistics. Near-duplicate similarity is
`1 - mean_absolute_diff / 255` on a 16×16 representation against the last selected frame.
Sharpness is variance of a
discrete Laplacian. These metrics are deterministic but thresholds are dataset-specific. Start
with defaults, inspect `inputs/frame_qa.json`, and tune only with evidence.

## Local setup and commands

Install FFmpeg/FFprobe and COLMAP, then verify them:

```bash
ffmpeg -version
ffprobe -version
colmap -h
uv run recon2sim adapters healthcheck \
  --config configs/colmap.yaml \
  --input examples/real_video
```

Put one video under `examples/real_video/` (the repository contains only placement guidance), then
run GPU SIFT:

```bash
uv run recon2sim run \
  --input examples/real_video \
  --config configs/colmap.yaml \
  --run-dir runs/real_video_colmap
```

CPU mode:

```bash
uv run recon2sim run \
  --input examples/real_video \
  --config configs/colmap_cpu.yaml \
  --run-dir runs/real_video_colmap_cpu
```

The adapter executes `feature_extractor`, sequential or exhaustive matching, and `mapper` with
argument lists. Video defaults to sequential matching (`overlap: 10`, loop detection disabled).
Exhaustive matching may help unordered small photo sets but scales poorly. Auto-detection reads
the installed subcommand help and maps GPU mode to current `FeatureExtraction.use_gpu` /
`FeatureMatching.use_gpu` or the corresponding legacy SIFT option names.

## Docker setup

Docker is optional:

```bash
docker build -t reconevery/colmap:phase1 docker/colmap
docker version
docker image inspect reconevery/colmap:phase1
uv run recon2sim adapters healthcheck \
  --config configs/colmap_docker.example.yaml \
  --input examples/real_video
uv run recon2sim run \
  --input examples/real_video \
  --config configs/colmap_docker.example.yaml \
  --run-dir runs/real_video_colmap_docker
```

The supplied config is CPU mode. Setting `use_gpu: true` adds `docker run --gpus all` and requires
a compatible NVIDIA container runtime and CUDA-capable image. Docker mounts the run read-only at
`/run` and the attempt COLMAP workspace writable at `/workspace`. No input is copied into the
image.

## Native workspace and typed output

Each attempt initially writes:

```text
work/camera_recovery/attempt_N/camera/
├── reconstruction.json
├── diagnostics.json
└── colmap/
    ├── database.db
    ├── logs/
    ├── sparse/<candidate>/
    └── workspace_manifest.json
```

Only a validated complete set is promoted to `camera/`. Failure leaves the attempt intact and the
previous canonical `camera/` unchanged.

The internal reader handles the native little-endian camera, image, 2D observation, sparse point,
and track records. It supports `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, and
`OPENCV`, preserving radial/tangential coefficients. Unsupported camera names and multiple used
camera IDs fail clearly without deleting raw files.

The implementation follows COLMAP's published output format and camera-model table. It is an
independent reader rather than copied helper code; COLMAP itself is BSD-licensed and remains a
separate executable.

COLMAP can emit `sparse/0`, `sparse/1`, etc. Recon2Sim ranks them by registered frame count,
registration ratio, sparse points, mean track length, reprojection error, and stable ID. The best
must meet `min_registered_frames` and `min_registration_ratio`. All candidates and rejection
reasons appear in `camera/diagnostics.json`.

## Coordinate conversion and scale

Given COLMAP's world-to-camera rotation `R` and translation `t`:

```text
R_world_from_camera = R^T
t_world_from_camera = -R^T t
```

COLMAP `wxyz` quaternions are normalized before matrix conversion; Recon2Sim output is normalized
`xyzw`. Unit tests cover identity, translation, 90° rotation, inversion, normalization, and
round-trip matrix transpose.

Monocular SfM has an unknown global similarity transform. Phase 1 does not fabricate meters or an
up direction. It emits `scale_ambiguous`, `arbitrary_scale`, `colmap_unaligned`, and
`colmap_arbitrary`. A trajectory export preserves these labels.

## Inspecting a run

```bash
uv run recon2sim ingest inspect runs/real_video_colmap
uv run recon2sim camera inspect runs/real_video_colmap
uv run recon2sim camera colmap-stats runs/real_video_colmap
uv run recon2sim camera export-trajectory \
  runs/real_video_colmap --output trajectory.json
```

Camera inspection reports input/selected/registered frames, registration ratio, camera model,
intrinsics/distortion, point count, scale/world status, and warnings.

## Failure recovery and troubleshooting

- **FFmpeg/FFprobe missing:** install both binaries or set `ffmpeg_executable` /
  `ffprobe_executable`; rerun the healthcheck.
- **Unreadable input/no frames:** inspect FFprobe and extraction logs in the failed attempt. Check
  codec support and the input-mode decision.
- **All frames rejected:** inspect the attempt's `inputs/frame_qa.json`; loosen thresholds based on
  measured values, not guesses.
- **Feature/matching failure:** inspect the named subcommand stderr; use CPU mode if GPU SIFT is not
  supported.
- **No sparse model/low registration:** improve capture overlap/parallax/sharpness; consider
  exhaustive matching for unordered photos or adjust sequential overlap. Lower thresholds only
  when accepting the resulting quality risk.
- **Unsupported/multiple cameras:** select a supported camera model and keep one physical camera
  per Phase 1 run.
- **Timeout:** increase the stage timeout only after checking whether the process is progressing.
- **Stale or malformed files:** Recon2Sim rejects them in the attempt; canonical success remains
  untouched. Do not copy partial binaries manually into `camera/`.

Resume after correcting input/config:

```bash
uv run recon2sim run \
  --input examples/real_video \
  --config configs/colmap.yaml \
  --run-dir runs/real_video_colmap \
  --resume
```

Changing any input image/video byte invalidates ingest and its dependents. Editing an upstream
artifact also invalidates the producer/dependent chain through recorded hashes and execution
signatures.

Optional installed-tool probes are excluded from the default CPU gate. Run them explicitly with:

```bash
uv run pytest -o addopts= -m integration
```
