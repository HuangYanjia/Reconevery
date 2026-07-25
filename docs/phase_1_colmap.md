# Phase 1: Real Ingest and COLMAP Camera Recovery

Phase 1 makes frame ingest and sparse camera recovery real while preserving the typed filesystem
boundary. FFmpeg and COLMAP execute as subprocesses because their system, CUDA, and native-library
requirements must not enter the lightweight core Python environment.

## Input and capture

Pass a video file, a directory containing exactly one supported video, or a directory of JPEG/PNG
images. If a directory contains both video and images, set `input_mode` explicitly.

Capture recommendations:

- move the camera through space rather than rotating in one location;
- maintain strong overlap between adjacent views;
- avoid motion blur and abrupt exposure changes;
- avoid large moving objects;
- avoid transparent and reflective dominance;
- capture multiple heights and viewing angles;
- do not assume monocular scale is metric.

## Local execution

Install FFmpeg/FFprobe and COLMAP on `PATH`, then check them:

```bash
uv run recon2sim adapters healthcheck
ffmpeg -version
ffprobe -version
colmap -h
```

GPU SIFT:

```bash
uv run recon2sim run \
  --input examples/real_video \
  --config configs/colmap.yaml \
  --run-dir runs/real_video_colmap
```

CPU SIFT:

```bash
uv run recon2sim run \
  --input examples/real_video \
  --config configs/colmap_cpu.yaml \
  --run-dir runs/real_video_colmap_cpu
```

## Docker execution

The Dockerfile builds COLMAP 3.11.1 without CUDA and includes FFmpeg:

```bash
docker build -t reconevery/colmap:phase1 docker/colmap
docker version
docker image inspect reconevery/colmap:phase1
uv run recon2sim run \
  --input examples/real_video \
  --config configs/colmap_docker.example.yaml \
  --run-dir runs/real_video_colmap_docker
```

Docker mode mounts only the attempt workspace at `/workspace`; it embeds no user data or model
checkpoints. Health fails unless both Docker daemon access and image inspection succeed.

## Extraction and frame QA

Video mode runs FFprobe for stream metadata, hashes the original video, then uses an FFmpeg `fps`
filter, optional aspect-preserving max-edge resize, a frame limit, and six-digit names. Image mode
decodes JPEG/PNG with Pillow, writes deterministic RGB PNG, uses path order, and reads EXIF capture
time when present.

Each candidate receives:

- Laplacian variance as a sharpness score;
- mean grayscale brightness;
- grayscale intensity variance;
- normalized mean difference between 32x32 grayscale signatures;
- selected/rejected status and a rejection reason.

Defaults reject exact duplicates and extreme brightness, while `blur_threshold=0` disables blur
rejection. These are conservative defaults, not universal thresholds. Tune them for the camera,
motion, lighting, and subject. Rejected frames remain in `inputs/rejected_frames/` when
`keep_rejected_frames=true`; only selected frames enter the manifest.

## COLMAP stages

Commands are argument arrays with `shell=False`:

```text
colmap feature_extractor --database_path camera/colmap/database.db --image_path frames ...
colmap sequential_matcher --database_path camera/colmap/database.db ...
colmap mapper --database_path camera/colmap/database.db --image_path frames \
  --output_path camera/colmap/sparse ...
```

Sequential matching with overlap 10 is the video default. Use exhaustive matching for a small,
unordered image collection. `use_gpu=false` selects CPU SIFT extraction and matching.

The parser reads `cameras.bin`, `images.bin`, and `points3D.bin`; it does not scrape console output.
Supported models are `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, and `OPENCV`.
Distortion is retained. Phase 1 rejects multi-camera and unsupported models explicitly while
preserving the raw workspace.

## Coordinates and scale

COLMAP stores a world-to-camera transform:

```text
qvec(wxyz), tvec
  -> normalized rotation_world_to_camera
  -> transpose rotation and compute -R^T t
  -> transform_world_from_camera
  -> quaternion xyzw
```

Recon2Sim declares a right-handed `+X forward, +Y left, +Z up` convention and world-from-camera
transforms. Sparse monocular reconstruction still has arbitrary global gauge and scale. Phase 1
writes `scale_status="scale_ambiguous"` and never claims raw translations are metric. External
scale calibration should later change the status to `externally_scaled`.

## Selection, confidence, and diagnostics

Sparse models rank by registered frame count, sparse point count, lower mean reprojection error,
and deterministic model ID. Each candidate and rejection reason appears in
`camera/diagnostics.json`. Threshold failures name the configured minimum frame count and ratio.

Camera confidence is:

```text
0.60 * registration_ratio
+ 0.15 * min(registered_frames / 20, 1)
+ 0.15 * min(log10(sparse_points + 1) / 4, 1)
+ 0.10 / (1 + average_reprojection_error)
```

The reprojection term is zero when unavailable. Inspect and export with:

```bash
uv run recon2sim ingest inspect runs/real_video_colmap
uv run recon2sim camera inspect runs/real_video_colmap
uv run recon2sim camera colmap-stats runs/real_video_colmap
uv run recon2sim camera export-trajectory \
  runs/real_video_colmap --output trajectory.json
```

## Failure recovery

Every attempt writes to `work/<stage>/attempt_<N>`. The runner validates temporary outputs before
atomic file promotion. Stale files cannot satisfy a new attempt, and a failed attempt cannot
replace the previous successful result. Partial databases, sparse outputs, exact commands, stdout,
stderr, diagnostics, and the failed subcommand remain in the attempt workspace.

Troubleshooting order:

1. Run `recon2sim adapters healthcheck` for missing tools or Docker image.
2. Inspect `inputs/frame_qa.json` for all-frame rejection.
3. Inspect `camera/colmap/logs/` or the failed attempt workspace.
4. Inspect `camera/diagnostics.json` for thresholds and model selection.
5. Reduce frame rate/image size for resource limits, or improve capture overlap for registration.

Timeouts terminate the process group. Rerun after changing configuration; `--resume` reuses only
successful stages whose signature and output hashes remain current.
