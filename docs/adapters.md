# Adapters

Adapters isolate implementation environments from the core package. They expose `name`,
`version`, `healthcheck`, `prepare`, declared `expected_outputs`, and `run`. The runner validates
and hashes the union of declared and dynamic outputs inside the current attempt workspace.

Missing files, invalid typed JSON, malformed PNG/OBJ data, conflicting output declarations, and
zero-return processes that produced no new output all fail the attempt.

## Real ingest: `ffmpeg_ingest`

Input modes are `auto`, `video`, `image_directory`, and an explicit fixture-oriented `mock` mode.
Auto mode accepts one supported video or a deterministic recursive set of `.jpg`, `.jpeg`, and
`.png` files; mixed image/video inputs and multiple videos are rejected.

Video mode checks `ffmpeg -version` and `ffprobe -version`, probes JSON stream metadata, then runs
an argument-list equivalent of:

```text
ffmpeg -nostdin -hide_banner -loglevel info -n -i <video>
  -vf fps=<target>[,select=...][,scale=...]
  -frames:v <max> -start_number 0 -fps_mode vfr
  <attempt>/raw_frames/frame_%06d.png
```

The exact command, detected versions, probe payload, original video SHA-256, and logs are
retained. `-n` plus a fresh attempt workspace prevents overwrite.

Image-directory mode uses Pillow only: EXIF orientation is applied, RGB is normalized to PNG,
aspect ratio is preserved during optional resize, source order is path-sorted, and EXIF capture
times are used as relative timestamps when available. Unreadable supported files fail with their
path.

Frame QA downsamples grayscale data and computes:

- mean brightness;
- grayscale intensity variance;
- variance of a discrete Laplacian as a sharpness/blur score;
- similarity to the last selected frame from 16×16 mean absolute pixel difference;
- selected/rejected state and an explicit reason.

Defaults (`blur_threshold=0`, `duplicate_threshold=0.995`, brightness 5–250) are conservative
engineering defaults, not universal quality criteria. Tune them per capture. Selected frames go
to `frames/`; optionally retained rejections go to `diagnostics/rejected_frames/`. The typed
`inputs/frame_qa.json` covers every extracted candidate; `inputs/manifest.json` lists selected
frames only.

## COLMAP: `colmap_camera_recovery`

The adapter consumes `inputs/manifest.json` and normalized `frames/*.png`. Core Python does not
import COLMAP or PyCOLMAP. Local mode invokes the configured executable; Docker mode invokes the
Docker CLI and explicit mounts.

The command sequence is:

```text
colmap -h
colmap feature_extractor --database_path ... --image_path ...
  --ImageReader.camera_model <model>
  --ImageReader.single_camera 0|1
  --FeatureExtraction.use_gpu 0|1
colmap sequential_matcher|exhaustive_matcher --database_path ...
  --FeatureMatching.use_gpu 0|1
colmap mapper --database_path ... --image_path ... --output_path ...
  --Mapper.multiple_models 0|1
```

Sequential matching additionally sets overlap and loop detection. Arguments are lists and are
never evaluated by a shell. `gpu_flag_style: auto` inspects both subcommand help texts and chooses
modern `FeatureExtraction`/`FeatureMatching` flags or legacy
`SiftExtraction`/`SiftMatching` flags. An unknown CLI fails instead of guessing. The provided
Docker base is from the COLMAP 4.0.4 release period.

The adapter parses `cameras.bin`, `images.bin`, and `points3D.bin` without console scraping.
Supported camera models are `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, and `OPENCV`;
all distortion coefficients are retained. Unsupported named models fail after preserving the raw
attempt. Phase 1 rejects multiple used camera IDs rather than merging incompatible intrinsics.

Sparse models are ranked by registered frames, registration ratio, sparse points, mean track
length, reprojection error, and stable model ID. Configured minimum count/ratio are enforced.
Diagnostics record the winner and rejected candidates. Confidence is a documented diagnostic
score: 55% registration ratio, 20% frame support, 15% sparse-point support, and 10% inverse mean
reprojection error. It is not a calibrated probability.

Outputs are:

| Path | Contract |
| --- | --- |
| `camera/reconstruction.json` | intrinsics/distortion, registered poses, registered/unregistered IDs, confidence, convention, scale, provenance |
| `camera/diagnostics.json` | model ranking, thresholds, points, ratio, warnings |
| `camera/colmap/database.db` | raw feature/match database |
| `camera/colmap/sparse/**` | raw native sparse models |
| `camera/colmap/logs/**` | stdout/stderr per subcommand |
| `camera/colmap/workspace_manifest.json` | tool version, exact commands, config, input hashes, selected model |

## Local and Docker healthchecks

`recon2sim adapters healthcheck` actually executes FFmpeg/FFprobe version checks and local
`colmap -h`. With a Docker-configured pipeline it executes `docker version` and
`docker image inspect <image>`. Results state available/unavailable, resolved executable/image,
detected output where possible, and installation/remediation guidance.

Docker runs mount the canonical run directory read-only at `/run` and only the attempt's raw
workspace writable at `/workspace`. GPU mode adds `--gpus all`; CPU mode does not. The image in
`docker/colmap/` is optional and contains no data or checkpoints.

## Process and failure behavior

`AdapterConfig.env` is an allowlist; no other parent variables reach a subprocess. Timeout or
interrupt terminates the process group, escalating to kill after a grace period. Nonzero return,
timeout, missing database/model, bad frame names, malformed binary records, unsupported cameras,
low registration, and missing output are actionable failures. The stage manifest records the
failed COLMAP subcommand when applicable.

Every retry has a new attempt directory. Only fully validated output is promoted, so a failed
attempt cannot overwrite the prior camera result and a successful no-op cannot reuse stale files.

## Mock downstream contracts

The downstream mock adapters consume real manifests/cameras without pretending their own geometry
is real. Segmentation emits typed tracks/boxes/masks, global reconstruction emits a valid mock OBJ,
object reconstruction emits one typed result per track, and Scene IR assembly connects all of
those artifacts. No SAM 3 or real object/global reconstruction is present in Phase 1.
