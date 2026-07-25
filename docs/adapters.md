# Adapters

Adapters isolate implementation environments from the core package. FFmpeg, COLMAP, and official
SAM code remain out of process; SAM and PyTorch are not core dependencies.

## Contract

An adapter exposes:

- `name` and `version` for signatures and provenance;
- `healthcheck()`;
- `prepare(context)`;
- `expected_outputs(context)` with path, artifact type, media type, source type, validation mode,
  schema identifier, and optional Pydantic model;
- `run(context)` with additional dynamic outputs and metrics.
- optional `required_inputs(context)` returning typed `InputSpec` records.

The runner validates and hashes the union of declared and dynamic outputs. Missing files, malformed
JSON, invalid Scene IR, invalid PNG/OBJ content, and conflicting declarations fail the attempt.
`InputSpec` declares a safe attempt-relative destination, artifact type, required/optional status,
expected hash, source artifact or approved external path, and copy/reflink mode. Older adapters
temporarily retain full-ancestor materialization.

## Mock stage contracts

| Stage | Reads | Writes |
| --- | --- | --- |
| ingest | input PNG directory | typed manifest and copied PNG frames |
| camera recovery | manifest and frames | typed intrinsics, poses, convention, confidence, provenance |
| segmentation/tracking | manifest and camera JSON | typed tracks, per-frame boxes, valid PNG masks |
| global reconstruction | camera JSON | valid floor OBJ and typed metadata |
| object reconstruction | track JSON | one typed result per track and visual/collision OBJs |
| Scene IR assembly | all upstream typed artifacts | validated canonical Scene IR |
| compilation | Scene IR | mock package JSON and derived mesh |
| validation | Scene IR and package | typed validation report |
| export | package and validation report | typed export manifest |

The cabinet reconstruction result contains body and drawer parts in one articulation. There is no
independent drawer track or object result.

## Command adapters

`AdapterConfig.env` is an allowlist; the child receives no other environment variables. Commands
run in an isolated attempt workspace, with a new process group, configured timeout, and configured
retry count.
Each attempt preserves separate stdout, stderr, and command-result JSON files. Timeout handling
sends termination to the process group and escalates to kill if necessary.

`AdapterConfig.expected_outputs` declares command output paths and validation modes. A zero return
code with missing or invalid output is a failed attempt.

## FFmpeg ingest

`ffmpeg_ingest` detects `video` or `image_directory` input. Video mode checks FFmpeg and FFprobe,
records versions and exact arguments, extracts `frame_%06d.png`, and retains logs. Image mode uses
Pillow to validate JPEG/PNG input, apply `ImageOps.exif_transpose`, resize without changing aspect
ratio, normalize RGB PNG bytes, preserve path order, and use EXIF timestamps when present.

Both modes emit `inputs/manifest.json` and `inputs/frame_qa.json`. QA uses Laplacian variance,
brightness mean, grayscale variance, and normalized 32x32 grayscale difference. Defaults are
conservative and not universal quality thresholds.

## COLMAP camera recovery

`colmap_camera_recovery` consumes the normalized manifest and frames. It runs explicit argument
lists for `feature_extractor`, `sequential_matcher` or `exhaustive_matcher`, and `mapper`.
`use_gpu` controls SIFT extraction and matching flags.

The internal binary parser supports `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, and
`OPENCV`, retains distortion, and rejects multi-camera or unsupported results. Models rank by
registered frames, sparse points, reprojection error, then deterministic ID.

Local health checks run the configured `colmap -h`. Docker mode runs `docker version`,
`docker image inspect <image>`, and an in-container `colmap -h`; use `--config` on the CLI to
check the selected executable or image. Linux containers run with the host UID:GID by default,
and `docker_user` can override it. The workspace manifest records the inspected image identifier
when available. Docker Phase 1 is CPU-only and rejects `use_gpu=true`. Raw databases, every sparse
candidate, logs, commands, model diagnostics, and typed reconstruction remain under `camera/`.

## SAM 3 segmentation and tracking

`sam3_segmentation_tracking` reads only:

- `inputs/manifest.json` and `inputs/frame_qa.json`;
- selected `frames/*.png`;
- `camera/reconstruction.json`;
- the configured prompt YAML and optional seed masks.

It does not materialize `camera/colmap/database.db`, sparse binaries, COLMAP logs, camera
diagnostics, or unrelated downstream artifacts. The adapter validates prompts and anchors, writes
`observations/sam3_request.json`, invokes `local_worker`, `docker`, or `fake_worker`, validates raw
output, filters tracks, creates stable IDs, writes canonical binary masks, and renders previews.

The real worker is pinned to official Meta code commit
`46957e47805eaa273f4aa7bbbd25a88bca9108ce` and official `facebook/sam3.1` revision
`daa63191845a41281374e725f4c9e51c7a824460`. Local health checks verify the configured Python,
worker import, exact official commit, PyTorch/torchvision/CUDA versions, GPU, precision, and
checkpoint access. Docker health checks also verify the daemon, configured image ID, NVIDIA GPU
access, mounted cache/checkpoint, UID/GID execution, and the in-container worker health command.

The pinned official public video predictor supports text and box prompts, point initialization or
refinement, Object Multiplex joint instances, and forward/backward propagation. Its public request
API does not expose mask seeds. Reconevery validates the mask-seed contract but the real worker
returns an explicit unsupported-backend error rather than importing private SAM internals.

Canonicalization filters invalid dimensions, empty/non-binary masks, invalid or inconsistent
boxes, invalid scores, short/low-coverage tracks, duplicate object/frame observations, and
same-prompt or configured-synonym duplicates. Different semantic labels are never merged solely
because their masks overlap. Valid no-object results produce `tracks: []`.

## GenRecon global reconstruction

`genrecon_camera_package` reads only the selected COLMAP binary triplet and emits deterministic
text plus a typed manifest. `genrecon_global_reconstruction` supports `local_worker`, `docker`,
and `fake_worker`. The core imports no GenRecon, PyTorch, NumPy, trimesh, Open3D, or CUDA package.

The official worker verifies commit `eaf1468118d20469d17079a4a19737297d2ef87b`, the recursive
Eigen submodule, three official TUM checkpoint hashes, Python/PyTorch/torchvision/CUDA versions,
and required CUDA extensions. It invokes the official reconstruction and GLB scripts with
argument arrays and validates files independently of return codes.

Checkpoint `InputSpec` entries use `reference_only`: they contribute bytes and hashes to cache
signatures but are not copied. The attempt contains only registered frames and the normalized
camera package. See `docs/phase_3_genrecon.md` for commands and failure handling.

## Future real adapters

A real adapter must document and test inputs, outputs, schema IDs, command template, environment
allowlist, timeout, retries, GPU metadata, healthcheck, provenance, coordinate conversion, and
failure artifacts. It must emit the existing typed contract before any downstream stage accepts
its work.

Automatic VLM scene inventory, object-level reconstruction/fusion, SceneSmith, Blender, and
simulator integrations remain later, separate phases.
