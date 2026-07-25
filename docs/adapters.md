# Adapters

Adapters isolate implementation environments from the core package. Phase 1 adds dedicated
FFmpeg ingest and COLMAP camera recovery adapters; neither external tool is imported as Python.

## Contract

An adapter exposes:

- `name` and `version` for signatures and provenance;
- `healthcheck()`;
- `prepare(context)`;
- `expected_outputs(context)` with path, artifact type, media type, source type, validation mode,
  schema identifier, and optional Pydantic model;
- `run(context)` with additional dynamic outputs and metrics.

The runner validates and hashes the union of declared and dynamic outputs. Missing files, malformed
JSON, invalid Scene IR, invalid PNG/OBJ content, and conflicting declarations fail the attempt.

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

## Future real adapters

A real adapter must document and test inputs, outputs, schema IDs, command template, environment
allowlist, timeout, retries, GPU metadata, healthcheck, provenance, coordinate conversion, and
failure artifacts. It must emit the existing typed contract before any downstream stage accepts
its work.

The next adapter should be SAM 3 segmentation/tracking only. GenRecon, SceneSmith, Blender, and
simulator integrations remain later phases.
