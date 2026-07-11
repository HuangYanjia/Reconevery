# Architecture

Recon2Sim keeps orchestration and typed semantic contracts lightweight while isolating native
reconstruction tools behind files and subprocesses. Phase 1 makes ingest and sparse camera
recovery real; every later stage remains a deterministic CPU mock.

## Layers

1. Strict Pydantic models define intermediate artifacts and canonical Scene IR.
2. Adapters consume committed upstream artifacts and write only to an isolated attempt workspace.
3. FFmpeg and COLMAP run out of process with explicit arguments, environment allowlists,
   timeouts, logs, and tool healthchecks.
4. The runner validates the DAG, computes signatures, retries attempts, validates/hash outputs,
   promotes a complete attempt, and writes manifest state atomically.
5. The Typer CLI exposes execution, inspection, camera export, Scene IR validation, cleanup, and
   adapter healthchecks.

```text
video / JPEG+PNG directory / mock fixture
                 |
                 v
       ffmpeg_ingest / mock_ingest
       manifest + frame QA + normalized PNGs
                 |
                 v
 colmap_camera_recovery / mock_camera_recovery
 raw COLMAP workspace + typed cameras + diagnostics
                 |
        +--------+-----------------------------+
        v                                      v
 mock segmentation/tracking          mock global reconstruction
        |                                      |
 mock object reconstruction                    |
        +------------------+-------------------+
                           v
                  Scene IR assembly
                           |
             mock compilation -> validation -> export
```

Scene assembly reads the committed ingest manifest, camera artifact, tracks, global metadata,
and one object result per track. It never constructs a disconnected second scene.

## Attempt isolation and promotion

For every execution or retry, `StageContext.output_path()` targets:

```text
work/<stage>/attempt_<monotonic-number>/
```

Upstream reads use canonical run paths. The runner validates every declared/dynamic output inside
the attempt, computes its hash, prepares replacement files, backs up affected canonical files,
and promotes the set. Promotion failure restores the backups. An adapter returning zero without
new outputs fails; a prior canonical output is never considered evidence for the current attempt.

Failed workspaces, including logs and partial COLMAP databases/models, remain under `work/`.
They do not alter the last successful canonical files.

## DAG, retries, and cache

Before execution the runner rejects unknown dependencies, dependency cycles (with their path),
unknown or reversed stage ranges, and enabled stages whose disabled/out-of-range dependency has
no explicitly allowed intact artifacts.

Retries are `retries + 1` isolated attempts. A signature includes full stage configuration,
adapter name/version, seed, recursive source path/size/SHA-256 data, direct upstream artifact
hashes, and upstream execution signatures. A cache hit additionally requires every recorded
canonical output hash to match. It leaves `status=succeeded` and records
`last_execution=cache_hit`.

## Artifact roles

- `inputs/manifest.json`: selected normalized frames and source/extraction provenance.
- `inputs/frame_qa.json`: metrics and selection reason for every extracted candidate.
- `camera/colmap/**`: preserved native COLMAP database, sparse models, and command logs.
- `camera/reconstruction.json`: normalized typed interface used downstream.
- `camera/diagnostics.json`: model ranking, registration quality, scale/alignment warnings.
- `scene_ir/scene.json`: canonical cameras, frames, observations, objects, assets, relations,
  physics, confidence, provenance, scale status, and coordinate status.
- `reconstruction/**/*.obj`: referenced mock visual/collision meshes, not canonical state.
- `compiled/scene_package`: derived mock compiler result; simulator outputs remain empty.

Every manifest `ArtifactRecord` stores run-relative path, artifact/media type, SHA-256, byte size,
producer stage/adapter, source type, and schema ID when applicable. Input-source references are
separately documented as relative to the configured input root.

## Native-tool boundary

FFmpeg performs video decoding; Pillow performs JPEG/PNG normalization and QA. COLMAP runs the
feature extractor, matcher, and mapper. Core code parses `cameras.bin`, `images.bin`, and
`points3D.bin` itself into immutable typed structures. Raw native output is retained for audit and
future conversion.

Local execution calls configured binaries directly. Docker execution verifies `docker version`
and `docker image inspect`, mounts the canonical run read-only at `/run`, mounts only the current
COLMAP workspace writable at `/workspace`, and does not embed user data or download models.

## Coordinates and scale

COLMAP emits world-to-camera `qvec` in `wxyz` order plus `tvec`. Recon2Sim normalizes the
quaternion, converts it to a rotation matrix, transposes/inverts the rigid transform, computes
`-R^T t`, and emits world-from-camera quaternions in `xyzw` order.

This inversion does not infer gravity, world axes, or scale. Monocular output uses
`scale_ambiguous`, `arbitrary_scale`, `colmap_unaligned`, and `colmap_arbitrary`. Only an explicit
future alignment/scaling stage may convert it to the canonical right-handed +X forward, +Y left,
+Z up, meter convention.
