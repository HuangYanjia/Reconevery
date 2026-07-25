# Architecture

Recon2Sim separates orchestration and semantic contracts from heavyweight reconstruction or
simulation software. Phase 1 runs FFmpeg and COLMAP out of process while retaining deterministic
mocks downstream, so mandatory tests require no external executable or GPU.

## Layers

1. Strict Pydantic models define canonical Scene IR and typed intermediate artifacts.
2. Adapters read declared upstream files and write declared outputs.
3. The runner validates the DAG, computes signatures, executes retries, validates outputs, hashes
   artifacts, and commits manifest state atomically.
4. The Typer CLI exposes run, resume, inspection, validation, cleanup, and adapter healthchecks.

```text
video or image directory
  -> FFmpeg/Pillow normalization + frame QA
  -> COLMAP features -> matching -> sparse mapping
  -> strict COLMAP binary parsing
  -> typed CameraReconstruction
  -> existing downstream mock stages
```

The configured DAG is:

```text
ingest -> camera_recovery -> segmentation_tracking -> object_reconstruction --+
                         \-> global_reconstruction ---------------------------+-> scene_ir_assembly
  -> scene_compilation -> validation -> export
```

Scene IR assembly reads the ingest manifest, camera reconstruction, tracks, global reconstruction
metadata, and per-track object reconstruction results. It does not synthesize a disconnected
second scene.

## Artifact roles

- `scene_ir/scene.json`: canonical cameras, frames, observations, objects, assets, relations,
  physics, confidence, provenance, and coordinate convention.
- `reconstruction/**/*.obj`: visual or collision meshes referenced by Scene IR; never canonical
  semantic state by themselves.
- `compiled/scene_package`: derived mock compiler package.
- simulator outputs: a future derived product. Phase 0.1 emits none and records an empty list.

Every manifest `ArtifactRecord` contains relative path, artifact type, media type, SHA-256, byte
size, producer stage and adapter, source type, and schema identifier when applicable.

## Graph validation

Before execution the runner rejects unknown dependencies, cycles with the cycle path, unknown
`from-stage` or `until-stage` values, reversed ranges, and enabled stages whose dependency is
disabled. Reusing artifacts from a disabled dependency requires both intact successful artifacts
and the explicit `allow_existing_artifacts_for_disabled_dependencies` configuration flag.

Starting from a later stage is allowed only when every omitted dependency has an intact successful
artifact set in the run directory.

## Cache and resume

A stage signature includes:

- full stage and adapter configuration;
- adapter name and version;
- pipeline seed;
- recursive path, size, and SHA-256 snapshots for root input files;
- current hashes of direct dependency artifacts;
- upstream execution signatures.

A cache hit requires the signature and every recorded output hash to match. It leaves the stage
`succeeded` and sets `last_execution=cache_hit`. If an output was edited, the producer reruns and
receives a new execution signature, which invalidates all dependents even if deterministic output
bytes are restored.

## Execution safety

Retries are `retries + 1` total attempts. Each attempt writes only inside
`work/<stage>/attempt_<N>`. Previously successful transitive-ancestor artifacts are copied into
that workspace, the adapter runs there, and required outputs are validated before file-by-file
atomic promotion. Stale canonical files cannot satisfy validation. A failed or interrupted
attempt keeps its workspace and cannot overwrite the previous successful output set.

JSON is validated with a Pydantic model (or a generic typed JSON-object contract for command
adapters), PNGs and OBJs receive format checks, and no stage succeeds solely because a process
returned zero.

Command adapters receive only explicitly allowlisted environment variables. They capture stdout,
stderr, return code, duration, and timeout state in per-attempt files. Timed-out process groups are
terminated and failed attempt files remain available for debugging.

## Coordinate convention

The world frame is right-handed with +X forward, +Y left, and +Z up. Metric or externally scaled
distances use meters, quaternions use `xyzw`, and `transform_world_from_camera` maps camera
coordinates into world coordinates.
COLMAP `qvec,tvec` world-to-camera transforms are inverted to produce world-from-camera, and
`wxyz` is normalized and reordered to `xyzw`. Monocular COLMAP world scale has gauge freedom, so
Phase 1 marks it `scale_ambiguous`.
