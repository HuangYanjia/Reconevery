# Adapters

Adapters isolate implementation environments from the core package. Phase 0.1 contains only
deterministic mock adapters plus generic subprocess and future Docker command boundaries. No
heavyweight reconstruction package is imported or executed.

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
run in the run directory, with a new process group, configured timeout, and configured retry count.
Each attempt preserves separate stdout, stderr, and command-result JSON files. Timeout handling
sends termination to the process group and escalates to kill if necessary.

`AdapterConfig.expected_outputs` declares command output paths and validation modes. A zero return
code with missing or invalid output is a failed attempt.

## Future real adapters

A real adapter must document and test inputs, outputs, schema IDs, command template, environment
allowlist, timeout, retries, GPU metadata, healthcheck, provenance, coordinate conversion, and
failure artifacts. It must emit the existing typed contract before any downstream stage accepts
its work.

The first real adapter should be COLMAP camera recovery only: consume `inputs/manifest.json` and
`frames/*.png`, run out of process, and emit `camera/reconstruction.json`. SAM 3, GenRecon,
SceneSmith, Blender, and simulator integrations remain later phases.
