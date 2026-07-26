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
expected hash, source artifact or approved external path, copy/reflink mode, and whether the
producer execution signature participates in downstream cache identity. The latter defaults to
`true`; set it to `false` only when the declared file bytes are the complete semantic input.
Older adapters temporarily retain full-ancestor materialization.

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

## Object surface lifting

`object_surface_lifting` consumes typed camera reconstruction, the minimal selected-model COLMAP
text package, canonical object tracks and their referenced masks, global reconstruction metadata,
and an attempt-local reflink/copy of the global PLY. The unused GLB is validated by the core but
is not passed to the worker. It does not consume SAM raw output, COLMAP databases/models/logs,
GenRecon chunk tensors, or any checkpoint. Local and Docker workers receive only the stage
attempt root. Execution modes are `local_worker`, `docker`, and `fake_worker`.

The isolated worker undistorts masks for `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`,
and `OPENCV`, then uses exact homogeneous clip coordinates and nvdiffrast to recover
nearest-visible original face IDs. Conservative culling drops a triangle only when all vertices
are outside one clip half-space. Both `exact_face_vote_v1` and
`surface_sample_fusion_v2` are measured in each run. Canonical outputs contain compact
little-endian face-ID arrays, true geometric component areas, seam diagnostics, spatial sample
support, alignment diagnostics, partial PLY assets, reprojection metrics, and previews. The
adapter independently rejects hash mismatches, face bounds, non-finite/empty surfaces, path
escape, coordinate changes, malformed worker output, OOM, timeout, and unsupported camera
models.

`phase4_consistency_validation` checks lineage, registered-frame eligibility, compact array
integrity, original-face mapping, raw COLMAP semantics, selective materialization, absence of
collisions/completion claims, and Phase 4 Scene IR references.

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

## Camera-mesh alignment

`camera_mesh_alignment` supports `local_worker`, `docker`, and `fake_worker`. Its declared inputs
are the typed manifest/cameras, selected COLMAP text package, global metadata/worker manifest,
working and chunk transforms, camera debug record, and a reflink/copy of `mesh.ply`. The canonical
run is never mounted. The core imports no NumPy, SciPy, OpenCV, trimesh, PyTorch, CUDA, or
nvdiffrast.

Outputs under `reconstruction/alignment/` include the request, transform-chain audit, filtered
sparse observations, disjoint dataset split, candidates/iterations, accepted or rejected typed
alignment, per-camera/per-chunk diagnostics, and deterministic previews. `alignment.json` maps the
original Phase 3 mesh to the corrected arbitrary COLMAP frame. Cameras stay fixed.

`object_surface_lifting.alignment_policy` is `none`, `use_if_accepted`, or `require_accepted`.
When an accepted transform is available, the worker applies it to an in-memory vertex copy and
runs a second unaligned baseline for `object_lifting_comparison.json`. Rejected alignment retains
the Phase 4 path. `phase4_2_consistency_validation` rechecks hashes, held-out separation,
coordinate semantics, topology, face IDs, selective materialization, and capability boundaries.

## Future real adapters

A real adapter must document and test inputs, outputs, schema IDs, command template, environment
allowlist, timeout, retries, GPU metadata, healthcheck, provenance, coordinate conversion, and
failure artifacts. It must emit the existing typed contract before any downstream stage accepts
its work.

## Dense and measured geometry adapters

`dense_mvs` pins official COLMAP 4.0.4 and invokes `image_undistorter`,
`patch_match_stereo`, and `stereo_fusion` with argument arrays. It preserves official dense maps,
logs, and a typed reversible frame mapping. `measured_object_geometry` invokes an isolated
NumPy/OpenCV worker to map binary masks, backproject depth, validate samples across views, and fuse
surfels. Only declared attempt-local inputs are visible to either worker.

Automatic VLM inventory, hidden-surface completion, SceneSmith, Blender, and simulator
integrations remain later, separate phases.

Phase 5B adapters are `completion_evidence_package`,
`sam3d_object_candidates`, `trellis2_object_candidates`,
`measured_only_candidates`, `completion_candidate_registration`,
`completion_candidate_evaluation`, `completion_candidate_selection`, and
`phase5b_consistency_validation`. Heavy generation, registration, and rendering live
only under `workers/` or Docker.
