# Architecture

Recon2Sim separates orchestration and semantic contracts from heavyweight reconstruction or
simulation software. Phase 2 runs FFmpeg, COLMAP, and SAM workers out of process while retaining
deterministic mocks downstream, so mandatory tests require no external executable or GPU.

## Layers

1. Strict Pydantic models define canonical Scene IR and typed intermediate artifacts.
2. Adapters read declared upstream files and write declared outputs. Heavy adapters may use an
   optional typed `required_inputs()` declaration for selective materialization.
3. The runner validates the DAG, computes signatures, executes retries, validates outputs, hashes
   artifacts, and commits manifest state atomically.
4. The Typer CLI exposes run, resume, inspection, validation, cleanup, and adapter healthchecks.

```text
video or image directory
  -> FFmpeg/Pillow normalization + frame QA
  -> COLMAP features -> matching -> sparse mapping
  -> strict COLMAP binary parsing
  -> typed raw-gauge CameraReconstruction
  -> explicit text/box/point/mask prompt manifest
  -> isolated official SAM 3.1 worker
  -> canonical binary masks + deterministic object tracks
```

Production Phase 1 configs stop after camera recovery, and Phase 2 configs stop after segmentation.
Explicit `*_with_mock_downstream.yaml` integration demos continue through:

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

SAM's canonical artifact is `observations/object_tracks.json`. Raw model IDs and worker output
remain under `observations/raw/`; only validated `observations/masks/<object>/<frame>.png` masks
and typed observations are downstream semantic inputs.

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
- for legacy adapters, current hashes and execution signatures of direct dependencies;
- for selective adapters, only the ancestor or approved external files declared by
  `InputSpec`, including destination, source, size, and SHA-256;
- the producer execution signature for a declared input when
  `include_producer_signature=true`.

Producer signatures default to enabled so a semantic upstream configuration change
still invalidates dependent stages even when a fake or degenerate backend happens to
reproduce identical bytes. Evidence-split stages may disable that flag for an input
whose exact bytes are the complete contract. Phase 5B uses this content-only mode to
keep held-out evidence out of generation and registration signatures. Cache migration
from the legacy signature layout is allowed only when stage configuration, adapter
identity, seed, and every newly declared input path/hash/size match exactly.

A cache hit requires the signature and every recorded output hash to match. It leaves the stage
`succeeded` and sets `last_execution=cache_hit`. Execution signatures are derived from stage
inputs and output bytes; `execution_count` is audit metadata. A forced deterministic rerun that
reproduces identical bytes therefore does not invalidate downstream stages.

## Execution safety

Retries are `retries + 1` total attempts. Each attempt writes only inside
`work/<stage>/attempt_<N>`. Legacy adapters receive successful ancestor artifacts. Selective
adapters receive only typed `InputSpec` files resolved from ancestors or approved external
configuration. Paths and hashes are checked, reflink is preferred with copy fallback, writable
symlinks are not used, and canonical upstream hashes are checked after execution. The attempt
manifest records every materialized input.

Required outputs are validated before transactional promotion. Promotion backs up the complete
stage-owned output set and rolls it back on any mid-promotion failure. Stale canonical files
cannot satisfy validation. A failed or interrupted attempt keeps its workspace and cannot
overwrite the previous successful output set.

JSON is validated with a Pydantic model (or a generic typed JSON-object contract for command
adapters), PNGs and OBJs receive format checks, and no stage succeeds solely because a process
returned zero.

Command adapters receive only explicitly allowlisted environment variables. They capture stdout,
stderr, return code, duration, and timeout state in per-attempt files. Timed-out or interrupted
process groups receive TERM followed by KILL if the grace period expires. `KeyboardInterrupt` and
`SystemExit` are marked interrupted and re-raised without retry.

The SAM core adapter contains no model imports. It writes a typed request and launches either the
separate local worker, an NVIDIA Docker worker, or the deterministic fake worker. Worker logs are
credential-redacted before retention. The official checkpoint is never part of a request or
image, and credentials are passed only through allowlisted environment names.

GenRecon is a sibling branch, not a SAM dependent. `genrecon_camera_package` exports one selected
COLMAP model as deterministic text. `genrecon_global_reconstruction` declares only that package,
registered normalized frames, typed camera metadata, and read-only checkpoint references. The
runner hashes `reference_only` external inputs for signatures without copying multi-gigabyte
weights into attempts.

The observation-lineage digest hashes ordered frame ID/path/content tuples. It is carried by
ingest, camera, SAM, GenRecon, and consistency artifacts. The Phase 3 validator also audits
attempt materialization lists and coordinate metadata.

Object lifting is a downstream fusion stage over canonical camera, mask, and global mesh
artifacts. The runner reflinks or copies the global PLY into the isolated attempt; the unused GLB
is not a worker input. Local and Docker workers see only that attempt root, never the canonical
run. Retained mask PNGs and the minimal COLMAP text package are the only other large evidence
inputs. The worker rasterizes nearest-visible global faces, compares exact-face and spatial
surface-sample evidence, writes compact original face IDs and partial PLY assets, and returns
typed evidence. The core rechecks all hashes, face bounds, surface counts, coordinate metadata,
and Scene IR references before transactional promotion. Its enriched IR is stage-owned at
`scene_ir/phase4_scene.json`, avoiding cross-stage output ownership and preserving Phase 3 cache
validity.

## Coordinate convention

COLMAP `qvec,tvec` world-to-camera transforms are inverted to produce world-from-camera, and
`wxyz` is normalized and reordered to `xyzw`. That inversion does not align or scale the world.
Phase 1 metadata declares an arbitrary, unoriented COLMAP world; OpenCV camera axes
X-right/Y-down/Z-forward; arbitrary linear units; and ambiguous scale.

The canonical robot frame is right-handed +X forward, +Y left, +Z up in meters. A later alignment
stage must estimate gravity/orientation and external scale before changing the metadata and
translations to that canonical contract.

GenRecon may use a recorded invertible PCA working transform because its official chunker assumes
axis-aligned bounds. This remains unoriented preprocessing. Canonical PLY/GLB outputs are
transformed back into the original COLMAP world before promotion.

## Phase 4.2 alignment branch

`camera_mesh_alignment` is a sibling of segmentation: it depends only on camera recovery, the
minimal GenRecon camera package, and global reconstruction. Prompt changes therefore do not
invalidate it. The isolated worker receives sparse COLMAP text, transform-chain diagnostics, and
an attempt-local reflink/copy of the global PLY. It never receives SAM output, COLMAP databases,
GenRecon checkpoints, or raw generation tensors.

The worker audits coordinate round trips first, then evaluates identity and bounded Sim(3)
initializations. Training and validation frame/point IDs are disjoint. The core independently
checks held-out gates and publishes the original mesh plus a typed root transform; it never bakes
the PBR GLB by default. Object lifting consumes the transform with `use_if_accepted`, keeps
original face IDs, and records an unaligned/aligned comparison. The final validator writes
`scene_ir/phase4_2_scene.json` without mutating Phase 3 or Phase 4 source artifacts.

## Phase 5A measured branch

`camera_recovery -> dense_mvs` and
`segmentation_tracking + dense_mvs -> measured_object_geometry`. Dense MVS consumes only the
selected sparse model and registered normalized frames. Measured extraction consumes canonical
masks and dense maps, never GenRecon. Prompt changes rerun SAM and measured extraction but not
PatchMatch; GenRecon changes do not invalidate canonical measured geometry.

## Phase 5B completion branches

`completion_evidence_package` produces eligibility, disjoint evidence splits, anchor
crops, and fitting-only points. SAM 3D Objects and TRELLIS.2 generation run in
parallel. Registration consumes fitting evidence; only evaluation sees held-out masks
and dense depth. Lightweight selection applies gates, Pareto ranking, and license
policy. GenRecon is not a canonical dependency.

## Phase 5C multi-state branches

`articulation_capture` imports only typed artifacts from independently valid Phase 5A
state runs. `articulation_state_alignment` sees all state base geometry but explicitly
excludes movable parts. `articulation_motion` and `articulation_fitting` receive only
the generation/fitting state subset. Held-out state geometry is materialized only for
`articulation_evaluation`. Stable part IDs map to state-local SAM track IDs. The
selection boundary references exact candidate, assignment, fitted-model, and
evaluation hashes; Scene IR consumes the fitted base Sim(3) and refined joints.

ArtVIP and PartNet retrieval consume immutable local indices without network access.
Official Particulate is a parallel research candidate source. Selection retains the
measured links and adds validated visual links separately. The canonical result is a
typed kinematic bundle plus `scene_ir/phase5c_scene.json`, not a simulation asset.
# Phase 5C.2 identity boundary

Articulated visual assets carry an explicit asset-to-candidate-base transform.
The worker, selection artifact, and Scene IR consume the same hashed
representation. Held-out evaluation records immutable per-view camera, depth,
mask, link-coverage, and render identities.

# Phase 5C.3 reference-world boundary

Original measured anchors are immutable `reference_world` evidence on the object,
while articulated links contain only candidate-base or link-local visuals. Scene IR
and the typed kinematic bundle reference dedicated selected records by exact file
hash, preventing nested-record identity ambiguity and measured double transforms.

## Phase 6A canonical wrapper

Phase 6A adds:

```text
calibration_evidence
  -> world_calibration
  -> canonical_scene_wrapper
  -> phase6a_consistency_validation
```

The evidence stage materializes only declared camera, scene, image, mask, depth,
landmark, and sensor inputs. The numerical worker sees only its attempt root.
Candidate fitting and selection use fitting evidence; held-out evidence is
acceptance-only.

The canonical scene is a derived wrapper. Source COLMAP cameras, dense geometry,
measured assets, completion assets, and articulation artifacts retain their bytes.
Articulated object roots receive the world composition; link-local geometry and
joint-local axes, pivots, and q remain unchanged. A typed per-joint scale maps
object-local prismatic q to meters exactly once. Partial or rejected calibration
leaves the source coordinate convention unchanged.

## Phase 6B layered assembly

Phase 6B adds a calibration-optional, non-destructive branch:

```text
scene_assembly_inputs
  -> scene_assembly_plan
  -> layered_scene_bundle
  -> assembly_previews
  -> phase6b_consistency_validation
```

`scene_assembly_inputs` promotes only typed Scene IR, calibration, geometry,
selection, evaluation, license, camera, and lineage records. Its source adapter
derives candidate, representation, validation, license, calibration, and alignment
semantics from the exact upstream files; local assertions cannot override them. The
plan rejects unconnected lineages, resolves one of four world modes, and constructs
explicit asset-native, object, source-world, and assembly-world transforms.

The bundle stage retains source geometry and measured anchors under
`layered_no_carve_v1`. It emits separate research and deployment-eligible JSON
bundles with independent object decisions, a simulator-neutral compiler input
manifest with separate research/deployment views, aggregate per-object overlap
diagnostics, and a derived Scene IR. Heavy geometry loading and diagnostic GLB
rendering remain in `workers/scene_assembly`; preview changes do not invalidate the
deterministic assembly plan.
