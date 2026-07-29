# Recon2Sim Agent Guide

## Current scope

Phase 4.2 provides real FFmpeg ingest, out-of-process COLMAP recovery, isolated official SAM 3.1
tracking, isolated official GenRecon global reconstruction, and observation-grounded lifting
of SAM tracks onto visible global mesh faces plus held-out global Sim(3) audit. Do not import
COLMAP, SAM, GenRecon, NumPy,
OpenCV, trimesh, torch, nvdiffrast, CUDA runtimes, or checkpoint loaders into the core package.
Do not add VLM inventory, hidden-surface completion, SceneSmith, physics, simulators, or model
checkpoints. Heavy integrations belong behind filesystem adapters.

## Architecture map

- `src/recon2sim/ir`: canonical strict Pydantic Scene IR.
- `src/recon2sim/artifacts.py`: typed intermediate stage contracts.
- `src/recon2sim/adapters`: deterministic mocks plus isolated FFmpeg and COLMAP adapters.
- `src/recon2sim/colmap`: strict binary model parsing and coordinate conversion.
- `src/recon2sim/frame_qa.py`: deterministic CPU frame-quality metrics.
- `src/recon2sim/segmentation.py`: prompt validation, anchors, masks, IDs, QA, previews, and COCO.
- `src/recon2sim/genrecon.py`: lineage, COLMAP text export, lightweight mesh checks, and previews.
- `src/recon2sim/object_lifting.py`: compact face IDs, validation, summaries, and exports.
- `src/recon2sim/alignment.py`: lightweight Sim(3) validation, deterministic previews, and
  derived PLY export.
- `src/recon2sim/pipeline`: DAG validation, signatures, cache/resume, retries, and manifests.
- `workers/sam3`: isolated pinned official SAM runtime; never imported by core.
- `docker/sam3`: optional NVIDIA/CUDA worker image with no embedded checkpoint.
- `workers/genrecon`: isolated pinned official GenRecon runtime; never imported by core.
- `docker/genrecon`: optional CUDA 12.6/H100 worker image with no embedded checkpoint.
- `workers/object_lifting`: isolated distortion, rasterization, face evidence, and extraction.
- `docker/object-lifting`: optional CUDA/nvdiffrast image with no model checkpoints.
- `workers/alignment`: isolated transform audit, sparse-depth validation, and bounded Sim(3).
- `docker/alignment`: optional CUDA/nvdiffrast/SciPy image with no model checkpoints.
- `src/recon2sim/images.py`: dependency-free test PNG generation and validation.
- `src/recon2sim/storage`: atomic JSON, YAML, and text writes.
- `configs`, `schemas`, `examples`, `tests`, `docs`: reproducible Phase 0.1 assets.

## Required commands

```bash
uv sync --all-groups
uv run recon2sim --help
uv run recon2sim run --input examples/tabletop --config configs/mock.yaml --run-dir runs/tabletop_demo
uv run recon2sim validate-ir runs/tabletop_demo/scene_ir/scene.json
uv run recon2sim adapters healthcheck --config configs/colmap.yaml
uv run recon2sim adapters healthcheck --config configs/sam3_fake.yaml
uv run recon2sim run --input examples/tabletop --config configs/sam3_fake.yaml --run-dir runs/tabletop_sam3_fake
uv run recon2sim run --input examples/tabletop --config configs/phase3_e2e_fake.yaml --run-dir runs/phase3_e2e_fake
uv run recon2sim run --input examples/tabletop --config configs/phase4_e2e_fake.yaml --run-dir runs/phase4_e2e_fake
uv run recon2sim run --input examples/tabletop --config configs/phase4_2_e2e_fake.yaml --run-dir runs/phase4_2_e2e_fake
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

## Invariants

- Use real Pydantic v2, Typer, PyYAML, pytest, Ruff, and mypy; never add in-tree substitutes.
- Use explicit imports and explicit `__all__`; wildcard imports are prohibited.
- Every stage declares required outputs and validates them before success.
- Every attempt writes to `work/<stage>/attempt_<N>`; only validated outputs are promoted.
- Adapters may declare typed `InputSpec` inputs. Selective attempts materialize only those files,
  verify their hashes before copying, and recheck canonical upstream hashes after execution.
- Failed attempts retain their workspace and never replace previous canonical outputs.
- Store run artifact paths relative to the run directory and record hashes and producer metadata.
- Cache signatures include config, adapter name/version, seed, input bytes, upstream artifacts,
  and content-derived upstream execution signatures. `execution_count` is audit metadata only.
  Cache hits keep `status=succeeded`.
- Scene IR is canonical. Exported mesh files and simulator/compiler outputs are derived artifacts.
- The articulated cabinet is one object with `cabinet_body` and `cabinet_drawer` links. Do not add
  a second top-level drawer without designing and validating an explicit cross-reference model.
- Raw COLMAP coordinates use `world_frame=colmap_arbitrary`, `alignment_status=unoriented`,
  `camera_axes=x_right_y_down_z_forward`, `linear_units=arbitrary_units`, quaternion `xyzw`,
  world-from-camera transforms, and `scale_status=scale_ambiguous`. Only a later alignment and
  scaling stage may emit canonical +X-forward, +Y-left, +Z-up metric coordinates.
- SAM processes frames in ingest-manifest order. A missing camera pose does not invalidate a 2D
  mask, but must set `camera_pose_available=false`.
- Canonical masks are grayscale PNGs with exact values `0/255`, input-frame dimensions, nonzero
  area, mask-derived boxes, and run-relative deterministic paths.
- SAM raw IDs are diagnostic only. Canonical object IDs follow the documented semantic, prompt,
  first-frame, centroid, area, and raw-ID-final-tiebreak ordering.
- Credentials are environment-only and must never enter commands, requests, resolved config,
  provenance, diagnostics, or logs.
- SAM and GenRecon are parallel consumers of one ordered frame lineage. Prompt changes must not
  invalidate GenRecon when GenRecon consumes no SAM artifact.
- GenRecon receives only the selected COLMAP text package and registered normalized frames.
  Checkpoints are read-only references identified by SHA-256.
- A GenRecon PCA working transform is internal, reversible preprocessing. Final visual geometry
  must be returned to the original arbitrary, unoriented, scale-ambiguous COLMAP frame.
- Object lifting uses only registered cameras for 3D evidence. It preserves original global face
  IDs, allows cross-label overlap, resolves same-label instance conflicts deterministically, and
  never materializes raw COLMAP/SAM/GenRecon model workspaces.
- Object-lifting workers receive only their attempt root. The canonical run must not be mounted;
  the global mesh is an attempt-local reflink/copy and the unused GLB is not a worker input.
- Rasterization uses exact homogeneous clip coordinates. Never clamp nonpositive camera depth to
  make vertices renderable. Component area thresholds use true triangle surface area.
- Keep `exact_face_vote_v1` as a measured baseline beside `surface_sample_fusion_v2`. Surface
  samples may map only to original global face IDs and may not create bridging or hidden geometry.
- Phase 4 geometry is `partial_observation_supported`, `not_completed`, and `sim_ready=false`.
  It uses `GeometrySourceType.FUSED`, creates no collisions, and keeps unregistered SAM masks as
  valid 2D evidence only.
- Phase 4.2 never rewrites COLMAP cameras or the Phase 3 mesh. A global Sim(3) is an optional
  typed root transform accepted only on disjoint held-out evidence. Its scale remains arbitrary,
  orientation remains unoriented, topology and original global face IDs remain unchanged, and a
  rejected transform is a valid result.

## Change discipline

Behavior changes require tests and documentation. Typed artifact changes require regenerating the
checked-in schemas. Adapter changes must document inputs, outputs, schema identifiers, environment
allowlists, timeout, retry behavior, healthcheck, provenance, and tests.

Phase 5A dense MVS uses official pinned COLMAP out of process. Core code must not import pycolmap,
NumPy, OpenCV, Open3D, trimesh, PyTorch, or CUDA. Measured object geometry may consume canonical
SAM masks and official dense maps, but never GenRecon geometry. Keep measured and generated
hypotheses as separate Scene IR assets.

Observed geometry stays open and partial: do not close holes, infer hidden backsides, create
collisions, or claim simulation readiness. Completion and object reconstruction remain separately
reviewed future work.

Phase 5B completion must retain the measured asset, keep generation/fitting/held-out
evidence disjoint, and keep torch, NumPy, SAM 3D Objects, TRELLIS.2, and CUDA outside
the core package. Model code/revision/hash and license policy are mandatory. A
research-only candidate must not become production-selected. Do not add articulated
completion, collision, physics, metric/gravity claims, or simulator export.

Phase 5C articulation requires explicit object/part prompts and independent
static-state runs. State alignment uses static base/environment geometry only.
Stable part IDs are separate from semantic prompts and map explicitly to state-local
SAM track IDs; never assume canonical SAM IDs match across independent runs.
Generation, kinematic fitting, and held-out states are filesystem-isolated; a
held-out state may fit only its scalar joint position after graph, link assignment,
base transform, axis, and pivot are frozen. The fitted kinematic model, not the raw
candidate, is the selected source of truth. Single-state evidence is prior-only and
two-state evidence is only partially validated. ArtVIP and PartNet indices are local
and immutable. Particulate and PartField remain isolated; the current PartField
checkpoint is research-only. Never add collision, inertial, dynamics, metric,
gravity, or simulation-ready claims.

Orient each fitted joint axis toward measured motion exactly once. `axis_sign` is
provenance only and must not multiply `q_scale`. Articulated visuals must declare
candidate-base or link-local space with an exact candidate-base transform.
Passing held-out views require all mapped links, target masks, valid depth, and
per-frame camera/depth/mask/render hashes.
Original Phase 5A articulated anchors are `reference_world` evidence and remain on
the object evidence list; never attach them below a fitted candidate/link transform
without transforming the geometry bytes. Selected candidate, fitted model, link
assignment, and evaluation records must be dedicated files with exact content hashes.

Phase 6A calibration requires explicit metric, gravity, forward, and origin evidence.
Calibration candidate selection uses fitting evidence only; held-out evidence is
acceptance-only. The core must not import NumPy, SciPy, OpenCV, AprilTag, Open3D, or
trimesh. Only `accepted_full_canonical` may claim meters and canonical
`+X`-forward/`+Y`-left/`+Z`-up together. Source cameras and geometry remain immutable;
reference-world measured assets receive one wrapper transform. Scale prismatic
quantities exactly once and never scale revolute angles. Phase 6A must not add
collisions, physical properties, simulator export, or simulation-ready claims.
