# Roadmap

## Completed foundation

- Phase 0: lightweight architecture, filesystem DAG, canonical Scene IR, and CPU mocks.
- Phase 0.1: real dependencies and packaging, strict schemas and references, artifact-driven mock
  stages, valid PNG/OBJ outputs, DAG validation, byte-sensitive caching, retries, command isolation,
  output validation, expanded artifact records, real Typer CLI, CI, tests, and documentation.

## Completed Phase 1

- real video and image-directory ingest with normalized PNGs;
- FFmpeg/FFprobe subprocess execution and health checks;
- deterministic frame QA and typed reports;
- local/Docker COLMAP feature extraction, matching, and sparse mapping;
- strict binary parsing, pose inversion, camera mapping, and scale ambiguity;
- deterministic multi-model selection and typed diagnostics;
- attempt workspaces that reject stale output and protect previous results;
- fake-executable tests that keep mandatory CI independent of system tools.

## Phase 1.2 alignment and scale recovery

Add a typed alignment stage that consumes raw `colmap_arbitrary` camera reconstruction, estimates
gravity/canonical orientation and an external metric scale, transforms poses into the right-handed
+X-forward, +Y-left, +Z-up robot world, and emits new provenance. It must not relabel raw COLMAP
translations as meters without an observable scale reference.

## Phase 2 implementation

- typed text, box, point, and mask prompt manifests;
- selective attempt inputs that exclude raw COLMAP workspaces;
- isolated official SAM 3.1 Object Multiplex local/Docker worker;
- deterministic fake worker for the CPU-only gate;
- registered/unregistered-aware anchor selection;
- canonical binary masks, stable object IDs, track QA, diagnostics, and no-object results;
- deterministic contact sheet, timeline, frame previews, and COCO export;
- exact official code/checkpoint provenance and credential-safe healthchecks.

A real official SAM 3.1 checkpoint smoke has run on H100 and records the pinned code/checkpoint
identity, valid canonical masks, previews, COCO output, and cache hit.

## Phase 3 implementation

- deterministic observation-lineage digest across COLMAP, SAM, and GenRecon;
- minimal selected-model COLMAP text camera package;
- isolated official GenRecon local/Docker worker and CPU fake protocol;
- three official checkpoint hashes and exact Git/submodule identity;
- reversible internal chunking transform with outputs returned to raw COLMAP coordinates;
- typed global PBR scene, mesh diagnostics, Scene IR visual assets, and previews;
- cross-stage consistency and selective-materialization validation;
- real module-level and full COLMAP -> SAM -> GenRecon acceptance gates.

## Phase 4 implementation

- camera-accurate world-from-camera inversion and OpenCV-axis projection;
- exact homogeneous clipping and conservative frustum culling;
- deterministic mask undistortion for supported COLMAP camera models;
- nvdiffrast nearest-visible original face-ID rasterization with bounded face chunks;
- exact-face and spatial surface-sample multi-view evidence;
- cross-label overlap retention, true-area connected components, seam diagnostics, and compact
  face-index files;
- COLMAP sparse-point versus global-mesh depth alignment diagnostics;
- partial object PLY assets, reprojection metrics, typed uncertainty, and previews;
- Phase 4 Scene IR hypotheses and cross-stage consistency validation;
- CPU fake protocol, synthetic geometric tests, and real H100 surface-lifting smoke.

Real Phase 4.1 comparison retained a non-empty four-face v2 table surface and unresolved results
for three tracks. V2 did not improve the v1 IoU. Sparse-point/rendered-depth residuals identify
camera/global-mesh alignment as the dominant current bottleneck. The low quality remains explicit:
Phase 4 proves observation-grounded association, not complete or accurate object reconstruction.

## Phase 4.2 implementation

- immutable audit of COLMAP, GenRecon PCA/chunk, working, GLB, and canonical mesh transforms;
- consistent sparse-observation undistortion and exact depth rendering;
- deterministic disjoint training/validation frame and point splits;
- identity, extent, centroid, and right-handed PCA initialization records;
- robust bounded global Sim(3) fitting with held-out acceptance gates;
- explicit accepted, rejected, insufficient, and transform-chain-bug statuses;
- per-camera/per-chunk residual and local-structure diagnosis;
- root-transform Scene IR representation without rewriting cameras, mesh, topology, or PBR GLB;
- accepted-alignment object lifting plus an unaligned/aligned comparison;
- isolated GPU worker, CPU fake protocol, previews, CLI, and Phase 4.2 consistency validation.

Phase 4.2 answers whether a single global transform is sufficient. It intentionally does not
perform per-chunk or non-rigid correction and does not turn arbitrary scale into meters.

The recorded H100 audit classified the real Phase 4.1 lineage as `global_sim3_insufficient`.
A bounded candidate reduced held-out median normalized depth residual from approximately `0.662`
to `0.373`, but the `0.10` inlier fraction remained approximately `0.141` and residuals stayed
structured by chunk. The candidate was therefore retained as diagnostic evidence and not applied
to canonical object lifting.

## Phase 5A measured geometry

Official COLMAP dense MVS plus mask-constrained multi-view backprojection now provides measured
visible object surfels independently of GenRecon. Phase 5A does not complete hidden surfaces,
make objects watertight, infer physics, or establish metric/gravity coordinates.

## Recommended Phase 2.5

Design optional VLM-assisted scene inventory as a separate prompt-generation stage. It must
produce a reviewable prompt manifest rather than bypass Phase 2 prompt contracts, and must not
claim exhaustive vocabulary or physical classification without validation.

## Phase 5C articulated reconstruction

Phase 5C implements observation-grounded visual kinematic hypotheses for fixed,
prismatic, and revolute mechanisms from multiple static states. It includes local
license-aware retrieval, official Particulate research candidates, constrained
fitting, and held-out-state evaluation. It does not include deformables, collision,
dynamics, or production simulator export. Phase 5C.1 separates stable part IDs from
state-local SAM tracks, derives evidence from accepted alignments, and makes the
typed fitted kinematic model the selection/Scene IR source of truth.
Phase 5C.2 makes the axis/q sign convention single-valued, records candidate
visual asset spaces explicitly, binds held-out metrics to exact artifact
identities, and requires complete per-link render coverage.
Phase 5C.3 adds the reference-world measured-asset boundary, dedicated selected
files, exact Scene IR path/hash pairs, and transform-aware visual-only URDF previews.

## Phase 6A canonical metric world

Phase 6A implements evidence-grounded metric scale, gravity, forward, origin,
proper Sim(3) validation, immutable canonical wrappers, and camera/rigid/articulated
propagation. AprilTag, known-distance, external metric, IMU/up landmarks, and dense
floor evidence have typed contracts. Full canonical status requires disjoint
held-out acceptance; metric-only and rejected results remain truthful partial
outputs.

## Later phases

- A separately reviewed deformable-object phase.
- A separately reviewed SceneSmith or equivalent scene compiler adapter.
- A later simulation phase: collision generation, physics repair, and explicit simulator exports.

Each phase must preserve the canonical Scene IR boundary and add real-adapter tests without making
the CPU-only mock quality gate depend on GPUs, checkpoints, Docker, or network model downloads.

Phase 5B now covers ordinary rigid/static visual candidates from official SAM 3D
Objects and TRELLIS.2 with measured registration, held-out evaluation, and
license-aware selection. Collision generation, physical validation, SceneSmith, and
simulator compilation remain future work.

## Phase 6B layered scene assembly

Phase 6B builds research and deployment-eligible visual bundles from one coherent
lineage. It supports full, partial, rejected, or absent calibration, preserves
measured anchors and articulated local quantities, and reports overlap without
carving the global mesh. The result is a visual-only compiler input manifest, not a
simulator export. SceneSmith, collision generation, physics identification, and
simulation validation remain later work.
