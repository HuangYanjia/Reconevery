# Phase 4: Observation-Grounded Object Surface Lifting

## Capability

Phase 4 associates persistent SAM object instances with nearest-visible faces of the GenRecon
global mesh. Its output is partial evidence:

```text
COLMAP cameras + canonical SAM masks + GenRecon global mesh
  -> undistorted mask space
  -> visible original face IDs
  -> multi-view support and conflict handling
  -> partial object surface hypotheses
```

It does not generate hidden surfaces, replace objects, fill holes, create collisions, estimate
physics, recover metric scale, align gravity, or export a simulator world. Every hypothesis
states:

```text
geometry_status=partial_observation_supported
completion_status=not_completed
hidden_surface_completion=not_implemented
sim_ready=false
metric_scale_known=false
canonical_gravity_alignment_known=false
```

## Inputs and Isolation

The core adapter declares only:

```text
inputs/manifest.json
camera/reconstruction.json
observations/object_tracks.json
observations/masks/<retained-track>/<frame>.png
reconstruction/global/metadata.json
reconstruction/global/mesh.ply       # reference_only
reconstruction/global/scene.glb      # reference_only, previews only
scene_ir/scene.json
```

It excludes raw COLMAP databases/models/logs, SAM raw files/checkpoints, GenRecon checkpoints,
chunk tensors, and unrelated outputs. The runner checks reference hashes before and after worker
execution. The lightweight core imports no NumPy, OpenCV, trimesh, torch, CUDA, or nvdiffrast.

The isolated worker is in `workers/object_lifting`. It can share the Phase 3 GenRecon environment
but never imports GenRecon or initializes a generative model. `docker/object-lifting` provides a
checkpoint-free CUDA alternative.

## Distortion and Rasterization

Supported source camera models are `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, and
`OPENCV`. OpenCV creates a deterministic same-size undistortion map and pinhole matrix. Canonical
binary masks are remapped with nearest-neighbor sampling and then resized to `raster_scale`,
again with nearest-neighbor sampling. Output mask values remain binary.

For each registered frame the worker:

1. inverts `transform_world_from_camera`;
2. transforms global vertices into OpenCV camera axes;
3. computes scene-relative near/far planes;
4. culls global faces in `face_chunk_size` batches;
5. rasterizes candidates with nvdiffrast;
6. maps local raster triangle IDs back to original global face IDs;
7. retains only the nearest visible surface.

Full per-frame triangle-ID buffers are ephemeral and are not promoted.

## Mask Evidence

Each mask yields an eroded core, a boundary band, a dilated exclusion region, and an exterior
negative region. Pixel radii scale with raster resolution. Per-face evidence records visibility,
core and boundary positives, exterior negatives, positive/negative view counts, supporting-frame
range, depth, and support score:

```text
positive = core_weight * core_pixels + boundary_weight * boundary_pixels
negative = exterior_weight * exterior_pixels
score = positive / (positive + negative + epsilon)
```

Weights are deterministic heuristics, not learned probabilities. A face must satisfy configured
visible-pixel, positive-pixel, supporting-view, and score thresholds.

Distinct instances of the same normalized semantic label are exclusive. The higher score wins
when the margin is sufficient; otherwise the face becomes ambiguous for both. Different labels
may overlap, so `cabinet`, `drawer`, and `handle` are not suppressed merely due to overlap.

Connected components use shared global mesh vertex-index edges. Small components are removed
using face-count and relative-size thresholds. No generated geometry connects separated pieces.

## Outputs

```text
reconstruction/object_surfaces/
  request.json
  worker_manifest.json
  evidence_manifest.json
  face_assignment_manifest.json
  diagnostics.json
  preview_manifest.json
  objects/<object_id>/
    accepted_face_ids.bin
    ambiguous_face_ids.bin
    face_evidence.npz
    surface_mesh.ply
    surface_points.ply
  raw/rasterization_manifest.json
  previews/*.png
scene_ir/phase4_scene.json
validation/phase4_object_surface_consistency.json
```

Face arrays are sorted little-endian `uint32` or `uint64`, selected by the maximum global face
ID. Their count, range, dtype, mesh hash, and content hash are typed. Evidence NPZ array names,
shapes, dtypes, and per-array byte hashes are recorded. The core validates these without NumPy.

The enriched Scene IR uses `GeometrySourceType.FUSED`, because the surface combines generated
global geometry, measured cameras, and observed masks. It is stage-owned at
`scene_ir/phase4_scene.json`; Phase 3's `scene_ir/scene.json` remains unchanged for correct cache
ownership.

## Reprojection and Confidence

Accepted global face IDs are compared against the nearest-visible global face buffer in every
registered object observation. Per-frame metrics are IoU, precision, recall, rendered area, mask
area, false-positive area, and false-negative area. Unregistered masks remain valid 2D evidence
but never contribute to lifting or reprojection.

Confidence describes observation support only:

```text
0.30 * mean face support
+ 0.25 * median reprojection IoU
+ 0.20 * normalized supporting views
+ 0.15 * track coverage
+ 0.10 * largest component ratio
- conflict penalty
```

It says nothing about complete shape, hidden surfaces, material, physics, or metric accuracy.

## Commands

```bash
uv run recon2sim run \
  --input /absolute/path/to/scene \
  --config configs/phase4_e2e.yaml \
  --run-dir runs/phase4

uv run recon2sim objects inspect-surfaces runs/phase4
uv run recon2sim objects inspect-surface runs/phase4 table_0001
uv run recon2sim objects render-surface-previews runs/phase4
uv run recon2sim objects export-surface \
  runs/phase4 table_0001 --output table_partial.ply
uv run recon2sim objects export-face-ids \
  runs/phase4 table_0001 --output table_face_ids.bin
uv run recon2sim validation inspect-phase4 runs/phase4
uv run recon2sim validation verify-phase4 runs/phase4
```

Use `configs/phase4_e2e_fake.yaml` for CPU protocol validation. Use ignored `configs/local/`
copies with absolute worker paths for real inference.

## Failure Recovery

Malformed cameras, unsupported camera models, hash mismatches, invalid mesh coordinates or
indices, rasterizer errors, OOM, timeouts, path escape, corrupt compact arrays, and worker output
schema errors fail the stage. Failed attempts stay under `work/object_surface_lifting/attempt_N`.
Transactional promotion preserves the previous canonical object-surface set. Per-track
insufficient support is not a stage failure; it produces an `unresolved` hypothesis.
