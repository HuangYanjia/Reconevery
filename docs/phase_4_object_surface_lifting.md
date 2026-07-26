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
camera/genrecon_package/{package_manifest.json,images.txt,points3D.txt,registered_frames.json}
observations/object_tracks.json
observations/masks/<retained-track>/<frame>.png
reconstruction/global/metadata.json
reconstruction/global/mesh.ply       # reflink_or_copy into the attempt
scene_ir/scene.json
```

It excludes raw COLMAP databases/models/logs, SAM raw files/checkpoints, GenRecon checkpoints,
chunk tensors, the unused global GLB, and unrelated outputs. Local execution passes the attempt
as `--input-root`; Docker mounts only the attempt at `/workspace`. The runner verifies canonical
hashes before and after worker execution. The lightweight core imports no NumPy, OpenCV, trimesh,
torch, CUDA, or nvdiffrast.

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
3. constructs exact homogeneous clip coordinates without clamping camera depth;
4. conservatively culls global faces in `face_chunk_size` batches;
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
visible-pixel, positive-pixel, supporting-view, and score thresholds. The v1 result remains the
exact-face baseline.

`surface_sample_fusion_v2` additionally maps every positive visible pixel to its world-space
surface point and a scene-relative voxel derived from median global edge length. Positive and
exterior-negative samples accumulate in the same spatial cells across views. Supported cells map
only their directly sampled faces back to original GenRecon face IDs; no adjacency propagation or
new geometry is introduced. `face_evidence.npz` distinguishes direct sample, patch, and propagated
support, with propagated support currently fixed to zero.

Distinct instances of the same normalized semantic label are exclusive. The higher score wins
when the margin is sufficient; otherwise the face becomes ambiguous for both. Different labels
may overlap, so `cabinet`, `drawer`, and `handle` are not suppressed merely due to overlap.

Connected components use shared global mesh vertex-index edges. Triangle areas are computed in
the original arbitrary COLMAP coordinates. `min_component_faces` filters by count while
`min_relative_component_area` uses true surface-area ratio, not face-count ratio. A seam-aware
diagnostic may group likely duplicated boundary vertices by scale-normalized distance and normal
similarity, but it never changes face IDs or generates bridging triangles.

## Outputs

```text
reconstruction/object_surfaces/
  request.json
  worker_manifest.json
  evidence_manifest.json
  face_assignment_manifest.json
  diagnostics.json
  method_comparison.json
  camera_mesh_alignment.json
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

Quality is split into association precision, mask recall, reprojection IoU, multiview support,
surface connectedness, observed-surface coverage, association confidence, and completeness
confidence. Association confidence is:

```text
geometric_mean(face_or_patch_support, reprojection_precision, multiview_support)
- conflict penalty
```

`completeness_confidence` is always zero. Status is `accepted`, `partial`, `ambiguous`, or
`unresolved` using explicit reprojection and ambiguity thresholds. These values describe
association only, never complete shape, hidden surfaces, material, physics, or metric accuracy.

Camera/mesh alignment diagnostics report per-frame mesh coverage, finite depth, visible faces,
depth percentiles, and normalized residuals between selected COLMAP sparse points and rendered
mesh depth. `global_mesh_depth_contact_sheet.png`, `global_mesh_edge_overlay.png`, and
`sparse_point_vs_mesh_depth.png` distinguish projection/alignment failure from exact-face
granularity or missing geometry.

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
