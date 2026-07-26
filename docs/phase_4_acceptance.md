# Phase 4 Acceptance

## CPU Acceptance

Mandatory CI uses `configs/phase4_e2e_fake.yaml`. It requires:

```bash
uv sync --all-groups --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv run python scripts/generate_schema.py
```

The fake DAG runs ingest, COLMAP protocol, SAM protocol, GenRecon protocol, object lifting, and
Phase 4 consistency. An identical `--resume` command must cache-hit every stage. Tests cover
compact face arrays, camera inversion, projection, distortion contracts, nearest-face
occlusion, original face IDs, overlap policies, connected components, unresolved results,
selective inputs, upstream immutability, transactional preservation, worker failure classes,
CLI, schemas, and deterministic previews. Worker-environment tests additionally cover
CPU/nvdiffrast interior face-ID parity, camera/near-plane clipping, deterministic surface-sample
fusion, negative patch evidence, and seam-aware component diagnostics.

## Real H100 Acceptance

The real smoke reuses one complete Phase 3 lineage:

```text
16 normalized real images
12 real COLMAP registered cameras
4 official SAM 3.1 canonical tracks
real official GenRecon global mesh
-> real nvdiffrast object surface lifting
```

The local ignored configuration uses one NVIDIA H100 NVL, Python 3.10.20,
PyTorch 2.6.0+cu126, CUDA 12.6, nvdiffrast 0.4.0, OpenCV 5.0.0, NumPy 2.2.6,
and trimesh 4.12.2. No model checkpoint is loaded by Phase 4.

Validated Phase 4.1 comparison metrics on 2026-07-25:

```text
normalized frames                 16
registered/processed frames       12
SAM tracks                        4
registered canonical masks        46
global vertices                   3,642,276
global faces                      7,677,700
raster scale                      0.5
candidate faces per camera        2,882,353 to 3,554,779
v1 table_0002 faces/components    76 / 29
v1 table_0002 precision/recall    0.565832 / 0.004297
v1 table_0002 IoU                 0.004290
v2 accepted/ambiguous/unresolved  0 / 1 / 3 objects
v2 table_0002 faces/components    4 / 1
v2 table_0002 precision/recall    0.250000 / 0.000205
v2 table_0002 IoU                 0.000205
v2 ambiguous faces                205 total
same-label conflicts              0
different-label overlaps          0
unassigned global face ratio      0.9999994790
mesh pixel coverage mean          0.998635
sparse depth residual median      0.658454
sparse depth residual p90         0.791167
sparse depth inlier fraction      0.013636
v2 runtime (final versioned run)  97.26 seconds
peak GPU memory                   307,344,896 bytes
peak host memory                  approximately 2.88 GB
```

The smoke used two faces and true relative surface area 0.001 for component filtering. The exact
topology still contains many disconnected fragments; seam diagnostics do not invent bridges.

V2 produces valid original global face IDs, but it does not improve the v1 baseline. The rendered
mesh covers almost every pixel while sparse-point depth agrees within the configured threshold
for only about 1.36% of comparable samples. The depth scatter is far from `y=x`. The diagnosed
bottleneck is therefore camera/global-mesh depth alignment rather than exact-face granularity
alone. Three tracks remain correctly unresolved. This evidence validates the typed comparison
and uncertainty boundary; it does not demonstrate complete or high-quality object reconstruction.

## Manual Preview Review

Inspect:

```text
reconstruction/object_surfaces/previews/global_face_assignment.png
reconstruction/object_surfaces/previews/object_surface_contact_sheet.png
reconstruction/object_surfaces/previews/reprojection_contact_sheet.png
reconstruction/object_surfaces/previews/conflict_heatmap.png
reconstruction/object_surfaces/previews/global_mesh_depth_contact_sheet.png
reconstruction/object_surfaces/previews/global_mesh_edge_overlay.png
reconstruction/object_surfaces/previews/sparse_point_vs_mesh_depth.png
reconstruction/object_surfaces/previews/surface_sample_fusion.png
```

The reviewed raster did not show a whole-frame axis inversion, behind-camera leakage, or a wall
assigned to a small cup. It did show a near-full-frame depth surface whose depths disagree with
COLMAP sparse points, almost entirely unassigned global geometry, and a tiny table fragment.
Those visuals agree with the measured residuals and IoU.

`validation/phase4_object_surface_consistency.json` must pass and state:

```text
real_2d_tracks_lifted_to_global_3d=true
hidden_surface_completion_implemented=false
object_replacement_implemented=false
sim_ready_scene_implemented=false
metric_scale_known=false
canonical_gravity_alignment_known=false
```

The identical full Phase 4 command must then cache-hit ingest, camera recovery, segmentation,
camera package, global reconstruction, object lifting, and Phase 4 validation. Large run outputs
remain local and are not committed.
