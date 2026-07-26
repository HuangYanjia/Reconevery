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
CLI, schemas, and deterministic previews.

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

Validated real metrics on 2026-07-25:

```text
normalized frames                 16
registered/processed frames       12
SAM tracks                        4
registered canonical masks        46
global vertices                   6,385,868
global faces                      13,437,638
raster scale                      0.5
candidate faces per camera        3,359,325 to 4,355,976
accepted/ambiguous/unresolved     0 / 1 / 3 objects
accepted faces                    2 (table_0002)
ambiguous faces                   611
same-label conflicts              0
different-label overlaps          0
unassigned global face ratio      0.9999998512
mean accepted face support        approximately 1.0
mean reprojection IoU             approximately 0.00003
runtime                           approximately 21.4 seconds
peak GPU memory                   592,284,672 bytes
peak host memory                  approximately 3.88 GB
```

The component threshold used for this smoke was 2 faces and relative size 0.001 because the
GenRecon mesh contains duplicated/disconnected topology and the default 4-face/0.01 filter
removed every candidate. Default production thresholds remain conservative.

The result meets the non-empty acceptance condition but is a severe quality limitation: only a
two-face table fragment survives and its reprojection IoU is extremely low. Three tracks remain
correctly unresolved. This evidence validates camera-aware association and the typed failure/
uncertainty boundary; it does not demonstrate complete or high-quality object reconstruction.

## Manual Preview Review

Inspect:

```text
reconstruction/object_surfaces/previews/global_face_assignment.png
reconstruction/object_surfaces/previews/object_surface_contact_sheet.png
reconstruction/object_surfaces/previews/reprojection_contact_sheet.png
reconstruction/object_surfaces/previews/conflict_heatmap.png
```

The reviewed raster did not show a whole-frame axis inversion or geometry behind the camera.
No wall was assigned to a small cup. The diagnostic did show almost entirely unassigned global
geometry and a tiny table fragment, consistent with the measured IoU and disconnected topology.

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
