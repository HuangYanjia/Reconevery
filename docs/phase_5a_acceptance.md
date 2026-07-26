# Phase 5A Acceptance

## Fake acceptance

```bash
uv run recon2sim run \
  --input examples/tabletop \
  --config configs/phase5a_e2e_fake.yaml \
  --run-dir runs/phase5a_fake
uv run recon2sim run \
  --input examples/tabletop \
  --config configs/phase5a_e2e_fake.yaml \
  --run-dir runs/phase5a_fake --resume
```

The second run must report cache hits for ingest, sparse camera recovery, SAM,
dense MVS, measured object geometry, and validation.

## Real H100 acceptance

Use an uncommitted `configs/local/phase5a_h100.yaml` containing absolute worker
and COLMAP paths. Then run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run recon2sim adapters healthcheck \
  --config configs/local/phase5a_h100.yaml
CUDA_VISIBLE_DEVICES=0 uv run recon2sim run \
  --input /absolute/path/to/real-scene \
  --config configs/local/phase5a_h100.yaml \
  --run-dir runs/phase5a_h100
CUDA_VISIBLE_DEVICES=0 uv run recon2sim run \
  --input /absolute/path/to/real-scene \
  --config configs/local/phase5a_h100.yaml \
  --run-dir runs/phase5a_h100 --resume
```

Real acceptance requires verified official COLMAP 4.0.4, a meaningful majority
of geometric depth maps, valid normals and consistency graphs, a non-empty
fused cloud, exact mask mapping, and at least one object with non-empty
multi-view measured geometry. A table or cabinet may satisfy acceptance; a
textureless cup is allowed to remain unresolved.

Inspect:

```text
reconstruction/dense/previews/depth_contact_sheet.png
reconstruction/dense/previews/normal_contact_sheet.png
reconstruction/dense/previews/consistency_contact_sheet.png
reconstruction/dense/previews/fused_point_cloud.png
reconstruction/measured_objects/previews/depth_mask_contact_sheet.png
reconstruction/measured_objects/previews/measured_object_contact_sheet.png
reconstruction/measured_objects/previews/reprojection_contact_sheet.png
reconstruction/measured_objects/previews/object_point_clouds.png
```

Look for mask shift, mirrored cameras, foreground/background bleeding, points
behind cameras, track mixing, and artificial hole closure. Record actual quality
limitations; do not weaken validation to make an object appear resolved.

## Recorded H100 smoke

The Phase 5A development smoke used one NVIDIA H100 NVL, the exact COLMAP
4.0.4 commit `9c23f6942fe69962e06030905e77067c8673382f`, CUDA 12.6, and
GCC 11.4. It reused one real observation lineage containing 16 normalized
frames, 12 registered COLMAP cameras, four real SAM 3.1 tracks, and 50
canonical masks.

Dense reconstruction produced geometric depth maps, normal maps, and valid
consistency graphs for all 12 registered frames. Stereo fusion produced
104,961 finite points. The independently remapped RGB images differed from
COLMAP's undistorted images by at most 0.807 mean absolute intensity units,
below the configured threshold of 3.0.

Measured extraction retained 146,043 of 177,531 raw samples and fused them
into 18,382 surfels:

| Object | Status | Views | Validated samples | Surfels | Reprojection IoU |
| --- | --- | ---: | ---: | ---: | ---: |
| `cup_0001` | accepted | 12 | 5,351 | 519 | 0.740 |
| `cup_0002` | accepted | 12 | 4,600 | 493 | 0.850 |
| `table_0001` | accepted | 10 | 9,042 | 1,570 | 0.796 |
| `table_0002` | accepted | 12 | 127,050 | 15,800 | 0.867 |

The dense stage took 567.0 seconds in this instrumented run, including 544.2
seconds for PatchMatch, and recorded a peak GPU allocation of approximately
672 MiB. Measured extraction took 84.4 seconds. The Phase 5A consistency
validator passed, and an identical resumed command reported cache hits for
dense MVS, measured object geometry, and Phase 5A validation.

Manual preview inspection found coherent scene depth and normals, no systematic
mask-undistortion shift, and no mirrored or upside-down geometry. Cups remained
sparser than tables. These outputs are measured visible surfaces only:
completeness confidence is zero, no hidden surface is inferred, and the scene
is not simulation-ready.
