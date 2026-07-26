# Recon2Sim

Recon2Sim Phase 4.2 is a typed observation pipeline with real video/image ingest, frame QA,
out-of-process COLMAP sparse camera recovery, prompt-driven SAM 3.1 tracking, and isolated
official GenRecon global visual reconstruction. It lifts 2D tracks to partial visible surfaces
and audits a bounded global camera-mesh Sim(3). Heavy runtimes remain outside the lightweight
core. Hidden-surface completion, physical reconstruction, and simulator export are not implemented.

## Quickstart

Python 3.12 is pinned in `.python-version`. Install `uv`, then run:

```bash
uv sync --all-groups
uv run recon2sim --help
uv run recon2sim run \
  --input examples/tabletop \
  --config configs/mock.yaml \
  --run-dir runs/tabletop_demo
uv run recon2sim validate-ir runs/tabletop_demo/scene_ir/scene.json
```

For a directory containing one video, or an `images/` directory of JPEG/PNG files:

```bash
uv run recon2sim adapters healthcheck --config configs/colmap.yaml
uv run recon2sim run \
  --input examples/real_video \
  --config configs/colmap.yaml \
  --run-dir runs/real_video_colmap
uv run recon2sim ingest inspect runs/real_video_colmap
uv run recon2sim camera inspect runs/real_video_colmap
uv run recon2sim camera export-trajectory \
  runs/real_video_colmap --output trajectory.json
```

`configs/colmap.yaml` and `configs/colmap_cpu.yaml` intentionally stop after camera recovery.
`configs/colmap_with_mock_downstream.yaml` is the explicit integration demo for real ingest/COLMAP
followed by Phase 0.1 mocks. The optional pinned COLMAP 3.11.1 image is:

```bash
docker build -t reconevery/colmap:phase1 docker/colmap
```

Run the complete CPU-only fake SAM protocol, including canonical masks and previews:

```bash
uv run recon2sim adapters healthcheck --config configs/sam3_fake.yaml
uv run recon2sim run \
  --input examples/tabletop \
  --config configs/sam3_fake.yaml \
  --run-dir runs/tabletop_sam3_fake
uv run recon2sim segmentation inspect runs/tabletop_sam3_fake
uv run recon2sim segmentation render-preview runs/tabletop_sam3_fake
uv run recon2sim segmentation export-coco \
  runs/tabletop_sam3_fake --output runs/tabletop_sam3_fake/annotations.json
```

`configs/sam3.yaml` and `configs/sam3_docker.example.yaml` use the official gated
`facebook/sam3.1` checkpoint and stop after segmentation. Accept Meta's terms and provide
`HF_TOKEN`, an authorized cache, or a mounted local official checkpoint. See
[`docs/phase_2_sam3.md`](docs/phase_2_sam3.md).

Run the complete CPU-only fake Phase 3 protocol:

```bash
uv run recon2sim run \
  --input examples/tabletop \
  --config configs/phase3_e2e_fake.yaml \
  --run-dir runs/phase3_e2e_fake
uv run recon2sim reconstruction inspect-global runs/phase3_e2e_fake
uv run recon2sim reconstruction render-global-preview runs/phase3_e2e_fake
uv run recon2sim validation verify-phase3-e2e runs/phase3_e2e_fake
```

`configs/genrecon_only.yaml` runs real ingest, COLMAP, and GenRecon without making GenRecon
depend on SAM. `configs/phase3_e2e.yaml` adds real SAM and the cross-stage consistency validator.
Replace its `/absolute/path/...` placeholders locally. See
[`docs/phase_3_genrecon.md`](docs/phase_3_genrecon.md).

Resume without changing successful status:

```bash
uv run recon2sim run \
  --input examples/tabletop \
  --config configs/mock.yaml \
  --run-dir runs/tabletop_demo \
  --resume
```

A cache hit remains `"status": "succeeded"` and records
`"last_execution": "cache_hit"`. Changing input bytes, stage configuration, or an upstream
artifact invalidates the affected stage and all of its dependents.

## Mock data flow

The checked-in PNG frames are tiny generated test fixtures. The pipeline produces and consumes:

```text
examples/tabletop/frames/*.png
  -> inputs/manifest.json + frames/*.png
  -> camera/reconstruction.json
  -> observations/object_tracks.json + observations/masks/*.png
  -> reconstruction/global/{floor.obj,metadata.json}
     reconstruction/objects/{results.json,*.obj}
  -> scene_ir/scene.json
  -> compiled/scene_package/{package.json,mock_scene.obj}
  -> validation/report.json
  -> export_manifest.json
```

`scene_ir/scene.json` is the canonical semantic and physical scene. OBJ files are visual or
collision mesh artifacts referenced by the IR. `compiled/scene_package` is a mock compiler
output, and its package explicitly contains no simulator outputs. These are separate contracts.

The cabinet is one top-level articulated object. Its body and drawer are articulation links;
the drawer is not duplicated as an independent `ObjectInstance`.

## Real ingest and camera recovery

`ffmpeg_ingest` accepts a video file, a directory containing one video, or a deterministic
JPEG/PNG image collection. Video decoding uses FFmpeg and FFprobe as subprocesses. Frame QA
records Laplacian-variance sharpness, mean grayscale brightness, intensity variance, and a
downsampled-pixel duplicate score in `inputs/frame_qa.json`. Defaults are conservative and
require dataset-specific tuning.

`colmap_camera_recovery` runs feature extraction, sequential or exhaustive matching, mapping,
binary-model parsing, and deterministic sparse-model selection. It preserves `database.db`,
every sparse model, commands, and logs under `camera/colmap/`. See
[`docs/phase_1_colmap.md`](docs/phase_1_colmap.md) for setup and capture guidance.

## SAM 3 segmentation and tracking

`sam3_segmentation_tracking` consumes only the normalized manifest, frame QA, typed camera
reconstruction, selected frames, and a validated prompt manifest. It never copies the raw COLMAP
database, sparse models, or logs into the SAM attempt. Text, box, point, and mask prompt contracts
are typed; the pinned official public video API supports text, box, and point requests. It does
not expose mask-seed requests, so the real worker reports that limitation instead of using an
undocumented internal method.

The default real backend is official SAM 3.1 Object Multiplex at code commit
`46957e47805eaa273f4aa7bbbd25a88bca9108ce` and checkpoint revision
`daa63191845a41281374e725f4c9e51c7a824460`. Canonical tracks preserve raw model IDs separately,
assign stable semantic IDs such as `cup_0001`, and write one grayscale `0/255` PNG per visible
object/frame. Frames without a COLMAP pose remain valid 2D observations and carry
`camera_pose_available=false`.

## GenRecon global visual reconstruction

`genrecon_camera_package` serializes only the selected COLMAP model to deterministic
`cameras.txt`, `images.txt`, and `points3D.txt`. It preserves manifest order, remaps image and
camera IDs consistently, and references only registered normalized frames. It does not copy the
COLMAP database, logs, or rejected models.

`genrecon_global_reconstruction` invokes official GenRecon commit
`eaf1468118d20469d17079a4a19737297d2ef87b` in an isolated Python 3.10/CUDA worker. The three
official checkpoints come only from `https://kaldir.vc.cit.tum.de/genrecon/` and are identified
by SHA-256. The official pipeline also requires separately accepted access to gated
`facebook/dinov3-vitl16-pretrain-lvd1689m`; its resolved revision is recorded without storing
Hugging Face credentials. Required decoder assets from `microsoft/TRELLIS-image-large` and
`microsoft/TRELLIS.2-4B` are likewise cached and identified by exact resolved revisions. A
reversible PCA working transform may stabilize chunking, but it is not gravity alignment.
Canonical `scene.glb` and `mesh.ply` are returned to the original COLMAP arbitrary frame.

SAM and GenRecon are parallel evidence branches. Prompt changes rerun SAM and the consistency
validator, not GenRecon. The validator checks ordered frame hashes, registration sets, camera
hashes, coordinate semantics, selective materialization, and the explicit capability boundary.

## Observation-grounded object surfaces

`object_surface_lifting` projects the canonical GenRecon mesh into registered COLMAP cameras,
undistorts canonical SAM masks, and accumulates core, boundary, exterior-negative, visibility,
depth, and supporting-view evidence against original global face IDs. Phase 4.1 compares exact
face voting with scene-relative surface-sample fusion and diagnoses camera/global-mesh depth
alignment from the minimal selected COLMAP text package. Exact homogeneous clipping handles
near-plane and behind-camera triangles without moving them in front of the camera. The heavy
NumPy/OpenCV/PyTorch/nvdiffrast/trimesh implementation remains in
`workers/object_lifting`; the core adapter validates typed outputs, compact face-ID arrays,
extracted PLY meshes, hashes, lineage, coordinate semantics, and upstream immutability without
importing those packages.

Outputs under `reconstruction/object_surfaces/` are partial observation-supported hypotheses.
They preserve arbitrary COLMAP units and orientation, may be unresolved, contain no hidden
surface completion or collision geometry, and are explicitly `sim_ready=false`. Phase 4 writes
its enriched Scene IR to `scene_ir/phase4_scene.json` so it does not mutate the Phase 3-owned
`scene_ir/scene.json` and invalidate GenRecon's cache.

Local and Docker workers see only the isolated attempt workspace. The global PLY is reflinked or
copied into that workspace; the canonical run, unused GLB, raw COLMAP workspace, SAM raw output,
and GenRecon tensors are not mounted or materialized.

```bash
uv run recon2sim run \
  --input /path/to/scene \
  --config configs/phase4_e2e.yaml \
  --run-dir runs/phase4
uv run recon2sim objects inspect-surfaces runs/phase4
uv run recon2sim validation verify-phase4 runs/phase4
```

## Camera-mesh alignment audit

`camera_mesh_alignment` audits the complete COLMAP -> GenRecon working frame -> canonical mesh
transform chain before fitting anything. It prepares filtered COLMAP sparse observations in the
same undistorted projection space used by GenRecon and object lifting, separates training and
held-out cameras/point IDs, and evaluates identity plus bounded global Sim(3) candidates. A
candidate is accepted only when held-out depth, inlier, coverage, plausibility, and
point-to-surface gates all pass.

The original cameras, PLY, GLB, topology, and face IDs are immutable. An accepted transform is
stored in `reconstruction/alignment/alignment.json` and applied only to an in-memory vertex copy
during downstream rasterization. A rejected transform is a valid outcome. The fitted scale
remains an arbitrary gauge correction: it does not establish meters or gravity.

```bash
uv run recon2sim run \
  --input /path/to/scene \
  --config configs/phase4_2_e2e.yaml \
  --run-dir runs/phase4_2
uv run recon2sim alignment inspect runs/phase4_2
uv run recon2sim alignment inspect-transform-chain runs/phase4_2
uv run recon2sim alignment compare-object-lifting runs/phase4_2
uv run recon2sim validation verify-phase4-2 runs/phase4_2
```

## Coordinate convention

Raw Phase 1 COLMAP output is explicitly `world_frame="colmap_arbitrary"`,
`alignment_status="unoriented"`, `camera_axes="x_right_y_down_z_forward"`,
`linear_units="arbitrary_units"`, and `scale_status="scale_ambiguous"`. Poses are
`transform_world_from_camera`, quaternions are `xyzw`, and the unit-neutral `translation` values
remain in COLMAP's arbitrary gauge.

The canonical robot scene frame is a separate, later contract: right-handed +X forward, +Y left,
+Z up, with meters after external alignment and scale recovery. Phase 1 does not perform or imply
that conversion.

## Quality gate

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

Scene IR, prompt, SAM, GenRecon, global reconstruction, and consistency schemas under `schemas/`
are checked against their Pydantic models in tests. Regenerate them with:

```bash
uv run python scripts/generate_schema.py
```

## Adapter boundary

Core code imports only lightweight dependencies. Official SAM and GenRecon code, PyTorch, CUDA
packages, checkpoints, GPU rasterization, SciPy, and Sim(3) optimization exist only in isolated
workers or Docker images.
SceneSmith, hidden object completion, physics, simulators, MVS, NeRF, and automatic VLM prompt
inventory remain outside Phase 4.2.
