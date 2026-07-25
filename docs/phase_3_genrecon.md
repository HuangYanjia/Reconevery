# Phase 3: GenRecon Global Visual Reconstruction

## Capability Boundary

Phase 3 reconstructs one global PBR visual scene from posed normalized RGB frames. It does not
fuse SAM masks into object meshes, establish metric scale or gravity, generate collision
geometry, estimate physics, or export a simulator world.

SAM and GenRecon are parallel branches:

```text
ingest -> camera_recovery -> segmentation_tracking
                         \-> genrecon_camera_package -> global_reconstruction

segmentation_tracking + camera_recovery + global_reconstruction
  -> end_to_end_consistency_validation
```

A prompt change invalidates SAM and the validator. It does not invalidate GenRecon because
GenRecon consumes no SAM mask.

## Official Identity

- Repository: `https://github.com/kasothaphie/GenRecon`
- Commit: `eaf1468118d20469d17079a4a19737297d2ef87b`
- Recursive Eigen submodule:
  `o-voxel/third_party/eigen@21e4582d1739107337a03460c81412981130373e`
- Code license: MIT
- Official dependency licenses remain independent, including nvdiffrast and nvdiffrec.

The worker refuses a different Git or submodule commit.

Official checkpoints:

```text
https://kaldir.vc.cit.tum.de/genrecon/sparse_structure.pt
https://kaldir.vc.cit.tum.de/genrecon/shape_slat.pt
https://kaldir.vc.cit.tum.de/genrecon/texture_slat.pt
```

Each run records URL, filename, byte size, SHA-256, file timestamp, and cache/download mode.
Weights are never committed, embedded in Docker, copied into stage attempts, or uploaded as CI
artifacts.

The pinned official pipeline also loads the official gated runtime encoder
`facebook/dinov3-vitl16-pretrain-lvd1689m`. This is separate from the three GenRecon
checkpoints. Accept its Hugging Face terms before real inference, authenticate with an
implicitly discovered `HF_TOKEN` or cached login, and set `HF_HOME` when using a non-default
cache. Reconevery never puts the token in a request, resolved configuration, log, provenance,
or command argument. The worker records the exact resolved DINOv3 repository revision and
runs the official subprocess from that cached snapshot in offline mode.

## Local H100 Environment

The pinned official environment uses Python 3.10, CUDA toolkit 12.6, PyTorch 2.6.0,
torchvision 0.21.0, Flash-Attention 2.7.3, nvdiffrast, nvdiffrec, CuMesh, O-Voxel, and
FlexGEMM. Set:

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export TORCH_CUDA_ARCH_LIST=9.0
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$HOME/.cache/huggingface"
# Set HF_TOKEN in the environment only when a cached login is unavailable.
```

Clone recursively and detach the official checkout:

```bash
git clone --recursive https://github.com/kasothaphie/GenRecon /path/to/GenRecon
git -C /path/to/GenRecon checkout --detach \
  eaf1468118d20469d17079a4a19737297d2ef87b
git -C /path/to/GenRecon submodule update --init --recursive
```

Follow the pinned official `README.md` and `setup.sh` for the compiled dependencies, then install
the worker:

```bash
/path/to/genrecon-env/bin/python -m pip install -e workers/genrecon
```

Create an ignored local config from `configs/genrecon_only.yaml` or `configs/phase3_e2e.yaml`.
Use absolute paths for the worker Python, checkout, and three checkpoints. Then run:

```bash
uv run recon2sim adapters healthcheck --config configs/local/phase3_h100.yaml
uv run recon2sim run \
  --input /absolute/path/to/scene \
  --config configs/local/phase3_h100.yaml \
  --run-dir runs/phase3_h100
```

## Docker

Build from the repository root:

```bash
docker build -f docker/genrecon/Dockerfile -t reconevery/genrecon:phase3 .
```

Set `hf_cache_path` in the Docker configuration to an existing writable host directory.
Reconevery mounts it at `/hf-cache`; the host `HF_HOME` path is never reused as an invalid
in-container absolute path.

The image compiles H100 `sm_90` extensions but contains no weights. Configure the three host
checkpoint paths in `configs/genrecon_docker.example.yaml`. Runtime uses `--gpus all`, read-only
checkpoint mounts, the current Linux UID/GID, and a writable attempt mount.

## Camera Package

`genrecon_camera_package` consumes only:

```text
inputs/manifest.json
camera/reconstruction.json
camera/diagnostics.json
camera/colmap/sparse/<selected>/{cameras,images,points3D}.bin
```

It writes deterministic COLMAP text:

```text
camera/genrecon_package/
  cameras.txt
  images.txt
  points3D.txt
  registered_frames.json
  package_manifest.json
```

Camera and image IDs are deterministically remapped. Point-track image IDs are remapped with
them. Poses, intrinsics, distortion parameters, sparse point colors/errors/tracks, original IDs,
source hashes, and manifest ordering are retained. The package references normalized frames; it
does not duplicate them.

Master order is the ingest manifest order. Eligible GenRecon order is that order filtered by
`camera.registered_frame_ids`. Unregistered frames remain valid SAM inputs and are excluded from
GenRecon multi-view reconstruction.

## Observation Lineage

The frame-sequence digest is SHA-256 over ordered tuples:

```text
(frame_id, normalized run-relative path, normalized frame SHA-256)
```

COLMAP, SAM, camera package, GenRecon request/worker/metadata, and the consistency report carry
the same digest. Filenames alone are never treated as lineage proof.

## Coordinates and Scale

Inputs and canonical global outputs retain:

```text
world_frame=colmap_arbitrary
alignment_status=unoriented
camera_axes=x_right_y_down_z_forward
linear_units=arbitrary_units
scale_status=scale_ambiguous
transform_direction=world_from_camera
```

The worker may center and rotate inputs with deterministic PCA for the official axis-aligned
chunker. The two 4x4 matrices and numerical round-trip error are recorded. PCA is not gravity
alignment. Before promotion, `mesh.ply` and `scene.glb` are transformed back into the original
COLMAP frame.

The official argument name `radius_m` is retained for CLI compatibility, but its value operates
in arbitrary COLMAP units. Diagnostics call this out. Tune chunk/cleaning values per dataset.
If point cleaning removes all points or every chunk, the stage fails and preserves the attempt.

## Official Execution

The worker invokes argument arrays without a shell:

```bash
python reconstruct_scene.py \
  --mode Iphone \
  --path <ephemeral-package> \
  --output_path <raw-output> \
  --ss_ckpt <sparse_structure.pt> \
  --shape_ckpt <shape_slat.pt> \
  --tex_ckpt <texture_slat.pt> \
  --num_imgs_per_scene <N> \
  --chunk_size_factor 1.08 \
  --stat_std_ratio 3.0 \
  --radius_nb_points 7 \
  --radius_m 0.2 \
  --pipeline_config configs/pipelines/texture.json \
  --proj_batch_voxels 2048

python chunked_to_glb.py \
  --inputs <raw-output>/to_glb_inputs.pt \
  --chunk_inputs <raw-output>/chunk_inputs.pt \
  --output_dir <raw-output>
```

The official GLB script may return zero after handling an internal exception. Reconevery
therefore validates required tensors, non-empty finite PLY geometry, GLB structure, bounds,
code/checkpoint identities, view order, and hashes rather than trusting return codes alone.

## Outputs and Inspection

Canonical outputs are under `reconstruction/global/` and include request, checkpoint and worker
manifests, metadata, diagnostics, `scene.glb`, `mesh.ply`, raw official intermediates, and three
PNG previews. A Scene IR global visual asset is written to `scene_ir/scene.json`; no collision
asset is created.

```bash
uv run recon2sim reconstruction inspect-global runs/phase3_h100
uv run recon2sim reconstruction render-global-preview runs/phase3_h100
uv run recon2sim reconstruction export-global-mesh \
  runs/phase3_h100 --output global_scene.ply
uv run recon2sim validation inspect-phase3-e2e runs/phase3_h100
uv run recon2sim validation verify-phase3-e2e runs/phase3_h100
```

The COLMAP preview labels axes as arbitrary/unoriented. SAM previews are reused from the SAM
stage. Global previews are diagnostic projections and never semantic inputs.

## Failure Recovery

Failures preserve the attempt workspace, worker stdout/stderr, and partial official output.
Timeout or interruption terminates the process group. A failed retry cannot overwrite the last
successful canonical set. Common actionable failures are:

- wrong Git/submodule commit;
- checkpoint missing or SHA mismatch;
- DINOv3 terms not accepted, missing Hugging Face authentication, or cache access failure;
- PyTorch/CUDA version mismatch;
- CUDA extension import/ABI failure;
- GPU OOM;
- invalid camera package or frame order;
- no cleaned sparse points or chunks;
- missing official intermediate tensors;
- invalid/empty/non-finite PLY or GLB;
- non-invertible working transform.

Reduce selected views, `proj_batch_voxels`, or chunk count for OOM. Rebuild every CUDA extension
after changing PyTorch, CUDA, Python, or GPU architecture.
