# Phase 5A Dense MVS

## Role

Phase 5A adds a measured-geometry branch independent of GenRecon:

```text
normalized RGB + selected COLMAP sparse model
  -> official COLMAP image_undistorter
  -> official COLMAP patch_match_stereo (geometric consistency)
  -> official COLMAP stereo_fusion
```

The official source is <https://github.com/colmap/colmap>, release `4.0.4`,
commit `9c23f6942fe69962e06030905e77067c8673382f`, BSD-3-Clause.
The worker rejects another version. The core package does not import COLMAP,
pycolmap, NumPy, OpenCV, CUDA, or a dense reconstruction library.

## Workspace and lineage

Only registered normalized frames and the selected `cameras.bin`, `images.bin`,
and `points3D.bin` enter the attempt. The COLMAP database, rejected models, raw
SAM outputs, GenRecon tensors, and checkpoints are absent.

The dense workspace records a reversible mapping among:

- canonical frame ID and normalized PNG hash;
- original COLMAP image ID;
- dense workspace filename;
- source and undistorted dimensions;
- undistorted camera ID and PINHOLE intrinsics.

Registered frames retain `inputs/manifest.json` order. Unregistered frames are
never passed to PatchMatch.

## Exact undistortion

`image_undistorter` owns the dense camera output. Reconevery parses that output
and independently reconstructs the OpenCV remap from the original camera model:
`SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, or `OPENCV`.
The independently remapped RGB is compared with COLMAP's image. Canonical masks
later use the exact same map with nearest-neighbor sampling and are forced back
to values 0 and 255.

## Dense formats

Depth and normal files use the official header:

```text
width&height&channels&
<little-endian float32, column-major>
```

Geometric depth has one channel; normals have three. Consistency graphs have
the same header followed by little-endian int32 entries:

```text
column row source_count source_image_indices...
```

Parsers reject truncated payloads, inconsistent dimensions, invalid channel
counts, and out-of-range image indices.

## Commands

```bash
uv run recon2sim run \
  --input /absolute/path/to/scene \
  --config configs/dense_mvs.yaml \
  --run-dir runs/dense_scene

uv run recon2sim dense inspect runs/dense_scene
uv run recon2sim dense inspect-frame runs/dense_scene frame_000010
uv run recon2sim dense export-fused runs/dense_scene --output fused.ply
```

Official COLMAP 4.0.4 PatchMatchStereo is CUDA-only. In particular,
`gpu_index=-1` selects all available GPUs; it does not select a CPU backend.
`configs/dense_mvs_cpu.example.yaml` therefore demonstrates the deterministic
fake-worker protocol used by mandatory CPU-only CI. Real local and Docker
configurations reject `use_gpu=false` instead of silently using a GPU.

## Failure recovery

All commands use argument arrays and preserved stdout/stderr. Timeout or
interruption terminates the process group. A failed attempt remains under
`work/dense_mvs/attempt_N` and cannot replace the last canonical result.
Reduce `max_image_size` or `patchmatch_cache_size_gb` after GPU OOM.

Coordinates remain `colmap_arbitrary`, `unoriented`, `arbitrary_units`, and
`scale_ambiguous`.
