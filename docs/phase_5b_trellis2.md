# Official TRELLIS.2 backend

Reconevery pins:

```text
repository: https://github.com/microsoft/TRELLIS.2
commit: 75fbf0183001ed9876c8dbb35de6b68552ee08bd
checkpoint repository: microsoft/TRELLIS.2-4B
checkpoint revision: af44b45f2e35a493886929c6d786e563ec68364d
submodule o-voxel/third_party/eigen:
  21e4582d1739107337a03460c81412981130373e
```

The isolated worker loads the exact local snapshot offline, calls
`Trellis2ImageTo3DPipeline`, and exports the native PBR GLB through the official
O-Voxel path. Every configured runtime file is hashed before use.

The pinned pipeline also resolves two runtime model repositories. They are separate
model identities and must be prefetched and verified explicitly:

```text
facebook/dinov3-vitl16-pretrain-lvd1689m
  revision: ea8dc2863c51be0a264bab82070e3e8836b02d51

microsoft/TRELLIS-image-large
  revision: 25e0d31ffbebe4b5a97464dd851910efc3002d96
```

Set `runtime_model_revisions`, `runtime_model_hashes`, and matching
`generation_configuration.runtime_model_paths`. Healthcheck rejects a missing,
unhashed, or revision-mismatched runtime model.

Reconevery supplies a canonical RGBA crop whose alpha is the canonical SAM mask.
The worker therefore loads the official pipeline without initializing BiRefNet and
rejects opaque crops. The official TRELLIS.2 preprocessing path uses the supplied
alpha directly, so no third-party background-removal model runs or changes the
evidence.

The direct repository is MIT. Production selection remains disabled until all
transitive runtime licenses, including O-Voxel and CUDA extensions, are inventoried
and approved.

Clone recursively and verify the fixed checkout before installing the worker:

```bash
git clone --recursive https://github.com/microsoft/TRELLIS.2 \
  /absolute/path/to/TRELLIS.2
git -C /absolute/path/to/TRELLIS.2 checkout --detach \
  75fbf0183001ed9876c8dbb35de6b68552ee08bd
git -C /absolute/path/to/TRELLIS.2 submodule update --init --recursive

/absolute/path/to/trellis2-env/bin/python -m pip install -e \
  /absolute/path/to/TRELLIS.2
/absolute/path/to/trellis2-env/bin/python -m pip install -e workers/trellis2_objects
```

The healthcheck performs a real official import, verifies CUDA and every configured
snapshot hash, and reports the H100 and runtime versions. Inference uses one bounded
candidate at a time. O-Voxel GLB export is part of the official smoke; successful
latent sampling without a validated PBR GLB is reported as a backend failure.

`python -m trellis2_objects_worker render` accepts a typed target-camera request plus
`--input-root` and `--output-dir`. It renders the official PBR GLB geometry through
the same exact homogeneous nvdiffrast projection used for held-out evaluation and
writes RGBA, depth, valid-pixel, and hash-bearing manifest outputs.
