# Official SAM 3D Objects backend

Reconevery pins:

```text
repository: https://github.com/facebookresearch/sam-3d-objects
commit: f91db411c50efee93d8db7aeb323885650f6f722
checkpoint repository: facebook/sam-3d-objects
checkpoint revision: 05929e2a63f234014031f9941f4aabefea5f382e
```

Configure `official_checkout_path`, `checkpoint_root`, `pipeline_config`, and exact
checkpoint hashes under `generation_configuration`. Accept gated terms separately
and use `HF_TOKEN` or an authorized mounted cache. Tokens never belong in YAML,
requests, command arguments, logs, or provenance.

The worker calls the official `notebook.inference.Inference` path. It preserves the
native Gaussian PLY and any mesh actually exposed by the pinned output. It does not
invent a Gaussian-to-mesh conversion. The current evaluation path registers and
renders the official optional visual GLB with nvdiffrast while retaining the
Gaussian PLY as an independent native asset. It does not claim to use the official
Gaussian renderer.

The default policy records the SAM License as research-only and blocks production
selection pending explicit legal approval.

Preferred setup uses an exact detached checkout and a dedicated environment:

```bash
git clone https://github.com/facebookresearch/sam-3d-objects \
  /absolute/path/to/sam-3d-objects
git -C /absolute/path/to/sam-3d-objects checkout --detach \
  f91db411c50efee93d8db7aeb323885650f6f722

/absolute/path/to/sam3d-env/bin/python -m pip install -e \
  /absolute/path/to/sam-3d-objects
/absolute/path/to/sam3d-env/bin/python -m pip install -e workers/sam3d_objects
```

Healthcheck verifies the checkout, exact snapshot revision and file hashes, CUDA
availability, and official import. Missing terms/authentication, absent cached
files, commit mismatch, and CUDA/OOM failures are distinct errors. Partial Gaussian
or GLB output never becomes canonical unless the full candidate protocol validates.

`python -m sam3d_objects_worker render` accepts a typed target-camera request plus
`--input-root` and `--output-dir`. For the official optional visual GLB it writes
`rgba.png`, `depth.npy`, `valid.png`, and a hash-bearing `render_manifest.json`
through the same exact homogeneous nvdiffrast projection used by held-out
evaluation. The native Gaussian PLY remains an independent asset and is not
mislabeled as a triangle surface.
