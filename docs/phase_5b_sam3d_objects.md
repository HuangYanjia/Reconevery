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
invent a Gaussian-to-mesh conversion. The official optional visual GLB is the
explicit registration, evaluation, and selection representation. The native
Gaussian PLY is an independent representation and is rendered with the official
pinned `gaussian_render.py` gsplat path for target-camera parity diagnostics.

Gaussian rendering records the official source path, backend, commit, camera
conversion, valid Gaussian and alpha counts, runtime, peak GPU memory, and dependency
license identity. The pinned renderer does not provide reliable depth through this
path, so Gaussian depth is reported as unavailable rather than fabricated or copied
from the GLB.

`representation_parity.json` compares Gaussian and GLB silhouettes in the canonical
backend anchor camera, the registered anchor camera, and at least two fitting
cameras. It records silhouette IoU, bounding boxes, centroids, scale/orientation
diagnostics, and valid-pixel counts. Failure keeps both representations independent;
it never permits silent transfer of GLB transforms or held-out metrics to the
Gaussian.

The canonical anchor camera is not the full COLMAP image camera. The worker
reconstructs the exact normalized intrinsics used by the pinned official point-map
path, records them as `backend_anchor_camera`, and evaluates the 1024-pixel RGBA
crop in that camera. The official predicted layout is local-to-camera in PyTorch3D
axes; Reconevery applies the explicit `diag(-1, -1, 1)` conversion before OpenCV
rendering. The crop sanity result and the registered full-frame anchor result are
kept separate.

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
evaluation. For native Gaussian PLY it writes `rgba.png`, `valid.png`, and an
official-renderer manifest without a fabricated depth map. Both paths verify the
exact asset representation and official commit before rendering.
