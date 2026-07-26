# SAM 3D Objects worker

This isolated worker pins official `facebookresearch/sam-3d-objects` commit
`f91db411c50efee93d8db7aeb323885650f6f722` and official checkpoint revision
`05929e2a63f234014031f9941f4aabefea5f382e`.

Checkpoint access is gated. Accept the official terms and provide `HF_TOKEN` or a
mounted verified local checkpoint. Tokens are never command-line arguments or
request fields. Set `generation_configuration.official_checkout_path`,
`checkpoint_root`, and `pipeline_config`; every configured checkpoint file hash is
verified before inference.

The worker preserves the native Gaussian PLY and an official optional mesh when the
pinned output exposes one. It does not invent a mesh conversion.

Generation captures the exact normalized crop-camera intrinsics returned by the
pinned official point-map path without changing inference. The predicted layout is
recorded in its native PyTorch3D camera convention; target rendering applies the
explicit PyTorch3D-to-OpenCV axis conversion. This keeps canonical crop sanity
separate from registered full-frame COLMAP evaluation.

`python -m sam3d_objects_worker render` supports the native Gaussian PLY through the
official pinned `gaussian_render.py` gsplat backend and the optional visual GLB
through nvdiffrast. Gaussian depth is explicitly unavailable when the official path
does not expose it. Every render manifest identifies the exact representation,
renderer source, official commit, camera conversion, valid-pixel count, runtime, and
peak GPU memory.

The Gaussian and GLB are separate representations. Registration, evaluation, or
selection results may not move between them without a passing representation-parity
artifact.
