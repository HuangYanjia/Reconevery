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
