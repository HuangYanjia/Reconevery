# Reconevery SAM 3 worker

This package is an isolated GPU process. The lightweight `recon2sim` package
does not import it and does not depend on PyTorch.

The worker is pinned to:

- Meta SAM repository `https://github.com/facebookresearch/sam3`
- commit `46957e47805eaa273f4aa7bbbd25a88bca9108ce`
- default checkpoint `facebook/sam3.1`
- checkpoint revision `daa63191845a41281374e725f4c9e51c7a824460`
- Python 3.12, PyTorch 2.10.0, torchvision 0.25.0, CUDA 12.8

The worker pins `einops==0.8.2`, `pycocotools==2.0.11`, and
`psutil==7.2.2` because the pinned official predictor import path
requires them while the official core dependency list omits them.

Accept the official checkpoint terms first. Authenticate with `HF_TOKEN` in the
environment, use an authorized `HF_HOME` cache, or configure a readable local
official checkpoint. Tokens are never command arguments or request fields.

For a local environment, install the CUDA 12.8 PyTorch wheels, install the
official repository at the pinned commit, and then install this directory:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git
git -C sam3 checkout --detach 46957e47805eaa273f4aa7bbbd25a88bca9108ce
test "$(git -C sam3 rev-parse HEAD)" = \
  "46957e47805eaa273f4aa7bbbd25a88bca9108ce"
python -m pip install -e ./sam3
python -m pip install -e ./workers/sam3
```

Use `python -m sam3_worker healthcheck --config worker_config.json` before
inference. The healthcheck verifies the exact official code commit, runtime
versions, CUDA/GPU availability, and checkpoint access.

The exact commit is derived from PEP 610 VCS metadata or the editable Git
checkout. The worker does not trust an environment variable as code
provenance, and rejects non-editable local-directory installs whose commit
cannot be verified.

The official public predictor API supports text and box prompting, point
initialization/refinement, and forward/backward propagation. The pinned public
`build_sam3_predictor` request API does not expose mask-seed input; such a
request fails explicitly rather than calling undocumented internal methods.
