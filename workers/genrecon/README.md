# Reconevery GenRecon Worker

This package is installed only inside an isolated Python 3.10 GenRecon environment.
It is not a dependency of the lightweight `recon2sim` package.

Official identity:

- repository: `https://github.com/kasothaphie/GenRecon`
- commit: `eaf1468118d20469d17079a4a19737297d2ef87b`
- Eigen submodule: `21e4582d1739107337a03460c81412981130373e`
- license: MIT; CUDA/rendering dependencies retain their own licenses

After installing the exact official environment and CUDA extensions:

```bash
python -m pip install -e workers/genrecon
python -m genrecon_worker healthcheck --config /path/to/worker_config.json
python -m genrecon_worker infer \
  --request /path/to/run/reconstruction/global/request.json \
  --output-dir /path/to/run/reconstruction/global/raw
```

The worker:

1. verifies the official checkout and recursive submodule;
2. hashes all three official checkpoints;
3. verifies access to official gated `facebook/dinov3-vitl16-pretrain-lvd1689m`;
4. caches and records exact revisions for DINOv3, `microsoft/TRELLIS-image-large`, and
   `microsoft/TRELLIS.2-4B` without exposing credentials;
5. builds an ephemeral COLMAP/rgb input package;
6. applies a recorded reversible PCA working transform;
7. invokes the official `reconstruct_scene.py`;
8. invokes the official `chunked_to_glb.py`;
9. transforms the final PLY and GLB back to the original COLMAP arbitrary frame;
10. writes a typed worker manifest and structured diagnostics.

Checkpoint URLs and paths are never downloaded during image build.

The pinned official CLI exposes `--skip_point_cleaning` but omits that argument
from `IphoneChunker.__init__`. For that option only, the worker launches the
unchanged official script through `official_launcher.py`, which adds the missing
constructor parameter and delegates all reconstruction work to official code.
