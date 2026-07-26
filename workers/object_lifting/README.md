# Reconevery object-lifting worker

This isolated worker projects the Phase 3 global mesh into registered COLMAP
cameras, intersects nearest-visible original face IDs with canonical SAM masks,
and extracts partial observation-supported object surfaces. It does not load SAM,
GenRecon, DINO, or generative checkpoints.

Install it in the Phase 3 GPU environment:

```bash
/path/to/genrecon-env/bin/python -m pip install -e workers/object_lifting
```

The real backend requires CUDA, PyTorch, and `nvdiffrast`. The package keeps
those dependencies outside Reconevery's lightweight core because their exact
build is environment-specific.

```bash
CUDA_VISIBLE_DEVICES=0 /path/to/genrecon-env/bin/python \
  -m object_lifting_worker healthcheck --config worker_config.json
```

All camera transforms remain in the arbitrary, unoriented, scale-ambiguous
COLMAP frame. Extracted surfaces are incomplete and are not simulation-ready.
