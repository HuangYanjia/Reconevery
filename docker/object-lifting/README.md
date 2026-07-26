# Object-lifting GPU image

Build from the repository root:

```bash
docker build \
  -f docker/object-lifting/Dockerfile \
  -t reconevery/object-lifting:phase4 .
```

The image contains Python 3.10, PyTorch 2.6.0/cu126, and nvdiffrast commit
`253ac4fcea7de5f396371124af597e6cc957bfae`. It contains no SAM or GenRecon
checkpoint, user input, or generated run output.

Reconevery runs it with the current Linux UID/GID, one writable stage-attempt
mount at `/workspace`, and one configured NVIDIA GPU. The canonical run is
never mounted. The configuration-aware healthcheck creates a real CUDA raster
context; image existence alone is not reported as readiness.
