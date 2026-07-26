# Camera-mesh alignment GPU image

Build from the repository root:

```bash
docker build \
  -f docker/alignment/Dockerfile \
  -t reconevery/alignment:phase4.2 .
```

The image contains Python 3.10, PyTorch 2.6.0/cu126, SciPy, OpenCV, trimesh,
and nvdiffrast commit `253ac4fcea7de5f396371124af597e6cc957bfae`.
The object-lifting worker package is installed only to share the exact camera,
distortion, clipping, and rasterization implementation. No SAM or GenRecon
model is imported or initialized.

No checkpoint, access token, user input, or run output is embedded. Reconevery
runs the container as the current Linux UID/GID and mounts only the isolated
stage-attempt workspace at `/workspace`.
