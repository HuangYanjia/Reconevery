# Reconevery COLMAP image

Build the optional Phase 1 image from the repository root:

```bash
docker build -t reconevery/colmap:phase1 docker/colmap
docker version
docker image inspect reconevery/colmap:phase1
```

The base tag `colmap/colmap:20260427.6785` is pinned to an official image from the
COLMAP 4.0.4 release period. The image adds FFmpeg only. It contains no user data,
checkpoints, or downloadable models. The current official base is Linux/amd64; on
other architectures Docker may use emulation.
