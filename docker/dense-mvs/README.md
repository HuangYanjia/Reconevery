# Dense MVS image

Build from the repository root:

```bash
docker build -f docker/dense-mvs/Dockerfile -t reconevery/dense-mvs:phase5a .
docker run --rm --gpus all --entrypoint colmap reconevery/dense-mvs:phase5a -h
```

The image builds official COLMAP 4.0.4 at commit
`9c23f6942fe69962e06030905e77067c8673382f`. It contains no inputs or model
checkpoints. The adapter maps the attempt workspace and the host UID/GID.
