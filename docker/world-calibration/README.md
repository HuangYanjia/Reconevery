# Phase 6A world-calibration image

Build from the repository root:

```bash
docker build -f docker/world-calibration/Dockerfile \
  -t reconevery/world-calibration:phase6a .
```

The image builds only official `AprilRobotics/apriltag` commit
`0e16a12dd380fd607e4afd54712ee9b1ffb9ec8f` (BSD-2-Clause). It contains no
checkpoint, calibration input, credential, or run artifact.
