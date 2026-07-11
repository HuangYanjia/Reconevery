# Real video input placeholder

Place exactly one supported capture here, for example `capture.mp4`, then run:

```bash
uv run recon2sim run \
  --input examples/real_video \
  --config configs/colmap.yaml \
  --run-dir runs/real_video_colmap
```

No sample video is committed because Phase 1 should operate on the user's own capture and the
repository quality gate must not depend on a large binary fixture or external tool installation.
