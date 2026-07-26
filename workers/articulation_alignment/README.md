# Articulation alignment worker

This isolated worker aligns independently reconstructed static articulation states
with static base geometry only, then estimates measured fixed, prismatic, revolute,
or unknown motion. It never rewrites cameras or source geometry and never treats its
arbitrary COLMAP units as metric.

```bash
python -m articulation_alignment_worker healthcheck
python -m articulation_alignment_worker align --request request.json \
  --input-root /workspace --output-dir /workspace/reconstruction/articulation
python -m articulation_alignment_worker estimate-motion --request request.json \
  --input-root /workspace --output-dir /workspace/reconstruction/articulation
```
