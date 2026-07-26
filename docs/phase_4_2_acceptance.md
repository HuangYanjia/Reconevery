# Phase 4.2 acceptance

## CPU and fake protocol

```bash
uv sync --all-groups --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv run python scripts/generate_schema.py

uv run recon2sim run \
  --input examples/tabletop \
  --config configs/phase4_2_e2e_fake.yaml \
  --run-dir runs/phase4_2_fake
uv run recon2sim run \
  --input examples/tabletop \
  --config configs/phase4_2_e2e_fake.yaml \
  --run-dir runs/phase4_2_fake \
  --resume
uv run recon2sim validation verify-phase4-2 runs/phase4_2_fake
```

The fake worker covers identity, scale/rigid/full Sim(3), transform-chain failure, no validation
improvement, implausible transforms, insufficient evidence, correspondence collapse, local
deformation, hash mismatch, path escape, timeout, interruption, OOM, and malformed output.

## Real H100

Install `workers/object_lifting` and `workers/alignment` into the existing isolated Phase 3/4
Python 3.10 environment. Use a local ignored config derived from `configs/phase4_2_e2e.yaml`, one
H100 through `CUDA_VISIBLE_DEVICES=0`, and the same Phase 4.1 lineage:

```bash
uv run recon2sim adapters healthcheck --config configs/local/phase4_2_h100.yaml
uv run recon2sim run \
  --input /absolute/path/to/the/same/real-scene \
  --config configs/local/phase4_2_h100.yaml \
  --run-dir runs/phase4_2_h100
uv run recon2sim alignment inspect runs/phase4_2_h100
uv run recon2sim alignment render-previews runs/phase4_2_h100
uv run recon2sim alignment compare-object-lifting runs/phase4_2_h100
uv run recon2sim validation verify-phase4-2 runs/phase4_2_h100
```

Inspect all alignment previews. The transform may be accepted or rejected. Readiness requires a
truthful held-out result, unchanged camera/mesh hashes, completed object-lifting comparison,
passing consistency report, and cache hits on the identical resumed command.

## Recorded H100 result

The Phase 4.2 smoke reused the Phase 4.1 lineage: 16 normalized frames, 12 registered cameras,
approximately 2,461 COLMAP sparse points, four SAM tracks, 50 canonical masks, and the same
7,677,700-face GenRecon mesh. Six registered frames and disjoint point IDs were used for fitting;
six frames and disjoint point IDs were held out.

The transform chain coordinate round trips were below `1.4e-14`. Independent nvdiffrast renders
of the working-frame and COLMAP-frame paths had median relative depth error below `1e-7` and
silhouette IoU `1.0`. The best candidate had scale `0.926493`, rotation `11.7756` degrees, and
translation `0.141018` scene diagonals. On held-out evidence, median normalized depth residual
improved from `0.662246` to `0.372986`, p90 improved from `0.787134` to `0.530112`, and the `0.10`
inlier fraction improved from `0.011430` to `0.140841`. Coverage changed from `0.998527` to
`0.958830`.

The result is nevertheless `global_sim3_insufficient`: the post-fit inlier fraction remains low,
and residuals remain structured by GenRecon chunk. The candidate is recorded but not consumed by
object lifting. Cameras and original mesh bytes remained unchanged, the 21-check consistency
report passed, and the identical resumed command cache-hit alignment, object lifting, and both
Phase 4 validators.

No result establishes metric scale, gravity, hidden surfaces, collisions, or simulation readiness.
