# Reconevery camera-mesh alignment worker

This isolated worker audits the Phase 3 transform chain and tests one bounded
global similarity transform between the immutable GenRecon mesh and the fixed
COLMAP camera frame. It uses the Phase 4 projection and nvdiffrast code rather
than importing heavy dependencies into the Reconevery core.

Install it in the Phase 3/4 GPU environment:

```bash
python -m pip install -e workers/object_lifting
python -m pip install -e workers/alignment
```

Healthcheck and inference use filesystem-only typed protocols:

```bash
python -m alignment_worker healthcheck --config worker_config.json
python -m alignment_worker infer \
  --request reconstruction/alignment/request.json \
  --input-root /path/to/attempt \
  --output-dir reconstruction/alignment
```

Only the attempt workspace is exposed. The worker loads no SAM or GenRecon
checkpoint and never rewrites canonical cameras or mesh bytes.
