# Phase 5B acceptance

CPU acceptance:

```bash
uv run recon2sim run --input examples/tabletop \
  --config configs/phase5b_e2e_fake.yaml --run-dir runs/phase5b_fake
uv run recon2sim run --input examples/tabletop \
  --config configs/phase5b_e2e_fake.yaml --run-dir runs/phase5b_fake --resume
```

Useful inspection:

```bash
recon2sim completion inspect <run-dir>
recon2sim completion compare-candidates <run-dir> cup_0001
recon2sim completion explain-selection <run-dir> cup_0001
recon2sim completion render-previews <run-dir>
recon2sim validation verify-phase5b <run-dir>
```

Real readiness additionally requires one H100 smoke for both official backends,
verified code/checkpoint identities, native target-camera rendering, registration,
held-out evaluation, preview inspection, consistency, and full cache hits. The PR
remains draft when access or a compatible environment is unavailable.

Use a gitignored local configuration with absolute worker/checkpoint paths. Supply
credentials only in the environment, accept gated terms separately, and prefetch
official snapshots before enabling offline mode:

```bash
export HF_TOKEN=...  # entered in the shell, never placed in YAML or arguments
export HF_HOME="$HOME/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=0

uv run recon2sim adapters healthcheck \
  --config configs/local/phase5b_h100.yaml

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run recon2sim run \
  --input /absolute/path/to/phase5a-lineage \
  --config configs/local/phase5b_h100.yaml \
  --run-dir /absolute/path/to/phase5a-lineage

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run recon2sim run \
  --input /absolute/path/to/phase5a-lineage \
  --config configs/local/phase5b_h100.yaml \
  --run-dir /absolute/path/to/phase5a-lineage \
  --resume
```

Read `reconstruction/completion/diagnostics.json`, candidate worker logs, and
`validation/phase5b_rigid_completion.json` before declaring success. A backend
failure, license block, or candidate failing held-out gates is a valid unresolved
result, but it is not a successful official-backend smoke. Partial worker output
remains in the isolated attempt workspace; previous canonical output remains intact.
