# Phase 5C acceptance

CPU validation:

```bash
uv sync --all-groups --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv run python scripts/generate_schema.py
```

Fake protocol and resume:

```bash
uv run recon2sim run --input examples/tabletop \
  --config configs/phase5c_e2e_fake.yaml --run-dir runs/phase5c_fake
uv run recon2sim run --input examples/tabletop \
  --config configs/phase5c_e2e_fake.yaml --run-dir runs/phase5c_fake --resume
uv run recon2sim validation verify-phase5c runs/phase5c_fake
```

Real acceptance requires three independently valid Phase 5A state runs, accepted
static alignment, a non-empty measured movable part in every state, measured
prismatic/revolute evidence, official Particulate execution, explicit link
assignment, frozen held-out evaluation, consistency validation, preview inspection,
and an identical all-stage cache hit. Truthful rejection of all visual priors is
valid.

Before expensive real workers, `articulation preflight-capture` must pass for the
three Phase 5A runs and explicit stable-part/state-track mappings. Acceptance counts
the distinct accepted generation, fitting, and actually evaluated held-out states.
Missing held-out cameras, target masks, visible renders, dense depth, fitted-model
hashes, or joint-type-specific metrics fail closed. Selection and Scene IR must
reference the exact fitted model and evaluation hashes.

## Current real validation

The official Particulate module smoke completed on one H100 using the pinned code
and both pinned checkpoints. The official cabinet input produced one native
candidate with three links and two revolute joints. The measured worker runtime was
52.24 seconds, peak GPU memory was 61,559,799,808 bytes, peak host memory was
1,296,474,112 bytes, and no candidate failed. The run used the explicit configured
`+Z` Particulate working-axis prior; no alternate axis hypothesis or gravity
alignment was claimed.

This is module-level validation only. The current machine does not contain three
independent closed/intermediate/open Phase 5A runs for one articulated object and
does not contain registered local ArtVIP or PartNet-Mobility indices. Consequently
the real multi-state smoke, real retrieval comparison, preview inspection, and real
all-stage resume acceptance remain blocked. The pull request must stay draft until
those inputs are supplied and those checks pass.

Phase 5C does not generate collision, inertia, mass, friction, damping, motors,
metric scale, gravity alignment, or a simulation-ready asset.
