# Phase 2 Implementation Plan

Phase 2 replaces only the mock segmentation and tracking stage. It does not add
automatic scene inventory, 3D reconstruction, physical classification, world
alignment, metric scale recovery, or simulator export.

## Pinned official backend

- Official code: `https://github.com/facebookresearch/sam3`
- Code commit: `46957e47805eaa273f4aa7bbbd25a88bca9108ce`
- Default checkpoint repository: `facebook/sam3.1`
- Checkpoint revision: `daa63191845a41281374e725f4c9e51c7a824460`
- Default model mode: SAM 3.1 Object Multiplex
- Worker Python: 3.12
- PyTorch: 2.10.0
- torchvision: 0.25.0
- CUDA build/runtime: 12.8

The official checkpoint is gated. Reconevery will not download it during an
image build, bypass its terms, accept its license for a user, or persist access
tokens. A real smoke test is conditional on official access and compatible
NVIDIA hardware.

## Implementation order

1. Extend the adapter protocol with typed, optional `InputSpec` declarations.
   Resolve and hash only declared ancestor or approved external inputs, copy
   them without writable symlinks, record them in each attempt, and verify the
   canonical sources again after adapter execution. Existing adapters retain
   the full-ancestor fallback.
2. Add strict prompt, request, raw-worker, canonical-track, diagnostics, and
   preview artifact models. Keep prompt asset types as configured hints rather
   than measured physical truth.
3. Implement a deterministic fake worker using the same filesystem protocol as
   the real worker. Exercise success, empty results, multiple objects, malformed
   outputs, invalid masks, process failures, timeouts, and authentication/OOM
   classifications without importing SAM or requiring a GPU.
4. Implement the lightweight `sam3` core adapter. It validates prompt geometry
   against normalized frames, chooses anchors from frame QA and camera
   registration, writes a token-free request, invokes the isolated worker,
   validates raw results, applies deterministic track QA, assigns canonical
   object IDs, and writes binary PNG masks.
5. Generate deterministic mask-outline previews and a track timeline from
   canonical outputs. Add inspect, preview regeneration, and deterministic
   COCO-style export CLI commands that do not invoke the model.
6. Add the separate `workers/sam3` package. The real implementation follows the
   official pinned `build_sam3_predictor` request API and supports SAM 3.1
   multiplex by default, with text, point, box, and mask prompts mapped only
   where supported by that API. Unsupported strategy/version combinations fail
   before inference.
7. Add local-worker, Docker, and fake-worker healthchecks. Healthchecks verify
   configured executables/images, official imports and versions, checkpoint
   access mode, device/CUDA readiness, and token redaction.
8. Add a CUDA 12.8 Docker image that installs the official repository at the
   pinned commit plus the isolated worker, downloads no checkpoint, mounts
   cache/checkpoints at runtime, and maps the Linux host UID/GID.
9. Add production, Docker example, fake, and explicitly mixed downstream
   configurations plus a typed tabletop prompt manifest. Production DAGs end at
   segmentation tracking.
10. Regenerate schemas and update architecture, adapter, Scene IR, roadmap,
    setup, authentication, troubleshooting, and operating documentation.
11. Run Ruff, format checking, strict mypy, pytest, schema checks, the existing
    mock demo, a fake SAM pipeline, its resume pass, and CLI exports. Build or
    run the real model only when official checkpoint access and suitable GPU
    infrastructure are actually present.

## Canonicalization invariants

- Frame order comes from `inputs/manifest.json`, never filenames.
- Unregistered COLMAP frames remain valid 2D observations and are marked
  `camera_pose_available=false`.
- Canonical masks are relative-path grayscale PNG files with values exactly
  `0` or `255`, normalized-frame dimensions, nonzero area, and mask-derived
  boxes.
- Canonical object IDs are derived from normalized label, prompt, first visible
  manifest index, first-mask centroid, area, and raw ID only as a final
  tie-breaker.
- Empty valid model results succeed with `tracks: []`.
- Duplicate suppression stays within one prompt or an explicit synonym group.
- Raw model IDs and raw files remain available for diagnostics, but downstream
  stages consume only normalized artifacts.
- Credentials never enter commands, requests, resolved configuration,
  provenance, diagnostics, or retained logs.
