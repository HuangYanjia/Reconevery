# Phase 2: SAM 3 Segmentation and Tracking

Phase 2 answers a bounded prompt-driven question: given selected normalized frames, camera
registration metadata, and explicit semantic or visual prompts, which persistent 2D object
instances appear and what is each visible instance's binary mask?

It does not invent a complete scene vocabulary, recover metric scale, align gravity, reconstruct
3D geometry, classify physical properties, or export a simulator scene.

## Official version and access

Reconevery pins:

| Component | Version |
| --- | --- |
| Official repository | `https://github.com/facebookresearch/sam3` |
| Official code commit | `46957e47805eaa273f4aa7bbbd25a88bca9108ce` |
| Default checkpoint | `facebook/sam3.1` |
| Checkpoint revision | `daa63191845a41281374e725f4c9e51c7a824460` |
| Worker | `0.1.0` |
| Python | `3.12` |
| PyTorch | `2.10.0` |
| torchvision | `0.25.0` |
| CUDA build | `12.8` |

Meta's checkpoint is gated under the official SAM License. The user must accept the official
terms and either authenticate through `HF_TOKEN`, mount an authorized Hugging Face cache, or
provide a readable local official checkpoint. Reconevery does not accept terms, bypass gating,
download from mirrors, embed credentials, or include a checkpoint in its Docker image.

`HF_TOKEN` is passed only as an allowlisted environment variable. It is never placed in
`resolved_config.yaml`, a process argument, request JSON, provenance, diagnostics, or retained
logs. Worker stdout/stderr is redacted before it is retained.

## Architecture

```text
lightweight recon2sim core
  -> validate prompt manifest and input hashes
  -> select anchors from frame QA and camera registration
  -> write typed observations/sam3_request.json
  -> invoke isolated filesystem worker
  -> validate raw worker schema and files
  -> canonicalize IDs, masks, boxes, tracks, diagnostics
  -> render deterministic previews
```

Official SAM and PyTorch are imported only by `workers/sam3`. The core runtime dependencies remain
Pydantic, Pillow, PyYAML, and Typer.

Three execution modes share the same request/output protocol:

- `local_worker`: a separate Python 3.12 CUDA environment;
- `docker`: the optional NVIDIA CUDA image;
- `fake_worker`: a deterministic CPU test process that never imports or runs official SAM.

## Local worker setup

Create an isolated environment and install the exact CUDA build:

```bash
python3.12 -m venv .venv-sam3
.venv-sam3/bin/python -m pip install --upgrade pip
.venv-sam3/bin/python -m pip install \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git .deps/sam3
git -C .deps/sam3 checkout --detach 46957e47805eaa273f4aa7bbbd25a88bca9108ce
.venv-sam3/bin/python -m pip install .deps/sam3
.venv-sam3/bin/python -m pip install workers/sam3
```

Set `worker_python` in a local copy of `configs/sam3.yaml` to the absolute
`.venv-sam3/bin/python` path, then authenticate without placing the token in YAML:

```bash
export HF_TOKEN=...
export HF_HOME="$HOME/.cache/huggingface"
uv run recon2sim adapters healthcheck --config configs/sam3.yaml
```

The local healthcheck verifies the configured Python, worker command, official import and commit,
Python/PyTorch/torchvision/CUDA versions, CUDA availability, GPU name, precision, and authorized
checkpoint access. A worker is not healthy merely because its Python executable exists.

## Docker setup

Build the image from the repository root:

```bash
docker build \
  -f docker/sam3/Dockerfile \
  -t reconevery/sam3:phase2 \
  .
```

Copy `configs/sam3_docker.example.yaml`, replace the example cache path, and run:

```bash
export HF_TOKEN=...
uv run recon2sim adapters healthcheck --config configs/sam3_docker.example.yaml
uv run recon2sim run \
  --input /path/to/video-or-images \
  --config configs/sam3_docker.example.yaml \
  --run-dir runs/scene_sam3
```

Docker health checks run `docker version`, inspect the configured image, record its image ID, pass
`--gpus all`, map the host UID:GID on Linux, mount the cache/checkpoint, and run the real worker
healthcheck inside the container. The NVIDIA Container Toolkit and a compatible driver are
required. The image contains no checkpoint or user data.

For offline mode, set `offline: true` and mount either:

- a cache that already contains the exact repository revision and checkpoint file; or
- `local_checkpoint_path` pointing to an official checkpoint file.

An offline cache miss is an error, not a reason to contact a mirror.

## Prompt manifests

Production is explicitly prompt-driven. `configs/prompts/tabletop.yaml` is a minimal example.
Prompt IDs must be unique, labels nonblank, and exactly one prompt type must be present.

Text concept:

```yaml
- prompt_id: cup
  label: cup
  text: cup
  asset_type_hint: rigid
```

Box seed in absolute normalized-frame pixels:

```yaml
- prompt_id: cup_seed
  label: cup
  frame_id: frame_000010
  box_xyxy: [120, 80, 260, 310]
```

Positive and negative points:

```yaml
- prompt_id: handle_seed
  label: cabinet_handle
  frame_id: frame_000010
  points:
    - {x: 182, y: 144, label: 1}
    - {x: 210, y: 144, label: 0}
```

Binary mask seed:

```yaml
- prompt_id: cabinet_seed
  label: cabinet
  frame_id: frame_000010
  mask_path: prompts/cabinet_seed.png
```

Mask seeds must match the normalized frame dimensions and contain only `0` and `255`. The prompt
schema and fake worker support the full mask contract. The pinned official public
`build_sam3_predictor` request API exposes text, box, and point inputs but does not expose a mask
seed request. The real worker therefore rejects mask-seed inference explicitly instead of using
undocumented private model methods.

Optional prompt fields include confidence override, positive status, synonym group, instance
limit, notes, enabled status, and `asset_type_hint`. A hint is a configured semantic hint, not a
measured physical truth. Tracks may remain `unclassified`.

## Tracking strategy

All selected frames are already available, so Phase 2 defaults to offline/pre-loaded processing.
It does not default to streaming.

The default `detect_then_track` strategy:

1. reads frames in exactly the order stored in `inputs/manifest.json`;
2. selects one or more anchor frames;
3. applies each enabled text or visual prompt;
4. retains every official instance ID as a raw diagnostic ID;
5. uses official SAM 3.1 Object Multiplex for joint instances within a prompt;
6. requests forward, backward, or both-direction propagation;
7. validates and normalizes the returned masks and tracks.

Multiple semantic text concepts use separate official sessions because the pinned public API
resets semantic state when a new text concept is added. Object Multiplex still jointly tracks the
multiple instances found for each concept. This behavior is recorded rather than described as
cross-concept joint inference.

`full_video_text_prompt` is available only when every enabled prompt is text. Unsupported
strategy/prompt combinations fail before canonicalization.

## Anchor selection

Supported strategies are:

- `first_frame`;
- `first_registered_frame`;
- `best_quality_frame`;
- `best_quality_registered_frame`;
- `explicit`.

The default is `best_quality_registered_frame`. The deterministic score is:

```text
0.40 * blur / (blur + 100)
+ 0.25 * variance / (variance + 500)
+ 0.20 * max(0, 1 - abs(brightness - 127.5) / 127.5)
+ 0.15 * camera_pose_available
```

For a registered-only strategy, unregistered frames are removed before ranking. Ties use ingest
manifest order. Diagnostics record the score, reason, strategy, and pose availability.

## Camera pose availability

SAM may produce a valid mask on a frame that COLMAP did not register. Every canonical observation
records `camera_pose_available`. An unregistered frame retains its 2D mask, box, score, and track
membership, but a later 3D stage must not silently treat it as directly eligible for multi-view
fusion.

Phase 2 does not rotate, translate, align, scale, or reinterpret COLMAP poses. The raw world
remains `colmap_arbitrary`, `unoriented`, and `scale_ambiguous`.

## Canonical IDs and masks

Official raw object IDs are retained as `raw_model_object_id` but never exposed as canonical
identity. Tracks sort by:

1. normalized semantic label;
2. prompt ID;
3. first visible ingest-manifest index;
4. first-mask centroid X;
5. first-mask centroid Y;
6. first-mask area;
7. raw ID only as the final tie-breaker.

Labels are lowercased, trimmed, converted to safe underscore-separated identifiers, and mapped to
IDs such as `cup_0001`. Repeating canonicalization over identical raw input produces byte-identical
`object_tracks.json`.

Each visible observation writes:

```text
observations/masks/<object_id>/<frame_id>.png
```

Canonical masks are grayscale mode `L`, contain exactly `0` or `255`, match the normalized frame
dimensions, have nonzero area, and use deterministic PNG bytes. Area and `bbox_xywh` are computed
from the canonical mask and checked against the artifact. Raw logits or encodings remain under
`observations/raw/`. The official predictor's normalized `out_boxes_xywh` values are converted to
pixel-space `model_box_xyxy` and checked against the mask-derived canonical box.

Track confidence uses `0.8 * mean_frame_score + 0.2 * coverage_ratio`. An empty valid result uses
artifact confidence `1.0` to represent successful protocol validation, not object-presence
certainty.

## Track QA and empty results

Configurable deterministic QA includes score and mask thresholds, minimum mask area, maximum area
ratio, minimum observations, minimum coverage, same-prompt duplicate IoU, and model-box/mask IoU.

The adapter detects and reports:

- missing, empty, wrongly sized, or non-binary masks;
- invalid, non-finite, out-of-range, or inconsistent scores and boxes;
- unknown frames or prompts;
- duplicate object/frame observations;
- worker objects with no observations;
- short or low-coverage tracks;
- nearly identical tracks within the same prompt or synonym group.

Different labels such as `cabinet`, `drawer`, and `cabinet_handle` are not merged solely because
their masks overlap. Every dropped track has a reason code and explanation.

No matching object is a valid successful result. It produces `tracks: []`, typed diagnostics, and
previews. It is distinct from a missing checkpoint, failed worker, or malformed output.

## Selective input materialization

The SAM attempt receives only:

```text
inputs/manifest.json
inputs/frame_qa.json
camera/reconstruction.json
frames/*.png
observations/prompt_inputs/prompts.yaml
observations/prompt_inputs/masks/*.png  # when configured
```

It does not receive the COLMAP database, logs, sparse models, camera diagnostics, reconstruction
outputs, or compiler artifacts. The runner verifies source hashes before copying, prefers reflink
with copy fallback, avoids writable canonical symlinks, rechecks canonical upstream hashes after
the worker, and records materialized inputs in the attempt manifest.

Changing prompt-manifest bytes invalidates segmentation and its dependents, but not ingest or
camera recovery. The source YAML and any seed masks are promoted as immutable stage-owned
artifacts; `prompts.json` records each hash, and the inference request hashes the complete
normalized prompt artifact.

## Diagnostics and derived exports

A successful stage writes:

```text
observations/
  prompt_inputs/prompts.yaml
  prompt_inputs/masks/*.png
  prompts.json
  sam3_request.json
  worker_manifest.json
  object_tracks.json
  diagnostics.json
  masks/<object_id>/<frame_id>.png
  raw/worker_result.json
  raw/worker_manifest.json
  raw/logs/
  previews/contact_sheet.png
  previews/track_timeline.png
  previews/frames/<frame_id>.png
```

Previews draw deterministic mask outlines, IDs, labels, scores, frame IDs, track visibility, and
camera registration. They are diagnostic outputs, never semantic inputs.

Inspect and regenerate without running SAM:

```bash
uv run recon2sim segmentation inspect runs/scene_sam3
uv run recon2sim segmentation render-preview runs/scene_sam3
uv run recon2sim segmentation export-coco \
  runs/scene_sam3 --output runs/scene_sam3/annotations.json
```

COCO category, image, annotation, and track IDs are deterministic. COCO is a convenience export;
`object_tracks.json` plus canonical masks remain authoritative.

## Failure recovery

Every worker attempt runs inside `work/segmentation_tracking/attempt_<N>`. Commands are argument
arrays with no shell. stdout/stderr and partial raw output remain in the attempt. Timeout or user
interruption terminates the process group with TERM and KILL fallback. `KeyboardInterrupt` and
`SystemExit` are not retried.

Outputs are validated before transactional promotion. A failed, timed-out, unauthorized, OOM, or
malformed attempt cannot use stale outputs and cannot overwrite the previous successful
canonical observation set.

Common failures:

- **terms/authentication**: accept official terms and expose `HF_TOKEN` only in the environment;
- **offline cache miss**: prepopulate the exact pinned revision or mount a local official file;
- **commit mismatch**: reinstall official code at the exact pinned commit;
- **CUDA unavailable/driver mismatch**: verify NVIDIA driver, toolkit, and container runtime;
- **OOM**: reduce frames or tracked objects, or use a GPU with more memory;
- **unsupported precision**: use `bfloat16` for the pinned official video predictor;
- **mask seed unsupported**: use a supported text, box, or point request for this pinned backend;
- **no object**: inspect thresholds and diagnostics; this is not itself a worker failure.

## CPU fake validation

Mandatory CI never downloads a checkpoint, builds the GPU image, imports official SAM, or requires
Docker/GPU:

```bash
uv run recon2sim run \
  --input examples/tabletop \
  --config configs/sam3_fake.yaml \
  --run-dir runs/tabletop_sam3_fake
uv run recon2sim run \
  --input examples/tabletop \
  --config configs/sam3_fake.yaml \
  --run-dir runs/tabletop_sam3_fake \
  --resume
```

The second run must report cache hits for ingest, camera recovery, and segmentation. This validates
the filesystem protocol and canonicalization, not the official model.

## Real smoke acceptance

A real smoke may be claimed only when the official code and checkpoint actually execute and the
worker records the exact commit/revision. At least one prompt must complete with valid tracks or a
typed no-object result; canonical masks, previews, inspection, and a resume cache hit must succeed.

Fake-worker output, an image import check, or a Docker build is not a real SAM reconstruction.
