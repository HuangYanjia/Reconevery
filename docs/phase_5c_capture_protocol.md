# Phase 5C capture protocol

An articulation capture contains separate static-state Reconevery runs. The object
and camera may move between sweeps, but the movable part must remain stationary
inside each sweep. Never run COLMAP dense MVS over continuously moving articulation.

Each state must pass real ingest, COLMAP, SAM 3.1, dense MVS, measured geometry, and
the Phase 5A validator. The typed capture manifest records hashes for every lineage.
Stable `part_id` values come from an explicit schema `0.2.0` part manifest and remain
constant across states. `prompt_id` identifies only the semantic prompt. Every
capture-state record must map each stable part to that run's state-local canonical
SAM ID through `part_track_ids`; identical SAM IDs across independent runs are never
assumed. Legacy fake fixtures are migrated by the fake capture adapter, but real
captures require this explicit mapping. Each state therefore needs only Phase 5A outputs. Phase 5B articulated routing is a
separate `phase5b_selection` input to the capture adapter; it is not rerun for every
state. An explicit research override is recorded when no routing artifact is used.

The capture stage promotes only declared held-out evidence beneath
`reconstruction/articulation/measured_states/<state>/evidence`: camera
reconstruction, tracking, undistortion, rewritten depth manifest, and reflinked
depth maps. Raw COLMAP, SAM workspaces, checkpoints, and source run roots are not
visible to downstream workers.

For three states, the default split uses the first state for candidate generation,
intermediate states for kinematic fitting, and the last state for held-out
validation. Held-out geometry is absent from fitting worker workspaces.

Run preflight before expensive workers:

```bash
uv run recon2sim articulation preflight-capture \
  --capture-manifest configs/articulation_capture.yaml \
  --part-manifest configs/articulation_parts/cabinet_drawer.yaml
```

`articulation capture-template --object-id cabinet_0001
--states closed,half_open,open` emits a mapping template. Capture tier describes
configured states only. Effective evidence is downgraded from accepted alignment
states and becomes `multi_state_heldout_validated` only after a frozen candidate
passes held-out gates.
