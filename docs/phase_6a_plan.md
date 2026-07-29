# Phase 6A: Canonical Metric World Calibration

## Objective

Phase 6A estimates one evidence-backed positive-scale similarity transform:

```text
p_canonical_m = s * R * p_colmap + t
```

The accepted full-canonical contract is right handed, uses `+X` forward, `+Y`
left, `+Z` up, and measures linear quantities in meters. Source cameras,
geometry, measured assets, and articulation artifacts remain immutable. The
canonical scene is a typed wrapper over those source artifacts.

## Evidence and truthfulness

Supported metric evidence is:

- official AprilTag detections with an explicitly measured detection-edge size;
- explicitly observed and triangulated known-distance landmarks;
- typed external metric trajectories or depth.

Supported up evidence is:

- synchronized IMU gravity;
- an explicitly surveyed fiducial-board orientation;
- an explicit reconstructed bottom-to-top landmark vector;
- a measured dense floor plane with an explicit normal sign.

Manhattan-world estimates are diagnostic only. Object-size guesses and
unreferenced scale constants are invalid evidence.

Forward direction and origin are always explicit. A reference-camera forward
direction is accepted only when the configuration names that policy.

The worker reports partial results honestly:

```text
accepted_full_canonical
accepted_metric_only
accepted_gravity_only
rejected_inconsistent_metric_evidence
rejected_inconsistent_gravity_evidence
rejected_heldout_validation
insufficient_forward_evidence
insufficient_origin_evidence
insufficient_evidence
```

Only `accepted_full_canonical` may set `metric_scale_known` and canonical
alignment metadata together.

## Architecture

The lightweight core contains three filesystem adapters:

```text
calibration_evidence
world_calibration
canonical_scene_wrapper
phase6a_consistency_validation
```

Numerical work lives in `workers/world_calibration`. The core never imports
NumPy, SciPy, OpenCV, AprilTag, Open3D, trimesh, torch, or CUDA.

The worker receives only declared inputs in its attempt workspace. It never
receives the COLMAP database, rejected models, generative checkpoints, model
catalogs, or the canonical run root.

## Data separation

Calibration candidates are fitted and selected using fitting evidence only.
Held-out evidence is reserved for acceptance:

- AprilTag detections split deterministically by registered-frame order;
- landmark observations split per point before triangulation/reprojection;
- floor evidence split by deterministic spatial cells or source frames.

Changing held-out observations may change acceptance and diagnostics, but must
not change the fitted transform.

## Official AprilTag pin

The implementation records an exact commit from the official
`AprilRobotics/apriltag` repository, its license, detector family, tag ID,
configured detection-edge length, detector source identity, and every input
image hash. The official detector is invoked only inside the isolated worker.
Fake CI and synthetic geometry tests do not substitute for a real detector
acceptance run.

## Propagation

Calibration is represented as:

```text
immutable source artifact + accepted world transform
```

Cameras and object transforms compose with the accepted transform. Normals and
axes rotate only. Linear positions, prismatic positions, and verified linear
limits scale once. Revolute positions and angular limits remain radians.
Measured reference-world assets are not rewritten or double transformed.

## Acceptance sequence

1. Validate evidence identities and source hashes.
2. Produce a deterministic fitting/held-out split.
3. Fit metric scale, gravity, forward, and origin without held-out evidence.
4. Construct and validate a proper, invertible Sim(3).
5. Apply held-out gates without changing the fitted candidate.
6. Write the calibration artifact, diagnostics, previews, and canonical wrapper.
7. Validate source immutability and propagation invariants.
8. Repeat the identical run with `--resume`.

Without a real metric source plus accepted gravity, forward, and origin evidence,
the pull request remains draft and the output remains partial calibration.

## Exclusions

Phase 6A does not replace objects, fill geometry, generate collisions, infer
mass/inertia/friction/damping, compile SceneSmith, export production simulator
formats, run physics, or claim simulation readiness.
