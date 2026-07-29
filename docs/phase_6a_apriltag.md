# Phase 6A AprilTag Calibration

Phase 6A pins the official detector:

```text
repository: https://github.com/AprilRobotics/apriltag
commit: 0e16a12dd380fd607e4afd54712ee9b1ffb9ec8f
license: BSD-2-Clause
pose API: apriltag_pose.h::estimate_tag_pose
```

The isolated worker uses the official Python binding installed by the official
CMake project. It records corner order, family, ID, decision margin, hamming
distance, camera intrinsics, pose error, image path, and image SHA-256.
`image_sources` must identify registered, already-undistorted images with their
own dimensions, pinhole intrinsics, fitting/held-out role, and exact evidence
file hash. The worker rejects changed bytes or dimensions and writes the
official results to `calibration/apriltag_detections.json`.

At the pinned commit, the pose object points are ordered
`(-s,+s,0), (+s,+s,0), (+s,-s,0), (-s,-s,0)` in tag coordinates, and the
official Python detection exposes corners as `lb-rb-rt-lt`. The official pose is
`camera_from_tag`. Reconevery explicitly inverts it to
`tag_from_camera`; camera centers used by the Sim(3) fit are
`-R_tag_from_camera * t_camera_from_tag`. It does not reinterpret pose error as
an angular residual. Held-out angular error is calculated from the registered
camera orientation and the observed tag orientation.

`detection_edge_size_m` is the measured distance between detection corners at
the black/white border. It is not the paper size or outer printed border.

Registered tag detections are split before solving. Fitting detections estimate
the Sim(3); held-out detections test camera-center translation and orientation.
Held-out evidence never selects or changes the fitted transform.

A tag supplies gravity, forward, or origin only when an explicit surveyed
`world_contract` declares signed tag-frame up/forward axes, tag-center origin,
mounting description, angular uncertainty, and origin uncertainty. The worker
derives COLMAP-frame axes and origin from the fitting tag solution, then writes
`calibration/apriltag_world_derivation.json` with exact fitting/held-out pose
hashes and held-out translation/orientation residuals. Users do not paste
fiducial-derived COLMAP vectors manually. An arbitrarily mounted tag supplies
metric evidence but not a canonical world.

No checkpoint or token is used.

The worker healthcheck imports the official binding and fails when its shared
library cannot be loaded. Docker runs `ldconfig`; a local build must either
install the library into the system linker path or explicitly allow and set
`LD_LIBRARY_PATH` for the calibration stage.
