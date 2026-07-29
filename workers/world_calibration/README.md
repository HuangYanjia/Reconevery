# World calibration worker

This isolated worker estimates Phase 6A metric/canonical Sim(3) transforms. It is
the only Phase 6A component allowed to import NumPy, SciPy, OpenCV, or the
official AprilTag Python binding.

Official AprilTag source:

```text
repository: https://github.com/AprilRobotics/apriltag
commit: 0e16a12dd380fd607e4afd54712ee9b1ffb9ec8f
project version: 3.4.5
license: BSD-2-Clause
pose API: apriltag_pose.h::estimate_tag_pose
```

Build/install the official repository at that exact commit into this worker
environment. The detector uses the configured detection-edge length, not paper
or outer-border dimensions.

The verified local build used CMake 3.22.1 and GCC 11.4.0. The isolated
validation environment used Python 3.12.13, NumPy 2.5.1, SciPy 1.18.0, and
OpenCV 4.14.0.94. CUDA is not used. Runtime versions are recorded in
`calibration/diagnostics.json`; Docker or later builds record their own resolved
versions rather than inheriting these values.

The worker receives an attempt-local input root and never the canonical run
root. It has no model checkpoint and requires no credentials.
