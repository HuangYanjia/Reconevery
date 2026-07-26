# Dense MVS worker

This isolated worker executes the official COLMAP 4.0.4 dense pipeline:
`image_undistorter`, `patch_match_stereo`, then geometric `stereo_fusion`.
It verifies the exact release, validates official dense map formats, and checks
that independent OpenCV mask/RGB undistortion matches the COLMAP output.

Install this package into the environment containing the pinned official
COLMAP binary. No COLMAP library is imported by the Reconevery core.
