# Measured object geometry worker

This isolated worker reads official COLMAP geometric depth/normal/consistency
maps, maps canonical SAM masks into the exact undistorted frame, backprojects
visible samples with real COLMAP poses, validates them across observations, and
fuses measured surfels.

It does not import GenRecon or load any generative checkpoint. It does not close
holes, create hidden surfaces, or claim metric scale.
