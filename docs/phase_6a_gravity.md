# Phase 6A Gravity Evidence

Accepted up evidence, from strongest to weakest, is synchronized IMU gravity,
surveyed fiducial orientation, explicit bottom-to-top reconstructed landmarks,
and a measured dense floor plane with an explicit sign.

Video rotation metadata is not IMU evidence. A dominant plane without a floor
mask is not floor evidence. GenRecon geometry is not the primary floor source.

Floor fitting records the mask/source hashes, point count, spatial extent,
normal, sign policy, fitting residual, held-out residual, and held-out normal
error. Floor evidence establishes orientation only; it does not establish
metric scale.

High-trust gravity records that disagree beyond the configured angular gate are
rejected rather than averaged. Manhattan-world estimates remain diagnostic
semantic priors and cannot establish gravity by themselves.
