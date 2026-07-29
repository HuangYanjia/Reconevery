# Articulation evaluation worker

This isolated worker explicitly assigns observed parts to candidate links, fits one
global Sim(3) and constrained joint positions on fitting states, freezes structure,
then evaluates held-out states. It never changes cameras, source measurements, link
geometry, graph topology, or joint type. Missing camera-render evidence yields failed
silhouette gates rather than fabricated overlap.

Fitting receives only accepted fitting states and records a bounded fitting-view
reprojection IoU. Held-out states are absent from that workspace. During held-out
evaluation, a bounded one-dimensional optimizer may infer only each scalar joint
position from the measured part cloud; the fitted base Sim(3), axis, pivot, graph,
and visual geometry remain frozen.

For held-out states it consumes only the Phase 5C evidence package: real COLMAP
camera poses, the official dense undistortion manifest, geometric depth maps, and
canonical part masks. Link meshes are rendered with exact homogeneous projection
through nvdiffrast, then classified against dense scene depth for visibility,
occlusion, negative space, and front-of-scene violations.
