# Completion evaluation worker

This isolated CUDA worker prepares fitting-only measured evidence, registers candidate
geometry with a proper positive-scale Sim(3), and renders frozen candidates into
held-out COLMAP cameras. Mesh rendering uses nvdiffrast and the same homogeneous
projection contract as Phase 4.

The worker never loads a generative checkpoint. Candidate generation and held-out
evaluation remain separate processes and evidence sets.

Fitting evidence follows the Phase 5A mask-core, COLMAP consistency, SAM score,
relative depth-discontinuity, and multi-view agreement rules. Dense arrays use the
official Fortran-order COLMAP layout for both scalar depth and three-channel normals.
An open fitting-only measured mesh exercises the same renderer and held-out metric
path as generated mesh candidates.

Registration, evaluation, and selection assets are explicit. Anchor sanity,
fitting-view metrics, held-out metrics, and per-frame pixel/depth/bounding-box
diagnostics classify export, rendering, registration, overfit, depth, and
negative-space failures separately.

The worker's full-frame anchor sanity uses the frozen registration transform.
Backend canonical crop sanity is produced separately by the official renderer using
the captured official crop intrinsics; a crop-local layout is never projected
directly through the full COLMAP image camera.
