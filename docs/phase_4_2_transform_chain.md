# Phase 4.2 transform-chain audit

The audit names every available coordinate stage:

1. original arbitrary COLMAP world;
2. GenRecon PCA working world;
3. chunk-local world;
4. official reconstructed working output;
5. GLB conversion output;
6. canonical Phase 3 mesh returned to the COLMAP world.

Each stage records its 4x4 forward/inverse transform, determinant, orthogonality error, uniform
scale, translation, round-trip error, and mesh/camera/sparse bounds. Sparse points, camera centers,
camera basis vectors, and sampled mesh vertices must survive COLMAP -> working -> COLMAP within
the configured tolerance.

When `mesh_working.ply` exists, the worker compares working-mesh/working-camera rendering with
canonical-mesh/original-camera rendering. Fresh Phase 3 runs preserve working PLY/GLB diagnostics
using reflink-or-copy. Older runs remain auditable from the recorded invertible transform, with an
explicit warning that raw pre-canonical geometry was unavailable.

The camera contract stays `transform_world_from_camera`, OpenCV axes (x right, y down, z forward),
and raw arbitrary COLMAP world. A transform-chain failure is reported before Sim(3) fitting; a
global correction must not hide it.
