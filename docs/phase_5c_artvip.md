# Phase 5C ArtVIP retrieval

ArtVIP retrieval uses a prebuilt local index. Inference never downloads assets. Each
index record contains category, links, joints, bounds, native units/up axis, file
hashes, a normalized `ArticulatedCandidate` bundle, and a per-asset license record.

Ranking uses semantic category, part count, joint type, measured proportions, and
motion direction. RGB appearance is not a primary signal. ArtVIP assets remain
non-production-selectable until the exact repository and asset license receive
project-policy approval.

The lightweight adapter ranks index metadata first and materializes only the
configured top-K bundles and visual files. The isolated retrieval worker rewrites
those files into canonical candidate paths and verifies every hash. It never sees
the full local catalog.
