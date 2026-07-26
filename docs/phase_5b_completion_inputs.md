# Phase 5B completion inputs

Phase 5B treats Phase 5A measured geometry as immutable evidence. Each eligible
object receives three disjoint sets: generation anchors, registration fitting frames,
and held-out validation frames.

Anchor ranking combines SAM confidence, mask area, frame QA, dense-depth validity,
measured sample count, and deterministic camera-direction diversity. Crops use
canonical normalized RGB and SAM alpha, square transparent padding, no geometric
stretching, and exact source/crop homographies.

The worker-visible fitting stage contains no held-out mask or depth bytes. Changing
held-out evidence therefore leaves candidate generation and registration signatures
unchanged.

Phase 5B generation, registration, evaluation, selection, and validation use
content-only `InputSpec` signatures over their exact declared files. Training surfel
changes cannot invalidate a generator that consumes only eligibility, split, crop
metadata, and crop PNGs. A crop, seed, backend pin, checkpoint hash, candidate asset,
or fitting-evidence byte change still invalidates the appropriate stage and all
actual consumers. Other Reconevery stages keep producer-signature invalidation by
default.
