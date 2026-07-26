# Phase 4.2 implementation plan

1. Verify the Phase 4 squash merge and preserve all Phase 3/4 tests.
2. Define typed transform-chain, sparse-observation, split, candidate, alignment, diagnostic,
   comparison, preview, and consistency artifacts.
3. Add selective attempt inputs and an isolated local/Docker/fake alignment worker.
4. Audit COLMAP -> GenRecon working/chunk -> canonical transforms before optimization.
5. Prepare distortion-consistent sparse depth evidence and disjoint held-out splits.
6. Fit bounded robust global Sim(3) candidates and accept only on held-out gates.
7. Represent the result as original mesh plus typed root transform; never rewrite cameras or PBR.
8. Apply an accepted transform to object lifting in memory and record before/after metrics.
9. Add CLI, deterministic previews, schemas, docs, synthetic/fake tests, and consistency checks.
10. Run CPU gates, fake resume, real H100 audit/comparison, and update the draft PR truthfully.

The scope ends at global similarity audit/refinement. It excludes local deformation, completion,
physics, metric recovery, gravity alignment, simulator compilation, and Phase 5.
