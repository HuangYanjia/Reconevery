# Phase 5B implementation plan

1. Make Phase 5A surfel spacing deterministic under input reordering by using
   coordinate-hash sampling and nearest-neighbor statistics.
2. Add typed eligibility, license, evidence-split, crop, candidate, registration,
   held-out evaluation, selection, diagnostics, and consistency artifacts.
3. Build deterministic completion evidence packages with disjoint generation,
   fitting, and held-out views plus immutable training-only measured geometry.
4. Add isolated fake and official SAM 3D Objects and TRELLIS.2 candidate workers
   that preserve each backend's native representation and exact model identity.
5. Register frozen candidates to measured geometry using positive-scale Sim(3),
   then evaluate frozen transforms on held-out masks and dense depth.
6. Apply hard gates, Pareto filtering, deterministic ranking, and explicit
   research-versus-production license policy without silently selecting failures.
7. Retain measured Phase 5A assets in Scene IR, add selected visual completions as
   separate non-physical assets, and validate the full lineage and capability boundary.
8. Run CPU-only fake workers and synthetic tests before attempting both official
   H100 backends, real held-out evaluation, preview inspection, and resume validation.

Phase 5B stops at observation-grounded visual completion candidates. It does not
implement articulation, collision geometry, physical properties, metric scale,
gravity alignment, scene replacement, or simulator export.
