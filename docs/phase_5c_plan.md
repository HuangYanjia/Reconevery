# Phase 5C implementation plan

1. Add strict typed contracts for articulated eligibility, explicit object/part
   prompts, multi-state capture lineage, state alignment, measured part motion,
   articulated candidates, link assignment, fitting, held-out evaluation,
   license policy, selection, diagnostics, previews, and consistency.
2. Treat every capture state as an independent Phase 5A run. Validate its
   immutable hashes before using only configured static evidence to estimate a
   proper positive-scale Sim(3) into the reference-state frame.
3. Retain measured part geometry per state and estimate an analytic fixed,
   prismatic, revolute, or unknown joint hypothesis without forcing a model when
   residuals are inconsistent.
4. Enforce evidence tiers: one state is prior-only, two states are
   motion-supported without held-out validation, and three or more states use
   disjoint generation, fitting, and held-out state evidence.
5. Add isolated workers and fake protocols for state alignment, ArtVIP and
   research-only PartNet retrieval, official pinned Particulate inference, and
   constrained articulated fitting/evaluation. Core dependencies remain
   lightweight.
6. Normalize every source into explicit links, joints, visual assets, source
   hashes, reversible working transforms, and license records. Never import
   collision, inertial, friction, damping, motor, or dynamics claims.
7. Fit only a global candidate Sim(3), allowed joint coordinates, joint-axis
   sign, and tightly bounded axis/pivot refinements. Candidate topology, link
   geometry, cameras, and measured evidence remain frozen.
8. Evaluate frozen structures on held-out states and views, apply hard gates,
   Pareto filtering, deterministic ranking, and separate research versus
   production selection. A truthful rejection is a valid result.
9. Retain all measured geometry in Scene IR and add any selected articulation as
   a separate observation-grounded visual kinematic hypothesis with
   `sim_ready=false`.
10. Run CPU fake DAG and synthetic prismatic/revolute/leakage/license tests
    before official Particulate and real three-state H100 acceptance, preview
    inspection, and identical resume validation.

Phase 5C does not implement physical joint validation, collision generation,
dynamics identification, metric scale, gravity alignment, scene replacement, or
production simulator export.
