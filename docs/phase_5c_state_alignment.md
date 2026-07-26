# Phase 5C state alignment

Independent state runs have independent arbitrary COLMAP gauges.
`articulation_state_alignment` estimates `T_reference_from_state` as a proper
positive-scale Sim(3). It uses static environment and base/body evidence only.
Movable parts, people, manipulators, and transient objects are excluded.

Acceptance uses preconfigured static correspondence count, scene-relative median and
p90 residual, and held-out static inlier gates. Source state artifacts and cameras
are immutable. A failed state is excluded from joint estimation instead of being
absorbed into part motion.
