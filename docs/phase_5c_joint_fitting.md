# Phase 5C constrained joint fitting

Candidate fitting explicitly maps observed parts to one or more candidate links. It
fits one global candidate-base Sim(3), joint-axis sign, state positions, and only
small configured axis/pivot refinements.

The typed `FittedArticulatedKinematicModel` is the canonical fitted result. Link
assignment minimizes deterministic graph-role, joint-type, bounds, size-ratio,
axis, motion-range, placement, and multi-state geometry costs. Near-tied assignments
remain `ambiguous_link_assignment`. Prismatic candidate displacement is converted
through the fitted global Sim(3) scale; revolute position remains angular.

It does not permit non-uniform scale, link deformation, arbitrary per-state link
poses, camera changes, graph changes, or joint-type changes. Prismatic links translate
along one axis. Revolute links rotate about one fixed axis and pivot. Failure to
explain fitting states produces a rejected candidate.

Only accepted state alignments may enter fitting. The worker receives fitting-state
camera, dense-depth, and part-mask evidence but no held-out state evidence. After
point-based Sim(3) registration it renders up to
`maximum_fitting_views_per_state` views and records `fitting_part_iou`. Candidate
topology and joint type remain fixed.

At held-out evaluation time the base Sim(3), link geometry, joint graph, axis, and
pivot stay frozen. A bounded one-dimensional optimization may estimate only
`q(state, joint)` from the held-out measured part cloud.
