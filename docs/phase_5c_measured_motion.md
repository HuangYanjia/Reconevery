# Phase 5C measured motion

Measured part geometry is retained per state. After static alignment, rigid part
registration estimates relative transforms. The analytic baseline tests fixed,
prismatic, revolute, then unknown in that order with explicit residuals.

Prismatic evidence requires a stable direction, small rotation leakage, and low
orthogonal translation residual. Revolute evidence requires a common rotation axis
and a pivot line that explains translation. Observed state positions and ranges are
not mechanical limits. One state is prior-only; two states are partially supported;
only three or more states can be held-out validated.
