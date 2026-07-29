# Phase 5C PartNet-Mobility policy

PartNet-Mobility is indexed from a locally registered immutable dataset. The worker
does not call the SAPIEN service during inference. File hashes and source license are
required.

All PartNet-Mobility candidates are `research_evaluation_only` and
`production_selectable=false`. Dynamics, collision, inertial, damping, and friction
metadata are not imported as observed truth.

As with ArtVIP, each indexed record must provide a normalized candidate bundle and
hashed visual assets. Only deterministic top-K records are exposed to the retrieval
worker. Direct PartNet candidates and their Particulate-derived alternatives remain
separate candidates with separate provenance.
