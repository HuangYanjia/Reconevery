# Phase 5B license policy

Two modes are typed: `research_evaluation` and `production_candidate`. Each
candidate records code, checkpoint, dependency, and asset licenses; access
conditions; commercial-review status; and production selectability.

Selection reports both `best_research_candidate` and
`best_production_eligible_candidate`. A research winner that is not production
selectable remains an evaluation artifact but cannot become a production Scene IR
asset. Gated model access does not imply commercial approval.

The pinned TRELLIS.2 repository states that its code and model are MIT licensed.
Its `nvdiffrast`, `nvdiffrec`, O-Voxel, Kaolin, and other transitive runtime
dependencies remain separately inventoried. Consequently the default project
policy keeps TRELLIS.2 `production_selectable=false` until that inventory receives
an explicit project review.

Phase 5B performs no Objaverse or third-party CAD internet retrieval.
