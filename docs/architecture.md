# Architecture

Recon2Sim keeps the core Python package lightweight by treating heavyweight reconstruction and simulation tools as isolated adapters. Adapters may run in separate processes, Docker containers, or services, which prevents dependency conflicts and makes GPU requirements metadata rather than core assumptions.

The Phase 0 DAG is: ingest → camera recovery → segmentation/tracking → global and object reconstruction → Scene IR assembly → scene compilation → validation → export. Each stage writes inspectable artifacts and manifest status. Failures are persisted as `failed`, and handled exceptions never leave `running` statuses behind.

Scene IR is the source of truth because it stores typed cameras, observations, objects, assets, relations, physics metadata, provenance, and confidence. OBJ/GLB files are export assets; they do not encode enough semantics to drive physics or future repair.
