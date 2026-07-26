# Phase 5A implementation plan

1. Reserve Phase 4.2 validation frames and point IDs exclusively for final acceptance.
2. Pin the official COLMAP dense backend and record its executable, version, commit, and build.
3. Add typed dense-MVS requests, manifests, diagnostics, previews, and strict binary parsers.
4. Build selective attempt-local dense workspace preparation and fake/local/Docker workers.
5. Add typed measured-object requests, mask mapping, backprojection, view validation, surfel
   fusion, observed-only surfaces, and fake/local/Docker workers.
6. Integrate measured partial assets into a new Scene IR without depending on GenRecon.
7. Add optional measured/generated comparison, Phase 5A validation, CLI, configs, and schemas.
8. Run CPU gates and fake DAG/resume before the real COLMAP dense and measured-object smoke.

Phase 5A ends at visible measured partial geometry. It does not perform completion, replacement,
collision generation, physical inference, metric scaling, gravity alignment, or simulation export.
