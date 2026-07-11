# Phase 0.1 Foundation Hardening Plan

Phase 0.1 turns the deterministic CPU-only mock into a reliable contract test for future
adapters. It deliberately excludes COLMAP, SAM 3, GenRecon, SceneSmith, Blender, simulator
SDKs, model checkpoints, and network downloads.

1. Replace the in-tree dependency substitutes and custom build backend with Hatchling,
   Pydantic v2, Typer, PyYAML, pytest, Ruff, mypy, and reproducible `uv` dependency groups.
2. Strengthen the canonical Scene IR and intermediate artifact models with strict fields,
   numeric constraints, enum values, cross-reference checks, and a checked-in generated schema.
3. Make each mock stage validate and consume its declared upstream JSON/files, emit valid PNG,
   OBJ, and typed JSON artifacts, and assemble one connected Scene IR from those artifacts.
4. Validate the pipeline DAG before execution and make resume signatures depend on config,
   adapter identity, seed, input bytes, upstream execution signatures, and artifact hashes.
5. Add retries, isolated command environments, process timeout handling, captured logs, required
   output validation, output hashing, and detailed artifact records.
6. Replace the handwritten CLI dispatcher with real Typer behavior, add actionable errors, and
   cover constraints, schema drift, graph failures, cache invalidation, adapters, CLI, and the
   full mock pipeline in tests.
7. Add CPU-only GitHub Actions, update architecture and adapter documentation, run the complete
   local quality gate, exercise resume/invalidation manually, and publish a draft pull request.

Completion requires all commands listed in the Phase 0.1 request to pass from `uv sync
--all-groups`, with the mock demo remaining deterministic and free of heavyweight integrations.
