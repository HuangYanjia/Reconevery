# Phase 6B: Calibration-Optional Layered Scene Assembly

## Scope

Phase 6B assembles one reconstruction lineage into visual-only research and
deployment-eligible bundles. It consumes promoted typed artifacts from Phases 5A
through 6A. It does not run reconstruction, completion, articulation generation,
calibration, SceneSmith, collision generation, or physics identification.

The assembly is non-destructive:

- source geometry remains byte-identical;
- measured geometry remains authoritative evidence;
- selected completion geometry is a separate visual layer;
- rejected candidates are never inserted;
- the global context is never carved or hole-filled;
- preview GLBs are diagnostics, not canonical assets.

## World Policy

The default calibration policy is `use_full_canonical_if_available`.

| Phase 6A status | Assembly world mode | Units | Alignment |
| --- | --- | --- | --- |
| `accepted_full_canonical` | `canonical_metric` | meters | canonical |
| `accepted_metric_only` | `metric_unoriented` | meters | unoriented |
| `accepted_gravity_only` | `source_arbitrary` | arbitrary | unoriented |
| insufficient or rejected | `source_arbitrary` | arbitrary | unoriented |

`require_full_canonical` fails closed unless Phase 6A accepted a full canonical
world. `preserve_source_world` always uses an identity world wrapper. Phase 6A's
current `accepted_gravity_only` contract intentionally has no accepted transform,
so the default policy preserves source coordinates and records
`gravity_evidence_available_but_no_typed_orientation_transform`. A future
`gravity_aligned_arbitrary_scale` mode requires a separate exact typed orientation
transform; Phase 6B never derives one locally from unbound evidence.

## Stages

```text
scene_assembly_inputs
  -> scene_assembly_plan
  -> layered_scene_bundle
  -> assembly_previews
  -> phase6b_consistency_validation
```

`scene_assembly_inputs` selectively materializes only declared typed artifacts and
visual assets. Machine-local paths are removed from the canonical manifest. A
source normalization pass parses exact Phase 5B rigid selection/evaluation/native
representation records, Phase 5C selected/fitted/evaluated articulation records,
license sources, cameras, accepted alignments, and Phase 6A calibration/wrapper
records. An accepted Phase 5C alignment is also bound to its exact capture manifest,
child/reference state IDs, camera hashes, and frame-sequence digests. Global context
is bound to the promoted Phase 3 reconstruction metadata, Scene IR representation,
worker/license record, and exact GLB or PLY bytes. Duplicated local semantic flags
are rejected on disagreement.

`scene_assembly_plan` validates one coherent lineage, resolves the optional
calibration policy, chooses independent research and deployment visual decisions
per object, and composes:

```text
asset_to_assembly_world =
    source_world_to_assembly_world
    @ object_to_source_world
    @ asset_to_object
```

Reference-world measured assets use the world wrapper directly. Candidate-base and
link-local assets use their object/link hierarchy. The world transform is never
applied through both paths.

`layered_scene_bundle` writes distinct research and deployment-eligible bundles,
overlap diagnostics, a simulator-neutral compiler input manifest, and a Phase 6B
Scene IR reference. Both bundles remain visual-only and non-simulation-ready.

The Phase 6B Scene IR preserves the source Scene IR coordinate metadata and all
numeric camera/object/world-space values. Its assembly reference states the exact
source-to-assembly transform and whether geometry, camera poses, and object roots
require that transform. The assembly plan and compiler manifest define the assembly
world; they never relabel untransformed source coordinates as meters or canonical.

`assembly_previews` runs in an isolated worker. Preview settings affect only this
stage and downstream validation, not the assembly plan.

## Lineage

Every source record declares:

- lineage ID;
- frame-sequence digest;
- camera reconstruction hash;
- source Scene IR hash;
- source world;
- an accepted typed inter-lineage transform when applicable.

Unconnected lineage IDs are rejected. Phase 5C state lineages may be connected only
by accepted state-alignment records whose capture manifest, child and reference
state IDs, camera hashes, and frame-sequence digests all match. Merely finding assets
under the same filesystem root is not lineage evidence.

## Object Decisions

Each object receives a decision set with one deterministic research decision and
one deterministic deployment decision:

```text
selected_deployment_candidate
selected_research_candidate
measured_only
global_context_only
deferred_no_valid_candidate
deferred_license_blocked
deferred_articulated_unresolved
ignored
```

Measured anchors are retained even when a visual completion is selected. A research
candidate must have passed its observation-validation gates and permit research
evaluation. A deployment candidate must additionally be production-selectable. The
two selected candidate IDs may differ. The compiler manifest repeats the two
bundle-local decision and articulated hierarchy views explicitly.

Measured reconstruction assets use the explicit project-owned
`user_measured_evidence` rights policy unless a future promoted measured-rights
record replaces it. Candidate and global-context rights are always derived from
their exact upstream license-bearing artifact.

## Outputs

Canonical JSON outputs live below `assembly/`. Preview PNGs and GLBs are diagnostic
outputs. `scene_ir/phase6b_layered_scene.json` retains the source Scene IR and adds
exact assembly path/hash references plus an explicit source/assembly coordinate
contract. `compiler_input_manifest.json` repeats the source convention, assembly
convention, exact source-to-assembly transform, and compile-time transform flags.

All outputs state:

```text
visual_only = true
collision_ready = false
physical_validation = not_implemented
sim_ready = false
```

## Acceptance

Mandatory CI uses deterministic fake inputs and a fake preview worker. Real
acceptance runs the tabletop/rigid and drawer/articulated lineages independently,
inspects previews, validates exact path/hash identities, and repeats the same command
with `--resume`. No unrelated lineages may be combined.
