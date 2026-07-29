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
| `accepted_gravity_only` | `gravity_aligned_arbitrary_scale` | arbitrary | gravity aligned |
| insufficient or rejected | `source_arbitrary` | arbitrary | unoriented |

`require_full_canonical` fails closed unless Phase 6A accepted a full canonical
world. `preserve_source_world` always uses an identity world wrapper.

## Stages

```text
scene_assembly_inputs
  -> scene_assembly_plan
  -> layered_scene_bundle
  -> assembly_previews
  -> phase6b_consistency_validation
```

`scene_assembly_inputs` selectively materializes only declared typed artifacts and
visual assets. Machine-local paths are removed from the canonical manifest.

`scene_assembly_plan` validates one coherent lineage, resolves the optional
calibration policy, chooses one primary visual decision per object, and composes:

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
by accepted state-alignment records. Merely finding assets under the same filesystem
root is not lineage evidence.

## Object Decisions

Each object receives exactly one deterministic decision:

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
evaluation. A deployment candidate must additionally be production-selectable.

## Outputs

Canonical JSON outputs live below `assembly/`. Preview PNGs and GLBs are diagnostic
outputs. `scene_ir/phase6b_layered_scene.json` retains the source Scene IR and adds
exact assembly path/hash references.

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
