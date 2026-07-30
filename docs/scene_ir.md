# Scene IR

`SceneIR` is the canonical semantic and physical representation. Meshes, compiler packages, and
future simulator files are referenced or derived artifacts; none replaces the IR.

## Top-level contract

- `schema_version`: current value `0.1.1`; legacy `0.1.0` payloads remain readable.
- `metadata`: scene identity, source, coordinate convention, and provenance.
- `cameras`: intrinsics, per-frame `transform_world_from_camera` poses, and scale status.
- `frames`: frame paths, timestamps, camera references, bounding boxes, mask paths, and confidence.
- `objects`: unclassified, static, rigid, articulated, deformable, fluid/particle, or ignored
  instances with physics and asset references.
- `geometry_assets`, `material_assets`, `collision_assets`: typed referenced assets.
- `relations`: validated object-to-object semantic or physical relations.
- `validation`: optional validation report.

All models reject extra fields. Numeric bounds include confidence in `[0, 1]`, non-negative mass,
positive image dimensions and focal lengths, and restitution in `[0, 1]`.

## Reference validation

Scene validation enforces:

- unique object, camera, frame, and asset IDs;
- valid geometry, material, and collision references from objects and links;
- valid relation subjects and objects;
- valid frame camera and observation object references;
- camera poses that reference known frames;
- unique articulation link and joint IDs;
- joint parent and child links that exist and differ;
- an articulation on every articulated object and none on other object types.

## Canonical articulated cabinet

The mock cabinet is one top-level `ObjectInstance` with `asset_type="articulated"`. Its
`cabinet_body` and `cabinet_drawer` are `Link` records connected by the
`cabinet_drawer_slide` prismatic joint. The drawer is not a second top-level object and no
`part_of` relation is used to imply duplicate identity.

## Segmentation semantics

Phase 2 segmentation does not measure final physical type. A prompt may carry
`asset_type_hint`, whose source is explicitly `configured_semantic_hint`. Tracks may omit the hint
or use `unclassified`; neither is a measured claim. The production Phase 2 DAG stops before Scene
IR assembly. The explicit mixed/mock demo may use hints to exercise older downstream contracts,
and labels every subsequent geometry result as mock.

Every canonical segmentation observation records whether its frame has a registered camera pose.
An unregistered frame remains a valid 2D mask and box, but cannot silently become eligible for
multi-view 3D fusion. Segmentation never modifies or reinterprets the COLMAP coordinate convention.

## Coordinates

`CoordinateConvention` explicitly records `world_frame`, `alignment_status`, `camera_axes`,
`linear_units`, `scale_status`, handedness, quaternion order, and transform direction. Transform
translations use the unit-neutral field `translation`. Legacy payloads containing `world_axes`,
`units`, `camera_transform_direction`, or `translation_m` remain readable but serialize in the
new form.

Raw monocular COLMAP output uses:

- `world_frame="colmap_arbitrary"` and `alignment_status="unoriented"`;
- `camera_axes="x_right_y_down_z_forward"` and right handedness;
- `linear_units="arbitrary_units"` and `scale_status="scale_ambiguous"`;
- quaternion order `xyzw` and transform direction `world_from_camera`.

The canonical robot frame is a distinct aligned contract: right-handed +X forward, +Y left, +Z up
with meters. Raw COLMAP poses must not be labeled canonical or metric. The same coordinate metadata
is propagated into the Scene IR metadata and camera record by the explicit mock-downstream demo.

## JSON Schema

Phase 3 adds generated global visual `GeometryAsset` records for PBR GLB and inspection PLY.
Each carries its own coordinate convention and scale status. No `CollisionAsset` is inferred from
the GenRecon mesh, and no physical truth is attached to generated triangles.

Phase 4 adds optional `geometry_status`, `completion_status`, and `sim_ready` fields without
changing older payload requirements. Resolved surface hypotheses use `source=fused`,
`asset_type=unclassified`, `geometry_status=partial_observation_supported`,
`completion_status=not_completed`, and `sim_ready=false`. Prompt asset-type hints remain typed
evidence and are not promoted to measured physical truth. Phase 4.1 splits association precision,
mask recall, reprojection IoU, multiview support, connectedness, observed coverage, association
confidence, and completeness confidence in the typed evidence artifact. Completeness confidence
remains zero and no collision asset is created.

Phase 4.2 extends a geometry asset with optional `source_asset_id`,
`alignment_transform_path`, and `geometry_alignment_status`. An aligned asset is a wrapper around
the original Phase 3 visual asset, not a rewritten mesh. Its transform is expressed in the same
`colmap_arbitrary` frame, remains `unoriented` and `scale_ambiguous`, and is never marked
simulation-ready. `scene_ir/phase4_2_scene.json` retains original global and partial-surface assets
and adds alignment-aware global wrappers only when held-out gates accept the transform.

Pydantic v2 generates Scene IR plus segmentation, GenRecon, object-surface, reconstruction, and
consistency schemas under `schemas/`, including nested `$defs`, enums, required fields, numeric
constraints, and `additionalProperties: false` behavior.

```bash
uv run python scripts/generate_schema.py
```

Tests compare the complete checked-in schema to `SceneIR.model_json_schema()` and separately assert
major properties, enum values, numeric bounds, and strict-object behavior.

Phase 5A adds `source=measured`, `geometry_status=partial_measured` assets with
`completion_status=not_completed`, `sim_ready=false`, and ambiguous scale. A measured
point/surfel asset and a generated or fused Phase 4 hypothesis may coexist on the same object.
Neither silently replaces the other, and no measured asset is promoted to collision geometry.

Phase 5B may add a separate `visual_completion_candidate` with
`geometry_status=complete_visual_candidate`,
`completion_status=selected_by_observation_validation`,
`observation_grounded=true`, `physical_validation=not_implemented`,
`collision_ready=false`, and `sim_ready=false`. Usage policy and production
selectability are explicit; the `measured_anchor` remains attached to the object.

Phase 5C adds typed `Link`, `Joint`, and `Articulation` evidence. Joints may be fixed,
prismatic, or revolute in Scene IR; unknown/continuous priors remain in Phase 5C
artifacts until they meet the stricter IR contract. Observed positions and ranges are
separate from candidate mechanical limits. Measured part assets remain
`partial_measured`. A selected generated link is
`articulated_visual_candidate` and `selected_by_multi_state_validation`. All Phase 5C
objects keep `physical_validation=not_implemented`, `collision_ready=false`, and
`sim_ready=false`. Selected objects use the fitted candidate-base Sim(3); joints use
fitted/refined axes and pivots, visual formats come from actual suffixes, and the
articulation records exact fitting/evaluation paths and hashes. Measured link assets
remain top-level `reference_world` evidence when a fitted candidate transform is
active; retrieved/generated links contain only `candidate_base` or `link_local`
visuals. Link-local measured evidence would require a newly transformed derived
asset with its own hash and provenance.

ArtVIP and PartNet-Mobility link visuals use `GeometrySourceType.RETRIEVED`.
Particulate link visuals use `GeometrySourceType.GENERATED`. The geometry asset and
its provenance record must agree; source-family identity is never inferred from a
filename.
# Articulated asset spaces

Phase 5C.3 distinguishes `reference_world`, `candidate_base`, and `link_local`.
Original Phase 5A point clouds are immutable `reference_world` evidence and do not
carry a candidate-base transform. Candidate and link-local assets carry the same
explicit transform used during fitting, rendering, preview export, and selection.

Scene IR references dedicated selected-candidate, fitted-model, link-assignment,
evaluation, identity-manifest, and kinematic-bundle files. Every declared SHA-256 is
the hash of the bytes at its declared path, never a digest of a nested parent record.

# Phase 6A canonical coordinates

`scene_ir/phase6a_canonical_scene.json` is canonical only after
`accepted_full_canonical`. Its coordinate convention is right handed,
`canonical_x_forward_y_left_z_up`, meters, and `metric_scale_known`.

The accompanying `calibration/canonical_scene_wrapper.json` stores the exact source
Scene IR, camera reconstruction, calibration, fiducial or landmark derivation, and per-asset
wrapper paths/hashes. Phase 6A Scene IR metadata repeats the exact source,
calibration, and wrapper identities. Every retained geometry asset is marked as
source-space. Reference-world/global assets require that exact wrapper after full
acceptance; candidate-base/link-local assets declare hierarchy-root composition
instead and must not receive the wrapper twice.
Geometry URIs and source bytes are retained. A rejected or partial calibration produces a derived Scene IR
with the original arbitrary/unoriented convention and no accepted canonical claim.

Phase 5C object transforms map object local to COLMAP world. Joint axes, pivots,
and prismatic q are object-local; revolute q is radians. Phase 6A left-composes the
object root only and leaves all joint-local values unchanged. The wrapper records
`prismatic_position_scale_to_m = calibration scale * source object scale` so a
downstream compiler converts q exactly once. Reference-world measured assets are
never baked or double transformed.

A landmark-derived canonical scene cannot be interpreted from its top-level
meter metadata alone. Consumers must verify and apply the referenced wrapper to
source-space geometry. The exact O/U/R derivation remains auditable through the
Scene IR calibration record instead of being reduced to unbound up, forward,
and origin values.

# Phase 6B layered scene

`scene_ir/phase6b_layered_scene.json` retains the source Scene IR and adds exact
path/hash references for the assembly plan, research bundle, deployment-eligible
bundle, compiler input manifest, and source Scene IR. Its top-level coordinate
metadata and all numeric cameras, object roots, geometry relations, and articulations
remain in source space. The assembly reference separately records the assembly world
mode, coordinate convention, exact source-to-assembly transform, and whether
geometry, cameras, and object roots require that transform.

Consequently, full-canonical or metric-only assembly does not relabel untransformed
source numbers as meters. `accepted_gravity_only` also remains source-arbitrary
because Phase 6A currently supplies evidence but no accepted orientation transform.

Reference-world measured assets receive the assembly world wrapper directly.
Candidate-base and link-local visuals remain under their object hierarchy, so the
wrapper is applied at the root exactly once. Measured anchors remain authoritative
evidence even when a validated visual completion is present.

The Scene IR is visual-only. It records unresolved objects, candidate evaluations,
and license exclusions while keeping `collision_ready=false`,
`physical_validation=not_implemented`, and `sim_ready=false`. Diagnostic preview
GLBs are not canonical geometry or future collision inputs.

The referenced plan and compiler manifest preserve independent research and
deployment object decisions. Candidate identity, representation, license, fitted
articulation, lineage, and calibration values are normalized from exact upstream
typed artifacts. Scene IR references never promote a local manifest assertion into
selection or calibration evidence. Phase 5C state connections include the exact
capture-state camera/digest chain, and global-context representations include the
exact Phase 3 source geometry identity.
