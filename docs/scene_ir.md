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
