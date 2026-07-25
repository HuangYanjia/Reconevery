# Scene IR

`SceneIR` is the canonical semantic and physical representation. Meshes, compiler packages, and
future simulator files are referenced or derived artifacts; none replaces the IR.

## Top-level contract

- `schema_version`: current value `0.1.1`; legacy `0.1.0` payloads remain readable.
- `metadata`: scene identity, source, coordinate convention, and provenance.
- `cameras`: intrinsics, per-frame `transform_world_from_camera` poses, and scale status.
- `frames`: frame paths, timestamps, camera references, bounding boxes, mask paths, and confidence.
- `objects`: static, rigid, or articulated instances with physics and asset references.
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

Pydantic v2 generates `schemas/scene_ir.schema.json`, including nested `$defs`, enums, required
fields, numeric constraints, and `additionalProperties: false` behavior.

```bash
uv run python scripts/generate_schema.py
```

Tests compare the complete checked-in schema to `SceneIR.model_json_schema()` and separately assert
major properties, enum values, numeric bounds, and strict-object behavior.
