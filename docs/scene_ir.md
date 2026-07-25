# Scene IR

`SceneIR` is the canonical semantic and physical representation. Meshes, compiler packages, and
future simulator files are referenced or derived artifacts; none replaces the IR.

## Top-level contract

- `schema_version`: current literal version `0.1.0`.
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

- right-handed;
- +X forward, +Y left, +Z up;
- meters;
- quaternion fields ordered `(x, y, z, w)`;
- camera transforms are world-from-camera.

The same `CoordinateConvention` appears in camera reconstruction output and Scene IR metadata.
Camera `scale_status` is one of `metric_scale_known`, `scale_ambiguous`, or `externally_scaled`.
Monocular COLMAP uses `scale_ambiguous`; its translations are consistent arbitrary reconstruction
units until external scaling is applied. The `translation_m` name remains for schema compatibility
and must be interpreted together with scale status.

## JSON Schema

Pydantic v2 generates `schemas/scene_ir.schema.json`, including nested `$defs`, enums, required
fields, numeric constraints, and `additionalProperties: false` behavior.

```bash
uv run python scripts/generate_schema.py
```

Tests compare the complete checked-in schema to `SceneIR.model_json_schema()` and separately assert
major properties, enum values, numeric bounds, and strict-object behavior.
