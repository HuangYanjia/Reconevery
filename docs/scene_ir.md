# Scene IR

`SceneIR` is the canonical semantic and physical representation. Raw COLMAP models, meshes,
compiler packages, trajectory exports, and future simulator files are referenced or derived
artifacts; none replaces the IR.

## Top-level contract

- `schema_version`: current literal version `0.1.0`.
- `metadata`: scene identity, source, coordinate convention, scale/alignment status, provenance.
- `cameras`: intrinsics, distortion, and registered `transform_world_from_camera` poses.
- `frames`: selected frame paths/timestamps, camera references, object observations and masks.
- `objects`: static, rigid, or articulated instances with physics and asset references.
- geometry/material/collision asset tables.
- validated object-to-object relations.
- optional validation report.

All models reject extra fields. Numeric bounds cover confidence `[0, 1]`, non-negative mass,
positive dimensions/focal lengths, valid material channels, and restitution `[0, 1]`.

## Reference validation

Scene validation enforces unique object/camera/frame/asset IDs; valid geometry, material,
collision, relation, camera, and observation references; pose frame existence; unique link/joint
IDs; existing and distinct joint endpoints; an articulation on articulated objects; and no
articulation on non-articulated objects.

The mock cabinet is one top-level articulated `ObjectInstance`. `cabinet_body` and
`cabinet_drawer` are links connected by `cabinet_drawer_slide`; the drawer is not a second object.

## Coordinates and transforms

The aligned Recon2Sim convention is:

- right-handed world;
- +X forward, +Y left, +Z up;
- meters;
- quaternion order `(x, y, z, w)`;
- camera transforms are world-from-camera.

COLMAP instead estimates an arbitrary world orientation and monocular scale, while its image pose
is world-to-camera `qvec` `(w, x, y, z)` plus `tvec`. Phase 1 normalizes `qvec`, constructs
`R_world_to_camera`, and inverts it:

```text
R_world_from_camera = transpose(R_world_to_camera)
t_world_from_camera = -R_world_from_camera * tvec
```

The resulting rotation is emitted as normalized `xyzw`. Inversion alone does not align gravity
or establish meters. Therefore a monocular COLMAP scene explicitly carries:

```json
{
  "scale_status": "scale_ambiguous",
  "world_frame_status": "colmap_unaligned",
  "coordinate_convention": {
    "world_axes": "colmap_arbitrary",
    "handedness": "right",
    "units": "arbitrary_scale",
    "quaternion_order": "xyzw",
    "camera_transform_direction": "world_from_camera"
  }
}
```

Despite the backward-compatible field name `translation_m`, values under this explicit ambiguous
convention are reconstruction units and must not be consumed as meters. A later alignment stage
may emit `externally_scaled` or `metric_scale_known` and the aligned axes/units only when supported
by real evidence.

## Camera intermediate contract

`CameraReconstruction` preserves one supported physical camera in Phase 1, including its COLMAP
model name, mapped intrinsics, all distortion terms, registered poses, registered and unregistered
frame IDs, diagnostic confidence, convention, scale/world status, and provenance. Registered IDs
must exactly equal pose frame IDs and cannot overlap unregistered IDs.

`CameraDiagnostics` is not canonical scene state. It records sparse model selection, registration
ratio, point count, confidence inputs, warnings, and failure thresholds for audit.

## JSON Schema

Pydantic v2 generates `schemas/scene_ir.schema.json` with properties, required fields, `$defs`,
enum values, numeric constraints, and `additionalProperties: false`:

```bash
uv run python scripts/generate_schema.py
```

Tests compare the complete checked-in document with `SceneIR.model_json_schema()` and separately
assert major fields, enum definitions, strictness, and numeric constraints.
