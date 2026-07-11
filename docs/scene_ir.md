# Scene IR

Top-level fields:
- `schema_version`: IR version string.
- `metadata`: scene id, name, units, source, and provenance.
- `cameras`: camera intrinsics and per-frame poses.
- `frames`: frame observations and object observations.
- `objects`: typed instances such as static floor, rigid cup, or articulated cabinet.
- `geometry_assets`, `material_assets`, `collision_assets`: referenced files and provenance.
- `relations`: semantic/physical relations like `supported_by` and `part_of`.
- `validation`: optional validation report.

Example relation:
```json
{"relation_type":"supported_by","subject_id":"cup","object_id":"table"}
```
