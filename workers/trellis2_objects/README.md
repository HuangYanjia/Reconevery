# TRELLIS.2 object worker

The isolated worker pins official `microsoft/TRELLIS.2` commit
`75fbf0183001ed9876c8dbb35de6b68552ee08bd` and checkpoint revision
`af44b45f2e35a493886929c6d786e563ec68364d`.
The recursive `o-voxel/third_party/eigen` submodule is pinned and verified at
`21e4582d1739107337a03460c81412981130373e`.

All checkpoint files are prefetched from `microsoft/TRELLIS.2-4B`, hashed, and used
offline. Runtime DINOv3 and `microsoft/TRELLIS-image-large` snapshots are pinned,
hashed, and verified independently. Because Reconevery supplies a non-opaque
canonical RGBA crop, the worker skips the otherwise unused BiRefNet constructor and
uses the official alpha-preserving preprocessing path.

The worker preserves the official PBR GLB produced through
`o_voxel.postprocess.to_glb`. The repository is MIT, but the production policy
remains blocked until direct and transitive dependency licenses have been reviewed.
