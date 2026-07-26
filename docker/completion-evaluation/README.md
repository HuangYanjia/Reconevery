# Completion evaluation container

This image contains only registration, dense-depth scoring, and rendering
dependencies. It contains no SAM 3D Objects or TRELLIS.2 checkpoint and never
initializes a generative model. Reconevery mounts only the current attempt workspace.

The runtime user defaults to `1000:1000` and is configurable with `HOST_UID` and
`HOST_GID` build arguments. Registration and evaluation receive one explicitly
configured NVIDIA device; no model cache is needed.
