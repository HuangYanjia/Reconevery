# TRELLIS.2 container

The image builds the exact official commit and CUDA extensions. It deliberately
contains no model snapshot. Mount the verified `microsoft/TRELLIS.2-4B` snapshot
read-only, use one NVIDIA GPU, and provide no token on the command line.

Dependency licenses must be reviewed before setting `production_selectable=true`.

The runtime user defaults to `1000:1000`; override `HOST_UID` and `HOST_GID` at
build time. Reconevery requests exactly the configured GPU and mounts model/cache
paths read-only under `/models` or `/cache`. It never mounts the canonical run root.
