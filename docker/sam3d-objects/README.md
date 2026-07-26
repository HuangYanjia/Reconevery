# SAM 3D Objects container

Build from the repository root. The image contains the exact official code commit but
no checkpoint, token, or user data. Mount an authorized checkpoint cache read-only
and run with `--gpus device=0`; Reconevery adds the host UID/GID and mounts only the
stage attempt workspace.

Set `--build-arg HOST_UID=$(id -u) --build-arg HOST_GID=$(id -g)` when the default
`1000:1000` does not match the host. Model mounts configured through
`docker_model_mounts` are read-only and may target only `/models` or `/cache`.
