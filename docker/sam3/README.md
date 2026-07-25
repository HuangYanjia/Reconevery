# SAM 3 GPU image

Build from the repository root:

```bash
docker build -f docker/sam3/Dockerfile -t reconevery/sam3:phase2 .
```

The image installs Meta's official repository at commit
`46957e47805eaa273f4aa7bbbd25a88bca9108ce`, CUDA 12.8 PyTorch 2.10.0,
torchvision 0.25.0, and the isolated Reconevery worker. It does not download or
contain a checkpoint, token, user frame, or cache.

Run through `configs/sam3_docker.example.yaml`. On Linux the adapter passes
`--user <host-uid>:<host-gid>` so canonical outputs are owned by the invoking
user. It mounts:

- the stage attempt at `/workspace`;
- an optional authorized Hugging Face cache at `/model-cache`;
- an optional local official checkpoint as a read-only file under
  `/checkpoints`.

The adapter passes allowlisted credential environment variable names with
Docker `-e NAME`; values never appear in the command or retained request.
Docker execution requires the NVIDIA Container Toolkit and `device: cuda`.

The Dockerfile healthcheck verifies imports and pinned runtime packages only.
Use the configuration-aware command below to verify Docker, the image ID, GPU
access, official code import, and checkpoint access:

```bash
uv run recon2sim adapters healthcheck \
  --config configs/sam3_docker.example.yaml
```
