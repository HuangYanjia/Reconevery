# GenRecon GPU Image

Build from the repository root:

```bash
docker build -f docker/genrecon/Dockerfile -t reconevery/genrecon:phase3 .
```

The image pins official GenRecon commit
`eaf1468118d20469d17079a4a19737297d2ef87b`, Python 3.10, PyTorch 2.6.0,
torchvision 0.21.0, CUDA 12.6, Flash-Attention 2.7.3, and exact implementation-time
commits for the CUDA extensions. It builds H100 (`sm_90`) kernels.

No checkpoint or user input is embedded. Mount the three official checkpoint files
read-only through `configs/genrecon_docker.example.yaml`. Reconevery runs the container
with `--gpus all` and the current Linux UID/GID, and mounts only the stage attempt plus
the configured checkpoint files.

The image healthcheck imports the official package and every required CUDA extension.
The configuration-aware Reconevery healthcheck additionally verifies Git and submodule
commits, checkpoint hashes, CUDA availability, and the visible GPU.

Real inference additionally requires accepted access to official gated
`facebook/dinov3-vitl16-pretrain-lvd1689m`. Pass `HF_TOKEN` only as an environment variable
and configure the writable host `hf_cache_path` shown in
`configs/genrecon_docker.example.yaml`; Reconevery mounts it at `/hf-cache` and sets the
in-container `HF_HOME`. Credentials must not be placed in the image, configuration file, or
command arguments.
