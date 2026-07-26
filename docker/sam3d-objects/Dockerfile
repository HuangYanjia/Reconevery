FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG SAM3D_COMMIT=f91db411c50efee93d8db7aeb323885650f6f722
ARG NVDIFFRAST_COMMIT=253ac4fcea7de5f396371124af597e6cc957bfae
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3.10 python3.10-venv python3-pip build-essential libgl1 && \
    rm -rf /var/lib/apt/lists/*
RUN python3.10 -m pip install --no-cache-dir --upgrade pip && \
    git clone https://github.com/facebookresearch/sam-3d-objects /opt/sam3d-objects && \
    git -C /opt/sam3d-objects checkout --detach "${SAM3D_COMMIT}" && \
    test "$(git -C /opt/sam3d-objects rev-parse HEAD)" = "${SAM3D_COMMIT}"
WORKDIR /opt/sam3d-objects
RUN python3.10 -m pip install --no-cache-dir -e .
RUN git clone https://github.com/NVlabs/nvdiffrast.git /opt/nvdiffrast && \
    git -C /opt/nvdiffrast checkout --detach "${NVDIFFRAST_COMMIT}" && \
    test "$(git -C /opt/nvdiffrast rev-parse HEAD)" = "${NVDIFFRAST_COMMIT}" && \
    python3.10 -m pip install --no-cache-dir /opt/nvdiffrast --no-build-isolation && \
    python3.10 -m pip install --no-cache-dir "trimesh>=4,<5"
COPY workers/sam3d_objects /opt/reconevery-worker
RUN python3.10 -m pip install --no-cache-dir -e /opt/reconevery-worker
RUN groupadd --gid "${HOST_GID}" reconevery && \
    useradd --uid "${HOST_UID}" --gid "${HOST_GID}" --create-home reconevery && \
    mkdir -p /workspace && chown reconevery:reconevery /workspace
USER reconevery
WORKDIR /workspace
HEALTHCHECK --interval=30s --timeout=20s --retries=3 CMD \
  python3.10 -c "import torch, nvdiffrast.torch, trimesh; assert torch.cuda.is_available()"
ENTRYPOINT ["python3.10", "-m", "sam3d_objects_worker"]
