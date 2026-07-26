FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TRELLIS2_COMMIT=75fbf0183001ed9876c8dbb35de6b68552ee08bd
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3.10 python3.10-venv python3-pip build-essential && \
    rm -rf /var/lib/apt/lists/*
RUN python3.10 -m pip install --no-cache-dir --upgrade pip && \
    git clone --recursive https://github.com/microsoft/TRELLIS.2 /opt/trellis2 && \
    git -C /opt/trellis2 checkout --detach "${TRELLIS2_COMMIT}" && \
    git -C /opt/trellis2 submodule update --init --recursive && \
    test "$(git -C /opt/trellis2 rev-parse HEAD)" = "${TRELLIS2_COMMIT}"
WORKDIR /opt/trellis2
RUN bash -lc '. ./setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm'
COPY workers/trellis2_objects /opt/reconevery-worker
RUN python3.10 -m pip install --no-cache-dir -e /opt/reconevery-worker
RUN groupadd --gid "${HOST_GID}" reconevery && \
    useradd --uid "${HOST_UID}" --gid "${HOST_GID}" --create-home reconevery && \
    mkdir -p /workspace && chown reconevery:reconevery /workspace
USER reconevery
WORKDIR /workspace
HEALTHCHECK --interval=30s --timeout=20s --retries=3 CMD \
  python3.10 -c "import torch, nvdiffrast.torch, trimesh; assert torch.cuda.is_available()"
ENTRYPOINT ["python3.10", "-m", "trellis2_objects_worker"]
