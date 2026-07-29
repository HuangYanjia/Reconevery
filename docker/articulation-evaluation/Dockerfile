FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG HOST_UID=1000
ARG HOST_GID=1000
ARG NVDIFFRAST_COMMIT=253ac4fcea7de5f396371124af597e6cc957bfae
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3.10 python3-pip build-essential libgl1 && \
    rm -rf /var/lib/apt/lists/*
RUN python3.10 -m pip install --no-cache-dir --upgrade pip && \
    python3.10 -m pip install --no-cache-dir \
      torch==2.6.0 torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cu124
RUN git clone https://github.com/NVlabs/nvdiffrast.git /opt/nvdiffrast && \
    git -C /opt/nvdiffrast checkout --detach "${NVDIFFRAST_COMMIT}" && \
    test "$(git -C /opt/nvdiffrast rev-parse HEAD)" = "${NVDIFFRAST_COMMIT}" && \
    python3.10 -m pip install --no-cache-dir /opt/nvdiffrast --no-build-isolation
COPY workers/articulation_evaluation /opt/reconevery-worker
RUN python3.10 -m pip install --no-cache-dir /opt/reconevery-worker --no-deps && \
    python3.10 -m pip install --no-cache-dir \
      'numpy>=2.0,<3' 'scipy>=1.15,<2' 'trimesh>=4.8,<5' \
      'pillow>=11,<13' 'opencv-python-headless>=4.10,<5'
RUN groupadd --gid "${HOST_GID}" reconevery && \
    useradd --uid "${HOST_UID}" --gid "${HOST_GID}" --create-home reconevery && \
    mkdir -p /workspace && chown reconevery:reconevery /workspace
USER reconevery
WORKDIR /workspace
ENTRYPOINT ["python3.10", "-m", "articulation_evaluation_worker"]
