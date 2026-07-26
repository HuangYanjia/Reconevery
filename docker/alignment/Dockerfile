FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG NVDIFFRAST_COMMIT=253ac4fcea7de5f396371124af597e6cc957bfae

ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST=9.0
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    ninja-build \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel \
    && python3.10 -m pip install \
      torch==2.6.0 torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cu126 \
    && python3.10 -m pip install \
      numpy==2.2.6 \
      scipy==1.15.3 \
      opencv-python-headless==5.0.0.93 \
      pillow==12.0.0 \
      pydantic==2.12.4 \
      trimesh==4.12.2

RUN git clone https://github.com/NVlabs/nvdiffrast.git /opt/nvdiffrast \
    && git -C /opt/nvdiffrast checkout --detach "${NVDIFFRAST_COMMIT}" \
    && test "$(git -C /opt/nvdiffrast rev-parse HEAD)" = "${NVDIFFRAST_COMMIT}" \
    && python3.10 -m pip install /opt/nvdiffrast --no-build-isolation

COPY workers/object_lifting /opt/reconevery-object-lifting-worker
COPY workers/alignment /opt/reconevery-alignment-worker
RUN python3.10 -m pip install /opt/reconevery-object-lifting-worker \
    && python3.10 -m pip install /opt/reconevery-alignment-worker

WORKDIR /workspace

HEALTHCHECK --interval=30s --timeout=20s --retries=3 CMD \
  python3.10 -c "import torch, scipy, nvdiffrast.torch, cv2, trimesh; assert torch.version.cuda"

ENTRYPOINT ["python3.10", "-m", "alignment_worker"]
