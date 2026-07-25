FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GENRECON_COMMIT=eaf1468118d20469d17079a4a19737297d2ef87b
ARG NVDIFFRAST_COMMIT=253ac4fcea7de5f396371124af597e6cc957bfae
ARG NVDIFFREC_COMMIT=b296927cc7fd01c2ac1087c8065c4d7248f72da4
ARG CUMESH_COMMIT=12289e1062f0603f2f0d0771b02e1395d247f26f
ARG FLEXGEMM_COMMIT=6dd94a859c26ee8246888502eada3dd8ad85532e

ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST=9.0
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
    libx11-6 \
    ninja-build \
    python3.10 \
    python3.10-dev \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel \
    && python3.10 -m pip install \
      torch==2.6.0 torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cu126 \
    && python3.10 -m pip install \
      imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh \
      transformers==4.57.3 gradio==6.0.1 tensorboard pandas lpips zstandard \
      kornia timm pillow plyfile open3d \
    && python3.10 -m pip install \
      git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8 \
    && python3.10 -m pip install \
      https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

RUN git clone --recursive https://github.com/kasothaphie/GenRecon.git /opt/GenRecon \
    && git -C /opt/GenRecon checkout --detach "${GENRECON_COMMIT}" \
    && git -C /opt/GenRecon submodule update --init --recursive \
    && test "$(git -C /opt/GenRecon rev-parse HEAD)" = "${GENRECON_COMMIT}" \
    && test "$(git -C /opt/GenRecon/o-voxel/third_party/eigen rev-parse HEAD)" = \
      "21e4582d1739107337a03460c81412981130373e"

RUN git clone https://github.com/NVlabs/nvdiffrast.git /opt/extensions/nvdiffrast \
    && git -C /opt/extensions/nvdiffrast checkout --detach "${NVDIFFRAST_COMMIT}" \
    && python3.10 -m pip install /opt/extensions/nvdiffrast --no-build-isolation \
    && git clone https://github.com/JeffreyXiang/nvdiffrec.git /opt/extensions/nvdiffrec \
    && git -C /opt/extensions/nvdiffrec checkout --detach "${NVDIFFREC_COMMIT}" \
    && python3.10 -m pip install /opt/extensions/nvdiffrec --no-build-isolation \
    && git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /opt/extensions/CuMesh \
    && git -C /opt/extensions/CuMesh checkout --detach "${CUMESH_COMMIT}" \
    && git -C /opt/extensions/CuMesh submodule update --init --recursive \
    && python3.10 -m pip install /opt/extensions/CuMesh --no-build-isolation \
    && git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /opt/extensions/FlexGEMM \
    && git -C /opt/extensions/FlexGEMM checkout --detach "${FLEXGEMM_COMMIT}" \
    && git -C /opt/extensions/FlexGEMM submodule update --init --recursive \
    && python3.10 -m pip install /opt/extensions/FlexGEMM --no-build-isolation \
    && python3.10 -m pip install /opt/GenRecon/o-voxel --no-build-isolation --no-deps

COPY workers/genrecon /opt/reconevery-genrecon-worker
RUN python3.10 -m pip install /opt/reconevery-genrecon-worker

ENV PYTHONPATH=/opt/GenRecon
WORKDIR /workspace

HEALTHCHECK --interval=30s --timeout=20s --retries=3 CMD \
  python3.10 -c "import genrecon, flash_attn, cumesh, flex_gemm, o_voxel, nvdiffrast.torch, nvdiffrec_render"

ENTRYPOINT ["python3.10", "-m", "genrecon_worker"]
