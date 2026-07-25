FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG SAM3_COMMIT=46957e47805eaa273f4aa7bbbd25a88bca9108ce

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        git \
        python3.12 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:${PATH}
ENV PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        torch==2.10.0 \
        torchvision==0.25.0 \
        --index-url https://download.pytorch.org/whl/cu128

RUN git clone https://github.com/facebookresearch/sam3.git /opt/sam3 \
    && git -C /opt/sam3 checkout --detach ${SAM3_COMMIT} \
    && test "$(git -C /opt/sam3 rev-parse HEAD)" = "${SAM3_COMMIT}" \
    && python -m pip install --no-cache-dir -e /opt/sam3

COPY workers/sam3 /opt/reconevery-sam3-worker
RUN python -m pip install --no-cache-dir \
        huggingface-hub==0.36.0 \
        pillow==12.0.0 \
        pydantic==2.12.5 \
    && python -m pip install --no-cache-dir --no-deps /opt/reconevery-sam3-worker

WORKDIR /workspace

# This checks the installed code only. Configuration-aware GPU/checkpoint
# readiness is checked by `recon2sim adapters healthcheck --config ...`.
HEALTHCHECK --interval=30s --timeout=20s --start-period=10s --retries=3 \
    CMD ["python", "-m", "sam3_worker", "importcheck"]

ENTRYPOINT ["python", "-m", "sam3_worker"]
