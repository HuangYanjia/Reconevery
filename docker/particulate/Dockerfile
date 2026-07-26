FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PARTICULATE_COMMIT=dee37a75c449f324d9989993461ee09eaccc1686
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3.10 python3-pip build-essential libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
RUN python3.10 -m pip install --no-cache-dir --upgrade pip && \
    git clone --recursive https://github.com/RuiningLi/particulate /opt/particulate && \
    git -C /opt/particulate checkout --detach "${PARTICULATE_COMMIT}" && \
    git -C /opt/particulate submodule update --init --recursive && \
    test "$(git -C /opt/particulate rev-parse HEAD)" = "${PARTICULATE_COMMIT}" && \
    python3.10 -m pip install --no-cache-dir \
      -r /opt/particulate/requirements.txt && \
    python3.10 -m pip install --no-cache-dir transformers==4.46.3
COPY workers/particulate /opt/reconevery-worker
RUN python3.10 -m pip install --no-cache-dir /opt/reconevery-worker
RUN groupadd --gid "${HOST_GID}" reconevery && \
    useradd --uid "${HOST_UID}" --gid "${HOST_GID}" --create-home reconevery && \
    mkdir -p /workspace /models && \
    chown -R reconevery:reconevery /workspace /models /opt/particulate
USER reconevery
WORKDIR /workspace
ENTRYPOINT ["python3.10", "-m", "particulate_worker"]
