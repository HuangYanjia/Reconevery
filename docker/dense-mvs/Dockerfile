FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS builder

ARG COLMAP_COMMIT=9c23f6942fe69962e06030905e77067c8673382f
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake git ninja-build \
    libboost-graph-dev libboost-program-options-dev libceres-dev libcgal-dev \
    libeigen3-dev libflann-dev libfreeimage-dev libglew-dev libgoogle-glog-dev \
    libmetis-dev libopenimageio-dev libsqlite3-dev python3.10 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/colmap/colmap.git /opt/colmap-src \
    && git -C /opt/colmap-src checkout --detach "${COLMAP_COMMIT}" \
    && test "$(git -C /opt/colmap-src rev-parse HEAD)" = "${COLMAP_COMMIT}"
RUN cmake -S /opt/colmap-src -B /opt/colmap-build -GNinja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/colmap \
    -DGUI_ENABLED=OFF -DTESTS_ENABLED=OFF -DCUDA_ENABLED=ON \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    && cmake --build /opt/colmap-build --target colmap_exe \
    && cmake --install /opt/colmap-build

COPY workers/dense_mvs /opt/reconevery-dense-mvs
RUN python3.10 -m pip install --no-cache-dir /opt/reconevery-dense-mvs

ENV PATH=/opt/colmap/bin:${PATH}
ENTRYPOINT ["python3.10", "-m", "dense_mvs_worker"]
