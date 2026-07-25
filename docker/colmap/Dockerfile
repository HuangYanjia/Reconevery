FROM ubuntu:24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG COLMAP_VERSION=3.11.1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libboost-graph-dev \
        libboost-program-options-dev \
        libboost-system-dev \
        libcgal-dev \
        libceres-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libglew-dev \
        libgoogle-glog-dev \
        libmetis-dev \
        libsqlite3-dev \
        libsuitesparse-dev \
        ninja-build \
        qtbase5-dev \
    && git clone --branch "${COLMAP_VERSION}" --depth 1 \
        https://github.com/colmap/colmap.git /src/colmap \
    && cmake -S /src/colmap -B /src/colmap/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/colmap \
        -DCUDA_ENABLED=OFF \
        -DGUI_ENABLED=OFF \
    && cmake --build /src/colmap/build \
    && cmake --install /src/colmap/build

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG COLMAP_VERSION=3.11.1

LABEL org.opencontainers.image.title="Reconevery Phase 1 COLMAP"
LABEL org.opencontainers.image.version="${COLMAP_VERSION}"
LABEL org.opencontainers.image.source="https://github.com/HuangYanjia/Reconevery"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libboost-graph1.83.0 \
        libboost-program-options1.83.0 \
        libceres4 \
        libflann1.9 \
        libfreeimage3 \
        libglew2.2 \
        libgoogle-glog0v6 \
        libmetis5 \
        libqt5core5t64 \
        libqt5gui5t64 \
        libsqlite3-0 \
        libsuitesparseconfig7 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/colmap /opt/colmap
ENV PATH="/opt/colmap/bin:${PATH}"
WORKDIR /workspace

RUN colmap -h >/dev/null && ffmpeg -version >/dev/null
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD colmap -h >/dev/null && ffmpeg -version >/dev/null || exit 1
