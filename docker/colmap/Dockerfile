ARG COLMAP_BASE_IMAGE=colmap/colmap:20260427.6785
FROM ${COLMAP_BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
USER root
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="Reconevery Phase 1 COLMAP runtime" \
      org.opencontainers.image.description="Pinned COLMAP 4.0.4-era runtime with FFmpeg; no models or user data" \
      org.opencontainers.image.version="phase1"

WORKDIR /workspace
ENTRYPOINT []
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD colmap -h >/dev/null && ffmpeg -version >/dev/null || exit 1
CMD ["colmap", "-h"]
