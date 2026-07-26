FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY workers/measured_geometry /opt/reconevery-measured-geometry
RUN python3.10 -m pip install --no-cache-dir /opt/reconevery-measured-geometry

ENTRYPOINT ["python3.10", "-m", "measured_geometry_worker"]
