FROM python:3.12-slim

ARG APRILTAG_COMMIT=0e16a12dd380fd607e4afd54712ee9b1ffb9ec8f

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

COPY workers/world_calibration /worker
RUN pip install --no-cache-dir /worker

RUN git clone https://github.com/AprilRobotics/apriltag.git /opt/apriltag \
    && git -C /opt/apriltag checkout "$APRILTAG_COMMIT" \
    && cmake -S /opt/apriltag -B /opt/apriltag/build -DBUILD_PYTHON_WRAPPER=ON \
    && cmake --build /opt/apriltag/build --parallel \
    && cmake --install /opt/apriltag/build \
    && ldconfig

ENTRYPOINT ["python", "-m", "world_calibration_worker"]
