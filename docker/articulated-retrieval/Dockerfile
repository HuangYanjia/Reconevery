FROM python:3.10-slim

ARG HOST_UID=1000
ARG HOST_GID=1000
COPY workers/articulated_retrieval /opt/reconevery-worker
RUN python -m pip install --no-cache-dir /opt/reconevery-worker
RUN groupadd --gid "${HOST_GID}" reconevery && \
    useradd --uid "${HOST_UID}" --gid "${HOST_GID}" --create-home reconevery && \
    mkdir -p /workspace && chown reconevery:reconevery /workspace
USER reconevery
WORKDIR /workspace
ENTRYPOINT ["python", "-m", "articulated_retrieval_worker"]
