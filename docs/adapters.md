# Adapters

Adapters expose healthcheck, prepare, run, and collect behavior. Future COLMAP, MapAnything, SAM 3, GenRecon, and SceneSmith integrations should be wrappers that read declared input directories and write declared outputs. Do not import their Python packages into `src/recon2sim`.

For each adapter document command templates, inputs, outputs, environment variable names, timeout, GPU memory metadata, health check, and provenance mapping. `configs/full.example.yaml` contains nonfunctional templates.
