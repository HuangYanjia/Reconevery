# Completion evaluation worker

This isolated CUDA worker prepares fitting-only measured evidence, registers candidate
geometry with a proper positive-scale Sim(3), and renders frozen candidates into
held-out COLMAP cameras. Mesh rendering uses nvdiffrast and the same homogeneous
projection contract as Phase 4.

The worker never loads a generative checkpoint. Candidate generation and held-out
evaluation remain separate processes and evidence sets.
