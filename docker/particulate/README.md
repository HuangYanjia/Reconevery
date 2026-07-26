# Official Particulate image

The image pins official Particulate commit
`dee37a75c449f324d9989993461ee09eaccc1686` with Python 3.10, PyTorch 2.4.0,
and CUDA 12.4 according to the official requirements.
The image additionally pins Transformers 4.46.3 because the official unbounded
dependency currently resolves to a 5.x release that cannot import with PyTorch 2.4.

Checkpoints are not embedded. Mount `model.pt` and
`model_objaverse.ckpt` read-only under `/models`, pass their paths in the worker
configuration, and keep Hugging Face offline during inference. PartField licensing
restricts the current stack to research evaluation.
