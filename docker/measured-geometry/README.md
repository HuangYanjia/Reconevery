# Measured geometry image

This image contains only NumPy/OpenCV/Pillow geometry processing. It loads no
SAM or GenRecon checkpoint and receives only the attempt-local masks and dense
COLMAP maps declared by `InputSpec`.

```bash
docker build -f docker/measured-geometry/Dockerfile \
  -t reconevery/measured-geometry:phase5a .
```
