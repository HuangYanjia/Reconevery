from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def load_inference_class(checkout: Path) -> type[Any]:
    module_path = checkout / "notebook" / "inference.py"
    if not module_path.is_file():
        raise RuntimeError("verified official checkout does not contain notebook/inference.py")
    sys.path.insert(0, str(checkout))
    spec = importlib.util.spec_from_file_location(
        "_reconevery_official_sam3d_inference",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the official SAM 3D Objects inference module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    inference = getattr(module, "Inference", None)
    if inference is None:
        raise RuntimeError("official SAM 3D Objects module does not expose Inference")
    return inference
