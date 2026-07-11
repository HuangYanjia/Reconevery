from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from recon2sim.config import load_config
from recon2sim.pipeline import PipelineRunner


@pytest.fixture()
def input_dir(tmp_path: Path) -> Path:
    destination = tmp_path / "input"
    shutil.copytree(Path("examples/tabletop"), destination)
    return destination


@pytest.fixture()
def completed_run(tmp_path: Path, input_dir: Path) -> Path:
    run_dir = tmp_path / "run"
    PipelineRunner(load_config(Path("configs/mock.yaml")), input_dir, run_dir).run()
    return run_dir


@pytest.fixture()
def scene_payload(completed_run: Path) -> dict[str, Any]:
    return json.loads((completed_run / "scene_ir" / "scene.json").read_text(encoding="utf-8"))
