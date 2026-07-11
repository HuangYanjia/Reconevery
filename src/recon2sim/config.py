from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class AdapterConfig(BaseModel):
    name: str
    config: dict[str, Any] = Field(default_factory=dict)
    command: list[str] | None = None
    timeout_s: float = 60
    retries: int = 0
    required_gpu_memory_gb: float = 0
    env: list[str] = Field(default_factory=list)


class StageConfig(BaseModel):
    enabled: bool = True
    adapter: AdapterConfig
    depends_on: list[str] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    seed: int = 7
    stages: dict[str, StageConfig]
    paths: dict[str, str] = Field(default_factory=dict)


def load_config(path: Path) -> PipelineConfig:
    with path.open("r", encoding="utf-8") as file:
        return PipelineConfig.model_validate(yaml.safe_load(file))
