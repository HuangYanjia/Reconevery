from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import yaml
from pydantic import Field, field_validator
from recon2sim.ir import StrictModel


class OutputConfig(StrictModel):
    path: str
    artifact_type: str = "adapter_output"
    media_type: str = "application/octet-stream"
    source_type: str = "command"
    schema_identifier: str | None = None
    validation: Literal["exists", "json", "png", "obj", "scene_ir"] = "exists"

    @field_validator("path")
    @classmethod
    def relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("expected output paths must be relative to the run directory")
        return value


class AdapterConfig(StrictModel):
    name: str = Field(min_length=1)
    config: dict[str, Any] = Field(default_factory=dict)
    command: list[str] | None = None
    timeout_s: float = Field(default=60, gt=0)
    retries: int = Field(default=0, ge=0)
    required_gpu_memory_gb: float = Field(default=0, ge=0)
    env: list[str] = Field(default_factory=list)
    expected_outputs: list[OutputConfig] = Field(default_factory=list)

    @field_validator("command")
    @classmethod
    def nonempty_command(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and (not value or any(not part for part in value)):
            raise ValueError("command must contain non-empty arguments")
        return value

    @field_validator("env")
    @classmethod
    def unique_environment_names(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("allowed environment variable names must be unique")
        if any(not value.isidentifier() for value in values):
            raise ValueError("environment allowlist entries must be variable names")
        return values


class StageConfig(StrictModel):
    enabled: bool = True
    adapter: AdapterConfig
    depends_on: list[str] = Field(default_factory=list)

    @field_validator("depends_on")
    @classmethod
    def unique_dependencies(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("stage dependencies must be unique")
        return values


class PipelineConfig(StrictModel):
    seed: int = 7
    stages: dict[str, StageConfig] = Field(min_length=1)
    paths: dict[str, str] = Field(default_factory=dict)
    allow_existing_artifacts_for_disabled_dependencies: bool = False

    @field_validator("stages")
    @classmethod
    def valid_stage_names(cls, values: dict[str, StageConfig]) -> dict[str, StageConfig]:
        invalid = [name for name in values if re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None]
        if invalid:
            raise ValueError(f"stage names must be safe identifiers: {invalid}")
        return values


def load_config(path: Path) -> PipelineConfig:
    if not path.is_file():
        raise FileNotFoundError(f"pipeline config does not exist or is not a file: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = cast(dict[str, Any], yaml.safe_load(file))
    if not isinstance(payload, dict):
        raise ValueError(f"pipeline config must contain a mapping: {path}")
    return PipelineConfig.model_validate(payload)
