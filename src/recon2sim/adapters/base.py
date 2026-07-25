from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol

from pydantic import BaseModel, Field, field_validator

from recon2sim.config import StageConfig
from recon2sim.ir import StrictModel

OutputValidation = Literal["exists", "json", "png", "obj", "scene_ir"]
InputMaterialization = Literal["reflink_or_copy", "copy", "reference_only"]


class ArtifactRecord(StrictModel):
    relative_path: str = Field(min_length=1)
    artifact_type: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    producer_stage: str = Field(min_length=1)
    producer_adapter: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    schema_identifier: str | None = None

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("artifact record paths must be relative to the run directory")
        return value


@dataclass(frozen=True)
class OutputSpec:
    relative_path: str
    artifact_type: str
    media_type: str
    source_type: str
    validation: OutputValidation = "exists"
    schema_identifier: str | None = None
    model: type[BaseModel] | None = None


@dataclass(frozen=True)
class InputSpec:
    relative_path: str
    artifact_type: str
    required: bool = True
    expected_sha256: str | None = None
    source_path: Path | None = None
    source_artifact_path: str | None = None
    materialization_mode: InputMaterialization = "reflink_or_copy"


@dataclass(frozen=True)
class HealthcheckResult:
    ok: bool
    message: str


@dataclass(frozen=True)
class StageResult:
    outputs: list[OutputSpec] = field(default_factory=list)
    metrics: dict[str, float | int | str | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class StageContext:
    stage_name: str
    input_dir: Path
    run_dir: Path
    canonical_run_dir: Path
    config: StageConfig
    seed: int
    attempt: int = 1

    def path(self, *parts: str) -> Path:
        return self.run_dir.joinpath(*parts)

    def canonical_path(self, *parts: str) -> Path:
        return self.canonical_run_dir.joinpath(*parts)


class Adapter(Protocol):
    name: str
    version: str

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult: ...

    def prepare(self, context: StageContext) -> None: ...

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]: ...

    def run(self, context: StageContext) -> StageResult: ...
