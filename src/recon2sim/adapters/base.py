from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from recon2sim.config import StageConfig


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    kind: str
    source: str


@dataclass(frozen=True)
class HealthcheckResult:
    ok: bool
    message: str


@dataclass(frozen=True)
class StageResult:
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    metrics: dict[str, float | int | str | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class StageContext:
    stage_name: str
    input_dir: Path
    run_dir: Path
    config: StageConfig
    seed: int

    def path(self, *parts: str) -> Path:
        return self.run_dir.joinpath(*parts)

    def signature(self) -> str:
        payload = {
            "stage": self.stage_name,
            "input": str(self.input_dir.resolve()),
            "config": self.config.model_dump(mode="json"),
            "seed": self.seed,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class Adapter(Protocol):
    name: str

    def healthcheck(self) -> HealthcheckResult: ...
    def prepare(self, context: StageContext) -> None: ...
    def run(self, context: StageContext) -> StageResult: ...
    def collect(self, context: StageContext) -> list[ArtifactRecord]: ...
