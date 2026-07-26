from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class WorkerRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    seed: int
    output_directory: str
    registration_configuration: dict[str, Any] | None = None
    evaluation_configuration: dict[str, Any] | None = None
