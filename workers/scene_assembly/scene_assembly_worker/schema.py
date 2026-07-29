from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    assembly_plan_path: str
    assembly_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    research_bundle_path: str
    research_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    deployment_bundle_path: str
    deployment_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    overlap_diagnostics_path: str
    overlap_diagnostics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preview_configuration: dict[str, object]
    output_directory: str
    diagnostic_only: bool
    fake_mode: str
    seed: int


__all__ = ["PreviewRequest"]
