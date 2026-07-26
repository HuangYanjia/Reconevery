from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CandidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    run_id: str
    object_id: str
    semantic_label: str
    asset_type_hint: str | None = None
    eligibility_status: str
    backend: str
    official_repository: str
    official_code_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    checkpoint_repository: str
    checkpoint_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    checkpoint_hashes: dict[str, str]
    runtime_model_revisions: dict[str, str] = Field(default_factory=dict)
    runtime_model_hashes: dict[str, dict[str, str]] = Field(default_factory=dict)
    license_policy: dict[str, Any]
    anchor_frame_id: str
    anchor_crop_path: str
    anchor_crop_sha256: str
    anchor_crop_transform: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]
    source_frame_sha256: str
    source_mask_sha256: str
    generation_seed: int
    generation_configuration: dict[str, Any]
    output_directory: str
