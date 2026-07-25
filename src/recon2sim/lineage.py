from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Protocol


class FrameIdentity(Protocol):
    frame_id: str
    relative_path: str
    sha256: str


def frame_sequence_digest(frames: Iterable[FrameIdentity]) -> str:
    """Hash ordered normalized-frame identities without depending on JSON formatting."""
    payload = [(frame.frame_id, frame.relative_path, frame.sha256) for frame in frames]
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["frame_sequence_digest"]
