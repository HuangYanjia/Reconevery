from __future__ import annotations


def overlap_summary(report: dict[str, object]) -> str:
    diagnostics = report.get("diagnostics", [])
    count = len(diagnostics) if isinstance(diagnostics, list) else 0
    return f"overlap diagnostics: {count}"


__all__ = ["overlap_summary"]
