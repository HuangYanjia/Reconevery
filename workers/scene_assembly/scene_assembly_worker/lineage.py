from __future__ import annotations


def lineage_summary(plan: dict[str, object]) -> str:
    reference = plan.get("lineage_report")
    return f"lineage report: {reference}" if reference is not None else "lineage unavailable"


__all__ = ["lineage_summary"]
