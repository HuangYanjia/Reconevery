from __future__ import annotations


def snapshot_summary(plan: dict[str, object]) -> str:
    decisions = plan.get("decisions", [])
    count = (
        sum(
            1
            for item in decisions
            if isinstance(item, dict) and item.get("articulated_kinematic_bundle") is not None
        )
        if isinstance(decisions, list)
        else 0
    )
    return f"articulated snapshots: {count}"


__all__ = ["snapshot_summary"]
