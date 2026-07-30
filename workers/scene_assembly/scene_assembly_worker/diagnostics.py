from __future__ import annotations


def object_lines(record: dict[str, object]) -> list[str]:
    decisions = record.get("object_decisions", record.get("decisions", []))
    if not isinstance(decisions, list):
        return []
    return [
        f"{item.get('object_id')}: {item.get('status')}"
        for item in decisions
        if isinstance(item, dict)
    ]


__all__ = ["object_lines"]
