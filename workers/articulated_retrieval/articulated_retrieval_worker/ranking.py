from __future__ import annotations


def rank_records(
    records: list[dict[str, object]],
    *,
    semantic_label: str,
    observed_joint_types: set[str],
    observed_part_count: int,
) -> list[tuple[float, dict[str, float], dict[str, object]]]:
    ranked = []
    normalized_label = semantic_label.casefold()
    for record in records:
        category = str(record.get("category", "")).casefold()
        semantic = (
            1.0
            if category == normalized_label
            else (0.6 if category in normalized_label or normalized_label in category else 0.0)
        )
        candidate_joint_types = {
            str(value) for value in record.get("joint_types", []) if isinstance(value, str)
        }
        joint = (
            len(candidate_joint_types & observed_joint_types) / max(len(observed_joint_types), 1)
            if observed_joint_types
            else 0.5
        )
        link_count = int(record.get("link_count", 0))
        part = 1.0 / (1.0 + abs(link_count - (observed_part_count + 1)))
        terms = {
            "semantic_category": semantic,
            "joint_type": joint,
            "part_count": part,
            "rgb_appearance": 0.0,
        }
        score = 0.50 * semantic + 0.35 * joint + 0.15 * part
        ranked.append((score, terms, record))
    return sorted(ranked, key=lambda item: (-item[0], str(item[2].get("asset_id", ""))))
