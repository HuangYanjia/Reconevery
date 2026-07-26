from __future__ import annotations

from pathlib import Path


def write_patch_match_config(
    path: Path,
    *,
    ordered_frame_ids: list[str],
    filename_by_frame: dict[str, str],
    mode: str,
    explicit_source_ids: dict[str, list[str]],
    neighbor_count: int,
) -> None:
    known = set(ordered_frame_ids)
    if set(filename_by_frame) != known:
        raise ValueError("dense image filenames do not match registered frame IDs")
    lines: list[str] = []
    for index, frame_id in enumerate(ordered_frame_ids):
        lines.append(filename_by_frame[frame_id])
        if mode == "auto":
            lines.append("__auto__, 20")
            continue
        if mode == "sequential_neighbors":
            ranked = sorted(
                (
                    (abs(candidate_index - index), candidate_index, candidate)
                    for candidate_index, candidate in enumerate(ordered_frame_ids)
                    if candidate != frame_id
                )
            )
            sources = [candidate for _, _, candidate in ranked[:neighbor_count]]
        elif mode == "explicit":
            sources = explicit_source_ids.get(frame_id, [])
            unknown = set(sources) - known
            if unknown:
                raise ValueError(
                    f"explicit PatchMatch sources for {frame_id} contain unknown IDs: "
                    f"{sorted(unknown)}"
                )
            if frame_id in sources:
                raise ValueError(f"explicit PatchMatch sources for {frame_id} include itself")
        else:
            raise ValueError(f"unsupported PatchMatch source-view mode {mode!r}")
        if not sources:
            raise ValueError(f"PatchMatch source list is empty for {frame_id}")
        lines.append(", ".join(filename_by_frame[source] for source in sources))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["write_patch_match_config"]
