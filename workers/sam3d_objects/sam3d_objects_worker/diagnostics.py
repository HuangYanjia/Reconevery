def classify_error(message: str) -> str:
    lowered = message.lower()
    if "out of memory" in lowered:
        return "oom"
    if "gated" in lowered or "unauthorized" in lowered or "access denied" in lowered:
        return "gated_access"
    if "checkpoint" in lowered:
        return "checkpoint"
    return "worker_failure"
