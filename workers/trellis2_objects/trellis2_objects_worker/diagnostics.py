def classify_error(message: str) -> str:
    lowered = message.lower()
    if "out of memory" in lowered:
        return "oom"
    if "checkpoint" in lowered or "model" in lowered:
        return "runtime_asset"
    if "cuda" in lowered:
        return "cuda"
    return "worker_failure"
