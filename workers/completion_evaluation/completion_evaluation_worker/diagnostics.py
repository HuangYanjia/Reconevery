def classify_error(message: str) -> str:
    lowered = message.lower()
    if "out of memory" in lowered:
        return "oom"
    if "nvdiffrast" in lowered:
        return "rasterizer"
    if "candidate" in lowered:
        return "candidate"
    return "worker_failure"
