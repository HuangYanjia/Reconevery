import json
from typing import Any


def safe_load(s: Any) -> Any:
    if hasattr(s, "read"):
        s = s.read()
    return json.loads(s)


def safe_dump(data: Any, sort_keys: bool = False) -> str:
    return json.dumps(data, indent=2, sort_keys=sort_keys)
