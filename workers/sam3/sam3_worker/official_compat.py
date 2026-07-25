from __future__ import annotations

import inspect
from typing import Any


def official_propagation_directions(
    tracking_direction: str,
) -> tuple[str, ...]:
    if tracking_direction == "forward_backward":
        return ("forward", "backward")
    if tracking_direction in {"forward", "backward"}:
        return (tracking_direction,)
    raise RuntimeError(f"unsupported tracking direction: {tracking_direction}")


def apply_sam31_start_session_compatibility(predictor: Any) -> bool:
    init_state = predictor.model.init_state
    if "offload_state_to_cpu" in inspect.signature(init_state).parameters:
        return False

    def compatible_init_state(*args: Any, **kwargs: Any) -> Any:
        offload_state = kwargs.pop("offload_state_to_cpu", False)
        if offload_state:
            raise RuntimeError(
                "the pinned SAM 3.1 Multiplex model does not support offload_state_to_cpu"
            )
        return init_state(*args, **kwargs)

    predictor.model.init_state = compatible_init_state
    return True
