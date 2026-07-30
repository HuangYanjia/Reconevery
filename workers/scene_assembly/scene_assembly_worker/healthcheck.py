from __future__ import annotations

import json


def healthcheck() -> None:
    print(
        json.dumps(
            {
                "ok": True,
                "worker": "scene_assembly_worker",
                "visual_only": True,
                "collision_generation": False,
                "physics_identification": False,
            },
            sort_keys=True,
        )
    )


__all__ = ["healthcheck"]
