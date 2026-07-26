from __future__ import annotations

import argparse
import json
from pathlib import Path

from completion_evaluation_worker.healthcheck import run_healthcheck


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["healthcheck", "prepare-evidence", "register", "evaluate"],
    )
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.action == "healthcheck":
        print(json.dumps(run_healthcheck(), sort_keys=True))
        return
    if args.input_root is None or args.output_dir is None:
        parser.error("--input-root and --output-dir are required")
    from completion_evaluation_worker.inference import run_action

    run_action(args.action, args.request, args.input_root, args.output_dir)


if __name__ == "__main__":
    main()
