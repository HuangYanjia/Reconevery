from __future__ import annotations

import argparse
import json
from pathlib import Path

from dense_mvs_worker.healthcheck import healthcheck
from dense_mvs_worker.inference import infer


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    health = commands.add_parser("healthcheck")
    health.add_argument("--config", type=Path, required=True)
    inference = commands.add_parser("infer")
    inference.add_argument("--request", type=Path, required=True)
    inference.add_argument("--input-root", type=Path, required=True)
    inference.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = (
        healthcheck(args.config)
        if args.command == "healthcheck"
        else infer(args.request, args.input_root, args.output_dir)
    )
    print(json.dumps(result, sort_keys=True))
    return 0
