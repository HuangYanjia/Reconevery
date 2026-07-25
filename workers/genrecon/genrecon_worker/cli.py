from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from genrecon_worker.healthcheck import run_healthcheck
from genrecon_worker.inference import run_inference
from genrecon_worker.schema import WorkerConfiguration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconevery isolated GenRecon worker")
    commands = parser.add_subparsers(dest="command", required=True)
    health = commands.add_parser("healthcheck")
    health.add_argument("--config", type=Path, required=True)
    inference = commands.add_parser("infer")
    inference.add_argument("--request", type=Path, required=True)
    inference.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "healthcheck":
        config = WorkerConfiguration.model_validate_json(args.config.read_text(encoding="utf-8"))
        try:
            result = run_healthcheck(config)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(json.dumps(result, sort_keys=True))
        return 0
    run_inference(args.request, args.output_dir)
    return 0
