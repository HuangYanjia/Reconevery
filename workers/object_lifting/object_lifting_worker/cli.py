from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from object_lifting_worker.healthcheck import healthcheck_json
from object_lifting_worker.inference import run_inference


def main() -> int:
    parser = argparse.ArgumentParser(prog="object_lifting_worker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    health = subparsers.add_parser("healthcheck")
    health.add_argument("--config", type=Path, required=True)
    infer = subparsers.add_parser("infer")
    infer.add_argument("--request", type=Path, required=True)
    infer.add_argument("--input-root", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "healthcheck":
            print(healthcheck_json(args.config))
            return 0
        run_inference(args.request, args.input_root, args.output_dir)
        return 0
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(
            json.dumps({"error": type(exc).__name__, "message": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
