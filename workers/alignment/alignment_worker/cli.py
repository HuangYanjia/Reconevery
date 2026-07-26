from __future__ import annotations

import argparse
from pathlib import Path

from alignment_worker.healthcheck import print_healthcheck
from alignment_worker.inference import run_inference


def main() -> int:
    parser = argparse.ArgumentParser(prog="alignment-worker")
    commands = parser.add_subparsers(dest="command", required=True)
    health = commands.add_parser("healthcheck")
    health.add_argument("--config", type=Path, required=True)
    infer = commands.add_parser("infer")
    infer.add_argument("--request", type=Path, required=True)
    infer.add_argument("--input-root", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "healthcheck":
        print_healthcheck(args.config)
        return 0
    run_inference(args.request, args.input_root, args.output_dir)
    return 0
