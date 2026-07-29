from __future__ import annotations

import argparse
import json
from pathlib import Path

from world_calibration_worker.healthcheck import health
from world_calibration_worker.solver import solve


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="world_calibration_worker")
    subparsers = value.add_subparsers(dest="action", required=True)
    health_parser = subparsers.add_parser("healthcheck")
    health_parser.add_argument("--request", type=Path)
    solve_parser = subparsers.add_parser("solve")
    solve_parser.add_argument("--request", type=Path, required=True)
    solve_parser.add_argument("--input-root", type=Path, required=True)
    solve_parser.add_argument("--output-dir", type=Path, required=True)
    return value


def main() -> int:
    arguments = parser().parse_args()
    if arguments.action == "healthcheck":
        result = health()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    solve(arguments.request, arguments.input_root, arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
