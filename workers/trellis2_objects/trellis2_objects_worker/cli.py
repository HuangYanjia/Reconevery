from __future__ import annotations

import argparse
import json
from pathlib import Path

from trellis2_objects_worker.healthcheck import run_healthcheck
from trellis2_objects_worker.inference import infer
from trellis2_objects_worker.native_render import render_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["healthcheck", "generate", "render"])
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.action == "healthcheck":
        result = run_healthcheck(args.request)
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(0 if result.get("available") else 1)
    if args.input_root is None or args.output_dir is None:
        parser.error("--input-root and --output-dir are required")
    if args.action == "render":
        render_candidate(args.request, args.input_root, args.output_dir)
        return
    infer(args.request, args.input_root, args.output_dir)


if __name__ == "__main__":
    main()
