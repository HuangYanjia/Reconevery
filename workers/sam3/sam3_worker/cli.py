from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sam3_worker.healthcheck import format_healthcheck
from sam3_worker.inference import run_inference


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconevery isolated SAM 3 worker")
    subparsers = parser.add_subparsers(dest="action", required=True)
    health = subparsers.add_parser("healthcheck")
    health.add_argument("--config", type=Path, required=True)
    infer = subparsers.add_parser("infer")
    infer.add_argument("--request", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    subparsers.add_parser("importcheck")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.action == "healthcheck":
            print(format_healthcheck(args.config))
        elif args.action == "infer":
            run_inference(args.request, args.output_dir)
        else:
            import sam3
            import torch
            import torchvision

            print(
                json.dumps(
                    {
                        "sam3": sam3.__version__,
                        "torch": torch.__version__,
                        "torchvision": torchvision.__version__,
                    },
                    sort_keys=True,
                )
            )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
