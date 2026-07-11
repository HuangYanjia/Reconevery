from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if hasattr(record, "stage"):
            data["stage"] = record.stage
        return json.dumps(data, sort_keys=True)


def configure_run_logger(run_dir: Path) -> logging.Logger:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("recon2sim.run")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_dir / "run.jsonl", encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger
