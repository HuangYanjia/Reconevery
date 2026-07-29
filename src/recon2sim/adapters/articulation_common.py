from __future__ import annotations

from pathlib import Path

from recon2sim.adapters.base import HealthcheckResult, StageContext
from recon2sim.adapters.completion_common import (
    CompletionWorkerConfig,
    completion_healthcheck,
    worker_command,
)
from recon2sim.adapters.ingest import ProcessExecutionError, run_process


class ArticulationWorkerConfig(CompletionWorkerConfig):
    pass


def articulation_healthcheck(
    context: StageContext | None,
    config_type: type[ArticulationWorkerConfig],
    *,
    worker_name: str,
) -> HealthcheckResult:
    return completion_healthcheck(context, config_type, worker_name=worker_name)


def run_articulation_worker(
    context: StageContext,
    config: ArticulationWorkerConfig,
    *,
    action: str,
    request_path: str,
    output_directory: str,
    log_name: str,
) -> None:
    command = worker_command(
        context,
        config,
        action,
        request_path,
        output_directory,
    )
    try:
        run_process(
            command,
            context=context,
            name=log_name,
            log_directory="reconstruction/articulation/raw/logs",
        )
    except ProcessExecutionError as exc:
        stderr = exc.result.stderr.lower()
        if "out of memory" in stderr or "cuda oom" in stderr:
            raise RuntimeError(f"{log_name} failed: gpu_out_of_memory") from exc
        raise


def load_json(path: Path) -> object:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "ArticulationWorkerConfig",
    "articulation_healthcheck",
    "load_json",
    "run_articulation_worker",
]
