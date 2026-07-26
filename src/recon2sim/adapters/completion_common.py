from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from recon2sim.adapters.base import HealthcheckResult, StageContext
from recon2sim.adapters.ingest import allowed_environment, resolve_executable
from recon2sim.ir import StrictModel
from recon2sim.storage import atomic_write_json

CompletionExecutionMode = Literal["local_worker", "docker", "fake_worker"]


def resolve_worker_python(value: str) -> str | None:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        return str(candidate.absolute()) if candidate.is_file() else None
    return shutil.which(value)


class CompletionWorkerConfig(StrictModel):
    execution_mode: CompletionExecutionMode
    worker_python: str = "python"
    worker_module: str
    worker_script: str | None = None
    docker_executable: str = "docker"
    docker_image: str
    docker_gpus: str = "device=0"
    docker_model_mounts: dict[str, str] = Field(default_factory=dict)
    fake_mode: str = "success"

    @field_validator("docker_model_mounts")
    @classmethod
    def valid_read_only_model_mounts(cls, value: dict[str, str]) -> dict[str, str]:
        for source, destination in value.items():
            if not Path(source).expanduser().is_absolute():
                raise ValueError("Docker model mount sources must be absolute")
            target = Path(destination)
            if not target.is_absolute() or not (
                target == Path("/models")
                or target == Path("/cache")
                or Path("/models") in target.parents
                or Path("/cache") in target.parents
            ):
                raise ValueError("Docker model mounts must target /models or /cache")
        return value

    @model_validator(mode="after")
    def isolated_worker(self) -> CompletionWorkerConfig:
        if self.execution_mode == "fake_worker":
            if self.worker_script is None:
                raise ValueError("fake completion execution requires worker_script")
            return self
        if self.execution_mode == "local_worker":
            python = resolve_worker_python(self.worker_python)
            if python is None:
                raise ValueError(f"configured worker Python {self.worker_python!r} was not found")
            root = Path(python).absolute().parent.parent
            if not (root / "pyvenv.cfg").is_file() and not (root / "conda-meta").is_dir():
                raise ValueError("completion worker must use an isolated environment")
            if root.resolve() == Path(sys.prefix).resolve():
                raise ValueError("completion worker must not use the core environment")
        return self


def local_worker_command(
    config: CompletionWorkerConfig,
    action: str,
    request_path: Path,
) -> list[str]:
    python = resolve_worker_python(config.worker_python)
    if python is None:
        raise RuntimeError(f"configured worker Python {config.worker_python!r} was not found")
    if config.execution_mode == "fake_worker":
        assert config.worker_script is not None
        script = Path(config.worker_script)
        if not script.is_absolute():
            script = Path.cwd() / script
        if not script.is_file():
            raise RuntimeError(f"fake completion worker does not exist: {script}")
        return [python, str(script.resolve()), action, "--request", str(request_path)]
    return [python, "-m", config.worker_module, action, "--request", str(request_path)]


def worker_command(
    context: StageContext,
    config: CompletionWorkerConfig,
    action: str,
    request_relative_path: str,
    output_relative_path: str,
) -> list[str]:
    request = Path(request_relative_path)
    if config.execution_mode != "docker":
        return [
            *local_worker_command(config, action, context.path(*request.parts)),
            "--input-root",
            str(context.run_dir.resolve()),
            "--output-dir",
            str(context.path(*Path(output_relative_path).parts).resolve()),
        ]
    docker = resolve_executable(config.docker_executable)
    if docker is None:
        raise RuntimeError("Docker executable was not found")
    user = (
        ["--user", f"{os.getuid()}:{os.getgid()}"]
        if hasattr(os, "getuid") and hasattr(os, "getgid")
        else []
    )
    model_mounts: list[str] = []
    canonical = context.canonical_run_dir.resolve()
    for source, destination in sorted(config.docker_model_mounts.items()):
        host = Path(source).expanduser().resolve()
        if not host.exists():
            raise RuntimeError(f"Docker model mount source does not exist: {host}")
        if host == canonical or canonical in host.parents or host in canonical.parents:
            raise RuntimeError("Docker model mounts cannot expose the canonical run directory")
        model_mounts.extend(["-v", f"{host}:{destination}:ro"])
    forwarded_environment = [
        argument
        for name in (
            "HF_TOKEN",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
        )
        if name in os.environ and name in context.config.adapter.env
        for argument in ("-e", name)
    ]
    return [
        docker,
        "run",
        "--rm",
        "--gpus",
        config.docker_gpus,
        *user,
        *forwarded_environment,
        "-v",
        f"{context.run_dir.resolve()}:/workspace:rw",
        *model_mounts,
        "-w",
        "/workspace",
        config.docker_image,
        action,
        "--request",
        f"/workspace/{request.as_posix()}",
        "--input-root",
        "/workspace",
        "--output-dir",
        f"/workspace/{output_relative_path}",
    ]


def completion_healthcheck(
    context: StageContext | None,
    config_type: type[CompletionWorkerConfig],
    *,
    worker_name: str,
) -> HealthcheckResult:
    if context is None:
        return HealthcheckResult(False, f"{worker_name} healthcheck requires --config")
    try:
        config = config_type.model_validate(context.config.adapter.config)
    except ValueError as exc:
        return HealthcheckResult(False, f"invalid {worker_name} configuration: {exc}")
    with tempfile.TemporaryDirectory(prefix=f"reconevery-{worker_name}-health-") as temp:
        health_path = Path(temp) / "health.json"
        atomic_write_json(
            health_path,
            {
                "worker_name": worker_name,
                **config.model_dump(mode="json"),
            },
        )
        if config.execution_mode == "docker":
            docker = resolve_executable(config.docker_executable)
            if docker is None:
                return HealthcheckResult(False, "Docker executable was not found")
            model_mounts = [
                argument
                for source, destination in sorted(config.docker_model_mounts.items())
                for argument in ("-v", f"{Path(source).expanduser().resolve()}:{destination}:ro")
            ]
            user = (
                ["--user", f"{os.getuid()}:{os.getgid()}"]
                if hasattr(os, "getuid") and hasattr(os, "getgid")
                else []
            )
            forwarded_environment = [
                argument
                for name in (
                    "HF_TOKEN",
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                )
                if name in os.environ and name in context.config.adapter.env
                for argument in ("-e", name)
            ]
            command = [
                docker,
                "run",
                "--rm",
                "--gpus",
                config.docker_gpus,
                *user,
                *forwarded_environment,
                "-v",
                f"{health_path.parent.resolve()}:/health:ro",
                *model_mounts,
                config.docker_image,
                "healthcheck",
                "--request",
                "/health/health.json",
            ]
        else:
            try:
                command = local_worker_command(config, "healthcheck", health_path)
            except RuntimeError as exc:
                return HealthcheckResult(False, str(exc))
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=min(context.config.adapter.timeout_s, 120),
                check=False,
                env=allowed_environment(context),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return HealthcheckResult(False, f"{worker_name} healthcheck failed: {exc}")
        output = result.stdout.strip() or result.stderr.strip()
        return HealthcheckResult(result.returncode == 0, output or f"{worker_name} unavailable")
