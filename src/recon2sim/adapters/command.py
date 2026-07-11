from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any

from recon2sim.adapters.base import HealthcheckResult, OutputSpec, StageContext, StageResult
from recon2sim.artifacts import CommandResultArtifact
from recon2sim.ir import SceneIR
from recon2sim.storage import atomic_write_json, atomic_write_text


class CommandExecutionError(RuntimeError):
    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


class CommandAdapter:
    name = "command"
    version = "0.1.0"

    def healthcheck(self) -> HealthcheckResult:
        return HealthcheckResult(True, "subprocess execution is available")

    def prepare(self, context: StageContext) -> None:
        context.run_dir.mkdir(parents=True, exist_ok=True)

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        stdout_relative = f"logs/{context.stage_name}.attempt_{context.attempt}.stdout.log"
        stderr_relative = f"logs/{context.stage_name}.attempt_{context.attempt}.stderr.log"
        result_relative = f"logs/{context.stage_name}.attempt_{context.attempt}.command.json"
        outputs = [
            OutputSpec(
                stdout_relative,
                "command_stdout",
                "text/plain",
                "command",
            ),
            OutputSpec(
                stderr_relative,
                "command_stderr",
                "text/plain",
                "command",
            ),
            OutputSpec(
                result_relative,
                "command_execution",
                "application/json",
                "command",
                validation="json",
                schema_identifier="recon2sim/command-result/0.1.0",
                model=CommandResultArtifact,
            ),
        ]
        for configured in context.config.adapter.expected_outputs:
            model = SceneIR if configured.validation == "scene_ir" else None
            outputs.append(
                OutputSpec(
                    relative_path=configured.path,
                    artifact_type=configured.artifact_type,
                    media_type=configured.media_type,
                    source_type=configured.source_type,
                    validation=configured.validation,
                    schema_identifier=configured.schema_identifier,
                    model=model,
                )
            )
        return outputs

    def run(self, context: StageContext) -> StageResult:
        command = context.config.adapter.command
        if not command:
            raise ValueError("command adapter requires a non-empty command list")

        stdout_relative = f"logs/{context.stage_name}.attempt_{context.attempt}.stdout.log"
        stderr_relative = f"logs/{context.stage_name}.attempt_{context.attempt}.stderr.log"
        result_relative = f"logs/{context.stage_name}.attempt_{context.attempt}.command.json"
        allowed_environment = {
            name: os.environ[name] for name in context.config.adapter.env if name in os.environ
        }

        start = time.monotonic()
        process = subprocess.Popen(
            command,
            cwd=context.run_dir,
            env=allowed_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=context.config.adapter.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
        duration = time.monotonic() - start

        atomic_write_text(context.path(stdout_relative), stdout)
        atomic_write_text(context.path(stderr_relative), stderr)
        command_result = CommandResultArtifact(
            stage=context.stage_name,
            attempt=context.attempt,
            command=command,
            return_code=process.returncode,
            duration_s=duration,
            timed_out=timed_out,
            stdout_path=stdout_relative,
            stderr_path=stderr_relative,
        )
        atomic_write_json(context.path(result_relative), command_result)
        details: dict[str, Any] = command_result.model_dump(mode="json")

        if timed_out:
            raise CommandExecutionError(
                f"command for stage {context.stage_name!r} timed out after "
                f"{context.config.adapter.timeout_s} seconds",
                details,
            )
        if process.returncode != 0:
            raise CommandExecutionError(
                f"command for stage {context.stage_name!r} exited with return code "
                f"{process.returncode}; see {stderr_relative}",
                details,
            )

        return StageResult(
            metrics={"return_code": process.returncode, "duration_s": duration},
        )


class DockerCommandAdapter(CommandAdapter):
    name = "docker_command"

    def healthcheck(self) -> HealthcheckResult:
        return HealthcheckResult(
            True,
            "Docker command adapter is configured; Docker is not invoked by the healthcheck",
        )
