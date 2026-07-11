from __future__ import annotations

import shutil
import subprocess
from typing import Any

from recon2sim.adapters.base import HealthcheckResult, OutputSpec, StageContext, StageResult
from recon2sim.adapters.process import ExternalProcessError, run_external_process
from recon2sim.artifacts import CommandResultArtifact
from recon2sim.ir import SceneIR
from recon2sim.storage import atomic_write_json


class CommandExecutionError(RuntimeError):
    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


class CommandAdapter:
    name = "command"
    version = "0.1.0"

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return HealthcheckResult(True, "subprocess execution is available")

    def prepare(self, context: StageContext) -> None:
        context.attempt_dir.mkdir(parents=True, exist_ok=True)

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
        process_error: ExternalProcessError | None = None
        try:
            process_result = run_external_process(
                command,
                cwd=context.attempt_dir,
                timeout_s=context.config.adapter.timeout_s,
                environment_names=context.config.adapter.env,
                stdout_path=context.output_path(stdout_relative),
                stderr_path=context.output_path(stderr_relative),
                command_name=f"command for stage {context.stage_name!r}",
            )
        except ExternalProcessError as exc:
            process_error = exc
            process_result = exc.result
        command_result = CommandResultArtifact(
            stage=context.stage_name,
            attempt=context.attempt,
            command=command,
            return_code=process_result.return_code,
            duration_s=process_result.duration_s,
            timed_out=process_result.timed_out,
            interrupted=process_result.interrupted,
            stdout_path=context.workspace_relative(stdout_relative),
            stderr_path=context.workspace_relative(stderr_relative),
        )
        atomic_write_json(context.output_path(result_relative), command_result)
        details: dict[str, Any] = command_result.model_dump(mode="json")

        if process_error is not None:
            raise CommandExecutionError(str(process_error), details) from process_error

        return StageResult(
            metrics={
                "return_code": process_result.return_code,
                "duration_s": process_result.duration_s,
            },
        )


class DockerCommandAdapter(CommandAdapter):
    name = "docker_command"

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        executable = shutil.which("docker")
        if executable is None:
            return HealthcheckResult(False, "Docker CLI was not found on PATH")
        try:
            completed = subprocess.run(
                [executable, "version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return HealthcheckResult(False, f"Docker engine check failed: {exc}")
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            summary = detail[0] if detail else "docker version returned no diagnostic"
            return HealthcheckResult(False, f"Docker engine unavailable: {summary}")
        return HealthcheckResult(True, f"Docker engine available via {executable}")
