from __future__ import annotations

import subprocess

from recon2sim.adapters.base import ArtifactRecord, HealthcheckResult, StageContext, StageResult


class CommandAdapter:
    name = "command"

    def healthcheck(self) -> HealthcheckResult:
        return HealthcheckResult(True, "subprocess available")

    def prepare(self, context: StageContext) -> None:
        context.run_dir.mkdir(parents=True, exist_ok=True)

    def run(self, context: StageContext) -> StageResult:
        cmd = context.config.adapter.command
        if not cmd:
            raise ValueError("command adapter requires command list")
        completed = subprocess.run(
            cmd,
            cwd=context.run_dir,
            text=True,
            capture_output=True,
            timeout=context.config.adapter.timeout_s,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"command failed: {completed.stderr.strip()}")
        return StageResult(
            metrics={"returncode": completed.returncode, "stdout": completed.stdout[:500]}
        )

    def collect(self, context: StageContext) -> list[ArtifactRecord]:
        return []


class DockerCommandAdapter(CommandAdapter):
    name = "docker_command"

    def healthcheck(self) -> HealthcheckResult:
        return HealthcheckResult(
            True, "docker command adapter configured; docker not invoked during healthcheck"
        )
