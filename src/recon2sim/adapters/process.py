from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from recon2sim.storage import atomic_write_text


@dataclass(frozen=True)
class ProcessResult:
    arguments: list[str]
    return_code: int
    duration_s: float
    timed_out: bool
    interrupted: bool
    stdout_path: Path
    stderr_path: Path
    stdout: str
    stderr: str


class ExternalProcessError(RuntimeError):
    def __init__(self, message: str, result: ProcessResult) -> None:
        super().__init__(message)
        self.result = result
        self.details = {
            "command": result.arguments,
            "return_code": result.return_code,
            "duration_s": result.duration_s,
            "timed_out": result.timed_out,
            "interrupted": result.interrupted,
            "stdout_path": str(result.stdout_path),
            "stderr_path": str(result.stderr_path),
        }


def allowed_environment(names: list[str]) -> dict[str, str]:
    """Return only explicitly permitted variables from the parent environment."""
    return {name: os.environ[name] for name in names if name in os.environ}


def run_external_process(
    arguments: list[str],
    *,
    cwd: Path,
    timeout_s: float,
    environment_names: list[str],
    stdout_path: Path,
    stderr_path: Path,
    command_name: str,
) -> ProcessResult:
    if not arguments or any(not argument for argument in arguments):
        raise ValueError(f"{command_name} requires a non-empty argument list")
    cwd.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    try:
        process = subprocess.Popen(
            arguments,
            cwd=cwd,
            env=allowed_environment(environment_names),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        atomic_write_text(stdout_path, "")
        atomic_write_text(stderr_path, str(exc))
        raise RuntimeError(
            f"could not start {command_name}: {exc}; command was {arguments!r}"
        ) from exc

    timed_out = False
    interrupted = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process_group(process, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            _terminate_process_group(process, signal.SIGKILL)
            stdout, stderr = process.communicate()
    except KeyboardInterrupt:
        interrupted = True
        _terminate_process_group(process, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            _terminate_process_group(process, signal.SIGKILL)
            stdout, stderr = process.communicate()

    duration = time.monotonic() - start
    atomic_write_text(stdout_path, stdout)
    atomic_write_text(stderr_path, stderr)
    result = ProcessResult(
        arguments=list(arguments),
        return_code=process.returncode,
        duration_s=duration,
        timed_out=timed_out,
        interrupted=interrupted,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stdout=stdout,
        stderr=stderr,
    )
    if timed_out:
        raise ExternalProcessError(
            f"{command_name} timed out after {timeout_s:g} seconds; see {stderr_path}",
            result,
        )
    if interrupted:
        raise ExternalProcessError(
            f"{command_name} was interrupted by the user; partial output is preserved in {cwd}",
            result,
        )
    if process.returncode != 0:
        raise ExternalProcessError(
            f"{command_name} exited with return code {process.returncode}; see {stderr_path}",
            result,
        )
    return result


def _terminate_process_group(process: subprocess.Popen[str], signal_number: signal.Signals) -> None:
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        pass
