from __future__ import annotations

import os
import signal
import subprocess


def terminate_process_group(
    process: subprocess.Popen[str],
    *,
    grace_period_s: float = 2.0,
) -> tuple[str, str]:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.communicate(timeout=grace_period_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.communicate()
