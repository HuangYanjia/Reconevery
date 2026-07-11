from __future__ import annotations

import shutil
import subprocess

import pytest


@pytest.mark.integration
@pytest.mark.requires_ffmpeg
@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and FFprobe are not installed",
)
def test_installed_ffmpeg_tools_report_versions() -> None:
    for executable in ("ffmpeg", "ffprobe"):
        completed = subprocess.run(
            [executable, "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert completed.returncode == 0
        assert executable in (completed.stdout or completed.stderr).lower()


@pytest.mark.integration
@pytest.mark.requires_colmap
@pytest.mark.skipif(shutil.which("colmap") is None, reason="COLMAP is not installed")
def test_installed_colmap_reports_help() -> None:
    completed = subprocess.run(
        ["colmap", "-h"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0
    assert "colmap" in (completed.stdout or completed.stderr).lower()
