from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from recon2sim.adapters.ingest import detect_input_mode, deterministic_frame_name
from recon2sim.artifacts import FrameQualityReport, IngestManifest, InputSourceType
from recon2sim.config import AdapterConfig, PipelineConfig, StageConfig
from recon2sim.pipeline import PipelineRunner


def _write_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (24, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _ingest_config(**adapter_config: object) -> PipelineConfig:
    return PipelineConfig(
        stages={
            "ingest": StageConfig(
                adapter=AdapterConfig(name="ffmpeg_ingest", config=dict(adapter_config))
            )
        }
    )


def _write_executable(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(0o755)
    return path


def _fake_video_tools(tmp_path: Path, mode: str) -> tuple[Path, Path]:
    ffprobe = _write_executable(
        tmp_path / f"ffprobe_{mode}",
        """
import json
import sys
if '-version' in sys.argv:
    print('ffprobe fake-1.0')
else:
    print(json.dumps({'streams': [{'codec_type': 'video', 'width': 16, 'height': 12}],
                      'format': {'duration': '1.0'}}))
""",
    )
    ffmpeg = _write_executable(
        tmp_path / f"ffmpeg_{mode}",
        f"""
import sys
import time
from pathlib import Path
if '-version' in sys.argv:
    print('ffmpeg fake-1.0')
    raise SystemExit(0)
mode = {mode!r}
if mode == 'failure':
    print('simulated decode failure', file=sys.stderr)
    raise SystemExit(7)
if mode == 'timeout':
    time.sleep(5)
if mode == 'stale':
    raise SystemExit(0)
path = Path(sys.argv[-1].replace('%06d', '000000'))
path.parent.mkdir(parents=True, exist_ok=True)
if mode == 'malformed':
    path.write_bytes(b'not a png')
else:
    from PIL import Image
    Image.new('RGB', (16, 12), (100, 120, 140)).save(path)
""",
    )
    return ffprobe, ffmpeg


def test_input_mode_detection_and_deterministic_names(tmp_path: Path) -> None:
    images = tmp_path / "images"
    _write_image(images / "b.JPG", (20, 30, 40))
    _write_image(images / "a.png", (50, 60, 70))
    detection = detect_input_mode(images)
    assert detection.mode == "image_directory"
    assert [path.name for path in detection.sources] == ["a.png", "b.JPG"]
    assert deterministic_frame_name(0) == "frame_000000.png"
    assert deterministic_frame_name(42) == "frame_000042.png"

    video = tmp_path / "capture.mov"
    video.write_bytes(b"not decoded during detection")
    assert detect_input_mode(video).mode == "video"


def test_auto_detection_rejects_mixed_sources(tmp_path: Path) -> None:
    _write_image(tmp_path / "frame.png", (80, 80, 80))
    (tmp_path / "video.mp4").write_bytes(b"video")
    with pytest.raises(ValueError, match="both videos and images"):
        detect_input_mode(tmp_path)


def test_detection_rejects_unsupported_image_formats(tmp_path: Path) -> None:
    (tmp_path / "capture.webp").write_bytes(b"unsupported image bytes")
    with pytest.raises(ValueError, match="accepts JPEG and PNG only.*capture.webp"):
        detect_input_mode(tmp_path)


def test_image_directory_normalization_and_frame_qa(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    _write_image(input_dir / "z.jpg", (100, 110, 120), (40, 20))
    _write_image(input_dir / "a.png", (160, 170, 180), (20, 40))
    run_dir = tmp_path / "run"
    PipelineRunner(
        _ingest_config(
            input_mode="image_directory",
            resize_max_edge=10,
            duplicate_threshold=None,
        ),
        input_dir,
        run_dir,
    ).run()

    manifest = IngestManifest.model_validate_json(
        (run_dir / "inputs" / "manifest.json").read_text(encoding="utf-8")
    )
    report = FrameQualityReport.model_validate_json(
        (run_dir / "inputs" / "frame_qa.json").read_text(encoding="utf-8")
    )
    assert manifest.source_type is InputSourceType.IMAGE_DIRECTORY
    assert [frame.source_file_reference for frame in manifest.frames] == ["a.png", "z.jpg"]
    assert [frame.relative_path for frame in manifest.frames] == [
        "frames/frame_000000.png",
        "frames/frame_000001.png",
    ]
    assert max(manifest.frames[0].width, manifest.frames[0].height) == 10
    assert report.selected_count == 2
    assert report.dropped_count == 0
    assert all(entry.mean_brightness > 0 for entry in report.entries)
    assert all(entry.intensity_variance >= 0 for entry in report.entries)


def test_duplicate_frame_is_rejected_but_retained_for_diagnostics(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    _write_image(input_dir / "000.png", (100, 100, 100))
    _write_image(input_dir / "001.png", (100, 100, 100))
    _write_image(input_dir / "002.png", (180, 180, 180))
    run_dir = tmp_path / "run"
    PipelineRunner(
        _ingest_config(input_mode="image_directory", duplicate_threshold=0.999),
        input_dir,
        run_dir,
    ).run()

    manifest = IngestManifest.model_validate_json(
        (run_dir / "inputs" / "manifest.json").read_text(encoding="utf-8")
    )
    report = FrameQualityReport.model_validate_json(
        (run_dir / "inputs" / "frame_qa.json").read_text(encoding="utf-8")
    )
    assert [frame.frame_id for frame in manifest.frames] == [
        "frame_000000",
        "frame_000002",
    ]
    rejected = report.entries[1]
    assert rejected.is_duplicate is True
    assert rejected.rejection_reason == "near_duplicate"
    assert rejected.rejected_path == "diagnostics/rejected_frames/frame_000001.png"
    assert (run_dir / rejected.rejected_path).is_file()


def test_all_rejected_frames_leave_qa_in_failed_attempt(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    _write_image(input_dir / "black.png", (0, 0, 0))
    run_dir = tmp_path / "run"
    with pytest.raises(RuntimeError, match="frame QA rejected every"):
        PipelineRunner(_ingest_config(input_mode="image_directory"), input_dir, run_dir).run()
    qa_path = run_dir / "work" / "ingest" / "attempt_1" / "inputs" / "frame_qa.json"
    report = FrameQualityReport.model_validate_json(qa_path.read_text(encoding="utf-8"))
    assert report.selected_count == 0
    assert report.entries[0].rejection_reason == "brightness_below_minimum"


def test_video_ingest_uses_ffprobe_and_ffmpeg_subprocesses(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    video = input_dir / "capture.mp4"
    video.write_bytes(b"deterministic fake video bytes")
    ffprobe = _write_executable(
        tmp_path / "fake_ffprobe",
        """
import json
import sys
if '-version' in sys.argv:
    print('ffprobe fake-1.0')
else:
    print(json.dumps({'streams': [{'codec_type': 'video', 'width': 32, 'height': 24}],
                      'format': {'duration': '1.0'}}))
""",
    )
    ffmpeg = _write_executable(
        tmp_path / "fake_ffmpeg",
        """
import sys
from pathlib import Path
from PIL import Image
if '-version' in sys.argv:
    print('ffmpeg fake-1.0')
    raise SystemExit(0)
pattern = sys.argv[-1]
for index, color in enumerate(((80, 90, 100), (130, 140, 150), (190, 180, 170))):
    path = Path(pattern.replace('%06d', f'{index:06d}'))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (32, 24), color).save(path)
""",
    )
    run_dir = tmp_path / "run"
    PipelineRunner(
        _ingest_config(
            input_mode="video",
            ffmpeg_executable=str(ffmpeg),
            ffprobe_executable=str(ffprobe),
            target_fps=3.0,
            max_frames=3,
            resize_max_edge=None,
            duplicate_threshold=None,
        ),
        input_dir,
        run_dir,
    ).run()

    manifest = IngestManifest.model_validate_json(
        (run_dir / "inputs" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.source_type is InputSourceType.VIDEO
    assert manifest.source_input_reference == "capture.mp4"
    assert manifest.source_sha256 is not None and len(manifest.source_sha256) == 64
    assert manifest.ffmpeg_version == "ffmpeg fake-1.0"
    assert manifest.ffprobe_version == "ffprobe fake-1.0"
    assert manifest.total_decoded_frames == 3
    assert [frame.original_frame_index for frame in manifest.frames] == [0, 1, 2]
    assert [frame.timestamp_s for frame in manifest.frames] == [0.0, 1 / 3, 2 / 3]
    assert "ffmpeg_command" in manifest.extraction_configuration
    assert (run_dir / "inputs" / "logs" / "ffmpeg_extract.stderr.log").is_file()
    assert (
        json.loads((run_dir / "manifest.json").read_text())["stages"]["ingest"]["status"]
        == "succeeded"
    )


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("failure", "FFmpeg frame extraction exited with return code 7"),
        ("timeout", "FFmpeg frame extraction timed out"),
        ("stale", "extracted no frames"),
        ("malformed", "could not decode extracted frame"),
    ],
)
def test_fake_ffmpeg_failure_modes_preserve_attempt(
    tmp_path: Path, mode: str, message: str
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "capture.mp4").write_bytes(b"fake video")
    ffprobe, ffmpeg = _fake_video_tools(tmp_path, mode)
    config = _ingest_config(
        input_mode="video",
        ffmpeg_executable=str(ffmpeg),
        ffprobe_executable=str(ffprobe),
        duplicate_threshold=None,
    )
    if mode == "timeout":
        config.stages["ingest"].adapter.timeout_s = 0.1
    run_dir = tmp_path / "run"
    with pytest.raises((RuntimeError, ValueError), match=message):
        PipelineRunner(config, input_dir, run_dir).run()
    attempt = run_dir / "work" / "ingest" / "attempt_1"
    assert attempt.is_dir()
    assert (attempt / "inputs" / "logs" / "ffmpeg_extract.stderr.log").is_file()
    assert not (run_dir / "inputs" / "manifest.json").exists()
