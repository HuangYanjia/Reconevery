from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from recon2sim.adapters.ingest import detect_input, deterministic_frame_name
from recon2sim.artifacts import FrameQualityReport, IngestManifest, InputSourceType
from recon2sim.config import AdapterConfig, PipelineConfig, StageConfig
from recon2sim.pipeline import PipelineRunner


def _ingest_config(**values: object) -> PipelineConfig:
    return PipelineConfig(
        stages={
            "ingest": StageConfig(
                adapter=AdapterConfig(
                    name="ffmpeg_ingest",
                    config={
                        "input_mode": "image_directory",
                        "target_fps": 2.0,
                        "max_frames": 20,
                        "resize_max_edge": 32,
                        "min_brightness": 0,
                        "max_brightness": 255,
                        "duplicate_threshold": 0,
                        **values,
                    },
                )
            )
        }
    )


def _image(path: Path, color: tuple[int, int, int], *, size: tuple[int, int] = (64, 40)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _fake_ffmpeg(path: Path, mode: str = "success") -> Path:
    source = f"""#!{sys.executable}
import json
import sys
import time
from pathlib import Path
from PIL import Image

MODE = {mode!r}
args = sys.argv[1:]
if args == ["-version"]:
    print("ffmpeg version 7.1 fake")
    raise SystemExit(0)
if "-show_entries" in args:
    print(json.dumps({{
        "streams": [{{"width": 64, "height": 40, "nb_frames": "9", "avg_frame_rate": "30/1"}}],
        "format": {{"duration": "3.0"}}
    }}))
    raise SystemExit(0)
if MODE == "timeout":
    time.sleep(5)
if MODE == "nonzero":
    print("simulated decode error", file=sys.stderr)
    raise SystemExit(11)
pattern = args[-1]
for index, color in enumerate(((20, 80, 140), (80, 140, 20), (140, 20, 80))):
    destination = Path(pattern.replace("%06d", f"{{index:06d}}"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 20), color).save(destination)
"""
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_input_mode_detection_and_deterministic_frame_names(tmp_path: Path) -> None:
    video_root = tmp_path / "video"
    video_root.mkdir()
    video = video_root / "capture.MOV"
    video.write_bytes(b"fixture")
    detected_video = detect_input(video_root)
    assert detected_video.source_type is InputSourceType.VIDEO
    assert detected_video.video == video

    image_root = tmp_path / "photos"
    _image(image_root / "images" / "b.png", (20, 30, 40))
    _image(image_root / "images" / "a.jpg", (50, 60, 70))
    detected_images = detect_input(image_root)
    assert detected_images.source_type is InputSourceType.IMAGE_DIRECTORY
    assert [path.name for path in detected_images.images] == ["a.jpg", "b.png"]
    assert deterministic_frame_name(0) == "frame_000000.png"
    assert deterministic_frame_name(123) == "frame_000123.png"
    with pytest.raises(ValueError, match="non-negative"):
        deterministic_frame_name(-1)


def test_auto_mode_rejects_ambiguous_inputs(tmp_path: Path) -> None:
    _image(tmp_path / "photo.png", (10, 20, 30))
    (tmp_path / "video.mp4").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="both videos and images"):
        detect_input(tmp_path)


def test_image_ingest_normalizes_resizes_and_rejects_duplicates(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    _image(input_dir / "images" / "001.jpg", (50, 100, 150), size=(80, 40))
    _image(input_dir / "images" / "002.png", (50, 100, 150), size=(80, 40))
    _image(input_dir / "images" / "003.png", (150, 100, 50), size=(80, 40))
    run_dir = tmp_path / "run"

    manifest_data = PipelineRunner(_ingest_config(), input_dir, run_dir).run()
    manifest = IngestManifest.model_validate_json(
        (run_dir / "inputs" / "manifest.json").read_text(encoding="utf-8")
    )
    report = FrameQualityReport.model_validate_json(
        (run_dir / "inputs" / "frame_qa.json").read_text(encoding="utf-8")
    )

    assert manifest.source_type is InputSourceType.IMAGE_DIRECTORY
    assert manifest.selected_frames == 2
    assert manifest.dropped_frames == 1
    assert [frame.frame_id for frame in manifest.frames] == ["frame_000000", "frame_000002"]
    assert [Image.open(run_dir / frame.relative_path).size for frame in manifest.frames] == [
        (32, 16),
        (32, 16),
    ]
    duplicate = report.entries[1]
    assert 85 < report.entries[0].mean_brightness < 95
    assert report.entries[0].blur_score == 0
    assert duplicate.near_duplicate is True
    assert duplicate.rejection_reason == "near_duplicate"
    assert (run_dir / duplicate.relative_path).is_file()
    assert manifest_data["stages"]["ingest"]["metrics"]["selected_frames"] == 2

    command_payload = json.loads((run_dir / "inputs" / "commands.json").read_text(encoding="utf-8"))
    assert command_payload == {"commands": []}


@pytest.mark.parametrize(
    ("orientation", "top_color"),
    [(6, "red"), (8, "blue")],
)
def test_image_ingest_applies_exif_orientation(
    tmp_path: Path,
    orientation: int,
    top_color: str,
) -> None:
    input_dir = tmp_path / "input"
    source = input_dir / "images" / "oriented.jpg"
    source.parent.mkdir(parents=True)
    image = Image.new("RGB", (40, 20), "red")
    for x in range(20, 40):
        for y in range(20):
            image.putpixel((x, y), (0, 0, 255))
    exif = Image.Exif()
    exif[274] = orientation
    image.save(source, exif=exif, quality=100, subsampling=0)

    run_dir = tmp_path / "run"
    PipelineRunner(
        _ingest_config(resize_max_edge=None),
        input_dir,
        run_dir,
    ).run()
    normalized = Image.open(run_dir / "frames" / "frame_000000.png")

    assert normalized.size == (20, 40)
    top = normalized.getpixel((10, 5))
    if top_color == "red":
        assert top[0] > top[2]
    else:
        assert top[2] > top[0]


def test_image_ingest_rejects_unreadable_image_actionably(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "broken.jpg").write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable or unsupported image.*broken.jpg"):
        PipelineRunner(_ingest_config(), input_dir, tmp_path / "run").run()


def test_fake_ffmpeg_video_ingest_records_metadata_commands_and_logs(tmp_path: Path) -> None:
    video = tmp_path / "capture.mp4"
    video.write_bytes(b"fake video bytes")
    fake = _fake_ffmpeg(tmp_path / "fake_ffmpeg")
    config = PipelineConfig(
        stages={
            "ingest": StageConfig(
                adapter=AdapterConfig(
                    name="ffmpeg_ingest",
                    env=["PATH"],
                    config={
                        "input_mode": "video",
                        "executable": str(fake),
                        "ffprobe_executable": str(fake),
                        "target_fps": 3,
                        "max_frames": 3,
                        "resize_max_edge": 32,
                        "min_brightness": 0,
                        "max_brightness": 255,
                    },
                )
            )
        }
    )
    run_dir = tmp_path / "run"
    PipelineRunner(config, video, run_dir).run()
    manifest = IngestManifest.model_validate_json(
        (run_dir / "inputs" / "manifest.json").read_text(encoding="utf-8")
    )
    commands = json.loads((run_dir / "inputs" / "commands.json").read_text(encoding="utf-8"))

    assert manifest.source_type is InputSourceType.VIDEO
    assert manifest.total_decoded_frames == 9
    assert manifest.selected_frames == 3
    assert manifest.ffmpeg_version == "ffmpeg version 7.1 fake"
    assert manifest.ffprobe_version == "ffmpeg version 7.1 fake"
    assert [frame.relative_path for frame in manifest.frames] == [
        "frames/frame_000000.png",
        "frames/frame_000001.png",
        "frames/frame_000002.png",
    ]
    assert [frame.original_frame_index for frame in manifest.frames] == [0, 10, 20]
    assert [record["name"] for record in commands["commands"]] == ["ffprobe", "ffmpeg"]
    assert (run_dir / "logs" / "ingest.ffmpeg.attempt_1.stderr.log").is_file()


def test_video_ingest_supports_relative_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "capture.mp4"
    video.write_bytes(b"fake video bytes")
    fake = _fake_ffmpeg(tmp_path / "fake_ffmpeg")
    config = PipelineConfig(
        stages={
            "ingest": StageConfig(
                adapter=AdapterConfig(
                    name="ffmpeg_ingest",
                    env=["PATH"],
                    config={
                        "input_mode": "video",
                        "executable": str(fake),
                        "ffprobe_executable": str(fake),
                        "min_brightness": 0,
                        "max_brightness": 255,
                    },
                )
            )
        }
    )
    monkeypatch.chdir(tmp_path)

    PipelineRunner(config, Path("capture.mp4"), Path("relative_run")).run()

    assert (tmp_path / "relative_run" / "frames" / "frame_000000.png").is_file()


@pytest.mark.parametrize(
    ("mode", "match"),
    [("nonzero", "return code 11"), ("timeout", "timed out")],
)
def test_fake_ffmpeg_failures_preserve_logs(tmp_path: Path, mode: str, match: str) -> None:
    video = tmp_path / "capture.mp4"
    video.write_bytes(b"fake video bytes")
    fake = _fake_ffmpeg(tmp_path / "fake_ffmpeg", mode)
    config = PipelineConfig(
        stages={
            "ingest": StageConfig(
                adapter=AdapterConfig(
                    name="ffmpeg_ingest",
                    timeout_s=0.1 if mode == "timeout" else 5,
                    env=["PATH"],
                    config={
                        "input_mode": "video",
                        "executable": str(fake),
                        "ffprobe_executable": str(fake),
                    },
                )
            )
        }
    )
    run_dir = tmp_path / "run"
    with pytest.raises(RuntimeError, match=match):
        PipelineRunner(config, video, run_dir).run()
    assert (run_dir / "logs" / "ingest.ffmpeg.attempt_1.stderr.log").is_file()
