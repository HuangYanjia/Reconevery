from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

WORKER_ROOT = Path(__file__).resolve().parents[1] / "workers" / "sam3"
sys.path.insert(0, str(WORKER_ROOT))

from sam3_worker import commit_verification  # noqa: E402
from sam3_worker.frame_preparation import (  # noqa: E402
    prepared_video_frames,
    resolve_worker_output_directory,
)
from sam3_worker.inference import _collect_outputs  # noqa: E402
from sam3_worker.official_compat import (  # noqa: E402
    apply_sam31_start_session_compatibility,
    official_propagation_directions,
)

EXPECTED_COMMIT = "46957e47805eaa273f4aa7bbbd25a88bca9108ce"


def _git_checkout(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / "README.md").write_text("official fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def test_editable_git_checkout_commit_is_verified(tmp_path: Path) -> None:
    checkout = tmp_path / "sam3"
    commit = _git_checkout(checkout)
    direct_url = json.dumps(
        {
            "url": checkout.as_uri(),
            "dir_info": {"editable": True},
        }
    )
    assert commit_verification.commit_from_direct_url(direct_url) == commit


def test_pep610_exact_vcs_commit_is_verified() -> None:
    direct_url = json.dumps(
        {
            "url": "https://github.com/facebookresearch/sam3.git",
            "vcs_info": {
                "vcs": "git",
                "requested_revision": EXPECTED_COMMIT,
                "commit_id": EXPECTED_COMMIT,
            },
        }
    )
    assert commit_verification.commit_from_direct_url(direct_url) == EXPECTED_COMMIT


def test_wrong_official_commit_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    wrong = "0" * 40
    monkeypatch.setattr(commit_verification, "installed_sam_commit", lambda: wrong)
    with pytest.raises(RuntimeError, match=f"is at {wrong}, expected {EXPECTED_COMMIT}"):
        commit_verification.require_official_commit(EXPECTED_COMMIT)


def test_unverifiable_local_install_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    direct_url = json.dumps(
        {
            "url": "file:///tmp/sam3",
            "dir_info": {},
        }
    )
    assert commit_verification.commit_from_direct_url(direct_url) is None
    monkeypatch.setattr(commit_verification, "installed_sam_commit", lambda: None)
    with pytest.raises(RuntimeError, match="could not be verified"):
        commit_verification.require_official_commit(EXPECTED_COMMIT)


def test_prepared_png_frames_preserve_order_dimensions_and_are_ephemeral(
    tmp_path: Path,
) -> None:
    frames = tmp_path / "frames"
    raw_output = tmp_path / "observations" / "raw"
    frames.mkdir()
    raw_output.mkdir(parents=True)
    Image.new("RGB", (7, 5), (20, 30, 40)).save(frames / "second.png")
    Image.new("RGB", (6, 4), (80, 90, 100)).save(frames / "first.png")

    with prepared_video_frames(
        tmp_path,
        ["frame_b", "frame_a"],
        ["frames/second.png", "frames/first.png"],
        {"frame_b": (7, 5), "frame_a": (6, 4)},
    ) as prepared:
        assert prepared.parent == tmp_path
        assert [path.name for path in sorted(prepared.iterdir())] == [
            "000000.png",
            "000001.png",
        ]
        assert (prepared / "000000.png").read_bytes() == (frames / "second.png").read_bytes()
        assert (prepared / "000001.png").read_bytes() == (frames / "first.png").read_bytes()
        with Image.open(prepared / "000000.png") as image:
            assert image.size == (7, 5)
        prepared_path = prepared

    assert not prepared_path.exists()
    assert not (raw_output / "video_frames").exists()


def test_worker_output_directory_is_anchored_and_cannot_escape(tmp_path: Path) -> None:
    assert resolve_worker_output_directory(tmp_path, Path("observations/raw")) == (
        tmp_path / "observations" / "raw"
    )
    with pytest.raises(RuntimeError, match="escapes the attempt workspace"):
        resolve_worker_output_directory(tmp_path, Path("../outside"))


def test_sam31_start_session_compatibility_filters_unsupported_false_flag() -> None:
    calls: list[dict[str, object]] = []

    class Model:
        def init_state(
            self,
            resource_path: str,
            offload_video_to_cpu: bool = False,
        ) -> dict[str, object]:
            calls.append(
                {
                    "resource_path": resource_path,
                    "offload_video_to_cpu": offload_video_to_cpu,
                }
            )
            return calls[-1]

    class Predictor:
        model = Model()

    predictor = Predictor()
    assert apply_sam31_start_session_compatibility(predictor)
    result = predictor.model.init_state(
        resource_path="/frames",
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    )
    assert result == {
        "resource_path": "/frames",
        "offload_video_to_cpu": False,
    }


def test_sam31_start_session_compatibility_rejects_state_offload() -> None:
    class Model:
        def init_state(
            self,
            resource_path: str,
            offload_video_to_cpu: bool = False,
        ) -> None:
            del resource_path, offload_video_to_cpu

    class Predictor:
        model = Model()

    predictor = Predictor()
    apply_sam31_start_session_compatibility(predictor)
    with pytest.raises(RuntimeError, match="does not support offload_state_to_cpu"):
        predictor.model.init_state(
            resource_path="/frames",
            offload_state_to_cpu=True,
        )


def test_forward_backward_uses_independent_official_propagations() -> None:
    assert official_propagation_directions("forward_backward") == (
        "forward",
        "backward",
    )
    assert official_propagation_directions("forward") == ("forward",)


def test_negative_official_absence_score_is_not_materialized(tmp_path: Path) -> None:
    tracks: defaultdict[str, dict[str, object]] = defaultdict(dict)
    _collect_outputs(
        tmp_path,
        tmp_path / "observations" / "raw",
        SimpleNamespace(
            frame_order=["frame_000000"],
            frame_dimensions={"frame_000000": (2, 2)},
        ),
        SimpleNamespace(prompt_id="drawer", label="drawer"),
        0,
        {
            "out_obj_ids": [0],
            "out_binary_masks": [[[True, True], [True, True]]],
            "out_probs": [-10000.0],
            "out_boxes_xywh": [[0.0, 0.0, 1.0, 1.0]],
        },
        tracks,
    )

    assert not tracks
    assert not (tmp_path / "observations" / "raw" / "masks").exists()


def test_official_confidence_above_one_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="above one"):
        _collect_outputs(
            tmp_path,
            tmp_path / "observations" / "raw",
            SimpleNamespace(
                frame_order=["frame_000000"],
                frame_dimensions={"frame_000000": (2, 2)},
            ),
            SimpleNamespace(prompt_id="drawer", label="drawer"),
            0,
            {
                "out_obj_ids": [0],
                "out_binary_masks": [[[True, True], [True, True]]],
                "out_probs": [1.5],
                "out_boxes_xywh": [[0.0, 0.0, 1.0, 1.0]],
            },
            defaultdict(dict),
        )
