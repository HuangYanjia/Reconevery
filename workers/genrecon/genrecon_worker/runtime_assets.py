from __future__ import annotations

from pathlib import Path

DINOV3_REPOSITORY = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def resolve_dinov3_revision() -> str:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    try:
        info = HfApi().model_info(DINOV3_REPOSITORY)
    except GatedRepoError as exc:
        raise RuntimeError(
            "official GenRecon requires accepted access to gated runtime model "
            f"{DINOV3_REPOSITORY}; accept its official Hugging Face terms and authenticate"
        ) from exc
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"could not verify official GenRecon runtime model {DINOV3_REPOSITORY}: {exc}"
        ) from exc
    revision = info.sha
    if not isinstance(revision, str) or len(revision) != 40:
        raise RuntimeError(
            f"official GenRecon runtime model returned an invalid revision: {revision!r}"
        )
    try:
        hf_hub_download(
            repo_id=DINOV3_REPOSITORY,
            filename="config.json",
            revision=revision,
        )
    except GatedRepoError as exc:
        raise RuntimeError(
            "official GenRecon runtime model metadata is public, but gated files are not "
            f"authorized for {DINOV3_REPOSITORY}; accept its terms and authenticate"
        ) from exc
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"could not read official GenRecon runtime model file at {revision}: {exc}"
        ) from exc
    return revision


def prepare_dinov3_runtime_asset(expected_revision: str) -> str:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    try:
        snapshot = Path(
            snapshot_download(
                repo_id=DINOV3_REPOSITORY,
                revision=expected_revision,
            )
        ).resolve()
    except GatedRepoError as exc:
        raise RuntimeError(
            "official GenRecon requires accepted access to gated runtime model "
            f"{DINOV3_REPOSITORY}; accept its official Hugging Face terms and authenticate"
        ) from exc
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"could not cache official GenRecon runtime model {DINOV3_REPOSITORY}: {exc}"
        ) from exc
    revision = snapshot.name
    if revision != expected_revision:
        raise RuntimeError(
            "official GenRecon DINOv3 revision changed between access verification and "
            f"download: expected {expected_revision}, resolved {revision}"
        )
    return revision
