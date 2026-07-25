from __future__ import annotations

from pathlib import Path

DINOV3_REPOSITORY = "facebook/dinov3-vitl16-pretrain-lvd1689m"
TRELLIS_IMAGE_REPOSITORY = "microsoft/TRELLIS-image-large"
TRELLIS_2_REPOSITORY = "microsoft/TRELLIS.2-4B"

RUNTIME_REPOSITORY_FILES = {
    DINOV3_REPOSITORY: (
        ".gitattributes",
        "LICENSE.md",
        "README.md",
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
    ),
    TRELLIS_IMAGE_REPOSITORY: (
        "ckpts/ss_dec_conv3d_16l8_fp16.json",
        "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
    ),
    TRELLIS_2_REPOSITORY: (
        "ckpts/shape_dec_next_dc_f16c32_fp16.json",
        "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        "ckpts/tex_dec_next_dc_f16c32_fp16.json",
        "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
    ),
}


def gated_access_error(error: Exception) -> RuntimeError:
    if "awaiting a review" in str(error).lower():
        return RuntimeError(
            "official GenRecon runtime model access request is pending review for "
            f"{DINOV3_REPOSITORY}; wait for the repository authors to approve the "
            "authenticated Hugging Face account"
        )
    return RuntimeError(
        "official GenRecon requires accepted access to gated runtime model "
        f"{DINOV3_REPOSITORY}; accept its official Hugging Face terms and authenticate"
    )


def resolve_repository_revision(repository: str, probe_file: str) -> str:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    try:
        info = HfApi().model_info(repository)
    except GatedRepoError as exc:
        raise gated_access_error(exc) from exc
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"could not verify official GenRecon runtime repository {repository}: {exc}"
        ) from exc
    revision = info.sha
    if not isinstance(revision, str) or len(revision) != 40:
        raise RuntimeError(
            f"official GenRecon runtime repository {repository} returned an invalid "
            f"revision: {revision!r}"
        )
    try:
        hf_hub_download(
            repo_id=repository,
            filename=probe_file,
            revision=revision,
        )
    except GatedRepoError as exc:
        raise gated_access_error(exc) from exc
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"could not read official GenRecon runtime file "
            f"{repository}/{probe_file} at {revision}: {exc}"
        ) from exc
    return revision


def resolve_runtime_repository_revisions() -> dict[str, str]:
    return {
        repository: resolve_repository_revision(repository, files[0])
        for repository, files in RUNTIME_REPOSITORY_FILES.items()
    }


def prepare_repository_snapshot(
    repository: str,
    expected_revision: str,
    files: tuple[str, ...],
) -> str:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    try:
        snapshot = Path(
            snapshot_download(
                repo_id=repository,
                revision="main",
                allow_patterns=list(files),
            )
        ).resolve()
    except GatedRepoError as exc:
        raise gated_access_error(exc) from exc
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"could not cache official GenRecon runtime repository {repository}: {exc}"
        ) from exc
    revision = snapshot.name
    if revision != expected_revision:
        raise RuntimeError(
            f"official GenRecon runtime repository {repository} changed between access "
            f"verification and download: expected {expected_revision}, resolved {revision}"
        )
    return revision


def prepare_official_runtime_assets() -> dict[str, str]:
    revisions = resolve_runtime_repository_revisions()
    for repository, files in RUNTIME_REPOSITORY_FILES.items():
        prepare_repository_snapshot(repository, revisions[repository], files)
    return revisions


def resolve_dinov3_revision() -> str:
    return resolve_repository_revision(DINOV3_REPOSITORY, "config.json")


def prepare_dinov3_runtime_asset(expected_revision: str) -> str:
    return prepare_repository_snapshot(
        DINOV3_REPOSITORY,
        expected_revision,
        RUNTIME_REPOSITORY_FILES[DINOV3_REPOSITORY],
    )
