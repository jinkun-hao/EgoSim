import os
from pathlib import Path


EGOSIM_STATE_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EGOSIM_STATE_REPO_ROOT = EGOSIM_STATE_PACKAGE_ROOT.parent


def package_path(*parts: str) -> Path:
    return EGOSIM_STATE_PACKAGE_ROOT.joinpath(*parts)


def repo_path(*parts: str) -> Path:
    return EGOSIM_STATE_REPO_ROOT.joinpath(*parts)


def artifacts_models_root() -> Path:
    """Default weight root: EgoSim/artifacts/models/."""
    return EGOSIM_STATE_REPO_ROOT.parents[2] / "artifacts" / "models"


def optional_env_path(env_var: str, default: str | Path | None = None) -> Path | None:
    raw = os.environ.get(env_var)
    if raw:
        return Path(raw).expanduser().resolve()
    if default is None:
        return None
    return Path(default).expanduser().resolve()


def required_model_path(env_var: str, default: Path) -> Path:
    resolved = optional_env_path(env_var, default=default)
    if resolved is None:
        raise FileNotFoundError(f"{env_var} is not set and no default path is configured.")
    if not resolved.exists():
        raise FileNotFoundError(
            f"Model weights not found at {resolved}. "
            f"Download the checkpoint and place it at this path, "
            f"or set {env_var} to your local copy."
        )
    return resolved


def get_sam3_checkpoint_path() -> Path:
    return required_model_path(
        "EGOSIM_STATE_SAM3_CHECKPOINT",
        artifacts_models_root() / "sam3" / "sam3.pt",
    )


def get_aot_checkpoint_path() -> Path:
    resolved = optional_env_path(
        "EGOSIM_STATE_AOT_CHECKPOINT",
        default=artifacts_models_root() / "aot" / "R50_DeAOTL_PRE_YTB_DAV.pth",
    )
    if resolved is None:
        raise FileNotFoundError("EGOSIM_STATE_AOT_CHECKPOINT is not set and no default path is configured.")
    return resolved


def get_droid_checkpoint_path() -> Path:
    return optional_env_path(
        "EGOSIM_STATE_DROID_CHECKPOINT",
        package_path("priors", "track_anything", "dorid_slam", "droid.pth"),
    )


def get_grounding_dino_checkpoint_path() -> Path:
    return required_model_path(
        "EGOSIM_STATE_GROUNDING_DINO_CHECKPOINT",
        artifacts_models_root() / "groundingdino" / "groundingdino_swint_ogc.pth",
    )


def get_grounding_dino_tokenizer_path() -> Path:
    return required_model_path(
        "EGOSIM_STATE_GROUNDING_DINO_TOKENIZER",
        artifacts_models_root() / "bert-base-uncased",
    )


def get_bert_base_uncased_path() -> Path:
    return required_model_path(
        "EGOSIM_STATE_BERT_BASE_UNCASED",
        artifacts_models_root() / "bert-base-uncased",
    )


def get_roberta_base_path() -> Path:
    return required_model_path(
        "EGOSIM_STATE_ROBERTA_BASE",
        artifacts_models_root() / "roberta-base",
    )


def get_dav3_metric_model_path() -> Path:
    return required_model_path(
        "EGOSIM_STATE_DA3_METRIC_MODEL",
        artifacts_models_root() / "depth-anything-3" / "DA3METRIC-LARGE",
    )


def get_dav3_nested_model_path() -> Path:
    return required_model_path(
        "EGOSIM_STATE_DA3_NESTED_MODEL",
        artifacts_models_root() / "depth-anything-3" / "DA3NESTED-GIANT-LARGE-1.1",
    )


def get_moge_model_path() -> Path:
    return required_model_path(
        "EGOSIM_STATE_MOGE_MODEL",
        artifacts_models_root() / "moge-2-vitl-normal",
    )


def get_unidepth_model_path() -> Path:
    return required_model_path(
        "EGOSIM_STATE_UNIDEPTH_MODEL",
        artifacts_models_root() / "unidepth-v2-vitl14",
    )


def get_video_depth_anything_checkpoint(model: str) -> Path:
    env_var = (
        "EGOSIM_STATE_VIDEO_DEPTH_ANYTHING_VITS"
        if model == "vits"
        else "EGOSIM_STATE_VIDEO_DEPTH_ANYTHING_VITL"
    )
    filename = (
        "video_depth_anything_vits.pth"
        if model == "vits"
        else "video_depth_anything_vitl.pth"
    )
    return required_model_path(
        env_var,
        artifacts_models_root() / "video-depth-anything" / filename,
    )
