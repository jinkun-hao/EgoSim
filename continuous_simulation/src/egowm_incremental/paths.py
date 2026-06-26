from __future__ import annotations

import os
import shutil
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent
EGOSIM_ROOT = PROJECT_ROOT.parent
# Backward-compatible aliases used by older path helpers.
EGOSIM_OPENSOURCE_ROOT = EGOSIM_ROOT
WANVIDEO_ROOT = EGOSIM_ROOT

_STANDARD_EGOWM_RUNNER = Path("egowm") / "inference" / "runner.py"


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_egosim_root() -> Path:
    return EGOSIM_ROOT


def get_egosim_opensource_root() -> Path:
    return EGOSIM_ROOT


def get_wanvideo_root() -> Path:
    return WANVIDEO_ROOT


def resolve_egosim_root(explicit: str | Path | None = None) -> Path | None:
    """Resolve the EgoSim repository root used for standard EgoSim imports."""
    if explicit not in (None, ""):
        root = Path(explicit).expanduser().resolve()
    else:
        env_raw = os.environ.get("EGOWM_ROOT", "").strip()
        root = (
            Path(env_raw).expanduser().resolve()
            if env_raw
            else EGOSIM_ROOT.resolve()
        )

    if (root / _STANDARD_EGOWM_RUNNER).exists():
        return root
    return None


resolve_egosim_opensource_root = resolve_egosim_root


def expand_path(raw_path: str | None, *, base_dir: Path) -> Path | None:
    if raw_path in (None, ""):
        return None
    expanded = Path(os.path.expandvars(os.path.expanduser(raw_path)))
    if not expanded.is_absolute():
        expanded = (base_dir / expanded).resolve()
    return expanded


def ensure_path(path: Path | None, *, kind: str) -> Path:
    if path is None:
        raise ValueError(f"Missing required {kind} path.")
    return path


def infer_conda_base() -> Path | None:
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        return Path(conda_exe).resolve().parent.parent

    conda_path = shutil.which("conda")
    if conda_path:
        return Path(conda_path).resolve().parent.parent

    mamba_exe = os.environ.get("MAMBA_EXE")
    if mamba_exe:
        return Path(mamba_exe).resolve().parent.parent

    return None


def resolve_conda_env_python(env_name: str) -> Path:
    conda_base = infer_conda_base()
    if conda_base is None:
        raise FileNotFoundError(
            f"Cannot resolve python for conda env '{env_name}' because conda is not available."
        )

    if env_name == "base":
        candidate = conda_base / "bin" / "python"
    else:
        candidate = conda_base / "envs" / env_name / "bin" / "python"

    if not candidate.exists():
        raise FileNotFoundError(
            f"Resolved python for env '{env_name}' does not exist: {candidate}"
        )
    return candidate
