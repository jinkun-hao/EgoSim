from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from .config import ProjectConfig
from .metadata import validate_metadata_csv
from .paths import PROJECT_ROOT, SRC_ROOT, WANVIDEO_ROOT, resolve_egosim_opensource_root
from .phrases import has_custom_scene_phrases
from .scene_runner import ModeSpec, SUPPORTED_MODES, requires_scene
from .wan_runner import build_quicktest_command


@dataclass
class QuicktestRun:
    mode: str
    output_dir: Path
    command: list[str]
    env: dict[str, str]


def _mode_config_path() -> Path:
    return PROJECT_ROOT / "configs" / "modes" / "quicktest.yaml"


def load_mode_specs() -> dict[str, ModeSpec]:
    with open(_mode_config_path(), "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    result: dict[str, ModeSpec] = {}
    for name, spec in (data.get("modes", {}) or {}).items():
        result[name] = ModeSpec(
            name=name,
            output_template=str(spec["output_template"]),
            enable_egosim_state=bool(spec["enable_egosim_state"]),
            enable_phrase_prediction=bool(spec["enable_phrase_prediction"]),
            recon_visualize=bool(spec.get("recon_visualize", False)),
            spatial_subsample_override=(
                int(spec["spatial_subsample_override"])
                if spec.get("spatial_subsample_override") is not None
                else None
            ),
            tsdf_voxel_size_override=(
                float(spec["tsdf_voxel_size_override"])
                if spec.get("tsdf_voxel_size_override") is not None
                else None
            ),
        )
    return result


def resolve_output_dir(config: ProjectConfig, mode_spec: ModeSpec, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (config.runtime.output_root / mode_spec.output_template.format(timestamp=timestamp)).resolve()


def _resolve_diffsynth_import_root() -> Path | None:
    diffsynth_path_raw = os.environ.get("DIFFSYNTH_PATH", "").strip()
    if diffsynth_path_raw:
        diffsynth_path = Path(diffsynth_path_raw).expanduser().resolve()
        if diffsynth_path.name == "diffsynth" and (diffsynth_path / "__init__.py").exists():
            return diffsynth_path.parent
        if (diffsynth_path / "diffsynth" / "__init__.py").exists():
            return diffsynth_path

    default_root = PROJECT_ROOT.parents[2]
    if (default_root / "diffsynth" / "__init__.py").exists():
        return default_root
    return None


def _resolve_standard_egowm_root() -> Path | None:
    return resolve_egosim_opensource_root()


def _resolve_torch_home() -> Path | None:
    torch_home_raw = os.environ.get("TORCH_HOME", "").strip()
    if torch_home_raw:
        torch_home = Path(torch_home_raw).expanduser().resolve()
        if torch_home.exists():
            return torch_home

    legacy_torch_home = WANVIDEO_ROOT / "inference" / "checkpoints"
    if legacy_torch_home.exists():
        return legacy_torch_home
    return None


def _merge_pythonpath(*paths: Path, existing: str) -> str:
    merged: list[str] = []
    seen: set[str] = set()
    for candidate in [*(str(path.resolve()) for path in paths), *existing.split(":")]:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return ":".join(merged)


def _prepend_path_entry(path: Path, *, existing: str) -> str:
    normalized = str(path.resolve()).strip()
    entries: list[str] = []
    seen: set[str] = set()
    for candidate in [normalized, *existing.split(":")]:
        value = candidate.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        entries.append(value)
    return ":".join(entries)


def build_runtime_env(config: ProjectConfig) -> dict[str, str]:
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYOPENGL_PLATFORM"] = config.runtime.pyopengl_platform
    env["TORCH_CUDA_ARCH_LIST"] = config.runtime.torch_cuda_arch_list
    pythonpath_roots = [SRC_ROOT]
    diffsynth_import_root = _resolve_diffsynth_import_root()
    if diffsynth_import_root is not None:
        pythonpath_roots.append(diffsynth_import_root)
    standard_egowm_root = _resolve_standard_egowm_root()
    if standard_egowm_root is not None:
        pythonpath_roots.append(standard_egowm_root)
    env["PYTHONPATH"] = _merge_pythonpath(*pythonpath_roots, existing=env.get("PYTHONPATH", ""))
    torch_home = _resolve_torch_home()
    if torch_home is not None:
        env["TORCH_HOME"] = str(torch_home)
    scene_python = config.runtime.python.scene.executable
    if scene_python is not None:
        env["PATH"] = _prepend_path_entry(scene_python.resolve().parent, existing=env.get("PATH", ""))
    if config.runtime.offline:
        env["MODELSCOPE_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
    if config.models.prior_depth_root is not None:
        env["PRIORDA_WEIGHTS_DIR"] = str(config.models.prior_depth_root)
    if config.runtime.unset_proxies:
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            env.pop(key, None)
    return env


def validate_project_config(config: ProjectConfig, mode: str, *, require_qwen: bool = False) -> None:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode: {mode}")

    required_paths = {
        "models.model_root": config.models.model_root,
        "data.dataset_root": config.data.dataset_root,
        "data.metadata_path": config.data.metadata_path,
        "backend.incremental_script": config.backend.incremental_script,
        "backend.egosim_state_script": config.backend.egosim_state_script,
    }
    if config.data.eval_set_path is not None:
        required_paths["data.eval_set_path"] = config.data.eval_set_path
    if require_qwen:
        required_paths["backend.predict_phrases_script"] = config.backend.predict_phrases_script
    for label, path in required_paths.items():
        if path is None or not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    validate_metadata_csv(config.data.metadata_path)
    config.runtime.python.wan.resolve_python(label="wan")

    if requires_scene(mode):
        config.runtime.python.scene.resolve_python(label="scene")
        if config.models.prior_depth_root is None or not config.models.prior_depth_root.exists():
            raise FileNotFoundError(
                "Scene-enabled modes require models.prior_depth_root to point to Prior-Depth-Anything weights."
            )

    if require_qwen:
        if config.models.qwen_vl_root is None or not config.models.qwen_vl_root.exists():
            raise FileNotFoundError(
                "Phrase-enabled modes require models.qwen_vl_root to exist."
            )


def prepare_quicktest_run(
    config: ProjectConfig,
    *,
    mode: str,
    output_dir: Path | None = None,
    extra_args: list[str] | None = None,
) -> QuicktestRun:
    mode_specs = load_mode_specs()
    if mode not in mode_specs:
        raise ValueError(f"Mode '{mode}' is not defined in {_mode_config_path()}")

    mode_spec = mode_specs[mode]
    phrase_prediction_enabled = (
        mode_spec.enable_phrase_prediction
        and not has_custom_scene_phrases(config.quicktest.scene_phrases)
    )
    for arg in extra_args or []:
        if arg == "--predict_phrases":
            phrase_prediction_enabled = True
        elif arg == "--no_predict_phrases":
            phrase_prediction_enabled = False

    validate_project_config(config, mode, require_qwen=phrase_prediction_enabled)
    wan_python = config.runtime.python.wan.resolve_python(label="wan")
    scene_python = (
        config.runtime.python.scene.resolve_python(label="scene")
        if requires_scene(mode)
        else wan_python
    )
    resolved_output = resolve_output_dir(config, mode_spec, output_dir)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    command = build_quicktest_command(
        config,
        mode_spec,
        wan_python=wan_python,
        scene_python=scene_python,
        output_dir=resolved_output,
        extra_args=extra_args or [],
    )

    return QuicktestRun(
        mode=mode,
        output_dir=resolved_output,
        command=command,
        env=build_runtime_env(config),
    )


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_quicktest(run: QuicktestRun, *, dry_run: bool = False) -> int:
    if dry_run:
        return 0

    proc = subprocess.Popen(run.command, env=run.env)
    return proc.wait()
