from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .phrases import DEFAULT_SCENE_PHRASES
from .paths import (
    PROJECT_ROOT,
    expand_path,
    resolve_conda_env_python,
)

DEFAULT_QWEN_VL_ROOT = "../artifacts/models/qwen-vl/Qwen3-VL-4B-Instruct"


@dataclass
class PythonEnvConfig:
    env_name: str | None = None
    executable: Path | None = None

    def resolve_python(self, *, label: str) -> Path:
        if self.executable is not None:
            if not self.executable.exists():
                raise FileNotFoundError(f"{label} python does not exist: {self.executable}")
            return self.executable
        if self.env_name:
            return resolve_conda_env_python(self.env_name)
        raise ValueError(
            f"{label} python is not configured. Set runtime.python.{label}.env_name or executable."
        )


@dataclass
class RuntimePythonConfig:
    wan: PythonEnvConfig
    scene: PythonEnvConfig


@dataclass
class RuntimeConfig:
    output_root: Path
    gpu_id: int = 0
    offline: bool = True
    torch_cuda_arch_list: str = "9.0"
    pyopengl_platform: str = "egl"
    unset_proxies: bool = True
    python: RuntimePythonConfig = field(
        default_factory=lambda: RuntimePythonConfig(
            wan=PythonEnvConfig(env_name="egosim"),
            scene=PythonEnvConfig(env_name="egosim-scene"),
        )
    )


@dataclass
class ModelsConfig:
    model_root: Path
    qwen_vl_root: Path | None = None
    prior_depth_root: Path | None = None


@dataclass
class DataConfig:
    dataset_root: Path
    metadata_path: Path
    dataset: str = "egodex"
    eval_set_path: Path | None = None


@dataclass
class BackendConfig:
    incremental_script: Path
    egosim_state_script: Path
    predict_phrases_script: Path


@dataclass
class QuicktestDefaults:
    part_prefix: str = "test_3dmem"
    skip_existing: bool = True
    use_long_prompt: bool = True
    only_multi_clip: bool = True
    num_inference_steps: int = 50
    cfg_scale: float = 1.0
    fps: int = 16
    egosim_state_pipeline: str = "dav3"
    spatial_subsample: int = 1
    temporal_subsample: int = 1
    use_color_depth_overlap: bool = False
    scene_phrases: list[str] = field(default_factory=lambda: list(DEFAULT_SCENE_PHRASES))


@dataclass
class ProjectConfig:
    config_path: Path
    runtime: RuntimeConfig
    models: ModelsConfig
    data: DataConfig
    backend: BackendConfig
    quicktest: QuicktestDefaults


def _nested(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def load_project_config(config_path: str | Path) -> ProjectConfig:
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent
    raw = _load_yaml(config_path)

    runtime_raw = raw.get("runtime", {})
    python_raw = _nested(raw, "runtime", "python", default={}) or {}
    models_raw = raw.get("models", {})
    data_raw = raw.get("data", {})
    backend_raw = raw.get("backend", {})
    quicktest_raw = raw.get("quicktest", {})

    runtime = RuntimeConfig(
        output_root=expand_path(runtime_raw.get("output_root", "./outputs"), base_dir=config_dir),
        gpu_id=int(runtime_raw.get("gpu_id", 0)),
        offline=bool(runtime_raw.get("offline", True)),
        torch_cuda_arch_list=str(runtime_raw.get("torch_cuda_arch_list", "9.0")),
        pyopengl_platform=str(runtime_raw.get("pyopengl_platform", "egl")),
        unset_proxies=bool(runtime_raw.get("unset_proxies", True)),
        python=RuntimePythonConfig(
            wan=PythonEnvConfig(
                env_name=_nested(python_raw, "wan", "env_name"),
                executable=expand_path(_nested(python_raw, "wan", "executable"), base_dir=config_dir),
            ),
            scene=PythonEnvConfig(
                env_name=_nested(python_raw, "scene", "env_name"),
                executable=expand_path(_nested(python_raw, "scene", "executable"), base_dir=config_dir),
            ),
        ),
    )

    backend_root = PROJECT_ROOT / "src" / "egowm_incremental"
    backend = BackendConfig(
        incremental_script=expand_path(
            backend_raw.get("incremental_script")
            or str(backend_root / "backends" / "incremental_inference.py"),
            base_dir=config_dir,
        ),
        egosim_state_script=expand_path(
            backend_raw.get("egosim_state_script")
            or str(backend_root / "backends" / "egosim_state_subprocess.py"),
            base_dir=config_dir,
        ),
        predict_phrases_script=expand_path(
            backend_raw.get("predict_phrases_script")
            or str(backend_root / "vendor" / "predict_phrases.py"),
            base_dir=config_dir,
        ),
    )

    config = ProjectConfig(
        config_path=config_path,
        runtime=runtime,
        models=ModelsConfig(
            model_root=expand_path(
                models_raw.get("model_root", "../EgoSim-14B"),
                base_dir=PROJECT_ROOT,
            ),
            qwen_vl_root=expand_path(
                models_raw.get("qwen_vl_root") or DEFAULT_QWEN_VL_ROOT,
                base_dir=config_dir,
            ),
            prior_depth_root=expand_path(
                models_raw.get("prior_depth_root", "../artifacts/models/priorda"),
                base_dir=config_dir,
            ),
        ),
        data=DataConfig(
            dataset=str(data_raw.get("dataset", "egodex")),
            dataset_root=expand_path(
                data_raw.get(
                    "dataset_root",
                    "../../tests/samples/mini_sample/continuous_generation",
                ),
                base_dir=config_dir,
            ),
            metadata_path=expand_path(
                data_raw.get(
                    "metadata_path",
                    # Same file as configs/project.yaml / CLI default incremental_inference --metadata_path
                    "../../tests/samples/mini_sample/continuous_generation/metadata.csv",
                ),
                base_dir=config_dir,
            ),
            eval_set_path=expand_path(data_raw.get("eval_set_path"), base_dir=config_dir),
        ),
        backend=backend,
        quicktest=QuicktestDefaults(
            part_prefix=str(quicktest_raw.get("part_prefix", "test_3dmem")),
            skip_existing=bool(quicktest_raw.get("skip_existing", True)),
            use_long_prompt=bool(quicktest_raw.get("use_long_prompt", True)),
            only_multi_clip=bool(quicktest_raw.get("only_multi_clip", True)),
            num_inference_steps=int(quicktest_raw.get("num_inference_steps", 50)),
            cfg_scale=float(quicktest_raw.get("cfg_scale", 1.0)),
            fps=int(quicktest_raw.get("fps", 16)),
            egosim_state_pipeline=str(quicktest_raw.get("egosim_state_pipeline", "dav3")),
            spatial_subsample=int(quicktest_raw.get("spatial_subsample", 1)),
            temporal_subsample=int(quicktest_raw.get("temporal_subsample", 1)),
            use_color_depth_overlap=bool(quicktest_raw.get("use_color_depth_overlap", False)),
            scene_phrases=[str(item) for item in quicktest_raw.get("scene_phrases", list(DEFAULT_SCENE_PHRASES))],
        ),
    )

    if config.models.model_root is None:
        raise ValueError("Configure models.model_root, matching the single-clip EgoWM runner.")
    if config.data.dataset_root is None or config.data.metadata_path is None:
        raise ValueError("Both data.dataset_root and data.metadata_path must be configured.")

    return config
