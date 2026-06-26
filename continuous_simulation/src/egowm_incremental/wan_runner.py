from __future__ import annotations

from pathlib import Path

from .config import ProjectConfig
from .phrases import has_custom_scene_phrases
from .scene_runner import ModeSpec


def _phrase_prediction_enabled(
    mode_spec: ModeSpec,
    extra_args: list[str],
    scene_phrases: list[str],
) -> bool:
    if has_custom_scene_phrases(scene_phrases):
        return False

    enabled = mode_spec.enable_phrase_prediction
    for arg in extra_args:
        if arg == "--predict_phrases":
            enabled = True
        elif arg == "--no_predict_phrases":
            enabled = False
    return enabled


def _filtered_extra_args(extra_args: list[str]) -> list[str]:
    return [arg for arg in extra_args if arg not in {"--predict_phrases", "--no_predict_phrases"}]


def build_quicktest_command(
    config: ProjectConfig,
    mode_spec: ModeSpec,
    *,
    wan_python: Path,
    scene_python: Path,
    output_dir: Path,
    extra_args: list[str],
) -> list[str]:
    quicktest = config.quicktest
    spatial_subsample = (
        mode_spec.spatial_subsample_override
        if mode_spec.spatial_subsample_override is not None
        else quicktest.spatial_subsample
    )
    phrase_prediction_enabled = _phrase_prediction_enabled(
        mode_spec,
        extra_args,
        quicktest.scene_phrases,
    )
    command = [
        str(wan_python),
        str(config.backend.incremental_script),
        "--dataset", config.data.dataset,
        "--model_root", str(config.models.model_root),
        "--dataset_root", str(config.data.dataset_root),
        "--metadata_path", str(config.data.metadata_path),
        "--part_prefix", quicktest.part_prefix,
        "--output_dir", str(output_dir),
        "--gpu_id", str(config.runtime.gpu_id),
        "--fps", str(quicktest.fps),
        "--num_inference_steps", str(quicktest.num_inference_steps),
        "--cfg_scale", str(quicktest.cfg_scale),
        "--egosim_state_python", str(scene_python),
        "--da3_python", str(scene_python),
        "--egosim_state_script", str(config.backend.egosim_state_script),
        "--predict_phrases_script", str(config.backend.predict_phrases_script),
        "--scene_phrases", *quicktest.scene_phrases,
    ]

    if config.data.eval_set_path is not None:
        command.extend(["--eval_set_path", str(config.data.eval_set_path)])

    if quicktest.only_multi_clip:
        command.append("--only_multi_clip")
    if quicktest.skip_existing:
        command.append("--skip_existing")
    if quicktest.use_long_prompt:
        command.append("--use_long_prompt")
    else:
        command.append("--no_use_long_prompt")

    if not mode_spec.enable_egosim_state:
        command.append("--no_egosim_state")
    else:
        command.extend([
            "--egosim_state_pipeline", quicktest.egosim_state_pipeline,
            "--spatial_subsample", str(spatial_subsample),
            "--temporal_subsample", str(quicktest.temporal_subsample),
        ])
        if not quicktest.use_color_depth_overlap:
            command.append("--no_use_color_depth_overlap")
        if mode_spec.tsdf_voxel_size_override is not None:
            command.extend(["--tsdf_voxel_size", str(mode_spec.tsdf_voxel_size_override)])

    if phrase_prediction_enabled:
        if config.models.qwen_vl_root is None:
            raise ValueError(
                f"Mode '{mode_spec.name}' requires models.qwen_vl_root in the config."
            )
        command.extend([
            "--predict_phrases",
            "--qwen_model_path", str(config.models.qwen_vl_root),
        ])
    else:
        command.append("--no_predict_phrases")

    if mode_spec.recon_visualize:
        command.extend(["--incremental_mode", "recon_visualize"])

    command.extend(_filtered_extra_args(extra_args))
    return command
