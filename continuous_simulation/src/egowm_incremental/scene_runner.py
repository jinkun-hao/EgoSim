from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_MODES = {
    "full",
    "recon_visualize",
}


@dataclass(frozen=True)
class ModeSpec:
    name: str
    output_template: str
    enable_egosim_state: bool
    enable_phrase_prediction: bool
    recon_visualize: bool = False
    spatial_subsample_override: int | None = None
    tsdf_voxel_size_override: float | None = None


def requires_scene(mode: str) -> bool:
    return mode in {"full", "recon_visualize"}


def requires_qwen(mode: str) -> bool:
    return mode in {"full", "recon_visualize"}
