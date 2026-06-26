#!/bin/bash
# Pipeline path configuration for data_process steps.
# Defaults are relative to the EgoSim repo; override any export before sourcing.

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_EGOSIM_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
_REPOS_DIR="${REPOS_DIR:-${_EGOSIM_ROOT}/../repos}"
_HF_MODELS="${HF_MODELS:-${HF_HOME:-${HOME}/.cache/huggingface}/models}"

export VIDEO_PATH="${VIDEO_PATH:-/path/to/your/video.mp4}"

export DA3_ROOT="${DA3_ROOT:-${_REPOS_DIR}/Depth-Anything-3}"
export HAMER_ROOT="${HAMER_ROOT:-${_REPOS_DIR}/hamer}"
export SAM3_ROOT="${SAM3_ROOT:-${_REPOS_DIR}/sam3}"

export DA3_MODEL="${DA3_MODEL:-${DA3_ROOT}/checkpoints/DA3NESTED-GIANT-LARGE-1.1}"
export SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-${SAM3_ROOT}/checkpoints/sam3.pt}"
export INPAINT_MODEL="${INPAINT_MODEL:-${_HF_MODELS}/Qwen-Image-Edit-2511}"
export VITDET_INIT_CHECKPOINT="${VITDET_INIT_CHECKPOINT:-${HAMER_ROOT}/_DATA/model_final_f05665.pkl}"
export MANO_PATH="${MANO_PATH:-${HAMER_ROOT}/_DATA/data/mano}"
export CAPTION_MODEL="${CAPTION_MODEL:-${_HF_MODELS}/Qwen2.5-VL-7B-Instruct}"

# export DEVICE=0
# export PYOPENGL_PLATFORM=egl
