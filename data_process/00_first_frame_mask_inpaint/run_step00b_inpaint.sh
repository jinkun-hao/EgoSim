#!/bin/bash
# Copyright (c) jiamingda (https://github.com/Luyitas)
set -euo pipefail
# ============================================================
# Step 00b: First-frame hand inpainting (Qwen-Image-Edit)
# conda env: da3
# ============================================================
#
# Usage:
#   source data_process/run_pipeline.env.sh
#   conda activate da3
#   bash data_process/00_first_frame_mask_inpaint/run_step00b_inpaint.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${VIDEO_PATH:?Set VIDEO_PATH in run_pipeline.env.sh and source it first}"
: "${INPAINT_MODEL:?Set INPAINT_MODEL in run_pipeline.env.sh and source it first}"
DEVICE="${DEVICE:-0}"

CLIP_ID="$(basename "${VIDEO_PATH%.*}")"
CLIP_DIR="${REPO_ROOT}/tests/samples/${CLIP_ID}"
PROC_DIR="${CLIP_DIR}/_proc"

LOG_DIR="${SCRIPT_DIR}/inpaint_logs_single"

echo "[Step 00b] Hand Inpainting"
echo "  Clip:       ${CLIP_ID}"
echo "  Input base: ${PROC_DIR}/sam_results"
echo "  Output:     ${PROC_DIR}/inpainted"
echo ""

CLIP_LIST=$(mktemp)
echo "${CLIP_ID}" > "${CLIP_LIST}"

CUDA_VISIBLE_DEVICES="${DEVICE}" python "${SCRIPT_DIR}/wanx_inpaint.py" \
    --video_list "${CLIP_LIST}" \
    --input_base "${PROC_DIR}/sam_results" \
    --output_base "${PROC_DIR}/inpainted" \
    --model_path "${INPAINT_MODEL}" \
    --device 0 \
    --log_dir "${LOG_DIR}"

rm -f "${CLIP_LIST}"
echo "[Step 00b] Done."
