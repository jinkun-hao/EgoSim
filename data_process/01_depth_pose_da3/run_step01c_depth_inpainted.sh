#!/bin/bash
# Copyright (c) jiamingda (https://github.com/Luyitas)
set -euo pipefail
# ============================================================
# Step 01c: DA3 depth on inpainted first frame (single clip)
# conda env: da3
# Working directory: Depth-Anything-3 repo root
# ============================================================
#
# Usage:
#   source data_process/run_pipeline.env.sh
#   conda activate da3
#   bash data_process/01_depth_pose_da3/run_step01c_depth_inpainted.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${VIDEO_PATH:?Set VIDEO_PATH in run_pipeline.env.sh and source it first}"
: "${DA3_ROOT:?Set DA3_ROOT in run_pipeline.env.sh and source it first}"
: "${DA3_MODEL:?Set DA3_MODEL in run_pipeline.env.sh and source it first}"
DEVICE="${DEVICE:-0}"

CLIP_ID="$(basename "${VIDEO_PATH%.*}")"
CLIP_DIR="${REPO_ROOT}/tests/samples/${CLIP_ID}"
PROC_DIR="${CLIP_DIR}/_proc"

INPAINT_CLIP_DIR="${PROC_DIR}/inpainted/${CLIP_ID}"
LOG_DIR="${SCRIPT_DIR}/_depth_inpaint_logs"

echo "[Step 01c] DA3 Depth on Inpainted Image"
echo "  Input:  ${INPAINT_CLIP_DIR}/hand_inpaint.png"
echo ""

cd "${DA3_ROOT}"
PYTHONPATH="${DA3_ROOT}:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="${DEVICE}" python "${SCRIPT_DIR}/process_depth_inpainted.py" \
    --clip_dir "${INPAINT_CLIP_DIR}" \
    --model_path "${DA3_MODEL}" \
    --log_dir "${LOG_DIR}"
echo "[Step 01c] Done."
