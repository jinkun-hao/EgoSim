#!/bin/bash
# Copyright (c) jiamingda (https://github.com/Luyitas)
set -euo pipefail
# ============================================================
# Step 01a: DA3 predict first-frame depth + full-sequence camera
# Single-clip annotation — configure VIDEO_PATH in run_pipeline.env.sh
# conda env: da3
# Working directory: Depth-Anything-3 repo root
# ============================================================
#
# Usage:
#   source data_process/run_pipeline.env.sh
#   conda activate da3
#   bash data_process/01_depth_pose_da3/run_step01a_da3_predict.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${VIDEO_PATH:?Set VIDEO_PATH in run_pipeline.env.sh and source it first}"
: "${DA3_ROOT:?Set DA3_ROOT in run_pipeline.env.sh and source it first}"
: "${DA3_MODEL:?Set DA3_MODEL in run_pipeline.env.sh and source it first}"
DEVICE="${DEVICE:-0}"

CLIP_ID="$(basename "${VIDEO_PATH%.*}")"
CLIP_DIR="${REPO_ROOT}/tests/samples/${CLIP_ID}"
PROC_DIR="${CLIP_DIR}/_proc"

echo "[Step 01a] DA3 Predict Depth + Camera"
echo "  Video:  ${CLIP_DIR}/video_16fps.mp4"
echo "  Output: ${PROC_DIR}/poses_da3"
echo ""

cd "${DA3_ROOT}"
PYTHONPATH="${DA3_ROOT}:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="${DEVICE}" python "${SCRIPT_DIR}/pred_multi_gpu_2.py" \
    --video_path "${CLIP_DIR}/video_16fps.mp4" \
    --clip_name "${CLIP_ID}" \
    --output_root "${PROC_DIR}/poses_da3" \
    --model_path "${DA3_MODEL}" \
    --batch_size 128 \
    --skip_check
echo "[Step 01a] Done."
