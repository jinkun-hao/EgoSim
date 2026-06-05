#!/bin/bash
# Copyright (c) jiamingda (https://github.com/Luyitas)
set -euo pipefail
# ============================================================
# Step 00a: SAM3 first-frame hand segmentation
# conda env: sam3
# ============================================================
#
# Usage:
#   source data_process/run_pipeline.env.sh
#   conda activate sam3
#   bash data_process/00_first_frame_mask_inpaint/run_step00a_sam_seg.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${VIDEO_PATH:?Set VIDEO_PATH in run_pipeline.env.sh and source it first}"
: "${SAM3_CHECKPOINT:?Set SAM3_CHECKPOINT in run_pipeline.env.sh and source it first}"
DEVICE="${DEVICE:-0}"

CLIP_ID="$(basename "${VIDEO_PATH%.*}")"
CLIP_DIR="${REPO_ROOT}/tests/samples/${CLIP_ID}"
PROC_DIR="${CLIP_DIR}/_proc"

echo "[Step 00a] SAM3 Hand Segmentation"
echo "  Video:      ${CLIP_DIR}/video.mp4"
echo "  Output dir: ${PROC_DIR}/sam_results"
echo "  Checkpoint: ${SAM3_CHECKPOINT}"
echo ""

CUDA_VISIBLE_DEVICES="${DEVICE}" python "${SCRIPT_DIR}/sam_seg.py" \
    --input_dir "${CLIP_DIR}/video.mp4" \
    --output_dir "${PROC_DIR}/sam_results" \
    --checkpoint "${SAM3_CHECKPOINT}" \
    --device 0

echo "[Step 00a] Done."
