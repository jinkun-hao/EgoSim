#!/bin/bash
# Copyright (c) jiamingda (https://github.com/Luyitas)
set -euo pipefail
# ============================================================
# Step 02c: Reconstruct full MANO (vertices + keypoints_3d)
# conda env: hamer
# Working directory: HaMeR repo root ($HAMER_ROOT)
# ============================================================
#
# Usage:
#   source data_process/run_pipeline.env.sh
#   conda activate hamer
#   bash data_process/02_mano_predict/run_step02c_reconstruct_full.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${VIDEO_PATH:?Set VIDEO_PATH in run_pipeline.env.sh and source it first}"
: "${HAMER_ROOT:?Set HAMER_ROOT in run_pipeline.env.sh and source it first}"
: "${MANO_PATH:?Set MANO_PATH in run_pipeline.env.sh and source it first}"
DEVICE="${DEVICE:-0}"

CLIP_ID="$(basename "${VIDEO_PATH%.*}")"
CLIP_DIR="${REPO_ROOT}/tests/samples/${CLIP_ID}"
PROC_DIR="${CLIP_DIR}/_proc"

export PYOPENGL_PLATFORM=osmesa

echo "[Step 02c] Reconstruct Full MANO"
echo "  Input:  ${PROC_DIR}/mano_filtered"
echo "  Output: ${PROC_DIR}/mano_full"
echo ""

cd "${HAMER_ROOT}"
PYTHONPATH="${HAMER_ROOT}:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="${DEVICE}" python "${SCRIPT_DIR}/reconstruct_full_mano.py" \
    --input_dir "${PROC_DIR}/mano_filtered" \
    --output_dir "${PROC_DIR}/mano_full" \
    --mano_path "${MANO_PATH}" \
    --device cuda \
    --no_resume

echo "[Step 02c] Done."
