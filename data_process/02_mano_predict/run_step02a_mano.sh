#!/bin/bash
# Copyright (c) jiamingda (https://github.com/Luyitas)
set -euo pipefail
# ============================================================
# Step 02a: HaMeR MANO hand pose prediction
# conda env: hamer
# Working directory: HaMeR repo root ($HAMER_ROOT)
# ============================================================
#
# Usage:
#   source data_process/run_pipeline.env.sh
#   conda activate hamer
#   bash data_process/02_mano_predict/run_step02a_mano.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${VIDEO_PATH:?Set VIDEO_PATH in run_pipeline.env.sh and source it first}"
: "${HAMER_ROOT:?Set HAMER_ROOT in run_pipeline.env.sh and source it first}"
: "${VITDET_INIT_CHECKPOINT:?Set VITDET_INIT_CHECKPOINT in run_pipeline.env.sh and source it first}"
DEVICE="${DEVICE:-0}"

CLIP_ID="$(basename "${VIDEO_PATH%.*}")"
CLIP_DIR="${REPO_ROOT}/tests/samples/${CLIP_ID}"
PROC_DIR="${CLIP_DIR}/_proc"

export PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

echo "[Step 02a] HaMeR MANO Prediction"
echo "  Clip:   ${CLIP_ID}"
echo "  Video:  ${CLIP_DIR}/video.mp4"
echo "  Output: ${PROC_DIR}/mano_annotations"
echo ""

CLIP_LIST=$(mktemp)
echo "${CLIP_ID}.mp4" > "${CLIP_LIST}"

cd "${HAMER_ROOT}"
PYTHONPATH="${HAMER_ROOT}:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="${DEVICE}" python "${SCRIPT_DIR}/egovid_annotate_batch.py" \
    --clip_list "${CLIP_LIST}" \
    --output_dir "${PROC_DIR}/mano_annotations" \
    --clips_dir "${CLIP_DIR}" \
    --vitdet_checkpoint "${VITDET_INIT_CHECKPOINT}" \
    --device cuda \
    --batch_size_videos 1 \
    --hand_batch_size 4096 \
    --body_batch_size 4 \
    --dataloader_workers 2 \
    --output_mode light \
    --no_resume

rm -f "${CLIP_LIST}"
echo "[Step 02a] Done."
