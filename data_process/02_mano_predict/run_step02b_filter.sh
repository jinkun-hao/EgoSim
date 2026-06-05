#!/bin/bash
# Copyright (c) jiamingda (https://github.com/Luyitas)
set -euo pipefail
# ============================================================
# Step 02b: MANO filtering (dedup, best hand, temporal outliers)
# conda env: hamer
# Working directory: HaMeR repo root ($HAMER_ROOT)
# ============================================================
#
# Usage:
#   source data_process/run_pipeline.env.sh
#   conda activate hamer
#   bash data_process/02_mano_predict/run_step02b_filter.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${VIDEO_PATH:?Set VIDEO_PATH in run_pipeline.env.sh and source it first}"
: "${HAMER_ROOT:?Set HAMER_ROOT in run_pipeline.env.sh and source it first}"
: "${MANO_PATH:?Set MANO_PATH in run_pipeline.env.sh and source it first}"
NUM_WORKERS="${NUM_WORKERS:-4}"

CLIP_ID="$(basename "${VIDEO_PATH%.*}")"
CLIP_DIR="${REPO_ROOT}/tests/samples/${CLIP_ID}"
PROC_DIR="${CLIP_DIR}/_proc"

export PYOPENGL_PLATFORM=osmesa

echo "[Step 02b] MANO Filter & Deduplicate"
echo "  Clip:       ${CLIP_ID}"
echo "  Annot dir:  ${PROC_DIR}/mano_annotations"
echo "  Output:     ${PROC_DIR}/mano_filtered"
echo ""

CLIP_LIST=$(mktemp)
echo "${CLIP_ID}" > "${CLIP_LIST}"

cd "${HAMER_ROOT}"
PYTHONPATH="${HAMER_ROOT}:${PYTHONPATH:-}" \
python "${SCRIPT_DIR}/filter_lightweight_duplicates.py" \
    --clip_list "${CLIP_LIST}" \
    --annot_dir "${PROC_DIR}/mano_annotations" \
    --video_dir "${CLIP_DIR}" \
    --output_dir "${PROC_DIR}/mano_filter_vis" \
    --output_annot_dir "${PROC_DIR}/mano_filtered" \
    --mano_path "${MANO_PATH}" \
    --num_workers "${NUM_WORKERS}" \
    --no_vis

rm -f "${CLIP_LIST}"
echo "[Step 02b] Done."
