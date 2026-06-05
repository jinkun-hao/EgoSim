#!/bin/bash
# Copyright (c) jiamingda (https://github.com/Luyitas)
set -euo pipefail
# ============================================================
# Step 02d: Hand skeleton visualization
# conda env: hamer (opencv + numpy only; no GPU required)
# ============================================================
#
# Usage:
#   source data_process/run_pipeline.env.sh
#   conda activate hamer
#   bash data_process/02_mano_predict/run_step02d_visualize_skeleton.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${VIDEO_PATH:?Set VIDEO_PATH in run_pipeline.env.sh and source it first}"

CLIP_ID="$(basename "${VIDEO_PATH%.*}")"
CLIP_DIR="${REPO_ROOT}/tests/samples/${CLIP_ID}"
PROC_DIR="${CLIP_DIR}/_proc"

echo "[Step 02d] Skeleton Visualization"
echo "  Video dir: ${CLIP_DIR}"
echo "  Annot dir: ${PROC_DIR}/mano_full"
echo "  Output:    ${CLIP_DIR}/skeleton_3d.mp4"
echo ""

python "${SCRIPT_DIR}/visualize_skeleton.py" \
    --video_dir "${CLIP_DIR}" \
    --annot_dir "${PROC_DIR}/mano_full" \
    --output_dir "${CLIP_DIR}" \
    --no_resume

# visualize_skeleton.py outputs <stem>_black.mp4 (stem = video_16fps).
# Rename to the filename expected by generate_metadata.py and runner.py.
BLACK_SRC="${CLIP_DIR}/video_16fps_black.mp4"
SKELETON_DST="${CLIP_DIR}/skeleton_3d.mp4"
if [ -f "${BLACK_SRC}" ]; then
    mv "${BLACK_SRC}" "${SKELETON_DST}"
    echo "[Step 02d] Renamed video_16fps_black.mp4 -> skeleton_3d.mp4"
else
    echo "[Step 02d] WARNING: expected ${BLACK_SRC} not found; skeleton_3d.mp4 not created"
    exit 1
fi

echo "[Step 02d] Done."
