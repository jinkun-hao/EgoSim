#!/bin/bash
# Copyright (c) jiamingda (https://github.com/Luyitas)
set -euo pipefail
# ============================================================
# Step 01d (mask): point-cloud mask video for one clip (for inference)
# conda env: da3
# ============================================================
#
# Usage:
#   source data_process/run_pipeline.env.sh
#   conda activate da3
#   bash data_process/01_depth_pose_da3/run_step01d_render_mask.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${VIDEO_PATH:?Set VIDEO_PATH in run_pipeline.env.sh and source it first}"

CLIP_ID="$(basename "${VIDEO_PATH%.*}")"
CLIP_DIR="${REPO_ROOT}/tests/samples/${CLIP_ID}"
PROC_DIR="${CLIP_DIR}/_proc"

export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "[Step 01d-mask] Render Point Cloud Mask Video"
echo "  Video:    ${CLIP_DIR}/video_16fps.mp4"
echo "  Pose dir: ${PROC_DIR}/poses_da3_smoothed/${CLIP_ID}"
echo "  Output:   ${CLIP_DIR}/pc_mask_video.mp4"
echo ""

mkdir -p "${CLIP_DIR}"

python "${SCRIPT_DIR}/render_16fps_aligned.py" \
    --video_path "${CLIP_DIR}/video_16fps.mp4" \
    --pose_dir "${PROC_DIR}/poses_da3_smoothed/${CLIP_ID}" \
    --intrinsics_path "${PROC_DIR}/inpainted/${CLIP_ID}/intrinsics_first_frame.npy" \
    --rgb_path "${PROC_DIR}/inpainted/${CLIP_ID}/hand_inpaint.png" \
    --depth_path "${PROC_DIR}/inpainted/${CLIP_ID}/depth_first_frame.npy" \
    --output_video "${CLIP_DIR}/pc_mask_video.mp4" \
    --overlay_video "${CLIP_DIR}/overlay_mask.mp4" \
    --mask_mode \
    --point_size 4.0 \
    --fps 16

echo "[Step 01d-mask] Done. Output: ${CLIP_DIR}/pc_mask_video.mp4"
