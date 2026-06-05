#!/bin/bash
# Copyright (c) jiamingda (https://github.com/Luyitas)
set -euo pipefail
# ============================================================
# Step 01d: Render point-cloud video for one clip (16fps colored overlay)
# conda env: da3
# ============================================================
#
# Usage:
#   source data_process/run_pipeline.env.sh
#   conda activate da3
#   bash data_process/01_depth_pose_da3/run_step01d_render.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${VIDEO_PATH:?Set VIDEO_PATH in run_pipeline.env.sh and source it first}"
: "${DA3_ROOT:?Set DA3_ROOT in run_pipeline.env.sh and source it first}"

CLIP_ID="$(basename "${VIDEO_PATH%.*}")"
CLIP_DIR="${REPO_ROOT}/tests/samples/${CLIP_ID}"
PROC_DIR="${CLIP_DIR}/_proc"

export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "[Step 01d] Render 16fps Point Cloud Video"
echo "  Video:    ${CLIP_DIR}/video_16fps.mp4"
echo "  Pose dir: ${PROC_DIR}/poses_da3_smoothed/${CLIP_ID}"
echo "  Output:   ${CLIP_DIR}/rendered_scene.mp4, ${CLIP_DIR}/overlay.mp4"
echo ""

mkdir -p "${CLIP_DIR}"

cd "${DA3_ROOT}/render_scene_first_frame"
PYTHONPATH="${DA3_ROOT}:${DA3_ROOT}/render_scene_first_frame:${PYTHONPATH:-}" \
python "${SCRIPT_DIR}/render_16fps_aligned.py" \
    --video_path "${CLIP_DIR}/video_16fps.mp4" \
    --pose_dir "${PROC_DIR}/poses_da3_smoothed/${CLIP_ID}" \
    --intrinsics_path "${PROC_DIR}/inpainted/${CLIP_ID}/intrinsics_first_frame.npy" \
    --rgb_path "${PROC_DIR}/inpainted/${CLIP_ID}/hand_inpaint.png" \
    --depth_path "${PROC_DIR}/inpainted/${CLIP_ID}/depth_first_frame.npy" \
    --output_video "${CLIP_DIR}/rendered_scene.mp4" \
    --overlay_video "${CLIP_DIR}/overlay.mp4" \
    --point_size 2.0 \
    --fps 16

echo "[Step 01d] Done."
