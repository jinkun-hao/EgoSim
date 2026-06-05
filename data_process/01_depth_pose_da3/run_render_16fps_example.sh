#!/bin/bash
# Copyright (c) jiamingda (https://github.com/Luyitas)
set -euo pipefail

# ============================================================
# Single-clip render helper (Steps 01d + 01d-mask)
#
# This script replaces the old EgoVid batch renderer
# (sharding / split_data.py / multi-worker). For one clip,
# run both point-cloud videos via the standard step scripts.
#
# Usage:
#   source data_process/run_pipeline.env.sh
#   conda activate da3
#   bash data_process/01_depth_pose_da3/run_render_16fps_example.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[info] Single-clip render: colored overlay + mask video"
bash "${SCRIPT_DIR}/run_step01d_render.sh"
bash "${SCRIPT_DIR}/run_step01d_render_mask.sh"
echo "[info] Done."
