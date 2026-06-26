#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/project.yaml}"
MODE="${1:-recon_visualize}"
shift || true

PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}" \
python -m egowm_incremental.cli run-incremental \
  --config "${CONFIG_PATH}" \
  --mode "${MODE}" \
  -- "$@"
