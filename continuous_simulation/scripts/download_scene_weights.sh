#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
ARTIFACTS_ROOT="${REPO_ROOT}/artifacts/models"
PRIORDA_SCRIPT="${PROJECT_ROOT}/scene_backend/egosim_state/scripts/download_priorda_weights.py"

WITH_QWEN=0
WITH_ALTERNATE=0
WITH_AOT=0
SKIP_SAM3=0

usage() {
  cat <<'EOF'
Download scene-reconstruction weights into EgoSim/artifacts/models/.

Usage:
  bash continuous_simulation/scripts/download_scene_weights.sh [options]

Options:
  --with-qwen               Also download Qwen3-VL-4B-Instruct
  --with-alternate          Also download UniDepth / Video Depth Anything / MoGe-2
  --with-aot                Also download DeAOT tracker weights (otherwise auto-fetched on first run)
  --skip-sam3               Skip facebook/sam3 (gated; download manually later)
  --all                     Download everything above
  -h, --help                Show this help

Required by default (dav3 scene pipeline):
  - Prior-Depth-Anything
  - SAM3
  - Depth Anything 3 (metric + nested)
  - GroundingDINO
  - BERT base uncased

Prerequisites:
  - huggingface-cli (or hf)
  - python with huggingface_hub (for PriorDA)
  - wget (for Video Depth Anything, when --with-alternate is used)
  - gdown (only when --with-aot is used)

Notes:
  - facebook/sam3 requires accepting the model license on Hugging Face first.
  - Existing files are skipped.
  - EgoSim-14B is not included; download it separately (see README).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-qwen)
      WITH_QWEN=1
      ;;
    --with-alternate)
      WITH_ALTERNATE=1
      ;;
    --with-aot)
      WITH_AOT=1
      ;;
    --skip-sam3)
      SKIP_SAM3=1
      ;;
    --all)
      WITH_QWEN=1
      WITH_ALTERNATE=1
      WITH_AOT=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $1" >&2
    exit 1
  fi
}

repo_has_weights() {
  local local_dir="$1"
  find "${local_dir}" \( -name '*.safetensors' -o -name '*.pth' -o -name '*.pt' -o -name '*.bin' \) \
    ! -path '*/.cache/*' -print -quit 2>/dev/null | grep -q .
}

hf_download_repo() {
  local repo_id="$1"
  local local_dir="$2"

  if [[ -d "${local_dir}" ]] && repo_has_weights "${local_dir}"; then
    echo "✓ Already present: ${local_dir}"
    return 0
  fi

  mkdir -p "${local_dir}"
  echo "→ Downloading ${repo_id} -> ${local_dir}"
  if command -v hf >/dev/null 2>&1; then
    hf download "${repo_id}" --local-dir "${local_dir}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${repo_id}" --local-dir "${local_dir}"
  else
    echo "ERROR: hf or huggingface-cli is required." >&2
    exit 1
  fi
}

hf_download_file() {
  local repo_id="$1"
  local filename="$2"
  local local_dir="$3"
  local output_path="${local_dir}/${filename}"

  if [[ -f "${output_path}" ]]; then
    echo "✓ Already present: ${output_path}"
    return 0
  fi

  mkdir -p "${local_dir}"
  echo "→ Downloading ${repo_id}/${filename} -> ${local_dir}"
  if command -v hf >/dev/null 2>&1; then
    hf download "${repo_id}" "${filename}" --local-dir "${local_dir}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${repo_id}" "${filename}" --local-dir "${local_dir}"
  else
    echo "ERROR: hf or huggingface-cli is required." >&2
    exit 1
  fi
}

download_url() {
  local url="$1"
  local output_path="$2"

  if [[ -f "${output_path}" ]]; then
    echo "✓ Already present: ${output_path}"
    return 0
  fi

  mkdir -p "$(dirname "${output_path}")"
  echo "→ Downloading ${url} -> ${output_path}"
  wget -q --show-progress -O "${output_path}" "${url}"
}

require_cmd python
if ! command -v huggingface-cli >/dev/null 2>&1 && ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: huggingface-cli or hf is required." >&2
  exit 1
fi

mkdir -p \
  "${ARTIFACTS_ROOT}/priorda" \
  "${ARTIFACTS_ROOT}/sam3" \
  "${ARTIFACTS_ROOT}/groundingdino" \
  "${ARTIFACTS_ROOT}/depth-anything-3" \
  "${ARTIFACTS_ROOT}/bert-base-uncased" \
  "${ARTIFACTS_ROOT}/aot"

echo "Artifact root: ${ARTIFACTS_ROOT}"
echo

echo "[1/6] Prior-Depth-Anything"
python "${PRIORDA_SCRIPT}" --output_dir "${ARTIFACTS_ROOT}/priorda"

echo
if [[ "${SKIP_SAM3}" == "1" ]]; then
  echo "[2/6] SAM3 (skipped)"
else
  echo "[2/6] SAM3"
  hf_download_file facebook/sam3 sam3.pt "${ARTIFACTS_ROOT}/sam3"
fi

echo
echo "[3/6] Depth Anything 3 (metric)"
hf_download_repo depth-anything/DA3METRIC-LARGE "${ARTIFACTS_ROOT}/depth-anything-3/DA3METRIC-LARGE"

echo
echo "[4/6] Depth Anything 3 (nested)"
hf_download_repo depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
  "${ARTIFACTS_ROOT}/depth-anything-3/DA3NESTED-GIANT-LARGE-1.1"

echo
echo "[5/6] GroundingDINO"
hf_download_file ShilongLiu/GroundingDINO groundingdino_swint_ogc.pth \
  "${ARTIFACTS_ROOT}/groundingdino"

echo
echo "[6/6] BERT base uncased (GroundingDINO tokenizer + text encoder)"
hf_download_repo google-bert/bert-base-uncased "${ARTIFACTS_ROOT}/bert-base-uncased"

if [[ "${WITH_AOT}" == "1" ]]; then
  echo
  echo "[optional] DeAOT tracker"
  AOT_PATH="${ARTIFACTS_ROOT}/aot/R50_DeAOTL_PRE_YTB_DAV.pth"
  if [[ -f "${AOT_PATH}" ]]; then
    echo "✓ Already present: ${AOT_PATH}"
  else
    require_cmd gdown
    mkdir -p "$(dirname "${AOT_PATH}")"
    echo "→ Downloading DeAOT -> ${AOT_PATH}"
    gdown "https://drive.google.com/uc?id=1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ" -O "${AOT_PATH}"
  fi
fi

if [[ "${WITH_QWEN}" == "1" ]]; then
  echo
  echo "[optional] Qwen3-VL-4B-Instruct"
  hf_download_repo Qwen/Qwen3-VL-4B-Instruct \
    "${ARTIFACTS_ROOT}/qwen-vl/Qwen3-VL-4B-Instruct"
fi

if [[ "${WITH_ALTERNATE}" == "1" ]]; then
  require_cmd wget

  echo
  echo "[optional] UniDepth v2 ViT-L"
  hf_download_repo lpiccinelli/unidepth-v2-vitl14 \
    "${ARTIFACTS_ROOT}/unidepth-v2-vitl14"

  echo
  echo "[optional] Video Depth Anything"
  download_url \
    "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth" \
    "${ARTIFACTS_ROOT}/video-depth-anything/video_depth_anything_vits.pth"
  download_url \
    "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth" \
    "${ARTIFACTS_ROOT}/video-depth-anything/video_depth_anything_vitl.pth"

  echo
  echo "[optional] MoGe-2 ViT-L normal"
  hf_download_repo Ruicheng/moge-2-vitl-normal \
    "${ARTIFACTS_ROOT}/moge-2-vitl-normal"
fi

echo
echo "Scene weight download finished."
echo "Weights are under: ${ARTIFACTS_ROOT}"
if [[ "${WITH_AOT}" != "1" ]]; then
  echo "DeAOT was skipped; it will auto-download on first scene run if missing."
fi
echo "EgoSim-14B is still required separately for video generation (see README)."
