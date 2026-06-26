#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCENE_ENV_NAME="${SCENE_ENV_NAME:-egosim-scene}"
USE_LOCKS="${USE_LOCKS:-0}"
DIFFSYNTH_PATH="${DIFFSYNTH_PATH:-}"
INSTALL_MOGE="${INSTALL_MOGE:-0}"
CONDA_CHANNELS="${CONDA_CHANNELS:-}"
EXPECTED_SCENE_CUDA="${EXPECTED_SCENE_CUDA:-12.8}"
SCENE_TORCH_VERSION="${SCENE_TORCH_VERSION:-2.7.0+cu128}"
SCENE_TORCHVISION_VERSION="${SCENE_TORCHVISION_VERSION:-0.22.0+cu128}"
SCENE_TORCH_INDEX_URL="${SCENE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
BUILD_TORCH_CUDA_ARCH_LIST="${BUILD_TORCH_CUDA_ARCH_LIST:-9.0}"
SCENE_BACKEND_EGOSIM_STATE_ROOT="${PROJECT_ROOT}/scene_backend/egosim_state"
SCENE_DEP_SAM3_DIR="${SCENE_BACKEND_EGOSIM_STATE_ROOT}/sam3"
SCENE_DEP_DA3_DIR="${SCENE_BACKEND_EGOSIM_STATE_ROOT}/Depth-Anything-3"
SAM3_REPO_URL="${SAM3_REPO_URL:-https://github.com/facebookresearch/sam3.git}"
DA3_REPO_URL="${DA3_REPO_URL:-https://github.com/ByteDance-Seed/Depth-Anything-3.git}"
TEMP_CONDARC=""

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is required but was not found in PATH."
  exit 1
fi

if [[ -z "${DIFFSYNTH_PATH}" ]]; then
  DEFAULT_DIFFSYNTH_PATH="$(cd "${PROJECT_ROOT}/../../.." && pwd)/diffsynth"
  if [[ -d "${DEFAULT_DIFFSYNTH_PATH}" ]]; then
    DIFFSYNTH_PATH="${DEFAULT_DIFFSYNTH_PATH}"
  fi
fi

if [[ -n "${DIFFSYNTH_PATH}" ]]; then
  if [[ -f "${DIFFSYNTH_PATH}/__init__.py" ]]; then
    :
  elif [[ -f "${DIFFSYNTH_PATH}/diffsynth/__init__.py" ]]; then
    :
  else
    echo "ERROR: DIFFSYNTH_PATH must point to the diffsynth package dir or its parent root: ${DIFFSYNTH_PATH}"
    exit 1
  fi
fi

cleanup() {
  if [[ -n "${TEMP_CONDARC}" && -f "${TEMP_CONDARC}" ]]; then
    rm -f "${TEMP_CONDARC}"
  fi
}

trap cleanup EXIT

prepare_condarc() {
  if [[ -z "${CONDA_CHANNELS}" ]]; then
    return
  fi

  TEMP_CONDARC="$(mktemp)"
  {
    echo "channels:"
    local normalized_channels="${CONDA_CHANNELS//,/ }"
    local channel
    for channel in ${normalized_channels}; do
      echo "  - ${channel}"
    done
  } > "${TEMP_CONDARC}"
}

run_conda() {
  if [[ -n "${TEMP_CONDARC}" ]]; then
    CONDARC="${TEMP_CONDARC}" conda "$@"
  else
    conda "$@"
  fi
}

prepare_condarc

create_env_from_yaml() {
  local env_name="$1"
  local yaml_path="$2"

  if conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
    run_conda env update -n "${env_name}" -f "${yaml_path}" --prune
  else
    run_conda env create -n "${env_name}" -f "${yaml_path}"
  fi
}

create_env_from_lock() {
  local env_name="$1"
  local lock_path="$2"
  if conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
    conda remove -y -n "${env_name}" --all
  fi
  run_conda create -y -n "${env_name}" --file "${lock_path}"
}

require_user_cloned_scene_dependency() {
  local dir="$1"
  local label="$2"
  local repo_url="$3"

  if [[ -d "${dir}" && -f "${dir}/pyproject.toml" ]]; then
    return 0
  fi

  cat <<EOF

ERROR: ${label} source tree not found at:
  ${dir}

This dependency is not bundled with EgoSim. Clone it to the path above, then re-run setup:

  git clone ${repo_url} "${dir}"

See continuous_simulation/README.md (Installation) for details.

EOF
  exit 1
}

require_user_cloned_scene_dependencies() {
  require_user_cloned_scene_dependency "${SCENE_DEP_SAM3_DIR}" "SAM3" "${SAM3_REPO_URL}"
  require_user_cloned_scene_dependency "${SCENE_DEP_DA3_DIR}" "Depth-Anything-3" "${DA3_REPO_URL}"
}

build_and_install_local_package() {
  local env_name="$1"
  local package_dir="$2"
  if [[ ! -d "${package_dir}" ]]; then
    echo "ERROR: local package directory not found: ${package_dir}"
    exit 1
  fi
  if [[ ! -f "${package_dir}/pyproject.toml" ]]; then
    echo "ERROR: ${package_dir} does not look like a Python package (missing pyproject.toml)."
    exit 1
  fi

  rm -rf "${package_dir}/dist"
  mkdir -p "${package_dir}/dist"
  TORCH_CUDA_ARCH_LIST="${BUILD_TORCH_CUDA_ARCH_LIST}" \
    conda run -n "${env_name}" python -m build --wheel --no-isolation --outdir "${package_dir}/dist" "${package_dir}"
  conda run -n "${env_name}" pip install --no-deps --force-reinstall "${package_dir}"/dist/*.whl
}

install_scene_backend_requirements() {
  local requirements_path="${SCENE_BACKEND_EGOSIM_STATE_ROOT}/envs/requirements.txt"
  local filtered_requirements
  filtered_requirements="$(mktemp)"

  python - "${requirements_path}" "${filtered_requirements}" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
skip_prefixes = ("torch==", "torchvision==")

with src.open("r", encoding="utf-8") as infile, dst.open("w", encoding="utf-8") as outfile:
    for line in infile:
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in skip_prefixes):
            continue
        outfile.write(line)
PY

  conda run -n "${SCENE_ENV_NAME}" pip install -r "${filtered_requirements}"
  rm -f "${filtered_requirements}"
}

install_scene_torch_cuda() {
  conda run -n "${SCENE_ENV_NAME}" pip install --no-deps --force-reinstall \
    --index-url "${SCENE_TORCH_INDEX_URL}" \
    "torch==${SCENE_TORCH_VERSION}" \
    "torchvision==${SCENE_TORCHVISION_VERSION}"
}

verify_scene_torch_cuda() {
  conda run -n "${SCENE_ENV_NAME}" python - "${EXPECTED_SCENE_CUDA}" <<'PY'
import sys

expected_cuda = sys.argv[1]

try:
    import torch
except ImportError as exc:
    raise SystemExit(f"Scene env is missing torch: {exc}")

actual_cuda = torch.version.cuda
if actual_cuda != expected_cuda:
    raise SystemExit(
        f"Scene env torch CUDA mismatch: expected {expected_cuda}, got {actual_cuda}. "
        "This usually means a pip install upgraded torch/torchvision to a different CUDA build."
    )

print(f"Verified scene torch {torch.__version__} with CUDA {actual_cuda}")
PY
}

verify_scene_runtime_imports() {
  conda run -n "${SCENE_ENV_NAME}" python - <<'PY'
import importlib

modules = [
    "pycocotools.mask",
    "decord",
    "addict",
    "evo.core.trajectory",
    "sam3.model_builder",
    "depth_anything_3.api",
    "egosim_state.pipeline.default",
    "transformers.models.qwen3_vl",
]

failures = []
for module_name in modules:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - setup validation
        failures.append(f"{module_name}: {exc}")

if failures:
    raise SystemExit(
        "Scene runtime import verification failed:\n  - " + "\n  - ".join(failures)
    )

print("Verified scene runtime imports for egosim_state, sam3, and Depth-Anything-3")
PY
}

require_user_cloned_scene_dependencies

if [[ "${USE_LOCKS}" == "1" ]]; then
  create_env_from_lock "${SCENE_ENV_NAME}" "${PROJECT_ROOT}/envs/locks/scene-linux-64.lock"
else
  create_env_from_yaml "${SCENE_ENV_NAME}" "${PROJECT_ROOT}/envs/conda/scene.yml"
fi

install_scene_torch_cuda
verify_scene_torch_cuda
conda run -n "${SCENE_ENV_NAME}" pip install -r "${PROJECT_ROOT}/envs/pip/scene.txt"
install_scene_backend_requirements
conda run -n "${SCENE_ENV_NAME}" pip install -e "${PROJECT_ROOT}"
verify_scene_torch_cuda

build_and_install_local_package "${SCENE_ENV_NAME}" "${SCENE_DEP_SAM3_DIR}"
verify_scene_torch_cuda
build_and_install_local_package "${SCENE_ENV_NAME}" "${SCENE_DEP_DA3_DIR}"
verify_scene_torch_cuda
if [[ "${INSTALL_MOGE}" == "1" ]]; then
  build_and_install_local_package "${SCENE_ENV_NAME}" "${SCENE_BACKEND_EGOSIM_STATE_ROOT}/MoGe"
  verify_scene_torch_cuda
fi
build_and_install_local_package "${SCENE_ENV_NAME}" "${SCENE_BACKEND_EGOSIM_STATE_ROOT}"
verify_scene_runtime_imports

echo "Environments are ready."
echo "  generation: use the Python environment installed by the main EgoSim repo"
echo "    (configure it in configs/project.yaml under runtime.python.wan)"
if [[ -n "${DIFFSYNTH_PATH}" ]]; then
  echo "  diffsynth runtime root: ${DIFFSYNTH_PATH}"
  echo "    (loaded via PYTHONPATH at runtime, not pip-installed)"
fi
echo "  scene: ${SCENE_ENV_NAME}"
echo "Installed scene backend CLI:"
echo "  egosim_state infer ..."
echo "  egosim_state visualize ..."
