#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: run_quicktest_mode.sh <recon_visualize|full> [extra args...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/project.yaml}"
ENV_PATHS_CONFIG="${ENV_PATHS_CONFIG:-${PROJECT_ROOT}/configs/env_paths.yaml}"
PARSER_PYTHON="${PARSER_PYTHON:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/${MODE}}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config does not exist: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${ENV_PATHS_CONFIG}" ]]; then
  echo "Env path config does not exist: ${ENV_PATHS_CONFIG}" >&2
  echo "Copy configs/env_paths.example.yaml to configs/env_paths.yaml and adjust it." >&2
  exit 1
fi

if ! command -v "${PARSER_PYTHON}" >/dev/null 2>&1; then
  echo "PARSER_PYTHON is not available on PATH: ${PARSER_PYTHON}" >&2
  exit 1
fi

LAUNCH_PYTHON="$(
  "${PARSER_PYTHON}" - "${ENV_PATHS_CONFIG}" <<'PY'
from pathlib import Path
import sys

config_path = Path(sys.argv[1])
current_section = None

for raw_line in config_path.read_text().splitlines():
    line = raw_line.split("#", 1)[0].rstrip()
    if not line.strip():
        continue

    if raw_line[:1].isspace():
        if current_section != "launch":
            continue
        stripped = line.strip()
        if not stripped.startswith("python:"):
            continue
        value = stripped.split(":", 1)[1].strip()
        if value[:1] in {'"', "'"} and value[-1:] == value[:1]:
            value = value[1:-1]
        print(value)
        break
    else:
        current_section = line[:-1].strip() if line.endswith(":") else None
PY
)"

if [[ -z "${LAUNCH_PYTHON}" ]]; then
  echo "launch.python is missing in ${ENV_PATHS_CONFIG}" >&2
  exit 1
fi

if [[ ! -x "${LAUNCH_PYTHON}" ]]; then
  echo "LAUNCH_PYTHON is not executable: ${LAUNCH_PYTHON}" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

CLI_ARGS=()
CLI_OUTPUT_DIR="${OUTPUT_DIR}"
while (($#)); do
  case "$1" in
    --output-dir)
      shift
      if (($#)); then
        CLI_OUTPUT_DIR="$1"
        shift
      else
        echo "--output-dir requires a value" >&2
        exit 1
      fi
      ;;
    --output-dir=*)
      CLI_OUTPUT_DIR="${1#--output-dir=}"
      shift
      ;;
    *)
      CLI_ARGS+=("$1")
      shift
      ;;
  esac
done

cd "${PROJECT_ROOT}"
"${LAUNCH_PYTHON}" -m egowm_incremental.cli print-command \
  --config "${CONFIG_PATH}" \
  --mode "${MODE}" \
  --output-dir "${CLI_OUTPUT_DIR}" \
  -- "${CLI_ARGS[@]}"

exec "${LAUNCH_PYTHON}" -m egowm_incremental.cli quicktest \
  --config "${CONFIG_PATH}" \
  --mode "${MODE}" \
  --output-dir "${CLI_OUTPUT_DIR}" \
  -- "${CLI_ARGS[@]}"
