#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

DATASETS_ROOT="${CLOVER_DATASETS_ROOT:-datasets}"
DATASETS="${CLOVER_DATASETS:-all}"
DATASET_SOURCE="${CLOVER_DATASET_SOURCE:-huggingface}"
MODELSCOPE_CACHE_DIR="${CLOVER_MODELSCOPE_CACHE_DIR:-}"
OVERWRITE="${CLOVER_DATASET_OVERWRITE:-0}"
DOWNLOAD_OVERWRITE="${CLOVER_DOWNLOAD_OVERWRITE:-0}"
INCLUDE_TABLEBENCH_VISUALIZATION="${CLOVER_INCLUDE_TABLEBENCH_VISUALIZATION:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"

truthy() {
  local value
  value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${value}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

DOWNLOAD_ARGS=(
  -m benchmarks.download
  --datasets-root "${DATASETS_ROOT}"
  --dataset-source "${DATASET_SOURCE}"
)

if [[ -n "${MODELSCOPE_CACHE_DIR}" ]]; then
  DOWNLOAD_ARGS+=(--modelscope-cache-dir "${MODELSCOPE_CACHE_DIR}")
fi

IFS=',' read -r -a DATASET_ITEMS <<< "${DATASETS}"
for dataset in "${DATASET_ITEMS[@]}"; do
  dataset="${dataset#"${dataset%%[![:space:]]*}"}"
  dataset="${dataset%"${dataset##*[![:space:]]}"}"
  if [[ -n "${dataset}" ]]; then
    DOWNLOAD_ARGS+=(--dataset "${dataset}")
  fi
done

if truthy "${OVERWRITE}"; then
  DOWNLOAD_ARGS+=(--overwrite)
fi

if truthy "${DOWNLOAD_OVERWRITE}"; then
  DOWNLOAD_ARGS+=(--download-overwrite)
fi

if truthy "${INCLUDE_TABLEBENCH_VISUALIZATION}"; then
  DOWNLOAD_ARGS+=(--include-tablebench-visualization)
fi

PYTHONWARNINGS="ignore" \
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_CMD[@]}" "${DOWNLOAD_ARGS[@]}" "$@"
