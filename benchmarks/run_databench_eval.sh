#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

REMOTE_BATCH_SIZE="${CLOVER_REMOTE_BATCH_SIZE:-8}"
LOCAL_BATCH_SIZE="${CLOVER_LOCAL_BATCH_SIZE:-8}"
MAX_WORKERS="${CLOVER_EVAL_MAX_WORKERS:-5}"
REMOTE_LLM_CONFIG="${CLOVER_REMOTE_LLM_CONFIG:-${CLOVER_LLM_CONFIG:-${LLM_CONFIG:-config/remote_llm_config.json}}}"
LOCAL_SLM_CONFIG="${CLOVER_LOCAL_SLM_CONFIG:-${CLOVER_SLM_CONFIG:-${SLM_CONFIG:-config/local_slm_config.json}}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"

PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}" \
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_CMD[@]}" eval.py \
    --eval \
    --remote-llm-config "${REMOTE_LLM_CONFIG}" \
    --local-slm-config "${LOCAL_SLM_CONFIG}" \
    --remote-batch-size "${REMOTE_BATCH_SIZE}" \
    --local-batch-size "${LOCAL_BATCH_SIZE}" \
    --max-workers "${MAX_WORKERS}" \
    "$@"
