#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

MAX_WORKERS="${CLOVER_EVAL_CONCURRENCY:-${CLOVER_EVAL_MAX_WORKERS:-5}}"
REMOTE_LLM_CONFIG="${CLOVER_REMOTE_LLM_CONFIG:-${CLOVER_LLM_CONFIG:-${LLM_CONFIG:-model_config/remote_llm_config.json}}}"
TABLEBENCH_INSTRUCTION_TYPE="${CLOVER_TABLEBENCH_INSTRUCTION_TYPE:-DP}"
EXECUTION_TIMEOUT_SECONDS="${CLOVER_EXECUTION_TIMEOUT_SECONDS:-20}"
PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"

PYTHONWARNINGS="ignore" \
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_CMD[@]}" -m benchmarks.eval \
    --tablebench-remote-only-baseline \
    --remote-llm-config "${REMOTE_LLM_CONFIG}" \
    --max-workers "${MAX_WORKERS}" \
    --tablebench-instruction-type "${TABLEBENCH_INSTRUCTION_TYPE}" \
    --execution-timeout-seconds "${EXECUTION_TIMEOUT_SECONDS}" \
    "$@"
