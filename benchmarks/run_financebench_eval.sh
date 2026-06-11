#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

MAX_PARALLEL_EXECUTION_UNITS="${CLOVER_MAX_PARALLEL_EXECUTION_UNITS:-64}"
SLM_SCHEDULER="${CLOVER_SLM_SCHEDULER:-tptt}"
MAX_PARALLEL_SLM_NODE_JOBS="${CLOVER_MAX_PARALLEL_SLM_NODE_JOBS:-64}"
MAX_PARALLEL_SLM_SEQUENCES="${CLOVER_MAX_PARALLEL_SLM_SEQUENCES:-64}"
MAX_PENDING_SLM_SEQUENCES="${CLOVER_MAX_PENDING_SLM_SEQUENCES:-1024}"
MAX_TPTT_LEAF_SEQUENCES_PER_TREE="${CLOVER_MAX_TPTT_LEAF_SEQUENCES_PER_TREE:-}"
TPTT_COALESCE_MS="${CLOVER_TPTT_COALESCE_MS:-}"
TPTT_PREFIX_TOKENS="${CLOVER_TPTT_PREFIX_TOKENS:-}"
MAX_RETRIES="${CLOVER_MAX_RETRIES:-1}"
MAX_WORKERS="${CLOVER_EVAL_CONCURRENCY:-${CLOVER_EVAL_MAX_WORKERS:-64}}"
REMOTE_LLM_CONFIG="${CLOVER_REMOTE_LLM_CONFIG:-${CLOVER_LLM_CONFIG:-${LLM_CONFIG:-model_config/remote_llm_config.json}}}"
LOCAL_SLM_CONFIG="${CLOVER_LOCAL_SLM_CONFIG:-${CLOVER_SLM_CONFIG:-${SLM_CONFIG:-model_config/local_slm_config.json}}}"
FINANCEBENCH_EXAMPLES_ROOT="${FINANCEBENCH_EXAMPLES_ROOT:-${REPO_ROOT}/datasets/financebench}"
PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"

SCHEDULER_ARGS=()
if [[ -n "${MAX_TPTT_LEAF_SEQUENCES_PER_TREE}" ]]; then
  SCHEDULER_ARGS+=(--max-tptt-leaf-sequences-per-tree "${MAX_TPTT_LEAF_SEQUENCES_PER_TREE}")
fi
if [[ -n "${TPTT_COALESCE_MS}" ]]; then
  SCHEDULER_ARGS+=(--tptt-coalesce-ms "${TPTT_COALESCE_MS}")
fi
if [[ -n "${TPTT_PREFIX_TOKENS}" ]]; then
  SCHEDULER_ARGS+=(--tptt-prefix-tokens "${TPTT_PREFIX_TOKENS}")
fi

PYTHONWARNINGS="ignore" \
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_CMD[@]}" -m benchmarks.eval \
    --financebench-eval \
    --remote-llm-config "${REMOTE_LLM_CONFIG}" \
    --local-slm-config "${LOCAL_SLM_CONFIG}" \
    --financebench-examples-root "${FINANCEBENCH_EXAMPLES_ROOT}" \
    --max-workers "${MAX_WORKERS}" \
    --max-retries "${MAX_RETRIES}" \
    --slm-scheduler "${SLM_SCHEDULER}" \
    --max-parallel-execution-units "${MAX_PARALLEL_EXECUTION_UNITS}" \
    --max-parallel-slm-node-jobs "${MAX_PARALLEL_SLM_NODE_JOBS}" \
    --max-parallel-slm-sequences "${MAX_PARALLEL_SLM_SEQUENCES}" \
    --max-pending-slm-sequences "${MAX_PENDING_SLM_SEQUENCES}" \
    "${SCHEDULER_ARGS[@]}" \
    "$@"
