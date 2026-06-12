#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

REMOTE_BATCH_SIZE="${CLOVER_REMOTE_BATCH_SIZE:-128}"
REMOTE_CONCURRENCY="${CLOVER_REMOTE_CONCURRENCY:-128}"
SLM_SCHEDULER="${CLOVER_SLM_SCHEDULER:-tptt}"
MAX_PARALLEL_EXECUTION_UNITS="${CLOVER_MAX_PARALLEL_EXECUTION_UNITS:-64}"
MAX_PARALLEL_SLM_NODE_JOBS="${CLOVER_MAX_PARALLEL_SLM_NODE_JOBS:-64}"
MAX_PARALLEL_SLM_SEQUENCES="${CLOVER_MAX_PARALLEL_SLM_SEQUENCES:-512}"
MAX_PENDING_SLM_SEQUENCES="${CLOVER_MAX_PENDING_SLM_SEQUENCES:-2048}"
MAX_TPTT_LEAF_SEQUENCES_PER_TREE="${CLOVER_MAX_TPTT_LEAF_SEQUENCES_PER_TREE:-}"
TPTT_COALESCE_MS="${CLOVER_TPTT_COALESCE_MS:-}"
TPTT_PREFIX_TOKENS="${CLOVER_TPTT_PREFIX_TOKENS:-}"
VALIDATION_MODE="${CLOVER_VALIDATION_MODE:-none}"
MAX_RETRIES="${CLOVER_MAX_RETRIES:-1}"
MAX_WORKERS="${CLOVER_EVAL_CONCURRENCY:-${CLOVER_EVAL_MAX_WORKERS:-64}}"
REMOTE_LLM_CONFIG="${CLOVER_REMOTE_LLM_CONFIG:-${CLOVER_LLM_CONFIG:-${LLM_CONFIG:-model_config/deepseek_remote_llm_config.json}}}"
LOCAL_SLM_CONFIG="${CLOVER_LOCAL_SLM_CONFIG:-${CLOVER_SLM_CONFIG:-${SLM_CONFIG:-model_config/qwen25_3b_instruct_local_slm_config.json}}}"
SYNTHESIZE_LLM_CONFIG="${CLOVER_SYNTHESIZE_LLM_CONFIG:-${LOCAL_SLM_CONFIG}}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"

ENERGY_PROFILE="${CLOVER_ENERGY_PROFILE:-}"
ENERGY_ARGS=()
if [ "${ENERGY_PROFILE}" = "true" ]; then
  ENERGY_ARGS+=(--energy-profile)
  ENERGY_ARGS+=(--energy-sample-ms "${CLOVER_ENERGY_SAMPLE_MS:-500}")
  ENERGY_ARGS+=(--energy-baseline-seconds "${CLOVER_ENERGY_BASELINE_SECONDS:-0.0}")
fi

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
    --tablebench-eval \
    --remote-llm-config "${REMOTE_LLM_CONFIG}" \
    --local-slm-config "${LOCAL_SLM_CONFIG}" \
    --synthesize-llm-config "${SYNTHESIZE_LLM_CONFIG}" \
    --remote-batch-size "${REMOTE_BATCH_SIZE}" \
    --remote-concurrency "${REMOTE_CONCURRENCY}" \
    --slm-scheduler "${SLM_SCHEDULER}" \
    --max-parallel-execution-units "${MAX_PARALLEL_EXECUTION_UNITS}" \
    --max-parallel-slm-node-jobs "${MAX_PARALLEL_SLM_NODE_JOBS}" \
    --max-parallel-slm-sequences "${MAX_PARALLEL_SLM_SEQUENCES}" \
    --max-pending-slm-sequences "${MAX_PENDING_SLM_SEQUENCES}" \
    "${SCHEDULER_ARGS[@]}" \
    --validation-mode "${VALIDATION_MODE}" \
    --max-retries "${MAX_RETRIES}" \
    --max-workers "${MAX_WORKERS}" \
    "${ENERGY_ARGS[@]}" \
    "$@"
