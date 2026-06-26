#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Orchestra baseline for CLOVER datasets.
#
# Edit the USER_* block for the server default run, or override any value with
# environment variables / command line arguments.
#
# Usage:
#   bash benchmarks/run_orchestra_baseline.sh [DATASET] [EDGE_MODEL_PATH] [runner options...]
#
# Examples:
#   bash benchmarks/run_orchestra_baseline.sh tablefact /root/autodl-tmp/models/Qwen2.5-3B-Instruct
#   CLOVER_ORCHESTRA_MODE=2agent bash benchmarks/run_orchestra_baseline.sh --max-cases 100
#
# Runner options after DATASET/MODEL are forwarded to
# benchmarks.baselines.table_agent_baselines, so they can override defaults below.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --------------------------- user-editable defaults ---------------------------
USER_DATASET="tablefact"                         # tablebench | wikitq | tablefact
# Server/virtual model path default. Override with CLOVER_EDGE_MODEL_PATH or
# the second positional argument when the deployed model path changes.
USER_EDGE_MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-3B-Instruct"
USER_ORCHESTRA_MODE="3agent"                    # 2agent | 3agent | both

# Case-level benchmark concurrency. Increase until the vLLM server is saturated.
USER_MAX_WORKERS="8"
USER_REPEAT_TIMES="1"
USER_MAX_ITERS="5"
USER_LINE_LIMIT="10"                            # N | inf

# Sampling / generation.
USER_TEMPERATURE="0.7"
USER_MAX_TOKENS="1024"

# vLLM endpoint and launch configuration. If the endpoint already responds, the
# common runner reuses it instead of starting a new server.
USER_VLLM_GPUS="0"
USER_VLLM_HOST="127.0.0.1"
USER_VLLM_PORT="8000"
USER_VLLM_SERVED_MODEL_NAME=""                  # empty = basename of model path
USER_VLLM_MAX_MODEL_LEN="16384"
USER_VLLM_GPU_MEMORY_UTILIZATION="0.88"
USER_VLLM_TENSOR_PARALLEL_SIZE=""               # empty = number of selected GPUs
USER_VLLM_MAX_NUM_SEQS="32"                     # vLLM sequence concurrency
USER_VLLM_DTYPE="auto"
USER_VLLM_ENABLE_PREFIX_CACHING="true"
USER_VLLM_WARMUP="true"
USER_VLLM_PERSIST_SERVER="false"
USER_VLLM_SERVER_ARGS=""                        # e.g. "--swap-space 4"

# Output / dataset choices.
USER_OUTPUT_ROOT="${REPO_ROOT}/benchmark/runs"
USER_RUN_NAME=""                                # empty = timestamped name
USER_OVERWRITE="false"
USER_WIKITQ_SPLIT="pristine-unseen-tables"
USER_TABLEFACT_SPLIT="test"
USER_TABLEFACT_SUBSET="small"
USER_BASELINE_TIMEOUT_SECONDS="1800"
# -----------------------------------------------------------------------------

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/run_orchestra_baseline.sh [DATASET] [EDGE_MODEL_PATH] [runner options...]

Main script/env configuration:
  CLOVER_BASELINE_DATASET       Overrides USER_DATASET.
  CLOVER_EDGE_MODEL_PATH        Overrides USER_EDGE_MODEL_PATH.
  CLOVER_ORCHESTRA_MODE         2agent | 3agent | both.
  CLOVER_MAX_WORKERS            Case-level concurrency.
  CLOVER_VLLM_GPUS              CUDA devices for vLLM, e.g. 0 or 0,1.
  CLOVER_VLLM_PORT              vLLM serving port.
  CLOVER_VLLM_MAX_MODEL_LEN     vLLM context length.
  CLOVER_VLLM_MAX_NUM_SEQS      vLLM sequence concurrency.
  CLOVER_BASELINE_TEMPERATURE   Sampling temperature.
  CLOVER_BASELINE_MAX_TOKENS    Max generated tokens per model call.

Examples:
  bash benchmarks/run_orchestra_baseline.sh tablefact /root/autodl-tmp/models/Qwen2.5-3B-Instruct --sample-size 500
  CLOVER_ORCHESTRA_MODE=both CLOVER_MAX_WORKERS=16 CLOVER_VLLM_MAX_NUM_SEQS=64 bash benchmarks/run_orchestra_baseline.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DATASET="${CLOVER_BASELINE_DATASET:-${USER_DATASET}}"
EDGE_MODEL_PATH="${CLOVER_EDGE_MODEL_PATH:-${USER_EDGE_MODEL_PATH}}"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  DATASET="$1"
  shift
fi
if [[ $# -gt 0 && "${1}" != -* ]]; then
  EDGE_MODEL_PATH="$1"
  shift
fi
EXTRA_ARGS=("$@")

normalize_orchestra_mode() {
  local raw="$1"
  local value
  value="$(
    printf '%s' "${raw}" \
    | tr '[:upper:]' '[:lower:]' \
    | tr '_' '-'
  )"
  case "${value}" in
    2agent|2-agent|two|two-agent) printf '%s' "2agent" ;;
    3agent|3-agent|three|three-agent) printf '%s' "3agent" ;;
    both) printf '%s' "both" ;;
    *)
      echo "Unsupported CLOVER_ORCHESTRA_MODE: ${raw}" >&2
      exit 2
      ;;
  esac
}

ORCHESTRA_MODE="$(normalize_orchestra_mode "${CLOVER_ORCHESTRA_MODE:-${USER_ORCHESTRA_MODE}}")"

MAX_WORKERS="${CLOVER_MAX_WORKERS:-${USER_MAX_WORKERS}}"
REPEAT_TIMES="${CLOVER_REPEAT_TIMES:-${USER_REPEAT_TIMES}}"
MAX_ITERS="${CLOVER_MAX_ITERS:-${USER_MAX_ITERS}}"
LINE_LIMIT="${CLOVER_LINE_LIMIT:-${USER_LINE_LIMIT}}"
OVERWRITE="${CLOVER_OVERWRITE:-${USER_OVERWRITE}}"

export CLOVER_VLLM_GPUS="${CLOVER_VLLM_GPUS:-${USER_VLLM_GPUS}}"
export CLOVER_VLLM_HOST="${CLOVER_VLLM_HOST:-${USER_VLLM_HOST}}"
export CLOVER_VLLM_PORT="${CLOVER_VLLM_PORT:-${USER_VLLM_PORT}}"
export CLOVER_VLLM_MAX_MODEL_LEN="${CLOVER_VLLM_MAX_MODEL_LEN:-${USER_VLLM_MAX_MODEL_LEN}}"
export CLOVER_VLLM_GPU_MEMORY_UTILIZATION="${CLOVER_VLLM_GPU_MEMORY_UTILIZATION:-${USER_VLLM_GPU_MEMORY_UTILIZATION}}"
export CLOVER_VLLM_MAX_NUM_SEQS="${CLOVER_VLLM_MAX_NUM_SEQS:-${USER_VLLM_MAX_NUM_SEQS}}"
export CLOVER_VLLM_DTYPE="${CLOVER_VLLM_DTYPE:-${USER_VLLM_DTYPE}}"
export CLOVER_VLLM_ENABLE_PREFIX_CACHING="${CLOVER_VLLM_ENABLE_PREFIX_CACHING:-${USER_VLLM_ENABLE_PREFIX_CACHING}}"
export CLOVER_VLLM_WARMUP="${CLOVER_VLLM_WARMUP:-${USER_VLLM_WARMUP}}"
export CLOVER_VLLM_PERSIST_SERVER="${CLOVER_VLLM_PERSIST_SERVER:-${USER_VLLM_PERSIST_SERVER}}"
export CLOVER_VLLM_SERVER_ARGS="${CLOVER_VLLM_SERVER_ARGS:-${USER_VLLM_SERVER_ARGS}}"
export CLOVER_BASELINE_TEMPERATURE="${CLOVER_BASELINE_TEMPERATURE:-${USER_TEMPERATURE}}"
export CLOVER_BASELINE_MAX_TOKENS="${CLOVER_BASELINE_MAX_TOKENS:-${USER_MAX_TOKENS}}"
export CLOVER_BASELINE_TIMEOUT_SECONDS="${CLOVER_BASELINE_TIMEOUT_SECONDS:-${USER_BASELINE_TIMEOUT_SECONDS}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${USER_OUTPUT_ROOT}}"
export WIKITQ_SPLIT="${WIKITQ_SPLIT:-${USER_WIKITQ_SPLIT}}"
export TABLEFACT_SPLIT="${TABLEFACT_SPLIT:-${USER_TABLEFACT_SPLIT}}"
export TABLEFACT_SUBSET="${TABLEFACT_SUBSET:-${USER_TABLEFACT_SUBSET}}"

apply_forwarded_runner_overrides_for_display() {
  local index arg value
  for ((index = 0; index < ${#EXTRA_ARGS[@]}; index++)); do
    arg="${EXTRA_ARGS[$index]}"
    case "${arg}" in
      --orchestra-mode)
        value="${EXTRA_ARGS[$((index + 1))]:-}"
        [[ -n "${value}" ]] && ORCHESTRA_MODE="$(normalize_orchestra_mode "${value}")"
        ;;
      --orchestra-mode=*)
        ORCHESTRA_MODE="$(normalize_orchestra_mode "${arg#*=}")"
        ;;
      --max-workers)
        value="${EXTRA_ARGS[$((index + 1))]:-}"
        [[ -n "${value}" ]] && MAX_WORKERS="${value}"
        ;;
      --max-workers=*) MAX_WORKERS="${arg#*=}" ;;
      --repeat-times)
        value="${EXTRA_ARGS[$((index + 1))]:-}"
        [[ -n "${value}" ]] && REPEAT_TIMES="${value}"
        ;;
      --repeat-times=*) REPEAT_TIMES="${arg#*=}" ;;
      --max-iters)
        value="${EXTRA_ARGS[$((index + 1))]:-}"
        [[ -n "${value}" ]] && MAX_ITERS="${value}"
        ;;
      --max-iters=*) MAX_ITERS="${arg#*=}" ;;
      --line-limit)
        value="${EXTRA_ARGS[$((index + 1))]:-}"
        [[ -n "${value}" ]] && LINE_LIMIT="${value}"
        ;;
      --line-limit=*) LINE_LIMIT="${arg#*=}" ;;
      --temperature)
        value="${EXTRA_ARGS[$((index + 1))]:-}"
        [[ -n "${value}" ]] && export CLOVER_BASELINE_TEMPERATURE="${value}"
        ;;
      --temperature=*) export CLOVER_BASELINE_TEMPERATURE="${arg#*=}" ;;
      --max-tokens)
        value="${EXTRA_ARGS[$((index + 1))]:-}"
        [[ -n "${value}" ]] && export CLOVER_BASELINE_MAX_TOKENS="${value}"
        ;;
      --max-tokens=*) export CLOVER_BASELINE_MAX_TOKENS="${arg#*=}" ;;
    esac
  done
}
apply_forwarded_runner_overrides_for_display

if [[ -z "${CLOVER_VLLM_TENSOR_PARALLEL_SIZE:-}" && -n "${USER_VLLM_TENSOR_PARALLEL_SIZE}" ]]; then
  export CLOVER_VLLM_TENSOR_PARALLEL_SIZE="${USER_VLLM_TENSOR_PARALLEL_SIZE}"
fi
if [[ -z "${CLOVER_VLLM_SERVED_MODEL_NAME:-}" && -n "${USER_VLLM_SERVED_MODEL_NAME}" ]]; then
  export CLOVER_VLLM_SERVED_MODEL_NAME="${USER_VLLM_SERVED_MODEL_NAME}"
fi
if [[ -z "${RUN_NAME:-}" && -n "${USER_RUN_NAME}" ]]; then
  export RUN_NAME="${USER_RUN_NAME}"
fi

RUNNER_ARGS=(
  "${DATASET}"
  "${EDGE_MODEL_PATH}"
  --orchestra-mode "${ORCHESTRA_MODE}"
  --max-workers "${MAX_WORKERS}"
  --repeat-times "${REPEAT_TIMES}"
  --max-iters "${MAX_ITERS}"
  --line-limit "${LINE_LIMIT}"
)
case "$(printf '%s' "${OVERWRITE}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on) RUNNER_ARGS+=(--overwrite) ;;
  0|false|no|n|off) ;;
  *)
    echo "Invalid CLOVER_OVERWRITE/USER_OVERWRITE: ${OVERWRITE}" >&2
    exit 2
    ;;
esac

echo "Orchestra baseline config" >&2
echo "  dataset: ${DATASET}" >&2
echo "  model: ${EDGE_MODEL_PATH}" >&2
echo "  mode: ${ORCHESTRA_MODE}" >&2
echo "  case workers: ${MAX_WORKERS}" >&2
echo "  vLLM GPUs: ${CLOVER_VLLM_GPUS}" >&2
echo "  vLLM endpoint: ${CLOVER_VLLM_HOST}:${CLOVER_VLLM_PORT}" >&2
echo "  vLLM max model len: ${CLOVER_VLLM_MAX_MODEL_LEN}" >&2
echo "  vLLM max num seqs: ${CLOVER_VLLM_MAX_NUM_SEQS}" >&2
echo "  temperature: ${CLOVER_BASELINE_TEMPERATURE}" >&2

exec bash "${SCRIPT_DIR}/run_table_agent_baseline.sh" orchestra \
  "${RUNNER_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
