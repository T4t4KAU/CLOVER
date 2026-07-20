#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Local vLLM launcher for CLOVER.
#
# Starts an OpenAI-compatible vLLM service and runs the selected evaluation.
# No model_config JSON files are required: runtime configs are generated from
# the settings below. The paper configuration uses the same checkpoint for the
# node-local and global roles, and the launcher reuses one server by default.
#
# EDGE1 and EDGE2 are legacy configuration labels. They denote restricted-
# context local repair and broader-context planning/replanning, respectively;
# they do not denote edge/cloud deployment or different model sizes.
#
# Demo: run the paper-style local configuration:
#   PYTHON_BIN=/root/miniconda3/envs/clover/bin/python \
#   CLOVER_EDGE1_MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-14B-Instruct \
#   CLOVER_EDGE1_GPUS=0 \
#   CLOVER_EDGE1_PORT=8000 \
#   CLOVER_EDGE1_MAX_MODEL_LEN=8192 \
#   CLOVER_DISABLE_THINKING=true \
#   CLOVER_EVAL_CONCURRENCY=16 \
#   CLOVER_EDGE2_CONCURRENCY=8 \
#   /Users/huangwenxuan/Documents/codes/CLOVER/benchmarks/run_vllm_eval_clover.sh tablebench --max-cases 100
#
# Usage:
#   bash benchmarks/run_vllm_eval_clover.sh [DATASET] [eval options...]
#
# Examples:
#   bash benchmarks/run_vllm_eval_clover.sh wikitq --max-cases 20
#   bash benchmarks/run_vllm_eval_clover.sh tablebench
#   MMQA_SPLIT=two_table bash benchmarks/run_vllm_eval_clover.sh mmqa --sample-size 100
# =============================================================================

# =============================================================================
# User settings
# Edit these values, then run: bash benchmarks/run_vllm_eval_clover.sh
# Every variable can also be overridden via environment variables (CLOVER_*).
# =============================================================================

# --- Dataset ---
DATASET="${CLOVER_EVAL_DATASET:-tablebench}"  # tablebench | wikitq | tablefact | mmqa

# --- Node-local role (legacy label: EDGE1) ---------------------------------
EDGE1_MODEL_PATH="${CLOVER_EDGE1_MODEL_PATH:-${CLOVER_EDGE_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-14B-Instruct}}"
EDGE1_GPUS="${CLOVER_EDGE1_GPUS:-${CLOVER_EDGE_GPUS:-0}}"  # comma-separated GPU ids, e.g. 0 or 0,1
EDGE1_GPU_MEM_UTIL="${CLOVER_EDGE1_GPU_MEM_UTIL:-${CLOVER_EDGE_GPU_MEM_UTIL:-}}"
EDGE1_PORT="${CLOVER_EDGE1_PORT:-${CLOVER_EDGE_PORT:-8000}}"
EDGE1_MAX_MODEL_LEN="${CLOVER_EDGE1_MAX_MODEL_LEN:-${CLOVER_EDGE_MAX_MODEL_LEN:-8192}}"
EDGE1_MAX_TOKENS="${CLOVER_EDGE1_MAX_TOKENS:-${CLOVER_EDGE_MAX_TOKENS:-3072}}"
EDGE1_TIMEOUT="${CLOVER_EDGE1_TIMEOUT:-${CLOVER_EDGE_TIMEOUT:-600}}"
EDGE1_DTYPE="${CLOVER_EDGE1_DTYPE:-${CLOVER_EDGE_DTYPE:-auto}}"
EDGE1_TENSOR_PARALLEL_SIZE="${CLOVER_EDGE1_TENSOR_PARALLEL_SIZE:-${CLOVER_EDGE_TENSOR_PARALLEL_SIZE:-}}"  # defaults to GPU count
EDGE1_TEMPERATURE="${CLOVER_EDGE1_TEMPERATURE:-0.3}"
EDGE1_TOP_P="${CLOVER_EDGE1_TOP_P:-1.0}"

# --- Global planning/replanning role (legacy label: EDGE2) -----------------
EDGE2_MODEL_PATH="${CLOVER_EDGE2_MODEL_PATH:-${CLOVER_EDGE_MODEL_PATH:-${EDGE1_MODEL_PATH}}}"
EDGE2_GPUS="${CLOVER_EDGE2_GPUS:-${CLOVER_EDGE_GPUS:-0}}"
EDGE2_GPU_MEM_UTIL="${CLOVER_EDGE2_GPU_MEM_UTIL:-${CLOVER_EDGE_GPU_MEM_UTIL:-}}"
EDGE2_PORT="${CLOVER_EDGE2_PORT:-8001}"
EDGE2_MAX_MODEL_LEN="${CLOVER_EDGE2_MAX_MODEL_LEN:-${CLOVER_EDGE_MAX_MODEL_LEN:-8192}}"
EDGE2_MAX_TOKENS="${CLOVER_EDGE2_MAX_TOKENS:-${CLOVER_EDGE_MAX_TOKENS:-3072}}"
EDGE2_TIMEOUT="${CLOVER_EDGE2_TIMEOUT:-${CLOVER_EDGE_TIMEOUT:-600}}"
EDGE2_DTYPE="${CLOVER_EDGE2_DTYPE:-${CLOVER_EDGE_DTYPE:-auto}}"
EDGE2_TENSOR_PARALLEL_SIZE="${CLOVER_EDGE2_TENSOR_PARALLEL_SIZE:-${CLOVER_EDGE_TENSOR_PARALLEL_SIZE:-}}"  # defaults to GPU count
EDGE2_TEMPERATURE="${CLOVER_EDGE2_TEMPERATURE:-0.3}"
EDGE2_TOP_P="${CLOVER_EDGE2_TOP_P:-0.9}"
FORCE_SEPARATE_EDGE_SERVERS="${CLOVER_FORCE_SEPARATE_EDGE_SERVERS:-false}"

# --- Common vLLM server ---
HOST="${CLOVER_VLLM_HOST:-127.0.0.1}"
ENABLE_PREFIX_CACHING="${CLOVER_VLLM_ENABLE_PREFIX_CACHING:-true}"
VLLM_MAX_NUM_SEQS="${CLOVER_VLLM_MAX_NUM_SEQS:-256}"
VLLM_SERVER_ARGS="${CLOVER_VLLM_SERVER_ARGS:-}"   # extra args, e.g. "--enforce-eager"
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-${CLOVER_VLLM_USE_FLASHINFER_SAMPLER:-0}}"
SERVER_READY_TIMEOUT="${CLOVER_VLLM_READY_TIMEOUT:-600}"
PERSIST_SERVER="${CLOVER_VLLM_PERSIST_SERVER:-false}"
WARMUP_SERVER="${CLOVER_VLLM_WARMUP:-true}"
DISABLE_THINKING="${CLOVER_DISABLE_THINKING:-false}"  # Qwen3/Qwen3.6: extra_body.chat_template_kwargs.enable_thinking=false

# --- Concurrency & retries ---
EDGE2_BATCH_SIZE="${CLOVER_EDGE2_BATCH_SIZE:-1}"               # query batch size; keep 1 to avoid batch-prompt errors
EDGE2_CONCURRENCY="${CLOVER_EDGE2_CONCURRENCY:-8}"             # concurrent EDGE2 requests
EVAL_CONCURRENCY="${CLOVER_EVAL_CONCURRENCY:-16}"               # overall eval workers (--max-workers)
MAX_PARALLEL_EXECUTION_UNITS="${CLOVER_MAX_PARALLEL_EXECUTION_UNITS:-8}"
MAX_PARALLEL_SLM_NODE_JOBS="${CLOVER_MAX_PARALLEL_SLM_NODE_JOBS:-32}"
MAX_PARALLEL_SLM_SEQUENCES="${CLOVER_MAX_PARALLEL_SLM_SEQUENCES:-32}"
MAX_PENDING_SLM_SEQUENCES="${CLOVER_MAX_PENDING_SLM_SEQUENCES:-64}"
SLM_SCHEDULER="${CLOVER_SLM_SCHEDULER:-tptt}"                  # tptt | fifo

# --- Agent retry budgets ---
AGENT_LOOP_MAX_ITERATIONS="${CLOVER_AGENT_LOOP_MAX_ITERATIONS:-4}"  # edge agent max iterations
EDGE2_MAX_RETRIES="${CLOVER_EDGE2_MAX_RETRIES:-3}"                 # EDGE2 agent max retries (--max-retries)
VALIDATION_MODE="${CLOVER_VALIDATION_MODE:-remote_supervisor}"     # remote_supervisor enables global repair; set none to disable

# --- TableFact direct verifier ---
# Only consumed by the TableFact evaluator. It keeps the full binary fact
# check in one deterministic vLLM call to preserve same-row evidence.
TABLEFACT_DIRECT_VERIFIER="${CLOVER_TABLEFACT_DIRECT_VERIFIER:-true}"
TABLEFACT_DIRECT_MAX_TOKENS="${CLOVER_TABLEFACT_DIRECT_MAX_TOKENS:-1024}"
TABLEFACT_DIRECT_TABLE_CHAR_LIMIT="${CLOVER_TABLEFACT_DIRECT_TABLE_CHAR_LIMIT:-24000}"
TABLEFACT_DIRECT_TEMPERATURE="${CLOVER_TABLEFACT_DIRECT_TEMPERATURE:-0.0}"
TABLEFACT_DIRECT_TOP_P="${CLOVER_TABLEFACT_DIRECT_TOP_P:-1.0}"
TABLEFACT_SECOND_PASS_VERIFIER="${CLOVER_TABLEFACT_SECOND_PASS_VERIFIER:-true}"
TABLEFACT_SECOND_PASS_MAX_TOKENS="${CLOVER_TABLEFACT_SECOND_PASS_MAX_TOKENS:-1024}"

# --- Generic table direct semantic probe ---
# Integrated into Supervisor synthesis as advisory evidence for answer/replan
# decisions. It is not a dataset-specific post-processing override.
TABLE_DIRECT_PROBE="${CLOVER_TABLE_DIRECT_PROBE:-true}"
TABLE_DIRECT_PROBE_MAX_TOKENS="${CLOVER_TABLE_DIRECT_PROBE_MAX_TOKENS:-384}"
TABLE_DIRECT_PROBE_TABLE_CHAR_LIMIT="${CLOVER_TABLE_DIRECT_PROBE_TABLE_CHAR_LIMIT:-20000}"
TABLE_DIRECT_PROBE_TEMPERATURE="${CLOVER_TABLE_DIRECT_PROBE_TEMPERATURE:-0.0}"
TABLE_DIRECT_PROBE_TOP_P="${CLOVER_TABLE_DIRECT_PROBE_TOP_P:-1.0}"

# --- Edge review / ablation ---
EDGE_REVIEW_MODE="${CLOVER_EDGE_REVIEW_MODE:-safe}"               # off | shadow | safe
EDGE_REVIEW_PROACTIVE="${CLOVER_EDGE_REVIEW_PROACTIVE:-true}"
ABLATION_VARIANT="${CLOVER_ABLATION_VARIANT:-full}"
ENABLE_EDGE_AGENT="${CLOVER_ENABLE_EDGE_AGENT:-true}"
ENABLE_EDGE_REPAIR="${CLOVER_ENABLE_EDGE_REPAIR:-${ENABLE_EDGE_AGENT}}"
ENABLE_TERMINAL_EDGE_REVIEW="${CLOVER_ENABLE_TERMINAL_EDGE_REVIEW:-${ENABLE_EDGE_AGENT}}"
ENABLE_CONTRACT_GATE="${CLOVER_ENABLE_CONTRACT_GATE:-true}"
ENABLE_NODE_REVIEW="${CLOVER_ENABLE_NODE_REVIEW:-true}"
ENABLE_CLOUD_RECOVERY="${CLOVER_ENABLE_CLOUD_RECOVERY:-true}"
ENABLE_CLOUD_REPLAN="${CLOVER_ENABLE_CLOUD_REPLAN:-${ENABLE_CLOUD_RECOVERY}}"
ENABLE_CLOUD_SYNTHESIS="${CLOVER_ENABLE_CLOUD_SYNTHESIS:-${ENABLE_CLOUD_RECOVERY}}"
ENABLE_STATIC_FAST_PATH="${CLOVER_ENABLE_STATIC_FAST_PATH:-true}"
ENABLE_STATIC_FINALIZATION="${CLOVER_ENABLE_STATIC_FINALIZATION:-true}"
ENABLE_OBSERVABLE_CLOSURE_CHECKER="${CLOVER_ENABLE_OBSERVABLE_CLOSURE_CHECKER:-true}"

# =============================================================================
# Internals (no user-serviceable parts below)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python 2>/dev/null || true)}"

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/run_vllm_eval_clover.sh [DATASET] [eval options...]

Starts local vLLM service(s) and runs the selected evaluation. The paper
configuration uses the same checkpoint for the local and global roles.
All model/server/concurrency settings are configured at the top of this script
or via CLOVER_* environment variables.

DATASET:
  tablebench | wikitq | tablefact | mmqa

Examples:
  bash benchmarks/run_vllm_eval_clover.sh
  bash benchmarks/run_vllm_eval_clover.sh wikitq --max-cases 20
  bash benchmarks/run_vllm_eval_clover.sh tablebench
  MMQA_SPLIT=three_table bash benchmarks/run_vllm_eval_clover.sh mmqa --sample-size 100

Environment variables (key ones):
  CLOVER_EDGE1_MODEL_PATH / CLOVER_EDGE2_MODEL_PATH    Model paths
  CLOVER_EDGE1_GPUS / CLOVER_EDGE2_GPUS                 GPU ids
  CLOVER_EDGE1_PORT / CLOVER_EDGE2_PORT                 vLLM ports
  CLOVER_EDGE1_MAX_MODEL_LEN / CLOVER_EDGE2_MAX_MODEL_LEN Context length
  CLOVER_VLLM_MAX_NUM_SEQS                            vLLM sequence concurrency
  CLOVER_EDGE1_TEMPERATURE / CLOVER_EDGE2_TEMPERATURE   Sampling temperature
  CLOVER_EVAL_CONCURRENCY / CLOVER_EDGE2_CONCURRENCY Eval parallelism
  CLOVER_AGENT_LOOP_MAX_ITERATIONS / CLOVER_EDGE2_MAX_RETRIES  Retry budgets
  CLOVER_DISABLE_THINKING                              Disable Qwen thinking chat template
  CLOVER_TABLEFACT_DIRECT_VERIFIER                    Use compact TableFact verifier
  CLOVER_TABLEFACT_SECOND_PASS_VERIFIER               Recover high-confidence false negatives
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# Optional positional DATASET override.
if [[ "${1:-}" != "" && "${1:-}" != --* ]]; then
  DATASET="$1"
  shift
fi
FORWARD_ARGS=("$@")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
normalize_bool() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) printf '%s' true ;;
    0|false|no|n|off) printf '%s' false ;;
    *) echo "Invalid boolean value: $1" >&2; exit 1 ;;
  esac
}

validate_gpus() {
  local label="$1" gpus="$2"
  gpus="$(printf '%s' "${gpus}" | tr -d '[:space:]')"
  if [[ ! "${gpus}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "Invalid ${label}: ${gpus}" >&2
    echo "Use comma-separated GPU ids, for example 0 or 0,1." >&2
    exit 1
  fi
  printf '%s' "${gpus}"
}

count_gpus() {
  local gpus="$1" count=0
  IFS=',' read -r -a _items <<< "${gpus}"
  for _ in "${_items[@]}"; do count=$((count + 1)); done
  printf '%s' "${count}"
}

resolve_tp() {
  local label="$1" tp="$2" gpu_count="$3"
  if [[ -z "${tp}" ]]; then
    tp="${gpu_count}"
  fi
  if [[ ! "${tp}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid ${label} tensor parallel size: ${tp}" >&2
    exit 1
  fi
  if (( tp > gpu_count )); then
    echo "${label} tensor parallel size (${tp}) exceeds visible GPUs (${gpu_count})." >&2
    exit 1
  fi
  printf '%s' "${tp}"
}

# -----------------------------------------------------------------------------
# Validate inputs
# -----------------------------------------------------------------------------
case "${DATASET}" in
  tablebench|wikitq|tablefact|mmqa) ;;
  *) echo "Unsupported dataset: ${DATASET}" >&2; exit 2 ;;
esac

# Empirical local 7B+3B defaults from the 2026-06-22 server sweep:
# - TableFact is harmed badly by local repair/retry and benefits from
#   deterministic EDGE2 generation.
# - TableBench/WikiTQ recover a few points from modest EDGE2 sampling and one
#   repair round, though they remain below the older DeepSeek-backed runs.
if [[ -z "${CLOVER_EDGE1_TEMPERATURE+x}" ]]; then
  case "${DATASET}" in
    tablefact|wikitq) EDGE1_TEMPERATURE="0.0" ;;
    tablebench) EDGE1_TEMPERATURE="0.3" ;;
    mmqa) EDGE1_TEMPERATURE="0.2" ;;
  esac
fi
if [[ -z "${CLOVER_EDGE1_TOP_P+x}" ]]; then
  case "${DATASET}" in
    tablefact|wikitq) EDGE1_TOP_P="1.0" ;;
    tablebench) EDGE1_TOP_P="1.0" ;;
    mmqa) EDGE1_TOP_P="0.9" ;;
  esac
fi
if [[ -z "${CLOVER_EDGE2_TEMPERATURE+x}" ]]; then
  case "${DATASET}" in
    tablefact) EDGE2_TEMPERATURE="0.0" ;;
    wikitq) EDGE2_TEMPERATURE="0.0" ;;
    tablebench) EDGE2_TEMPERATURE="0.3" ;;
    mmqa) EDGE2_TEMPERATURE="0.2" ;;
  esac
fi
if [[ -z "${CLOVER_EDGE2_TOP_P+x}" ]]; then
  case "${DATASET}" in
    tablefact|wikitq) EDGE2_TOP_P="1.0" ;;
    tablebench) EDGE2_TOP_P="0.9" ;;
    mmqa) EDGE2_TOP_P="0.9" ;;
  esac
fi
if [[ "${DATASET}" == "mmqa" && -z "${CLOVER_EDGE1_MAX_MODEL_LEN+x}" && -z "${CLOVER_EDGE_MAX_MODEL_LEN+x}" ]]; then
  EDGE1_MAX_MODEL_LEN="16384"
fi
if [[ "${DATASET}" == "mmqa" && -z "${CLOVER_EDGE2_MAX_MODEL_LEN+x}" && -z "${CLOVER_EDGE_MAX_MODEL_LEN+x}" ]]; then
  EDGE2_MAX_MODEL_LEN="16384"
fi
if [[ "${DATASET}" == "mmqa" && -z "${CLOVER_EDGE1_MAX_TOKENS+x}" && -z "${CLOVER_EDGE_MAX_TOKENS+x}" ]]; then
  EDGE1_MAX_TOKENS="2048"
fi
if [[ "${DATASET}" == "mmqa" && -z "${CLOVER_EDGE2_MAX_TOKENS+x}" && -z "${CLOVER_EDGE_MAX_TOKENS+x}" ]]; then
  EDGE2_MAX_TOKENS="2048"
fi
if [[ "${DATASET}" == "mmqa" && -z "${CLOVER_VLLM_MAX_NUM_SEQS+x}" ]]; then
  VLLM_MAX_NUM_SEQS="64"
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "No executable 'python' found. Activate the intended environment first." >&2
  exit 1
fi
if [[ ! "${VLLM_MAX_NUM_SEQS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid CLOVER_VLLM_MAX_NUM_SEQS: ${VLLM_MAX_NUM_SEQS}" >&2
  exit 1
fi

for model_path in "${EDGE1_MODEL_PATH}" "${EDGE2_MODEL_PATH}"; do
  if [[ "${model_path}" == /path/to/* ]]; then
    echo "Set EDGE1/EDGE2 model paths at the top of this script." >&2
    exit 1
  fi
  if [[ "${model_path}" == /* && ! -e "${model_path}" ]]; then
    echo "Model path not found: ${model_path}" >&2
    exit 1
  fi
done

EDGE1_GPUS="$(validate_gpus EDGE1_GPUS "${EDGE1_GPUS}")"
EDGE1_GPU_COUNT="$(count_gpus "${EDGE1_GPUS}")"
EDGE1_TENSOR_PARALLEL_SIZE="$(resolve_tp EDGE1 "${EDGE1_TENSOR_PARALLEL_SIZE}" "${EDGE1_GPU_COUNT}")"
EDGE2_GPUS="$(validate_gpus EDGE2_GPUS "${EDGE2_GPUS}")"
EDGE2_GPU_COUNT="$(count_gpus "${EDGE2_GPUS}")"
EDGE2_TENSOR_PARALLEL_SIZE="$(resolve_tp EDGE2 "${EDGE2_TENSOR_PARALLEL_SIZE}" "${EDGE2_GPU_COUNT}")"

PERSIST_SERVER="$(normalize_bool "${PERSIST_SERVER}")"
WARMUP_SERVER="$(normalize_bool "${WARMUP_SERVER}")"
DISABLE_THINKING="$(normalize_bool "${DISABLE_THINKING}")"
ENABLE_PREFIX_CACHING="$(normalize_bool "${ENABLE_PREFIX_CACHING}")"
FORCE_SEPARATE_EDGE_SERVERS="$(normalize_bool "${FORCE_SEPARATE_EDGE_SERVERS}")"
ENABLE_EDGE_AGENT="$(normalize_bool "${ENABLE_EDGE_AGENT}")"
ENABLE_EDGE_REPAIR="$(normalize_bool "${ENABLE_EDGE_REPAIR}")"
ENABLE_TERMINAL_EDGE_REVIEW="$(normalize_bool "${ENABLE_TERMINAL_EDGE_REVIEW}")"
ENABLE_CONTRACT_GATE="$(normalize_bool "${ENABLE_CONTRACT_GATE}")"
ENABLE_NODE_REVIEW="$(normalize_bool "${ENABLE_NODE_REVIEW}")"
ENABLE_CLOUD_RECOVERY="$(normalize_bool "${ENABLE_CLOUD_RECOVERY}")"
ENABLE_CLOUD_REPLAN="$(normalize_bool "${ENABLE_CLOUD_REPLAN}")"
ENABLE_CLOUD_SYNTHESIS="$(normalize_bool "${ENABLE_CLOUD_SYNTHESIS}")"
ENABLE_STATIC_FAST_PATH="$(normalize_bool "${ENABLE_STATIC_FAST_PATH}")"
ENABLE_STATIC_FINALIZATION="$(normalize_bool "${ENABLE_STATIC_FINALIZATION}")"
ENABLE_OBSERVABLE_CLOSURE_CHECKER="$(normalize_bool "${ENABLE_OBSERVABLE_CLOSURE_CHECKER}")"
EDGE_REVIEW_PROACTIVE="$(normalize_bool "${EDGE_REVIEW_PROACTIVE}")"
TABLEFACT_DIRECT_VERIFIER="$(normalize_bool "${TABLEFACT_DIRECT_VERIFIER}")"
TABLEFACT_SECOND_PASS_VERIFIER="$(normalize_bool "${TABLEFACT_SECOND_PASS_VERIFIER}")"
TABLE_DIRECT_PROBE="$(normalize_bool "${TABLE_DIRECT_PROBE}")"

if [[ ! "${EDGE_REVIEW_MODE}" =~ ^(off|shadow|safe)$ ]]; then
  echo "Invalid CLOVER_EDGE_REVIEW_MODE: ${EDGE_REVIEW_MODE}" >&2
  exit 1
fi

SHARE_EDGE_SERVER=false
if [[ "${FORCE_SEPARATE_EDGE_SERVERS}" != "true" \
    && "${EDGE1_MODEL_PATH}" == "${EDGE2_MODEL_PATH}" \
    && -z "${CLOVER_EDGE2_PORT+x}" ]]; then
  EDGE2_PORT="${EDGE1_PORT}"
  EDGE2_GPUS="${EDGE1_GPUS}"
  EDGE2_GPU_COUNT="${EDGE1_GPU_COUNT}"
  EDGE2_TENSOR_PARALLEL_SIZE="${EDGE1_TENSOR_PARALLEL_SIZE}"
  EDGE2_GPU_MEM_UTIL="${EDGE1_GPU_MEM_UTIL}"
  EDGE2_MAX_MODEL_LEN="${EDGE1_MAX_MODEL_LEN}"
  EDGE2_DTYPE="${EDGE1_DTYPE}"
  SHARE_EDGE_SERVER=true
fi
if [[ -z "${EDGE1_GPU_MEM_UTIL}" ]]; then
  if [[ "${SHARE_EDGE_SERVER}" != "true" && "${EDGE1_GPUS}" == "${EDGE2_GPUS}" ]]; then
    EDGE1_GPU_MEM_UTIL="0.45"
  else
    EDGE1_GPU_MEM_UTIL="0.90"
  fi
fi
if [[ -z "${EDGE2_GPU_MEM_UTIL}" ]]; then
  if [[ "${SHARE_EDGE_SERVER}" == "true" ]]; then
    EDGE2_GPU_MEM_UTIL="${EDGE1_GPU_MEM_UTIL}"
  elif [[ "${EDGE1_GPUS}" == "${EDGE2_GPUS}" ]]; then
    EDGE2_GPU_MEM_UTIL="0.45"
  else
    EDGE2_GPU_MEM_UTIL="0.90"
  fi
fi
if [[ "${EDGE1_PORT}" == "${EDGE2_PORT}" && "${SHARE_EDGE_SERVER}" != "true" ]]; then
  echo "EDGE1 and EDGE2 ports are both ${EDGE1_PORT}, but the servers are not shared." >&2
  echo "Use different ports or unset CLOVER_EDGE2_PORT when both model paths are identical." >&2
  exit 1
fi

# Dataset roots.
DATASETS_ROOT="${CLOVER_DATASETS_ROOT:-${REPO_ROOT}/datasets}"
TABLEBENCH_ROOT="${TABLEBENCH_ROOT:-${DATASETS_ROOT}/tablebench}"
WIKITQ_ROOT="${WIKITQ_ROOT:-${DATASETS_ROOT}/wikitq}"
TABLEFACT_ROOT="${TABLEFACT_ROOT:-${DATASETS_ROOT}/tablefact}"
MMQA_ROOT="${MMQA_ROOT:-${DATASETS_ROOT}/mmqa}"
WIKITQ_SPLIT="${WIKITQ_SPLIT:-pristine-unseen-tables}"
TABLEFACT_SPLIT="${TABLEFACT_SPLIT:-test}"
TABLEFACT_SUBSET="${TABLEFACT_SUBSET:-small}"
MMQA_SPLIT="${MMQA_SPLIT:-}"

case "${DATASET}" in
  tablebench) DATASET_ROOT="${TABLEBENCH_ROOT}" ;;
  wikitq) DATASET_ROOT="${WIKITQ_ROOT}" ;;
  tablefact) DATASET_ROOT="${TABLEFACT_ROOT}" ;;
  mmqa) DATASET_ROOT="${MMQA_ROOT}" ;;
esac
DATASET_CASE_MAXDEPTH=2
[[ "${DATASET}" == "mmqa" ]] && DATASET_CASE_MAXDEPTH=3
if ! find "${DATASET_ROOT}" -mindepth 2 -maxdepth "${DATASET_CASE_MAXDEPTH}" -name cases.jsonl \
    -print -quit 2>/dev/null | grep -q .; then
  echo "Converted ${DATASET} dataset not found: ${DATASET_ROOT}" >&2
  echo "Run: bash benchmarks/download_datasets.sh --dataset ${DATASET}" >&2
  exit 1
fi

# vLLM CLI discovery.
"${PYTHON_BIN}" - <<'PY'
import importlib.util, sys
if importlib.util.find_spec("vllm") is None:
    raise SystemExit(
        f"vLLM is not installed in {sys.executable}. "
        f"Install requirements first: {sys.executable} -m pip install -r requirements.txt"
    )
PY
VLLM_BIN="$("${PYTHON_BIN}" - <<'PY'
import sys
from pathlib import Path
candidate = Path(sys.executable).with_name("vllm")
print(candidate if candidate.is_file() else "")
PY
)"
if [[ -z "${VLLM_BIN}" ]]; then
  echo "vLLM CLI was not found next to ${PYTHON_BIN}." >&2
  exit 1
fi

# Output / run naming.
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/benchmark/runs}"
RUN_NAME="${RUN_NAME:-${DATASET}_vllm_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_ROOT}"
TMP_DIR="$(mktemp -d)"
EDGE1_SERVER_LOG="${OUTPUT_ROOT}/${RUN_NAME}_edge1_vllm.log"
EDGE2_SERVER_LOG="${OUTPUT_ROOT}/${RUN_NAME}_edge2_vllm.log"
EDGE1_BASE_URL="http://${HOST}:${EDGE1_PORT}/v1"
EDGE2_BASE_URL="http://${HOST}:${EDGE2_PORT}/v1"
EDGE1_SERVED_MODEL_NAME="${CLOVER_EDGE1_SERVED_MODEL_NAME:-$(basename "${EDGE1_MODEL_PATH}")}"
EDGE2_SERVED_MODEL_NAME="${CLOVER_EDGE2_SERVED_MODEL_NAME:-$(basename "${EDGE2_MODEL_PATH}")}"

EDGE1_PID=""
EDGE2_PID=""

cleanup() {
  if [[ "${PERSIST_SERVER}" != "true" ]]; then
    if [[ -n "${EDGE2_PID}" && "${EDGE2_PID}" != "${EDGE1_PID}" ]]; then
      kill "${EDGE2_PID}" >/dev/null 2>&1 || true
      wait "${EDGE2_PID}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${EDGE1_PID}" ]]; then
      kill "${EDGE1_PID}" >/dev/null 2>&1 || true
      wait "${EDGE1_PID}" >/dev/null 2>&1 || true
    fi
  fi
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

# -----------------------------------------------------------------------------
# vLLM server control
# -----------------------------------------------------------------------------
server_ready() {
  local base_url="$1"
  local expected_model="${2:-}"
  "${PYTHON_BIN}" - "${base_url}" "${expected_model}" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
base_url, expected = sys.argv[1:3]
with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=2) as r:
    payload = json.loads(r.read().decode("utf-8"))
if not expected:
    raise SystemExit(0)
model_ids = {
    str(item.get("id"))
    for item in payload.get("data", [])
    if isinstance(item, dict) and item.get("id") is not None
}
raise SystemExit(0 if expected in model_ids else 2)
PY
}

start_vllm() {
  local label="$1" model_path="$2" gpus="$3" tp="$4" port="$5"
  local mem_util="$6" max_model_len="$7" dtype="$8" served_name="$9" log_file="${10}"
  local base_url="http://${HOST}:${port}/v1"

  if server_ready "${base_url}" "${served_name}"; then
    echo "Using existing ${label} server: ${base_url}" >&2
    return 0
  fi
  if server_ready "${base_url}"; then
    echo "${label} endpoint ${base_url} is already serving a different model." >&2
    echo "Expected served model name: ${served_name}" >&2
    echo "Stop the existing vLLM server or choose a different CLOVER_${label}_PORT." >&2
    exit 1
  fi

  local extra_args=()
  if [[ -n "${VLLM_SERVER_ARGS}" ]]; then
    read -r -a extra_args <<< "${VLLM_SERVER_ARGS}"
  fi
  local cmd=(
    env CUDA_VISIBLE_DEVICES="${gpus}" VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER}"
    "${VLLM_BIN}" serve "${model_path}"
    --served-model-name "${served_name}"
    --host "${HOST}"
    --port "${port}"
    --dtype "${dtype}"
    --gpu-memory-utilization "${mem_util}"
    --tensor-parallel-size "${tp}"
    --max-model-len "${max_model_len}"
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
  )
  [[ "${ENABLE_PREFIX_CACHING}" == "true" ]] && cmd+=(--enable-prefix-caching)
  [[ "${#extra_args[@]}" -gt 0 ]] && cmd+=("${extra_args[@]}")

  echo "Starting ${label} vLLM" >&2
  echo "  model: ${model_path}" >&2
  echo "  GPUs: ${gpus} (tp=${tp})" >&2
  echo "  gpu_memory_utilization: ${mem_util}" >&2
  echo "  max_model_len: ${max_model_len}" >&2
  echo "  endpoint: ${base_url}" >&2
  echo "  log: ${log_file}" >&2
  "${cmd[@]}" >"${log_file}" 2>&1 &
  local pid="$!"

  for ((second = 0; second < SERVER_READY_TIMEOUT; second++)); do
    server_ready "${base_url}" "${served_name}" && break
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "${label} vLLM exited before becoming ready. See ${log_file}" >&2
      exit 1
    fi
    sleep 1
  done
  if ! server_ready "${base_url}" "${served_name}"; then
    echo "Timed out waiting for ${label} vLLM after ${SERVER_READY_TIMEOUT}s." >&2
    echo "See ${log_file}" >&2
    exit 1
  fi
  echo "${pid}"  # return pid via stdout
}

warmup_server() {
  local base_url="$1" served_name="$2" label="$3"
  if [[ "${WARMUP_SERVER}" != "true" ]]; then return 0; fi
  echo "Warming up ${label} vLLM model" >&2
  "${PYTHON_BIN}" - "${base_url}" "${served_name}" <<'PY' >/dev/null
import json, sys, urllib.request
base_url, model = sys.argv[1:3]
payload = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "Reply with OK."}],
    "temperature": 0,
    "max_tokens": 2,
    "stream": False,
}).encode("utf-8")
req = urllib.request.Request(
    base_url.rstrip("/") + "/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=180) as r:
    json.loads(r.read().decode("utf-8"))
PY
}

# -----------------------------------------------------------------------------
# Generate configs (no model_config/ files required)
# -----------------------------------------------------------------------------
write_local_config() {
  local out="$1"
  "${PYTHON_BIN}" - "${out}" "${EDGE1_SERVED_MODEL_NAME}" "${EDGE1_BASE_URL}" \
    "${EDGE1_MAX_TOKENS}" "${EDGE1_TIMEOUT}" "${EDGE1_TEMPERATURE}" "${EDGE1_TOP_P}" \
    "${AGENT_LOOP_MAX_ITERATIONS}" \
    "${EDGE_REVIEW_MODE}" "${EDGE_REVIEW_PROACTIVE}" "${ABLATION_VARIANT}" \
    "${ENABLE_EDGE_AGENT}" "${ENABLE_EDGE_REPAIR}" "${ENABLE_TERMINAL_EDGE_REVIEW}" \
    "${ENABLE_CONTRACT_GATE}" "${ENABLE_NODE_REVIEW}" "${ENABLE_CLOUD_RECOVERY}" \
    "${ENABLE_CLOUD_REPLAN}" "${ENABLE_CLOUD_SYNTHESIS}" \
    "${ENABLE_STATIC_FAST_PATH}" "${ENABLE_STATIC_FINALIZATION}" \
    "${ENABLE_OBSERVABLE_CLOSURE_CHECKER}" \
    "${TABLEFACT_DIRECT_VERIFIER}" "${TABLEFACT_DIRECT_MAX_TOKENS}" \
    "${TABLEFACT_DIRECT_TABLE_CHAR_LIMIT}" "${TABLEFACT_DIRECT_TEMPERATURE}" \
    "${TABLEFACT_DIRECT_TOP_P}" \
    "${TABLEFACT_SECOND_PASS_VERIFIER}" "${TABLEFACT_SECOND_PASS_MAX_TOKENS}" \
    "${TABLE_DIRECT_PROBE}" "${TABLE_DIRECT_PROBE_MAX_TOKENS}" \
    "${TABLE_DIRECT_PROBE_TABLE_CHAR_LIMIT}" \
    "${TABLE_DIRECT_PROBE_TEMPERATURE}" "${TABLE_DIRECT_PROBE_TOP_P}" \
    "${DISABLE_THINKING}" <<'PY'
import json, sys
from pathlib import Path

path, model, base_url = sys.argv[1:4]
max_tokens, timeout, temperature, top_p, agent_loop = sys.argv[4:9]
payload = {
    "provider": "local",
    "api_type": "chat_completions",
    "api_key": "EMPTY",
    "base_url": base_url,
    "model": model,
    "timeout": int(timeout),
    "node_timeout_seconds": int(timeout),
    "max_retries": 3,
    "max_tokens": int(max_tokens),
    "temperature": float(temperature),
    "top_p": float(top_p),
    "http2": False,
    "trust_env": False,
    "agent_loop_max_iterations": int(agent_loop),
    "edge_review_mode": sys.argv[9],
    "edge_review_proactive": sys.argv[10] == "true",
    "ablation_variant": sys.argv[11],
    "enable_edge_agent": sys.argv[12] == "true",
    "enable_edge_repair": sys.argv[13] == "true",
    "enable_terminal_edge_review": sys.argv[14] == "true",
    "enable_contract_gate": sys.argv[15] == "true",
    "enable_node_review": sys.argv[16] == "true",
    "enable_cloud_recovery": sys.argv[17] == "true",
    "enable_cloud_replan": sys.argv[18] == "true",
    "enable_cloud_synthesis": sys.argv[19] == "true",
    "enable_static_fast_path": sys.argv[20] == "true",
    "enable_static_finalization": sys.argv[21] == "true",
    "enable_observable_closure_checker": sys.argv[22] == "true",
    "enable_tablefact_direct_verifier": sys.argv[23] == "true",
    "tablefact_direct_max_tokens": int(sys.argv[24]),
    "tablefact_direct_table_char_limit": int(sys.argv[25]),
    "tablefact_direct_temperature": float(sys.argv[26]),
    "tablefact_direct_top_p": float(sys.argv[27]),
    "enable_tablefact_second_pass_verifier": sys.argv[28] == "true",
    "tablefact_second_pass_max_tokens": int(sys.argv[29]),
    "enable_table_direct_probe": sys.argv[30] == "true",
    "table_direct_probe_max_tokens": int(sys.argv[31]),
    "table_direct_probe_table_char_limit": int(sys.argv[32]),
    "table_direct_probe_temperature": float(sys.argv[33]),
    "table_direct_probe_top_p": float(sys.argv[34]),
    "edge_review_max_actions": 6,
    "edge_review_max_rows": 8,
    "edge_review_max_columns": 8,
    "edge_review_max_facts": 80,
    "tptt_coalesce_ms": 60,
    "tptt_prefix_tokens": 200,
    "max_tptt_leaf_sequences_per_tree": 128,
}
if sys.argv[35] == "true":
    payload["extra_body"] = {
        "chat_template_kwargs": {
            "enable_thinking": False,
        }
    }
Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

write_edge2_config() {
  local out="$1"
  "${PYTHON_BIN}" - "${out}" "${EDGE2_SERVED_MODEL_NAME}" "${EDGE2_BASE_URL}" \
    "${EDGE2_MAX_TOKENS}" "${EDGE2_TIMEOUT}" "${EDGE2_TEMPERATURE}" "${EDGE2_TOP_P}" \
    "${DISABLE_THINKING}" <<'PY'
import json, sys
from pathlib import Path

path, model, base_url = sys.argv[1:4]
max_tokens, timeout, temperature, top_p = sys.argv[4:8]
payload = {
    "provider": "local",
    "api_type": "chat_completions",
    "api_key": "EMPTY",
    "base_url": base_url,
    "model": model,
    "timeout": int(timeout),
    "max_retries": 3,
    "max_tokens": int(max_tokens),
    "temperature": float(temperature),
    "top_p": float(top_p),
    "http2": False,
    "trust_env": False,
}
if sys.argv[8] == "true":
    payload["extra_body"] = {
        "chat_template_kwargs": {
            "enable_thinking": False,
        }
    }
Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

# -----------------------------------------------------------------------------
# Print configuration
# -----------------------------------------------------------------------------
echo "=========================================" >&2
echo " CLOVER Local vLLM Configuration" >&2
echo "=========================================" >&2
echo " Dataset:              ${DATASET}" >&2
echo "" >&2
echo " EDGE1 local model:    ${EDGE1_MODEL_PATH}" >&2
echo "   GPUs:               ${EDGE1_GPUS} (tp=${EDGE1_TENSOR_PARALLEL_SIZE})" >&2
echo "   GPU mem util:       ${EDGE1_GPU_MEM_UTIL}" >&2
echo "   Endpoint:           ${EDGE1_BASE_URL}" >&2
echo "   Max model len:      ${EDGE1_MAX_MODEL_LEN}" >&2
echo "   Max tokens:         ${EDGE1_MAX_TOKENS}" >&2
echo "" >&2
echo " EDGE1 sampling:" >&2
echo "   Temperature:        ${EDGE1_TEMPERATURE}" >&2
echo "   Top-p:              ${EDGE1_TOP_P}" >&2
echo "" >&2
echo " EDGE2 supervisor:     ${EDGE2_MODEL_PATH}" >&2
echo "   Shared server:      ${SHARE_EDGE_SERVER}" >&2
echo "   GPUs:               ${EDGE2_GPUS} (tp=${EDGE2_TENSOR_PARALLEL_SIZE})" >&2
echo "   GPU mem util:       ${EDGE2_GPU_MEM_UTIL}" >&2
echo "   Endpoint:           ${EDGE2_BASE_URL}" >&2
echo "   Max model len:      ${EDGE2_MAX_MODEL_LEN}" >&2
echo "   Max tokens:         ${EDGE2_MAX_TOKENS}" >&2
echo "" >&2
echo " EDGE2 sampling:" >&2
echo "   Temperature:        ${EDGE2_TEMPERATURE}" >&2
echo "   Top-p:              ${EDGE2_TOP_P}" >&2
echo "" >&2
echo " Eval concurrency:     ${EVAL_CONCURRENCY} workers / ${EDGE2_CONCURRENCY} edge2" >&2
echo " vLLM max num seqs:    ${VLLM_MAX_NUM_SEQS}" >&2
echo " Disable thinking:     ${DISABLE_THINKING}" >&2
echo " Edge agent max iters: ${AGENT_LOOP_MAX_ITERATIONS}" >&2
echo " EDGE2 max retries:    ${EDGE2_MAX_RETRIES}" >&2
echo " Validation mode:      ${VALIDATION_MODE}" >&2
echo " Closure checker:      ${ENABLE_OBSERVABLE_CLOSURE_CHECKER}" >&2
echo " TableFact direct:     ${TABLEFACT_DIRECT_VERIFIER} (max_tokens=${TABLEFACT_DIRECT_MAX_TOKENS}, table_chars=${TABLEFACT_DIRECT_TABLE_CHAR_LIMIT})" >&2
echo " TableFact 2nd pass:   ${TABLEFACT_SECOND_PASS_VERIFIER} (max_tokens=${TABLEFACT_SECOND_PASS_MAX_TOKENS})" >&2
echo " Table direct probe:   ${TABLE_DIRECT_PROBE} (max_tokens=${TABLE_DIRECT_PROBE_MAX_TOKENS}, table_chars=${TABLE_DIRECT_PROBE_TABLE_CHAR_LIMIT})" >&2
echo " FlashInfer sampler:   ${VLLM_USE_FLASHINFER_SAMPLER}" >&2
echo "=========================================" >&2

# -----------------------------------------------------------------------------
# Start vLLM server(s)
# -----------------------------------------------------------------------------
EDGE1_PID="$(start_vllm \
  "EDGE1" "${EDGE1_MODEL_PATH}" "${EDGE1_GPUS}" "${EDGE1_TENSOR_PARALLEL_SIZE}" \
  "${EDGE1_PORT}" "${EDGE1_GPU_MEM_UTIL}" "${EDGE1_MAX_MODEL_LEN}" \
  "${EDGE1_DTYPE}" "${EDGE1_SERVED_MODEL_NAME}" "${EDGE1_SERVER_LOG}")"
warmup_server "${EDGE1_BASE_URL}" "${EDGE1_SERVED_MODEL_NAME}" "EDGE1"

if [[ "${SHARE_EDGE_SERVER}" == "true" ]]; then
  EDGE2_PID="${EDGE1_PID}"
else
  EDGE2_PID="$(start_vllm \
    "EDGE2" "${EDGE2_MODEL_PATH}" "${EDGE2_GPUS}" "${EDGE2_TENSOR_PARALLEL_SIZE}" \
    "${EDGE2_PORT}" "${EDGE2_GPU_MEM_UTIL}" "${EDGE2_MAX_MODEL_LEN}" \
    "${EDGE2_DTYPE}" "${EDGE2_SERVED_MODEL_NAME}" "${EDGE2_SERVER_LOG}")"
  warmup_server "${EDGE2_BASE_URL}" "${EDGE2_SERVED_MODEL_NAME}" "EDGE2"
fi

# -----------------------------------------------------------------------------
# Generate configs
# -----------------------------------------------------------------------------
LOCAL_CONFIG="${TMP_DIR}/edge_local_slm_config.json"
EDGE2_CONFIG="${TMP_DIR}/edge2_llm_config.json"
write_local_config "${LOCAL_CONFIG}"
write_edge2_config "${EDGE2_CONFIG}"

# -----------------------------------------------------------------------------
# Build eval command
# -----------------------------------------------------------------------------
EVAL_CMD=(
  "${PYTHON_BIN}" -m benchmarks.eval
  --output-root "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
  --remote-llm-config "${EDGE2_CONFIG}"
  --synthesize-llm-config "${EDGE2_CONFIG}"
  --local-slm-config "${LOCAL_CONFIG}"
  --remote-batch-size "${EDGE2_BATCH_SIZE}"
  --remote-concurrency "${EDGE2_CONCURRENCY}"
  --max-workers "${EVAL_CONCURRENCY}"
  --slm-scheduler "${SLM_SCHEDULER}"
  --max-parallel-execution-units "${MAX_PARALLEL_EXECUTION_UNITS}"
  --max-parallel-slm-node-jobs "${MAX_PARALLEL_SLM_NODE_JOBS}"
  --max-parallel-slm-sequences "${MAX_PARALLEL_SLM_SEQUENCES}"
  --max-pending-slm-sequences "${MAX_PENDING_SLM_SEQUENCES}"
  --max-retries "${EDGE2_MAX_RETRIES}"
  --validation-mode "${VALIDATION_MODE}"
)
case "${DATASET}" in
  tablebench)
    EVAL_CMD+=(
      --tablebench-eval
      --tablebench-root "${TABLEBENCH_ROOT}"
      --qtype FactChecking
      --qtype NumericalReasoning
    )
    ;;
  wikitq)
    EVAL_CMD+=(
      --wikitq-eval
      --wikitq-root "${WIKITQ_ROOT}"
      --wikitq-split "${WIKITQ_SPLIT}"
    )
    ;;
  tablefact)
    EVAL_CMD+=(
      --tablefact-eval
      --tablefact-root "${TABLEFACT_ROOT}"
      --tablefact-split "${TABLEFACT_SPLIT}"
      --tablefact-subset "${TABLEFACT_SUBSET}"
    )
    ;;
  mmqa)
    EVAL_CMD+=(
      --mmqa-eval
      --mmqa-root "${MMQA_ROOT}"
    )
    if [[ -n "${MMQA_SPLIT}" ]]; then
      EVAL_CMD+=(--mmqa-split "${MMQA_SPLIT}")
    fi
    ;;
esac
[[ "${#FORWARD_ARGS[@]}" -gt 0 ]] && EVAL_CMD+=("${FORWARD_ARGS[@]}")

echo "Running ${DATASET} evaluation" >&2
echo "  output: ${OUTPUT_ROOT}/${RUN_NAME}" >&2
echo "  edge2: local/${EDGE2_SERVED_MODEL_NAME} @ ${EDGE2_BASE_URL}" >&2
echo "  synthesize: local/${EDGE2_SERVED_MODEL_NAME} @ ${EDGE2_BASE_URL}" >&2
echo "  local: local/${EDGE1_SERVED_MODEL_NAME} @ ${EDGE1_BASE_URL}" >&2

EVAL_EXIT=0
PYTHONWARNINGS="ignore" \
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
"${EVAL_CMD[@]}" || EVAL_EXIT=$?

# -----------------------------------------------------------------------------
# Print compact summary to stdout
# -----------------------------------------------------------------------------
SUMMARY_FILE="${OUTPUT_ROOT}/${RUN_NAME}/run_summary.json"
if [[ -f "${SUMMARY_FILE}" ]]; then
  "${PYTHON_BIN}" - "${SUMMARY_FILE}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
brief = summary.get("brief_summary") or {}

remote_usage = summary.get("remote_token_usage") or {}
local_usage = summary.get("local_slm_token_usage") or {}
input_tokens = int(summary.get("input_tokens", 0) or 0)
output_tokens = int(summary.get("output_tokens", 0) or 0)
total_tokens = int(summary.get("total_tokens", 0) or 0)
if not input_tokens:
    input_tokens = int(remote_usage.get("input_tokens", 0) or 0) + int(local_usage.get("input_tokens", 0) or 0)
if not output_tokens:
    output_tokens = int(remote_usage.get("output_tokens", 0) or 0) + int(local_usage.get("output_tokens", 0) or 0)
if not total_tokens:
    total_tokens = int(remote_usage.get("total_tokens", 0) or 0) + int(local_usage.get("total_tokens", 0) or 0)
    if not total_tokens:
        total_tokens = input_tokens + output_tokens
total_cases = int(summary.get("total_cases", 0) or 0)
correct = summary.get("correct")
acc = summary.get("acc_pct", brief.get("acc_pct"))
avg_input_per_q = round(input_tokens / total_cases, 2) if total_cases else 0.0
avg_output_per_q = round(output_tokens / total_cases, 2) if total_cases else 0.0
avg_total_per_q = round(total_tokens / total_cases, 2) if total_cases else 0.0

print("\n===== Brief Summary =====")
print(f"{'Benchmark':<32}: {brief.get('benchmark') or summary.get('dataset')}")
print(f"{'Total Cases':<32}: {total_cases}")
print(f"{'Correct':<32}: {correct}")
print(f"{'Acc. (%)':<32}: {acc}")
print(f"{'Input Tokens':<32}: {input_tokens}")
print(f"{'Output Tokens':<32}: {output_tokens}")
print(f"{'Total Tokens':<32}: {total_tokens}")
print(f"{'Avg Input Tok / Query':<32}: {avg_input_per_q}")
print(f"{'Avg Output Tok / Query':<32}: {avg_output_per_q}")
print(f"{'Avg Total Tok / Query':<32}: {avg_total_per_q}")
print("=========================")
PY
else
  echo "Summary file not found: ${SUMMARY_FILE}" >&2
fi

exit ${EVAL_EXIT}
