#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Dual local vLLM launcher for CLOVER.
#
# Starts two independent vLLM OpenAI-compatible servers (on different ports,
# optionally on different GPUs) and runs the selected eval. No model_config
# JSON files are required — both the EDGE1 (local slm) and EDGE2
# (remote + synthesize) configs are generated on the fly from the settings
# below.
#
#   EDGE1  -> local slm  (edge agent, table reasoning)
#   EDGE2  -> remote + synthesize (cloud agent, repair)
#
# Demo: run the main experiment with two edge models / two vLLM services:
#   PYTHON_BIN=/Users/huangwenxuan/Documents/codes/CLOVER/.venv/bin/python \
#   CLOVER_EDGE1_MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-3B-Instruct \
#   CLOVER_EDGE2_MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-7B-Instruct \
#   CLOVER_EDGE1_GPUS=0 \
#   CLOVER_EDGE2_GPUS=1 \
#   CLOVER_EDGE1_PORT=8000 \
#   CLOVER_EDGE2_PORT=8001 \
#   CLOVER_EDGE1_MAX_MODEL_LEN=4096 \
#   CLOVER_EDGE2_MAX_MODEL_LEN=8192 \
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
# =============================================================================

# =============================================================================
# User settings
# Edit these values, then run: bash benchmarks/run_vllm_eval_clover.sh
# Every variable can also be overridden via environment variables (CLOVER_*).
# =============================================================================

# --- Dataset ---
DATASET="${CLOVER_EVAL_DATASET:-tablebench}"  # tablebench | wikitq | tablefact

# --- Edge model (local slm / edge agent) -----------------------------------
EDGE1_MODEL_PATH="${CLOVER_EDGE1_MODEL_PATH:-/path/to/Qwen2.5-3B-Instruct}"
EDGE1_GPUS="${CLOVER_EDGE1_GPUS:-0}"                # comma-separated GPU ids, e.g. 0 or 0,1
EDGE1_GPU_MEM_UTIL="${CLOVER_EDGE1_GPU_MEM_UTIL:-0.30}"  # fraction of GPU memory
EDGE1_PORT="${CLOVER_EDGE1_PORT:-8000}"
EDGE1_MAX_MODEL_LEN="${CLOVER_EDGE1_MAX_MODEL_LEN:-4096}"
EDGE1_MAX_TOKENS="${CLOVER_EDGE1_MAX_TOKENS:-512}"
EDGE1_TIMEOUT="${CLOVER_EDGE1_TIMEOUT:-600}"
EDGE1_DTYPE="${CLOVER_EDGE1_DTYPE:-auto}"
EDGE1_TENSOR_PARALLEL_SIZE="${CLOVER_EDGE1_TENSOR_PARALLEL_SIZE:-}"  # defaults to GPU count
# Edge sampling params (0 = deterministic; >0 adds diversity for edge repair).
EDGE1_TEMPERATURE="${CLOVER_EDGE1_TEMPERATURE:-0.0}"
EDGE1_TOP_P="${CLOVER_EDGE1_TOP_P:-1.0}"

# --- EDGE2 (remote + synthesize / cloud agent) -----------------------------
EDGE2_MODEL_PATH="${CLOVER_EDGE2_MODEL_PATH:-/path/to/Qwen2.5-7B-Instruct}"
EDGE2_GPUS="${CLOVER_EDGE2_GPUS:-0}"                # comma-separated GPU ids
EDGE2_GPU_MEM_UTIL="${CLOVER_EDGE2_GPU_MEM_UTIL:-0.67}"  # fraction of GPU memory
EDGE2_PORT="${CLOVER_EDGE2_PORT:-8001}"
EDGE2_MAX_MODEL_LEN="${CLOVER_EDGE2_MAX_MODEL_LEN:-8192}"
EDGE2_MAX_TOKENS="${CLOVER_EDGE2_MAX_TOKENS:-4096}"
EDGE2_TIMEOUT="${CLOVER_EDGE2_TIMEOUT:-600}"
EDGE2_DTYPE="${CLOVER_EDGE2_DTYPE:-auto}"
EDGE2_TENSOR_PARALLEL_SIZE="${CLOVER_EDGE2_TENSOR_PARALLEL_SIZE:-}"  # defaults to GPU count
# EDGE2 repair sampling params. Dataset-specific defaults below may override
# these values when the matching CLOVER_* variable is not set by the user.
EDGE2_TEMPERATURE="${CLOVER_EDGE2_TEMPERATURE:-0.3}"
EDGE2_TOP_P="${CLOVER_EDGE2_TOP_P:-0.9}"

# --- Common vLLM server ---
HOST="${CLOVER_VLLM_HOST:-127.0.0.1}"
ENABLE_PREFIX_CACHING="${CLOVER_VLLM_ENABLE_PREFIX_CACHING:-true}"
VLLM_SERVER_ARGS="${CLOVER_VLLM_SERVER_ARGS:-}"   # extra args, e.g. "--enforce-eager"
SERVER_READY_TIMEOUT="${CLOVER_VLLM_READY_TIMEOUT:-600}"
PERSIST_SERVER="${CLOVER_VLLM_PERSIST_SERVER:-false}"
WARMUP_SERVER="${CLOVER_VLLM_WARMUP:-true}"

# --- Concurrency & retries ---
EDGE2_BATCH_SIZE="${CLOVER_EDGE2_BATCH_SIZE:-16}"              # EDGE2 request batch size
EDGE2_CONCURRENCY="${CLOVER_EDGE2_CONCURRENCY:-8}"             # concurrent EDGE2 requests
EVAL_CONCURRENCY="${CLOVER_EVAL_CONCURRENCY:-16}"               # overall eval workers (--max-workers)
MAX_PARALLEL_EXECUTION_UNITS="${CLOVER_MAX_PARALLEL_EXECUTION_UNITS:-8}"
MAX_PARALLEL_SLM_NODE_JOBS="${CLOVER_MAX_PARALLEL_SLM_NODE_JOBS:-32}"
MAX_PARALLEL_SLM_SEQUENCES="${CLOVER_MAX_PARALLEL_SLM_SEQUENCES:-32}"
MAX_PENDING_SLM_SEQUENCES="${CLOVER_MAX_PENDING_SLM_SEQUENCES:-64}"
SLM_SCHEDULER="${CLOVER_SLM_SCHEDULER:-tptt}"                  # tptt | fifo

# --- Agent retry budgets ---
AGENT_LOOP_MAX_ITERATIONS="${CLOVER_AGENT_LOOP_MAX_ITERATIONS:-8}"  # edge agent max iterations
EDGE2_MAX_RETRIES="${CLOVER_EDGE2_MAX_RETRIES:-1}"                 # EDGE2 agent max retries (--max-retries)

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

Starts two local vLLM servers (EDGE1 + EDGE2) and runs the selected eval.
All model/server/concurrency settings are configured at the top of this script
or via CLOVER_* environment variables.

DATASET:
  tablebench | wikitq | tablefact

Examples:
  bash benchmarks/run_vllm_eval_clover.sh
  bash benchmarks/run_vllm_eval_clover.sh wikitq --max-cases 20
  bash benchmarks/run_vllm_eval_clover.sh tablebench

Environment variables (key ones):
  CLOVER_EDGE1_MODEL_PATH / CLOVER_EDGE2_MODEL_PATH   Model paths
  CLOVER_EDGE1_GPUS / CLOVER_EDGE2_GPUS               GPU ids per model
  CLOVER_EDGE1_GPU_MEM_UTIL / CLOVER_EDGE2_GPU_MEM_UTIL  GPU memory fraction
  CLOVER_EDGE1_PORT / CLOVER_EDGE2_PORT               vLLM ports
  CLOVER_EVAL_CONCURRENCY / CLOVER_EDGE2_CONCURRENCY Eval parallelism
  CLOVER_AGENT_LOOP_MAX_ITERATIONS / CLOVER_EDGE2_MAX_RETRIES  Retry budgets
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
  tablebench|wikitq|tablefact) ;;
  *) echo "Unsupported dataset: ${DATASET}" >&2; exit 2 ;;
esac

# Empirical local 7B+3B defaults from the 2026-06-22 server sweep:
# - TableFact is harmed badly by local repair/retry and benefits from
#   deterministic EDGE2 generation.
# - TableBench/WikiTQ recover a few points from modest EDGE2 sampling and one
#   repair round, though they remain below the older DeepSeek-backed runs.
if [[ -z "${CLOVER_EDGE2_TEMPERATURE+x}" ]]; then
  case "${DATASET}" in
    tablefact) EDGE2_TEMPERATURE="0.0" ;;
    tablebench|wikitq) EDGE2_TEMPERATURE="0.3" ;;
  esac
fi
if [[ -z "${CLOVER_EDGE2_TOP_P+x}" ]]; then
  case "${DATASET}" in
    tablefact) EDGE2_TOP_P="1.0" ;;
    tablebench|wikitq) EDGE2_TOP_P="0.9" ;;
  esac
fi
if [[ -z "${CLOVER_EDGE2_MAX_RETRIES+x}" ]]; then
  case "${DATASET}" in
    tablefact) EDGE2_MAX_RETRIES="0" ;;
    tablebench|wikitq) EDGE2_MAX_RETRIES="1" ;;
  esac
fi

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "No executable 'python' found. Activate the intended environment first." >&2
  exit 1
fi

for label in EDGE1 EDGE2; do
  model_var="${label}_MODEL_PATH"
  model_path="${!model_var}"
  if [[ "${model_path}" == /path/to/* ]]; then
    echo "Set ${label}_MODEL_PATH at the top of this script." >&2
    exit 1
  fi
  if [[ "${model_path}" == /* && ! -e "${model_path}" ]]; then
    echo "${label} model path not found: ${model_path}" >&2
    exit 1
  fi
done

EDGE1_GPUS="$(validate_gpus EDGE1_GPUS "${EDGE1_GPUS}")"
EDGE2_GPUS="$(validate_gpus EDGE2_GPUS "${EDGE2_GPUS}")"
EDGE1_GPU_COUNT="$(count_gpus "${EDGE1_GPUS}")"
EDGE2_GPU_COUNT="$(count_gpus "${EDGE2_GPUS}")"
EDGE1_TENSOR_PARALLEL_SIZE="$(resolve_tp EDGE1 "${EDGE1_TENSOR_PARALLEL_SIZE}" "${EDGE1_GPU_COUNT}")"
EDGE2_TENSOR_PARALLEL_SIZE="$(resolve_tp EDGE2 "${EDGE2_TENSOR_PARALLEL_SIZE}" "${EDGE2_GPU_COUNT}")"

PERSIST_SERVER="$(normalize_bool "${PERSIST_SERVER}")"
WARMUP_SERVER="$(normalize_bool "${WARMUP_SERVER}")"
ENABLE_PREFIX_CACHING="$(normalize_bool "${ENABLE_PREFIX_CACHING}")"
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
EDGE_REVIEW_PROACTIVE="$(normalize_bool "${EDGE_REVIEW_PROACTIVE}")"

if [[ ! "${EDGE_REVIEW_MODE}" =~ ^(off|shadow|safe)$ ]]; then
  echo "Invalid CLOVER_EDGE_REVIEW_MODE: ${EDGE_REVIEW_MODE}" >&2
  exit 1
fi

# Dataset roots.
DATASETS_ROOT="${CLOVER_DATASETS_ROOT:-${REPO_ROOT}/datasets}"
TABLEBENCH_ROOT="${TABLEBENCH_ROOT:-${DATASETS_ROOT}/tablebench}"
WIKITQ_ROOT="${WIKITQ_ROOT:-${DATASETS_ROOT}/wikitq}"
TABLEFACT_ROOT="${TABLEFACT_ROOT:-${DATASETS_ROOT}/tablefact}"
WIKITQ_SPLIT="${WIKITQ_SPLIT:-pristine-unseen-tables}"
TABLEFACT_SPLIT="${TABLEFACT_SPLIT:-test}"
TABLEFACT_SUBSET="${TABLEFACT_SUBSET:-small}"

case "${DATASET}" in
  tablebench) DATASET_ROOT="${TABLEBENCH_ROOT}" ;;
  wikitq) DATASET_ROOT="${WIKITQ_ROOT}" ;;
  tablefact) DATASET_ROOT="${TABLEFACT_ROOT}" ;;
esac
if ! find "${DATASET_ROOT}" -mindepth 2 -maxdepth 2 -name cases.jsonl \
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
EDGE1_SERVED_MODEL_NAME="$(basename "${EDGE1_MODEL_PATH}")"
EDGE2_SERVED_MODEL_NAME="$(basename "${EDGE2_MODEL_PATH}")"

EDGE1_PID=""
EDGE2_PID=""
STARTED_EDGE1=0
STARTED_EDGE2=0

cleanup() {
  if [[ "${PERSIST_SERVER}" != "true" ]]; then
    for pid_var in EDGE1_PID EDGE2_PID; do
      pid="${!pid_var}"
      if [[ -n "${pid}" ]]; then
        kill "${pid}" >/dev/null 2>&1 || true
        wait "${pid}" >/dev/null 2>&1 || true
      fi
    done
  fi
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

# -----------------------------------------------------------------------------
# vLLM server control
# -----------------------------------------------------------------------------
server_ready() {
  local base_url="$1"
  "${PYTHON_BIN}" - "${base_url}" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
with urllib.request.urlopen(sys.argv[1].rstrip("/") + "/models", timeout=2) as r:
    json.loads(r.read().decode("utf-8"))
PY
}

start_vllm() {
  local label="$1" model_path="$2" gpus="$3" tp="$4" port="$5"
  local mem_util="$6" max_model_len="$7" dtype="$8" served_name="$9" log_file="${10}"
  local base_url="http://${HOST}:${port}/v1"

  if server_ready "${base_url}"; then
    echo "Using existing ${label} server: ${base_url}" >&2
    return 0
  fi

  local extra_args=()
  if [[ -n "${VLLM_SERVER_ARGS}" ]]; then
    read -r -a extra_args <<< "${VLLM_SERVER_ARGS}"
  fi
  local cmd=(
    env CUDA_VISIBLE_DEVICES="${gpus}"
    "${VLLM_BIN}" serve "${model_path}"
    --served-model-name "${served_name}"
    --host "${HOST}"
    --port "${port}"
    --dtype "${dtype}"
    --gpu-memory-utilization "${mem_util}"
    --tensor-parallel-size "${tp}"
    --max-model-len "${max_model_len}"
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
    server_ready "${base_url}" && break
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "${label} vLLM exited before becoming ready. See ${log_file}" >&2
      exit 1
    fi
    sleep 1
  done
  if ! server_ready "${base_url}"; then
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
    "${ENABLE_STATIC_FAST_PATH}" "${ENABLE_STATIC_FINALIZATION}" <<'PY'
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
    "max_retries": 2,
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
    "edge_review_max_actions": 4,
    "edge_review_max_rows": 5,
    "edge_review_max_columns": 5,
    "edge_review_max_facts": 40,
    "tptt_coalesce_ms": 60,
    "tptt_prefix_tokens": 200,
    "max_tptt_leaf_sequences_per_tree": 128,
}
Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

write_edge2_config() {
  local out="$1"
  "${PYTHON_BIN}" - "${out}" "${EDGE2_SERVED_MODEL_NAME}" "${EDGE2_BASE_URL}" \
    "${EDGE2_MAX_TOKENS}" "${EDGE2_TIMEOUT}" "${EDGE2_TEMPERATURE}" "${EDGE2_TOP_P}" <<'PY'
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
    "max_retries": 2,
    "max_tokens": int(max_tokens),
    "temperature": float(temperature),
    "top_p": float(top_p),
    "http2": False,
    "trust_env": False,
}
Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

# -----------------------------------------------------------------------------
# Print configuration
# -----------------------------------------------------------------------------
echo "=========================================" >&2
echo " Dual Local vLLM Configuration" >&2
echo "=========================================" >&2
echo " Dataset:              ${DATASET}" >&2
echo "" >&2
echo " Edge model (local):   ${EDGE1_MODEL_PATH}" >&2
echo "   GPUs:               ${EDGE1_GPUS} (tp=${EDGE1_TENSOR_PARALLEL_SIZE})" >&2
echo "   GPU mem util:       ${EDGE1_GPU_MEM_UTIL}" >&2
echo "   Endpoint:           ${EDGE1_BASE_URL}" >&2
echo "   Max model len:      ${EDGE1_MAX_MODEL_LEN}" >&2
echo "   Max tokens:         ${EDGE1_MAX_TOKENS}" >&2
echo "   Temperature:        ${EDGE1_TEMPERATURE}" >&2
echo "   Top-p:              ${EDGE1_TOP_P}" >&2
echo "" >&2
echo " EDGE2 model:          ${EDGE2_MODEL_PATH}" >&2
echo "   GPUs:               ${EDGE2_GPUS} (tp=${EDGE2_TENSOR_PARALLEL_SIZE})" >&2
echo "   GPU mem util:       ${EDGE2_GPU_MEM_UTIL}" >&2
echo "   Endpoint:           ${EDGE2_BASE_URL}" >&2
echo "   Max model len:      ${EDGE2_MAX_MODEL_LEN}" >&2
echo "   Max tokens:         ${EDGE2_MAX_TOKENS}" >&2
echo "   Temperature:        ${EDGE2_TEMPERATURE}" >&2
echo "   Top-p:              ${EDGE2_TOP_P}" >&2
echo "" >&2
echo " Eval concurrency:     ${EVAL_CONCURRENCY} workers / ${EDGE2_CONCURRENCY} edge2" >&2
echo " Edge agent max iters: ${AGENT_LOOP_MAX_ITERATIONS}" >&2
echo " EDGE2 max retries:    ${EDGE2_MAX_RETRIES}" >&2
echo "=========================================" >&2

# -----------------------------------------------------------------------------
# Start both vLLM servers
# -----------------------------------------------------------------------------
EDGE1_PID="$(start_vllm \
  "edge" "${EDGE1_MODEL_PATH}" "${EDGE1_GPUS}" "${EDGE1_TENSOR_PARALLEL_SIZE}" \
  "${EDGE1_PORT}" "${EDGE1_GPU_MEM_UTIL}" "${EDGE1_MAX_MODEL_LEN}" \
  "${EDGE1_DTYPE}" "${EDGE1_SERVED_MODEL_NAME}" "${EDGE1_SERVER_LOG}")"
STARTED_EDGE1=1
warmup_server "${EDGE1_BASE_URL}" "${EDGE1_SERVED_MODEL_NAME}" "edge"

EDGE2_PID="$(start_vllm \
  "edge2" "${EDGE2_MODEL_PATH}" "${EDGE2_GPUS}" "${EDGE2_TENSOR_PARALLEL_SIZE}" \
  "${EDGE2_PORT}" "${EDGE2_GPU_MEM_UTIL}" "${EDGE2_MAX_MODEL_LEN}" \
  "${EDGE2_DTYPE}" "${EDGE2_SERVED_MODEL_NAME}" "${EDGE2_SERVER_LOG}")"
STARTED_EDGE2=1
warmup_server "${EDGE2_BASE_URL}" "${EDGE2_SERVED_MODEL_NAME}" "edge2"

# -----------------------------------------------------------------------------
# Generate configs
# -----------------------------------------------------------------------------
LOCAL_CONFIG="${TMP_DIR}/edge1_local_slm_config.json"
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
# Print brief summary (including combined per-case max context tokens) to stdout
# -----------------------------------------------------------------------------
SUMMARY_FILE="${OUTPUT_ROOT}/${RUN_NAME}/run_summary.json"
if [[ -f "${SUMMARY_FILE}" ]]; then
  "${PYTHON_BIN}" - "${SUMMARY_FILE}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
brief = summary.get("brief_summary") or {}
ctx_stats = summary.get("max_context_tokens_stats") or {}
remote_ctx = ctx_stats.get("remote") or {}
local_ctx = ctx_stats.get("local") or {}
combined_ctx = ctx_stats.get("combined") or {}

print("\n===== Brief Summary =====")
print(f"{'Benchmark':<32}: {brief.get('benchmark') or summary.get('dataset')}")
print(f"{'Cloud Model':<32}: {brief.get('cloud_model') or (summary.get('remote_llm') or {}).get('model')}")
print(f"{'Edge Model':<32}: {brief.get('edge_model') or (summary.get('local_slm') or {}).get('model')}")
print(f"{'Acc. (%)':<32}: {brief.get('acc_pct')}")
print(f"{'Cloud Tokens':<32}: {brief.get('cloud_tokens')}")
print(f"{'Edge Tokens':<32}: {brief.get('edge_tokens')}")
print(f"{'API Cost (USD)':<32}: {brief.get('api_cost_usd')}")
print(f"{'Cloud Tok/Q':<32}: {brief.get('cloud_tokens_per_q')}")
print(f"{'Edge Tok/Q':<32}: {brief.get('edge_tokens_per_q')}")
print(f"{'Calls/Q':<32}: {brief.get('calls_per_q')}")
print(f"{'API Cost/Q (USD)':<32}: {brief.get('api_cost_per_q_usd')}")
print(f"{'Avg Max Ctx Tok/Q (combined)':<32}: {round(combined_ctx.get('mean', 0.0), 2)}")
print(f"{'  remote  max/mean/min':<32}: {remote_ctx.get('max')} / {round(remote_ctx.get('mean', 0.0), 2)} / {remote_ctx.get('min')}")
print(f"{'  local   max/mean/min':<32}: {local_ctx.get('max')} / {round(local_ctx.get('mean', 0.0), 2)} / {local_ctx.get('min')}")
print(f"{'  combined max/mean/min':<32}: {combined_ctx.get('max')} / {round(combined_ctx.get('mean', 0.0), 2)} / {combined_ctx.get('min')}")
print("=========================")
PY
else
  echo "Summary file not found: ${SUMMARY_FILE}" >&2
fi

exit ${EVAL_EXIT}
