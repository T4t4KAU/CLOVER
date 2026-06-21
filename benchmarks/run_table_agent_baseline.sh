#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Edge-only ReAcTable / Orchestra baseline runner for CLOVER datasets.
#
# Usage:
#   bash benchmarks/run_table_agent_baseline.sh BASELINE DATASET EDGE_MODEL_PATH [options...]
#
# BASELINE:
#   reactable | orchestra
#
# DATASET:
#   tablebench | wikitq | tablefact
#
# The script starts or reuses a local OpenAI-compatible vLLM endpoint and then
# runs a single-model baseline with CLOVER's converted datasets and native
# metrics. No cloud model config or API key is used.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/run_table_agent_baseline.sh BASELINE DATASET EDGE_MODEL_PATH [options...]

BASELINE:
  reactable | orchestra

DATASET:
  tablebench | wikitq | tablefact

Examples:
  bash benchmarks/run_table_agent_baseline.sh reactable wikitq /root/autodl-tmp/models/Qwen2.5-3B-Instruct --sample-size 100
  bash benchmarks/run_table_agent_baseline.sh orchestra tablefact /root/autodl-tmp/models/Qwen2.5-3B-Instruct --max-workers 16

Common options forwarded to the Python runner:
  --max-cases N
  --sample-size N
  --case-id ID              Repeatable
  --seed N
  --max-workers N
  --repeat-times N
  --max-iters N
  --max-demo N              ReAcTable only
  --orchestra-mode 2agent|3agent|both
  --line-limit N|inf
  --output-root PATH
  --run-name NAME
  --overwrite
  --validate-only

Useful environment variables:
  CLOVER_VLLM_HOST=127.0.0.1
  CLOVER_VLLM_PORT=8000
  CLOVER_VLLM_GPUS=0
  CLOVER_VLLM_SERVED_MODEL_NAME=<name>
  CLOVER_VLLM_MAX_MODEL_LEN=16384
  CLOVER_VLLM_GPU_MEMORY_UTILIZATION=0.88
  CLOVER_VLLM_MAX_NUM_SEQS=32
  CLOVER_VLLM_SERVER_ARGS="--swap-space 4"
  CLOVER_VLLM_PERSIST_SERVER=false
  CLOVER_BASELINE_TEMPERATURE=0.7
  CLOVER_BASELINE_MAX_TOKENS=1024
  CLOVER_DATASETS_ROOT=<repo>/datasets
  PYTHON_BIN=<repo>/.venv/bin/python
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

BASELINE="${1:-}"
DATASET="${2:-}"
EDGE_MODEL_PATH="${3:-}"
if [[ -z "${BASELINE}" || -z "${DATASET}" || -z "${EDGE_MODEL_PATH}" ]]; then
  usage >&2
  exit 2
fi
shift 3
EXTRA_ARGS=("$@")

BASELINE="$(printf '%s' "${BASELINE}" | tr '[:upper:]' '[:lower:]')"
case "${BASELINE}" in
  reactable|orchestra) ;;
  reac-table|reac_table|react) BASELINE="reactable" ;;
  *)
    echo "Unsupported baseline: ${BASELINE}" >&2
    exit 2
    ;;
esac

DATASET="$(printf '%s' "${DATASET}" | tr '[:upper:]' '[:lower:]')"
case "${DATASET}" in
  tablebench|wikitq|tablefact) ;;
  wikitablequestions) DATASET="wikitq" ;;
  tabfact) DATASET="tablefact" ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

PYTHON_CMD=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_CMD=("${REPO_ROOT}/.venv/bin/python")
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=("$(command -v python)")
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=("$(command -v python3)")
elif command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=("$(command -v uv)" run python)
else
  echo "No Python found. Create/activate .venv with uv first." >&2
  exit 1
fi

if [[ "${#PYTHON_CMD[@]}" -eq 1 && ! -x "${PYTHON_CMD[0]}" ]]; then
  echo "Python is not executable: ${PYTHON_CMD[0]}" >&2
  exit 1
fi

if [[ "${EDGE_MODEL_PATH}" == /path/to/* ]]; then
  echo "Pass a real edge model path, e.g. /root/autodl-tmp/models/<model>." >&2
  exit 2
fi

HOST="${CLOVER_VLLM_HOST:-127.0.0.1}"
PORT="${CLOVER_VLLM_PORT:-8000}"
BASE_URL="http://${HOST}:${PORT}/v1"
SERVED_MODEL_NAME="${CLOVER_VLLM_SERVED_MODEL_NAME:-$(basename "${EDGE_MODEL_PATH}")}"
GPU_DEVICES="${CLOVER_VLLM_GPUS:-0}"
DTYPE="${CLOVER_VLLM_DTYPE:-auto}"
MAX_MODEL_LEN="${CLOVER_VLLM_MAX_MODEL_LEN:-}"
GPU_MEMORY_UTILIZATION="${CLOVER_VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
MAX_NUM_SEQS="${CLOVER_VLLM_MAX_NUM_SEQS:-}"
VLLM_SERVER_ARGS="${CLOVER_VLLM_SERVER_ARGS:-}"
SERVER_READY_TIMEOUT="${CLOVER_VLLM_READY_TIMEOUT:-600}"
PERSIST_SERVER="${CLOVER_VLLM_PERSIST_SERVER:-false}"
WARMUP_SERVER="${CLOVER_VLLM_WARMUP:-true}"
ENABLE_PREFIX_CACHING="${CLOVER_VLLM_ENABLE_PREFIX_CACHING:-true}"

DATASETS_ROOT="${CLOVER_DATASETS_ROOT:-${REPO_ROOT}/datasets}"
DATASET_ROOT="${DATASETS_ROOT}/${DATASET}"
if ! find "${DATASET_ROOT}" -mindepth 2 -maxdepth 2 -name cases.jsonl \
    -print -quit 2>/dev/null | grep -q .; then
  echo "Converted ${DATASET} dataset not found: ${DATASET_ROOT}" >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/benchmark/runs}"
RUN_NAME="${RUN_NAME:-${DATASET}_${BASELINE}_${TIMESTAMP}}"
SERVER_LOG="${SERVER_LOG:-${OUTPUT_ROOT}/${RUN_NAME}_vllm_server.log}"

normalize_bool() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) printf '%s' true ;;
    0|false|no|n|off) printf '%s' false ;;
    *)
      echo "Invalid boolean value: $1" >&2
      exit 1
      ;;
  esac
}
PERSIST_SERVER="$(normalize_bool "${PERSIST_SERVER}")"
WARMUP_SERVER="$(normalize_bool "${WARMUP_SERVER}")"
ENABLE_PREFIX_CACHING="$(normalize_bool "${ENABLE_PREFIX_CACHING}")"

GPU_DEVICES="$(printf '%s' "${GPU_DEVICES}" | tr -d '[:space:]')"
if [[ ! "${GPU_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "Invalid CLOVER_VLLM_GPUS: ${GPU_DEVICES}" >&2
  exit 1
fi
IFS=',' read -r -a GPU_DEVICE_ITEMS <<< "${GPU_DEVICES}"
GPU_COUNT="${#GPU_DEVICE_ITEMS[@]}"
TENSOR_PARALLEL_SIZE="${CLOVER_VLLM_TENSOR_PARALLEL_SIZE:-${GPU_COUNT}}"
export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"

if [[ -n "${MAX_NUM_SEQS}" && ! "${MAX_NUM_SEQS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid CLOVER_VLLM_MAX_NUM_SEQS: ${MAX_NUM_SEQS}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
TMP_DIR="$(mktemp -d)"
SERVER_PID=""
STARTED_SERVER=0

cleanup() {
  if [[ "${STARTED_SERVER}" == "1" && -n "${SERVER_PID}" \
      && "${PERSIST_SERVER}" != "true" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

server_ready() {
  "${PYTHON_CMD[@]}" - "${BASE_URL}" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1].rstrip("/") + "/models", timeout=2) as response:
    json.loads(response.read().decode("utf-8"))
PY
}

VLLM_CMD_PREFIX=()
if [[ -x "${REPO_ROOT}/.venv/bin/vllm" ]]; then
  VLLM_CMD_PREFIX=("${REPO_ROOT}/.venv/bin/vllm")
elif command -v vllm >/dev/null 2>&1; then
  VLLM_CMD_PREFIX=("$(command -v vllm)")
elif command -v uv >/dev/null 2>&1; then
  VLLM_CMD_PREFIX=("$(command -v uv)" run vllm)
else
  VLLM_CMD_PREFIX=()
fi

if server_ready; then
  echo "Using existing OpenAI-compatible server: ${BASE_URL}" >&2
else
  if [[ "${EDGE_MODEL_PATH}" == /* && ! -e "${EDGE_MODEL_PATH}" ]]; then
    echo "Edge model path not found for vLLM launch: ${EDGE_MODEL_PATH}" >&2
    exit 1
  fi
  if [[ "${#VLLM_CMD_PREFIX[@]}" -eq 0 ]]; then
    echo "vLLM CLI not found in .venv or PATH." >&2
    exit 1
  fi
  EXTRA_SERVER_ARGS=()
  if [[ -n "${VLLM_SERVER_ARGS}" ]]; then
    read -r -a EXTRA_SERVER_ARGS <<< "${VLLM_SERVER_ARGS}"
  fi
  VLLM_CMD=(
    "${VLLM_CMD_PREFIX[@]}" serve "${EDGE_MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${HOST}"
    --port "${PORT}"
    --dtype "${DTYPE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  )
  [[ -n "${MAX_MODEL_LEN}" ]] && VLLM_CMD+=(--max-model-len "${MAX_MODEL_LEN}")
  [[ -n "${MAX_NUM_SEQS}" ]] && VLLM_CMD+=(--max-num-seqs "${MAX_NUM_SEQS}")
  [[ "${ENABLE_PREFIX_CACHING}" == "true" ]] && VLLM_CMD+=(--enable-prefix-caching)
  [[ "${#EXTRA_SERVER_ARGS[@]}" -gt 0 ]] && VLLM_CMD+=("${EXTRA_SERVER_ARGS[@]}")

  echo "Starting vLLM edge model" >&2
  echo "  baseline: ${BASELINE}" >&2
  echo "  dataset: ${DATASET}" >&2
  echo "  model: ${EDGE_MODEL_PATH}" >&2
  echo "  endpoint: ${BASE_URL}" >&2
  echo "  log: ${SERVER_LOG}" >&2
  "${VLLM_CMD[@]}" >"${SERVER_LOG}" 2>&1 &
  SERVER_PID="$!"
  STARTED_SERVER=1

  for ((second = 0; second < SERVER_READY_TIMEOUT; second++)); do
    server_ready && break
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      echo "vLLM exited before becoming ready. See ${SERVER_LOG}" >&2
      exit 1
    fi
    sleep 1
  done
  if ! server_ready; then
    echo "Timed out waiting for vLLM after ${SERVER_READY_TIMEOUT}s. See ${SERVER_LOG}" >&2
    exit 1
  fi
fi

if [[ "${WARMUP_SERVER}" == "true" ]]; then
  echo "Warming up edge model" >&2
  "${PYTHON_CMD[@]}" - "${BASE_URL}" "${SERVED_MODEL_NAME}" <<'PY' >/dev/null
import json
import sys
import urllib.request

base_url, model = sys.argv[1:3]
payload = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "Reply with OK."}],
    "temperature": 0,
    "max_tokens": 2,
    "stream": False,
}).encode("utf-8")
request = urllib.request.Request(
    base_url.rstrip("/") + "/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=180) as response:
    json.loads(response.read().decode("utf-8"))
PY
fi

MODEL_CONFIG="${TMP_DIR}/edge_model_config.json"
LOCAL_MAX_TOKENS="${CLOVER_BASELINE_MAX_TOKENS:-1024}"
LOCAL_TIMEOUT_SECONDS="${CLOVER_BASELINE_TIMEOUT_SECONDS:-1800}"
BASELINE_TEMPERATURE="${CLOVER_BASELINE_TEMPERATURE:-}"
"${PYTHON_CMD[@]}" - "${MODEL_CONFIG}" "${SERVED_MODEL_NAME}" "${BASE_URL}" \
  "${LOCAL_MAX_TOKENS}" "${LOCAL_TIMEOUT_SECONDS}" "${BASELINE_TEMPERATURE}" <<'PY'
import json
import sys
from pathlib import Path

path, model, base_url, max_tokens, timeout, temperature = sys.argv[1:7]
payload = {
    "provider": "local",
    "api_type": "chat_completions",
    "api_key": "EMPTY",
    "base_url": base_url,
    "model": model,
    "timeout": int(timeout),
    "max_retries": 2,
    "max_tokens": int(max_tokens),
    "top_p": 1.0,
    "http2": False,
    "trust_env": False,
}
if temperature:
    payload["temperature"] = float(temperature)
Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

RUNNER_CMD=(
  "${PYTHON_CMD[@]}" -m benchmarks.baselines.table_agent_baselines
  --baseline "${BASELINE}"
  --dataset "${DATASET}"
  --model-config "${MODEL_CONFIG}"
  --datasets-root "${DATASETS_ROOT}"
  --dataset-root "${DATASET_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
  --max-tokens "${LOCAL_MAX_TOKENS}"
)
[[ -n "${BASELINE_TEMPERATURE}" ]] && RUNNER_CMD+=(--temperature "${BASELINE_TEMPERATURE}")

case "${DATASET}" in
  tablebench)
    RUNNER_CMD+=(--qtype FactChecking --qtype NumericalReasoning)
    ;;
  wikitq)
    RUNNER_CMD+=(--wikitq-split "${WIKITQ_SPLIT:-pristine-unseen-tables}")
    ;;
  tablefact)
    RUNNER_CMD+=(--tablefact-split "${TABLEFACT_SPLIT:-test}" --tablefact-subset "${TABLEFACT_SUBSET:-small}")
    ;;
esac
[[ "${#EXTRA_ARGS[@]}" -gt 0 ]] && RUNNER_CMD+=("${EXTRA_ARGS[@]}")

echo "Running ${BASELINE} baseline on ${DATASET}" >&2
PYTHONWARNINGS="ignore" \
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
"${RUNNER_CMD[@]}"
