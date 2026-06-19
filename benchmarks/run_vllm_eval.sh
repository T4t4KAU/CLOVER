#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# User settings
# Edit these values, then run: bash benchmarks/run_vllm_eval.sh
# Environment variables or positional arguments can still override them.
# =============================================================================
DATASET="${CLOVER_EVAL_DATASET:-tablebench}"  # tablebench | wikitq | tablefact
EDGE_MODEL_PATH="${CLOVER_EDGE_MODEL_PATH:-/path/to/Qwen2.5-3B-Instruct}"
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
GPU_DEVICES="${CLOVER_VLLM_GPUS:-0}"  # Example: "0,1"

# Use the same `python` executable that the user gets in the current shell.
PYTHON_BIN="${PYTHON_BIN:-$(command -v python 2>/dev/null || true)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/run_vllm_eval.sh [DATASET] [EDGE_MODEL_PATH] [eval options...]

DATASET:
  tablebench | wikitq | tablefact

Examples:
  # Use the settings at the top of this script:
  bash benchmarks/run_vllm_eval.sh

  # Temporarily override dataset/model:
  bash benchmarks/run_vllm_eval.sh tablebench /models/Qwen2.5-3B-Instruct --max-cases 10
  bash benchmarks/run_vllm_eval.sh wikitq Qwen/Qwen2.5-3B-Instruct --sample-size 100
  TABLEFACT_SUBSET=simple bash benchmarks/run_vllm_eval.sh tablefact /models/Qwen2.5-3B-Instruct

The script starts a vLLM OpenAI-compatible server for CLOVER's local edge
model, creates a temporary local model config, and launches the selected eval.
Cloud planning/synthesis uses CLOVER_REMOTE_LLM_CONFIG (DeepSeek by default).
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# Optional positional overrides preserve the previous invocation style.
if [[ "${1:-}" != "" && "${1:-}" != --* ]]; then
  DATASET="$1"
  shift
fi
if [[ "${1:-}" != "" && "${1:-}" != --* ]]; then
  EDGE_MODEL_PATH="$1"
  shift
fi

DATASET="$(printf '%s' "${DATASET}" | tr '[:upper:]' '[:lower:]')"
MODEL_PATH="${EDGE_MODEL_PATH}"
case "${DATASET}" in
  tablebench) ;;
  wikitq|wikitablequestions) DATASET="wikitq" ;;
  tablefact|tabfact) DATASET="tablefact" ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    usage >&2
    exit 2
    ;;
esac
EXTRA_EVAL_ARGS=("$@")

HOST="${CLOVER_VLLM_HOST:-${CLOVER_LOCAL_HOST:-127.0.0.1}}"
PORT="${CLOVER_VLLM_PORT:-${CLOVER_LOCAL_PORT:-8000}}"
BASE_URL="http://${HOST}:${PORT}/v1"
SERVED_MODEL_NAME="${CLOVER_VLLM_SERVED_MODEL_NAME:-clover-edge-model}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/benchmark/runs}"
RUN_NAME="${RUN_NAME:-${DATASET}_vllm_$(date +%Y%m%d_%H%M%S)}"
SERVER_LOG="${SERVER_LOG:-${OUTPUT_ROOT}/${RUN_NAME}_vllm_server.log}"

DATASETS_ROOT="${CLOVER_DATASETS_ROOT:-${REPO_ROOT}/datasets}"
TABLEBENCH_ROOT="${TABLEBENCH_ROOT:-${DATASETS_ROOT}/tablebench}"
WIKITQ_ROOT="${WIKITQ_ROOT:-${DATASETS_ROOT}/wikitq}"
TABLEFACT_ROOT="${TABLEFACT_ROOT:-${DATASETS_ROOT}/tablefact}"
WIKITQ_SPLIT="${WIKITQ_SPLIT:-pristine-unseen-tables}"
TABLEFACT_SPLIT="${TABLEFACT_SPLIT:-test}"
TABLEFACT_SUBSET="${TABLEFACT_SUBSET:-}"

REMOTE_LLM_CONFIG="${CLOVER_REMOTE_LLM_CONFIG:-${REPO_ROOT}/model_config/deepseek_remote_llm_config.json}"
SYNTHESIZE_LLM_CONFIG="${CLOVER_SYNTHESIZE_LLM_CONFIG:-${REMOTE_LLM_CONFIG}}"
LOCAL_MAX_TOKENS="${CLOVER_LOCAL_MAX_TOKENS:-512}"
LOCAL_TIMEOUT_SECONDS="${CLOVER_LOCAL_TIMEOUT_SECONDS:-600}"
AGENT_LOOP_MAX_ITERATIONS="${CLOVER_AGENT_LOOP_MAX_ITERATIONS:-8}"

DTYPE="${CLOVER_VLLM_DTYPE:-auto}"
MAX_MODEL_LEN="${CLOVER_VLLM_MAX_MODEL_LEN:-}"
GPU_MEMORY_UTILIZATION="${CLOVER_VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
ENABLE_PREFIX_CACHING="${CLOVER_VLLM_ENABLE_PREFIX_CACHING:-true}"
VLLM_SERVER_ARGS="${CLOVER_VLLM_SERVER_ARGS:-}"
SERVER_READY_TIMEOUT="${CLOVER_VLLM_READY_TIMEOUT:-600}"

REMOTE_BATCH_SIZE="${CLOVER_REMOTE_BATCH_SIZE:-16}"
REMOTE_CONCURRENCY="${CLOVER_REMOTE_CONCURRENCY:-8}"
EVAL_CONCURRENCY="${CLOVER_EVAL_CONCURRENCY:-16}"
MAX_PARALLEL_EXECUTION_UNITS="${CLOVER_MAX_PARALLEL_EXECUTION_UNITS:-8}"
MAX_PARALLEL_SLM_NODE_JOBS="${CLOVER_MAX_PARALLEL_SLM_NODE_JOBS:-32}"
MAX_PARALLEL_SLM_SEQUENCES="${CLOVER_MAX_PARALLEL_SLM_SEQUENCES:-32}"
MAX_PENDING_SLM_SEQUENCES="${CLOVER_MAX_PENDING_SLM_SEQUENCES:-64}"
SLM_SCHEDULER="${CLOVER_SLM_SCHEDULER:-tptt}"

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "The current shell has no executable 'python' command." >&2
  echo "Activate the intended environment first, then rerun this script." >&2
  exit 1
fi
if [[ "${MODEL_PATH}" == /path/to/* ]]; then
  echo "Set EDGE_MODEL_PATH at the top of this script or pass a model path." >&2
  exit 1
fi
if [[ "${MODEL_PATH}" == /* && ! -e "${MODEL_PATH}" ]]; then
  echo "Local edge model path not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -e "${MODEL_PATH}" && "${MODEL_PATH}" != */* ]]; then
  echo "Model must be a local path or Hugging Face model id: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${REMOTE_LLM_CONFIG}" ]]; then
  echo "Remote LLM config not found: ${REMOTE_LLM_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${SYNTHESIZE_LLM_CONFIG}" ]]; then
  echo "Synthesis LLM config not found: ${SYNTHESIZE_LLM_CONFIG}" >&2
  exit 1
fi

if grep -Eq '"api_key_env"[[:space:]]*:[[:space:]]*"DEEPSEEK_API_KEY"' \
    "${REMOTE_LLM_CONFIG}" "${SYNTHESIZE_LLM_CONFIG}" \
    && [[ -z "${DEEPSEEK_API_KEY}" ]]; then
  echo "DEEPSEEK_API_KEY is empty." >&2
  echo "Set it in the user settings block at the top of this script." >&2
  exit 1
fi
export DEEPSEEK_API_KEY

GPU_DEVICES="$(printf '%s' "${GPU_DEVICES}" | tr -d '[:space:]')"
if [[ ! "${GPU_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "Invalid GPU_DEVICES: ${GPU_DEVICES}" >&2
  echo "Use comma-separated GPU ids, for example 0 or 0,1." >&2
  exit 1
fi
IFS=',' read -r -a GPU_DEVICE_ITEMS <<< "${GPU_DEVICES}"
GPU_COUNT=0
for gpu_id in "${GPU_DEVICE_ITEMS[@]}"; do
  GPU_COUNT=$((GPU_COUNT + 1))
done
export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
TENSOR_PARALLEL_SIZE="${CLOVER_VLLM_TENSOR_PARALLEL_SIZE:-${GPU_COUNT}}"
if [[ ! "${TENSOR_PARALLEL_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid tensor parallel size: ${TENSOR_PARALLEL_SIZE}" >&2
  exit 1
fi
if ((TENSOR_PARALLEL_SIZE > GPU_COUNT)); then
  echo "Tensor parallel size (${TENSOR_PARALLEL_SIZE}) exceeds visible GPUs (${GPU_COUNT})." >&2
  exit 1
fi

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

"${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

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

mkdir -p "${OUTPUT_ROOT}"
TMP_DIR="$(mktemp -d)"
SERVER_PID=""
STARTED_SERVER=0

cleanup() {
  if [[ "${STARTED_SERVER}" == "1" && -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

server_ready() {
  "${PYTHON_BIN}" - "${BASE_URL}" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1].rstrip("/") + "/models", timeout=2) as response:
    json.loads(response.read().decode("utf-8"))
PY
}

if server_ready; then
  echo "Using existing OpenAI-compatible server: ${BASE_URL}" >&2
else
  EXTRA_SERVER_ARGS=()
  if [[ -n "${VLLM_SERVER_ARGS}" ]]; then
    read -r -a EXTRA_SERVER_ARGS <<< "${VLLM_SERVER_ARGS}"
  fi
  VLLM_CMD=(
    "${VLLM_BIN}" serve "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${HOST}"
    --port "${PORT}"
    --dtype "${DTYPE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  )
  [[ -n "${MAX_MODEL_LEN}" ]] && VLLM_CMD+=(--max-model-len "${MAX_MODEL_LEN}")
  [[ "${ENABLE_PREFIX_CACHING}" == "true" ]] && VLLM_CMD+=(--enable-prefix-caching)
  [[ "${#EXTRA_SERVER_ARGS[@]}" -gt 0 ]] && VLLM_CMD+=("${EXTRA_SERVER_ARGS[@]}")

  echo "Starting vLLM" >&2
  echo "  model: ${MODEL_PATH}" >&2
  echo "  python: ${PYTHON_BIN}" >&2
  echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}" >&2
  echo "  tensor_parallel_size: ${TENSOR_PARALLEL_SIZE}" >&2
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
    echo "Timed out waiting for vLLM after ${SERVER_READY_TIMEOUT}s." >&2
    echo "See ${SERVER_LOG}" >&2
    exit 1
  fi
fi

LOCAL_CONFIG="${TMP_DIR}/vllm_local_slm_config.json"
"${PYTHON_BIN}" - "${LOCAL_CONFIG}" "${SERVED_MODEL_NAME}" "${BASE_URL}" \
  "${LOCAL_MAX_TOKENS}" "${LOCAL_TIMEOUT_SECONDS}" \
  "${AGENT_LOOP_MAX_ITERATIONS}" <<'PY'
import json
import sys
from pathlib import Path

path, model, base_url = sys.argv[1:4]
payload = {
    "provider": "local",
    "api_type": "chat_completions",
    "api_key": "EMPTY",
    "base_url": base_url,
    "model": model,
    "timeout": int(sys.argv[5]),
    "node_timeout_seconds": int(sys.argv[5]),
    "max_retries": 2,
    "max_tokens": int(sys.argv[4]),
    "temperature": 0,
    "top_p": 1.0,
    "http2": False,
    "trust_env": False,
    "agent_loop_max_iterations": int(sys.argv[6]),
    "tptt_coalesce_ms": 60,
    "tptt_prefix_tokens": 200,
    "max_tptt_leaf_sequences_per_tree": 128,
}
Path(path).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

EVAL_CMD=(
  "${PYTHON_BIN}" -m benchmarks.eval
  --output-root "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
  --remote-llm-config "${REMOTE_LLM_CONFIG}"
  --synthesize-llm-config "${SYNTHESIZE_LLM_CONFIG}"
  --local-slm-config "${LOCAL_CONFIG}"
  --remote-batch-size "${REMOTE_BATCH_SIZE}"
  --remote-concurrency "${REMOTE_CONCURRENCY}"
  --max-workers "${EVAL_CONCURRENCY}"
  --slm-scheduler "${SLM_SCHEDULER}"
  --max-parallel-execution-units "${MAX_PARALLEL_EXECUTION_UNITS}"
  --max-parallel-slm-node-jobs "${MAX_PARALLEL_SLM_NODE_JOBS}"
  --max-parallel-slm-sequences "${MAX_PARALLEL_SLM_SEQUENCES}"
  --max-pending-slm-sequences "${MAX_PENDING_SLM_SEQUENCES}"
)
case "${DATASET}" in
  tablebench)
    EVAL_CMD+=(--tablebench-eval --tablebench-root "${TABLEBENCH_ROOT}")
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
    )
    [[ -n "${TABLEFACT_SUBSET}" ]] \
      && EVAL_CMD+=(--tablefact-subset "${TABLEFACT_SUBSET}")
    ;;
esac
[[ "${#EXTRA_EVAL_ARGS[@]}" -gt 0 ]] && EVAL_CMD+=("${EXTRA_EVAL_ARGS[@]}")

echo "Running ${DATASET} evaluation" >&2
PYTHONWARNINGS="ignore" \
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
"${EVAL_CMD[@]}"
