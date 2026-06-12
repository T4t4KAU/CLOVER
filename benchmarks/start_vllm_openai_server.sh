#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONWARNINGS="ignore"

PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_PATH="${CLOVER_VLLM_MODEL:-${CLOVER_LOCAL_MODEL:-Qwen/Qwen3-8B}}"
SERVED_MODEL_NAME="${CLOVER_VLLM_SERVED_MODEL_NAME:-${MODEL_PATH}}"
HOST="${CLOVER_VLLM_HOST:-${CLOVER_LOCAL_HOST:-127.0.0.1}}"
PORT="${CLOVER_VLLM_PORT:-${CLOVER_LOCAL_PORT:-8000}}"
DTYPE="${CLOVER_VLLM_DTYPE:-auto}"
MAX_MODEL_LEN="${CLOVER_VLLM_MAX_MODEL_LEN:-}"
GPU_MEMORY_UTILIZATION="${CLOVER_VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
CHAT_TEMPLATE="${CLOVER_VLLM_CHAT_TEMPLATE:-}"
ENABLE_PREFIX_CACHING="${CLOVER_VLLM_ENABLE_PREFIX_CACHING:-true}"
SERVER_ARGS="${CLOVER_VLLM_SERVER_ARGS:-}"

read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"
EXTRA_ARGS=()
if [ -n "${SERVER_ARGS}" ]; then
  read -r -a EXTRA_ARGS <<< "${SERVER_ARGS}"
fi

"${PYTHON_CMD[@]}" - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("vllm") is None:
    print(
        "vLLM is not installed. Install it first, for example: "
        f"{sys.executable} -m pip install vllm",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

echo "Starting vLLM OpenAI-compatible server" >&2
echo "  model: ${MODEL_PATH}" >&2
echo "  served_model_name: ${SERVED_MODEL_NAME}" >&2
echo "  url: http://${HOST}:${PORT}/v1" >&2
echo "  dtype: ${DTYPE}" >&2

CMD=(
  "${PYTHON_CMD[@]}"
  -m vllm.entrypoints.openai.api_server
  --model "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --dtype "${DTYPE}"
)

if [ -n "${MAX_MODEL_LEN}" ]; then
  CMD+=(--max-model-len "${MAX_MODEL_LEN}")
fi
if [ -n "${GPU_MEMORY_UTILIZATION}" ]; then
  CMD+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
fi
if [ "${ENABLE_PREFIX_CACHING}" = "true" ]; then
  CMD+=(--enable-prefix-caching)
fi
if [ -n "${CHAT_TEMPLATE}" ]; then
  CMD+=(--chat-template "${CHAT_TEMPLATE}")
fi
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

exec "${CMD[@]}"
