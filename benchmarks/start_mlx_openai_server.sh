#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONWARNINGS="ignore"

PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_PATH="${CLOVER_MLX_MODEL:-${CLOVER_LOCAL_MODEL:-Qwen/Qwen3-8B}}"
HOST="${CLOVER_MLX_HOST:-${CLOVER_LOCAL_HOST:-127.0.0.1}}"
PORT="${CLOVER_MLX_PORT:-${CLOVER_LOCAL_PORT:-8000}}"
MAX_TOKENS="${CLOVER_MLX_MAX_TOKENS:-4096}"
CHAT_TEMPLATE_ARGS="${CLOVER_MLX_CHAT_TEMPLATE_ARGS:-{\"enable_thinking\":false}}"
SERVER_ARGS="${CLOVER_MLX_SERVER_ARGS:-}"

read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"
EXTRA_ARGS=()
if [ -n "${SERVER_ARGS}" ]; then
  read -r -a EXTRA_ARGS <<< "${SERVER_ARGS}"
fi

"${PYTHON_CMD[@]}" - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("mlx_lm") is None:
    print(
        "mlx-lm is not installed. Install it first, for example: "
        f"{sys.executable} -m pip install mlx-lm",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

echo "Starting MLX OpenAI-compatible server" >&2
echo "  model: ${MODEL_PATH}" >&2
echo "  url: http://${HOST}:${PORT}/v1" >&2
echo "  max_tokens: ${MAX_TOKENS}" >&2
echo "  chat_template_args: ${CHAT_TEMPLATE_ARGS}" >&2

CMD=(
  "${PYTHON_CMD[@]}"
  -m mlx_lm server
  --model "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --max-tokens "${MAX_TOKENS}"
  --chat-template-args "${CHAT_TEMPLATE_ARGS}"
)

if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

exec "${CMD[@]}"
