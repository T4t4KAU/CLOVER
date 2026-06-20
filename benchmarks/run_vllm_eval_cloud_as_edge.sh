#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# User settings
# Edit these values, then run: bash benchmarks/run_vllm_eval_cloud_as_edge.sh
# Environment variables or CLI flags can still override them.
# =============================================================================
# --- Model configs ---
# Cloud (remote + synthesize) model config. Any non-DeepSeek config under
# model_config/ works, e.g. qwen25_3b_instruct_local_slm_config.json,
# qwen25_7b_instruct_local_slm_config.json, qwen3_4b_local_remote_config.json,
# doubao_remote_llm_config.json, remote_llm_config.json (OpenAI gpt-5.2).
REMOTE_CONFIG="${CLOVER_REMOTE_LLM_CONFIG:-model_config/qwen25_3b_instruct_local_slm_config.json}"
SYNTHESIZE_CONFIG="${CLOVER_SYNTHESIZE_LLM_CONFIG:-${REMOTE_CONFIG}}"

# --- Dataset & edge model ---
DATASET="${CLOVER_EVAL_DATASET:-tablebench}"  # tablebench | wikitq | tablefact
EDGE_MODEL_PATH="${CLOVER_EDGE_MODEL_PATH:-/path/to/Qwen2.5-3B-Instruct}"

# --- vLLM server ---
GPU_DEVICES="${CLOVER_VLLM_GPUS:-0}"                  # Example: "0,1"
HOST="${CLOVER_VLLM_HOST:-127.0.0.1}"
PORT="${CLOVER_VLLM_PORT:-8000}"
SERVED_MODEL_NAME="${CLOVER_VLLM_SERVED_MODEL_NAME:-clover-edge-model}"
DTYPE="${CLOVER_VLLM_DTYPE:-auto}"
MAX_MODEL_LEN="${CLOVER_VLLM_MAX_MODEL_LEN:-}"
GPU_MEMORY_UTILIZATION="${CLOVER_VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
TENSOR_PARALLEL_SIZE="${CLOVER_VLLM_TENSOR_PARALLEL_SIZE:-}"
ENABLE_PREFIX_CACHING="${CLOVER_VLLM_ENABLE_PREFIX_CACHING:-true}"
VLLM_SERVER_ARGS="${CLOVER_VLLM_SERVER_ARGS:-}"
SERVER_READY_TIMEOUT="${CLOVER_VLLM_READY_TIMEOUT:-600}"
PERSIST_SERVER="${CLOVER_VLLM_PERSIST_SERVER:-false}"
WARMUP_SERVER="${CLOVER_VLLM_WARMUP:-true}"

# --- API keys (only required if the chosen config references them) ---
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
DOUBAO_API_KEY="${DOUBAO_API_KEY:-}"

# =============================================================================
# Wrapper around run_vllm_eval.sh that swaps the cloud (remote + synthesize)
# model for a non-DeepSeek config, e.g. a local vLLM model or another vendor.
# Useful for end-to-end testing without a DeepSeek API key.
#
# All positional arguments and CLOVER_* environment variables not consumed here
# are forwarded to run_vllm_eval.sh unchanged.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/run_vllm_eval_cloud_as_edge.sh [options] [DATASET] [EDGE_MODEL_PATH] [eval options...]

Options:
  --remote-config <path>   JSON config for the cloud (remote + synthesize) model.
                           Defaults to model_config/qwen25_3b_instruct_local_slm_config.json.
  -h, --help               Show this help.

Examples:
  # Use the settings at the top of this script:
  bash benchmarks/run_vllm_eval_cloud_as_edge.sh

  # Override dataset/edge model positionally:
  bash benchmarks/run_vllm_eval_cloud_as_edge.sh wikitq /models/Qwen2.5-3B-Instruct --max-cases 10

  # Use Doubao as the cloud model:
  bash benchmarks/run_vllm_eval_cloud_as_edge.sh --remote-config model_config/doubao_remote_llm_config.json

Environment variables:
  CLOVER_REMOTE_LLM_CONFIG       Same as --remote-config.
  CLOVER_SYNTHESIZE_LLM_CONFIG   Override synthesize config (defaults to remote config).
  DEEPSEEK_API_KEY / OPENAI_API_KEY / DOUBAO_API_KEY
                                 Required only if the chosen config references them.
  All other CLOVER_* variables supported by run_vllm_eval.sh are forwarded.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# Parse --remote-config before forwarding the rest to run_vllm_eval.sh.
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote-config)
      [[ $# -lt 2 ]] && { echo "--remote-config requires a value" >&2; exit 2; }
      REMOTE_CONFIG="$2"
      shift 2
      ;;
    --remote-config=*)
      REMOTE_CONFIG="${1#--remote-config=}"
      shift
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

# Resolve to absolute path so the child script can find it regardless of cwd.
if [[ "${REMOTE_CONFIG}" != /* ]]; then
  REMOTE_CONFIG="${REPO_ROOT}/${REMOTE_CONFIG}"
fi
if [[ "${SYNTHESIZE_CONFIG}" != /* && "${SYNTHESIZE_CONFIG}" != "${REMOTE_CONFIG}" ]]; then
  SYNTHESIZE_CONFIG="${REPO_ROOT}/${SYNTHESIZE_CONFIG}"
fi

if [[ ! -f "${REMOTE_CONFIG}" ]]; then
  echo "Remote config not found: ${REMOTE_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${SYNTHESIZE_CONFIG}" ]]; then
  echo "Synthesize config not found: ${SYNTHESIZE_CONFIG}" >&2
  exit 1
fi

# Surface any API key requirements of the chosen config before launching vLLM.
require_api_key() {
  local env_name="$1"
  if grep -Eq "\"api_key_env\"[[:space:]]*:[[:space:]]*\"${env_name}\"" "${REMOTE_CONFIG}" "${SYNTHESIZE_CONFIG}" \
      && [[ -z "${!env_name:-}" ]]; then
    echo "${env_name} is empty but the chosen config requires it." >&2
    exit 1
  fi
}
require_api_key DEEPSEEK_API_KEY
require_api_key OPENAI_API_KEY
require_api_key DOUBAO_API_KEY

# Export model configs and API keys for the child script.
export CLOVER_REMOTE_LLM_CONFIG="${REMOTE_CONFIG}"
export CLOVER_SYNTHESIZE_LLM_CONFIG="${SYNTHESIZE_CONFIG}"
export DEEPSEEK_API_KEY OPENAI_API_KEY DOUBAO_API_KEY

# Export vLLM server settings for the child script.
export CLOVER_EVAL_DATASET="${DATASET}"
export CLOVER_EDGE_MODEL_PATH="${EDGE_MODEL_PATH}"
export CLOVER_VLLM_GPUS="${GPU_DEVICES}"
export CLOVER_VLLM_HOST="${HOST}"
export CLOVER_VLLM_PORT="${PORT}"
export CLOVER_VLLM_SERVED_MODEL_NAME="${SERVED_MODEL_NAME}"
export CLOVER_VLLM_DTYPE="${DTYPE}"
[[ -n "${MAX_MODEL_LEN}" ]] && export CLOVER_VLLM_MAX_MODEL_LEN="${MAX_MODEL_LEN}"
export CLOVER_VLLM_GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}"
[[ -n "${TENSOR_PARALLEL_SIZE}" ]] && export CLOVER_VLLM_TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}"
export CLOVER_VLLM_ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING}"
export CLOVER_VLLM_SERVER_ARGS="${VLLM_SERVER_ARGS}"
export CLOVER_VLLM_READY_TIMEOUT="${SERVER_READY_TIMEOUT}"
export CLOVER_VLLM_PERSIST_SERVER="${PERSIST_SERVER}"
export CLOVER_VLLM_WARMUP="${WARMUP_SERVER}"

echo "Cloud model config:      ${CLOVER_REMOTE_LLM_CONFIG}" >&2
echo "Synthesize model config: ${CLOVER_SYNTHESIZE_LLM_CONFIG}" >&2
echo "Edge model:              ${CLOVER_EDGE_MODEL_PATH}" >&2
echo "Dataset:                 ${CLOVER_EVAL_DATASET}" >&2
echo "vLLM endpoint:           http://${HOST}:${PORT}/v1" >&2
echo "GPUs:                    ${GPU_DEVICES}" >&2

exec bash "${SCRIPT_DIR}/run_vllm_eval.sh" ${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}
