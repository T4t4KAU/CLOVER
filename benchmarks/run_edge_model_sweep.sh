#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Edge model scale sweep (P0-3)
#
# Runs CLOVER with multiple Edge model sizes on the full datasets to measure
# how Edge model scale affects accuracy, cost, and Edge-only finalization.
#
# For each Edge model, runs two variants:
#   full      = Full CLOVER
#   no_static = w/o Static Fast Path + w/o Static Finalization (Edge-only)
#
# Usage:
#   bash benchmarks/run_edge_model_sweep.sh
#
# Override Edge models via CLOVER_EDGE_MODELS (colon-separated paths).
# =============================================================================

USER_PYTHON_BIN="python"
USER_DEEPSEEK_API_KEY=""
USER_REMOTE_LLM_CONFIG="model_config/deepseek_remote_llm_config.json"
USER_SYNTHESIZE_LLM_CONFIG="model_config/deepseek_remote_llm_config.json"
USER_VLLM_GPUS="0"
USER_VLLM_HOST="127.0.0.1"
USER_VLLM_PORT="8000"
USER_VLLM_GPU_MEMORY_UTILIZATION="0.88"
USER_VLLM_SERVER_ARGS=""
USER_VLLM_WARMUP="true"
USER_EDGE_REVIEW_PROACTIVE="true"
USER_MAX_RETRIES="1"
USER_OUTPUT_ROOT=""

# Edge models to sweep (colon-separated paths).
# Defaults can be overridden via CLOVER_EDGE_MODELS.
USER_EDGE_MODELS="/path/to/Qwen2.5-3B-Instruct:/path/to/Qwen2.5-7B-Instruct:/path/to/Qwen3-4B-Instruct"

# Datasets to sweep (space-separated).
USER_DATASETS="tablebench wikitq tablefact"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/run_edge_model_sweep.sh

Runs CLOVER (full + no_static) across multiple Edge model sizes on the full
datasets. Each (model, dataset, variant) combination is one run.

Environment variables:
  CLOVER_EDGE_MODELS=/path/to/3B:/path/to/7B:/path/to/4B
  CLOVER_EDGE_SWEEP_DATASETS="tablebench wikitq tablefact"
  CLOVER_EDGE_SWEEP_VARIANTS="full no_static"
  DEEPSEEK_API_KEY=...
  CLOVER_VLLM_GPUS=0
  CLOVER_VLLM_PORT=8000
  CLOVER_EDGE_SWEEP_OUTPUT_ROOT=/path/to/output
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-${USER_PYTHON_BIN}}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v "${PYTHON_BIN}" 2>/dev/null || true)"
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Activate the intended Python environment before running the sweep." >&2
  exit 1
fi

DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-${USER_DEEPSEEK_API_KEY}}"
if [[ -z "${DEEPSEEK_API_KEY}" ]]; then
  echo "Set USER_DEEPSEEK_API_KEY at the top of this script or export DEEPSEEK_API_KEY." >&2
  exit 2
fi
export DEEPSEEK_API_KEY

EDGE_MODELS="${CLOVER_EDGE_MODELS:-${USER_EDGE_MODELS}}"
DATASETS="${CLOVER_EDGE_SWEEP_DATASETS:-${USER_DATASETS}}"
VARIANTS="${CLOVER_EDGE_SWEEP_VARIANTS:-full no_static}"

export CLOVER_VLLM_GPUS="${CLOVER_VLLM_GPUS:-${USER_VLLM_GPUS}}"
export CLOVER_VLLM_HOST="${CLOVER_VLLM_HOST:-${USER_VLLM_HOST}}"
export CLOVER_VLLM_PORT="${CLOVER_VLLM_PORT:-${USER_VLLM_PORT}}"
export CLOVER_VLLM_GPU_MEMORY_UTILIZATION="${CLOVER_VLLM_GPU_MEMORY_UTILIZATION:-${USER_VLLM_GPU_MEMORY_UTILIZATION}}"
export CLOVER_VLLM_SERVER_ARGS="${CLOVER_VLLM_SERVER_ARGS:-${USER_VLLM_SERVER_ARGS}}"
export CLOVER_VLLM_WARMUP="${CLOVER_VLLM_WARMUP:-${USER_VLLM_WARMUP}}"
export CLOVER_EDGE_REVIEW_PROACTIVE="${CLOVER_EDGE_REVIEW_PROACTIVE:-${USER_EDGE_REVIEW_PROACTIVE}}"

REMOTE_LLM_CONFIG="${CLOVER_REMOTE_LLM_CONFIG:-${USER_REMOTE_LLM_CONFIG}}"
SYNTHESIZE_LLM_CONFIG="${CLOVER_SYNTHESIZE_LLM_CONFIG:-${USER_SYNTHESIZE_LLM_CONFIG}}"
[[ "${REMOTE_LLM_CONFIG}" == /* ]] \
  || REMOTE_LLM_CONFIG="${REPO_ROOT}/${REMOTE_LLM_CONFIG}"
[[ "${SYNTHESIZE_LLM_CONFIG}" == /* ]] \
  || SYNTHESIZE_LLM_CONFIG="${REPO_ROOT}/${SYNTHESIZE_LLM_CONFIG}"
export CLOVER_REMOTE_LLM_CONFIG="${REMOTE_LLM_CONFIG}"
export CLOVER_SYNTHESIZE_LLM_CONFIG="${SYNTHESIZE_LLM_CONFIG}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/benchmark/runs/edge_model_sweep_${TIMESTAMP}"
SWEEP_ROOT="${CLOVER_EDGE_SWEEP_OUTPUT_ROOT:-${USER_OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}}"
mkdir -p "${SWEEP_ROOT}"

# Validate variant names.
for variant in ${VARIANTS}; do
  case "${variant}" in
    full|no_static) ;;
    *)
      echo "Unsupported sweep variant: ${variant} (only full and no_static)" >&2
      exit 2
      ;;
  esac
done

# Parse Edge model paths and derive short names.
EDGE_MODEL_PATHS=()
EDGE_MODEL_NAMES=()
IFS=':' read -ra RAW_MODELS <<<"${EDGE_MODELS}"
for model_path in "${RAW_MODELS[@]}"; do
  [[ -z "${model_path}" ]] && continue
  if [[ "${model_path}" == /path/to/* ]]; then
    echo "Edit USER_EDGE_MODELS at the top of this script (or set CLOVER_EDGE_MODELS)." >&2
    exit 2
  fi
  EDGE_MODEL_PATHS+=("${model_path}")
  # Derive a short name from the basename (e.g., Qwen2.5-3B-Instruct -> Qwen2.5-3B).
  base="$(basename "${model_path}")"
  short="${base%-Instruct}"
  short="${short%-Chat}"
  EDGE_MODEL_NAMES+=("${short}")
done

if [[ "${#EDGE_MODEL_PATHS[@]}" -eq 0 ]]; then
  echo "No Edge models configured." >&2
  exit 2
fi

MAX_RETRIES="${CLOVER_ABLATION_MAX_RETRIES:-${USER_MAX_RETRIES}}"
PID_FILE="${SWEEP_ROOT}/vllm.pid"

cleanup() {
  if [[ -f "${PID_FILE}" ]]; then
    server_pid="$(tr -d '[:space:]' <"${PID_FILE}")"
    if [[ "${server_pid}" =~ ^[0-9]+$ ]]; then
      kill "${server_pid}" >/dev/null 2>&1 || true
      wait "${server_pid}" >/dev/null 2>&1 || true
    fi
    rm -f "${PID_FILE}"
  fi
}
trap cleanup EXIT

run_sweep_variant() {
  local dataset="$1"
  local edge_model_path="$2"
  local edge_model_name="$3"
  local variant="$4"

  local edge_agent="true"
  local edge_repair="true"
  local terminal_edge_review="true"
  local contract_gate="true"
  local node_review="true"
  local cloud_replan="true"
  local cloud_synthesis="true"
  local static_fast_path="true"
  local static_finalization="true"

  case "${variant}" in
    full) ;;
    no_static)
      static_fast_path="false"
      static_finalization="false"
      ;;
    *)
      echo "Unsupported variant: ${variant}" >&2
      exit 2
      ;;
  esac

  local edge_review_mode="safe"
  if [[ "${edge_agent}" != "true" || "${terminal_edge_review}" != "true" ]]; then
    edge_review_mode="off"
  fi

  local run_name="${dataset}_${edge_model_name}_${variant}"
  local run_output="${SWEEP_ROOT}/${run_name}"
  if [[ -f "${run_output}/run_summary.json" ]]; then
    echo "Skipping (already done): ${run_name}" >&2
    return 0
  fi

  echo "Running sweep: ${run_name}" >&2
  CLOVER_ABLATION_VARIANT="${variant}" \
  CLOVER_ENABLE_EDGE_AGENT="${edge_agent}" \
  CLOVER_ENABLE_EDGE_REPAIR="${edge_repair}" \
  CLOVER_ENABLE_TERMINAL_EDGE_REVIEW="${terminal_edge_review}" \
  CLOVER_ENABLE_CONTRACT_GATE="${contract_gate}" \
  CLOVER_ENABLE_NODE_REVIEW="${node_review}" \
  CLOVER_ENABLE_CLOUD_RECOVERY=true \
  CLOVER_ENABLE_CLOUD_REPLAN="${cloud_replan}" \
  CLOVER_ENABLE_CLOUD_SYNTHESIS="${cloud_synthesis}" \
  CLOVER_ENABLE_STATIC_FAST_PATH="${static_fast_path}" \
  CLOVER_ENABLE_STATIC_FINALIZATION="${static_finalization}" \
  CLOVER_EDGE_REVIEW_MODE="${edge_review_mode}" \
  CLOVER_VLLM_PERSIST_SERVER=true \
  CLOVER_VLLM_PID_FILE="${PID_FILE}" \
  OUTPUT_ROOT="${SWEEP_ROOT}" \
  RUN_NAME="${run_name}" \
  bash "${SCRIPT_DIR}/run_vllm_eval.sh" \
    "${dataset}" \
    "${edge_model_path}" \
    --validation-mode remote_supervisor \
    --max-retries "${MAX_RETRIES}" \
    --seed 20260619
}

# Record sweep metadata.
"${PYTHON_BIN}" - "${SWEEP_ROOT}" "${EDGE_MODEL_NAMES[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
names = sys.argv[2:]
metadata = {
    "edge_models": names,
    "datasets": ["tablebench", "wikitq", "tablefact"],
    "variants": ["full", "no_static"],
}
(root / "sweep_metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

for dataset in ${DATASETS}; do
  case "${dataset}" in
    tablebench|wikitq|tablefact) ;;
    *)
      echo "Skipping unsupported dataset: ${dataset}" >&2
      continue
      ;;
  esac
  for idx in "${!EDGE_MODEL_PATHS[@]}"; do
    edge_model_path="${EDGE_MODEL_PATHS[${idx}]}"
    edge_model_name="${EDGE_MODEL_NAMES[${idx}]}"
    for variant in ${VARIANTS}; do
      run_sweep_variant "${dataset}" "${edge_model_path}" "${edge_model_name}" "${variant}"
    done
  done
done

"${PYTHON_BIN}" -m benchmarks.summarize_edge_model_sweep \
  --sweep-root "${SWEEP_ROOT}" \
  --output-dir "${SWEEP_ROOT}"

echo "Edge model sweep completed: ${SWEEP_ROOT}" >&2
echo "Summary: ${SWEEP_ROOT}/edge_model_sweep.md" >&2
