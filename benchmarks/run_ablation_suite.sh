#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/run_ablation_suite.sh DATASET [EDGE_MODEL_PATH] [eval options...]

DATASET:
  tablebench | wikitq

This runs six variants on one fixed 100-case manifest:
  full, static, no_contract, end_review, one_shot, cloud_finalize

Examples:
  bash benchmarks/run_ablation_suite.sh wikitq /models/Qwen2.5-3B-Instruct
  bash benchmarks/run_ablation_suite.sh tablebench /models/Qwen2.5-3B-Instruct

Useful environment variables:
  CLOVER_ABLATION_SIZE=100
  CLOVER_ABLATION_SEED=20260619
  CLOVER_ABLATION_REGENERATE_MANIFEST=1
  CLOVER_ABLATION_OUTPUT_ROOT=/path/to/output
  CLOVER_EDGE_MODEL_PATH=/path/to/model
EOF
}

if [[ "$#" -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  [[ "$#" -ge 1 ]] && exit 0
  exit 2
fi

DATASET="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
shift
case "${DATASET}" in
  tablebench|wikitq) ;;
  wikitablequestions) DATASET="wikitq" ;;
  *)
    echo "Unsupported ablation dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

EDGE_MODEL_PATH="${CLOVER_EDGE_MODEL_PATH:-}"
if [[ "${1:-}" != "" && "${1:-}" != --* ]]; then
  EDGE_MODEL_PATH="$1"
  shift
fi
if [[ -z "${EDGE_MODEL_PATH}" ]]; then
  echo "Set CLOVER_EDGE_MODEL_PATH or pass EDGE_MODEL_PATH." >&2
  exit 2
fi
EXTRA_EVAL_ARGS=("$@")

PYTHON_BIN="${PYTHON_BIN:-$(command -v python 2>/dev/null || true)}"
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Activate the intended Python environment before running ablations." >&2
  exit 1
fi

DATASETS_ROOT="${CLOVER_DATASETS_ROOT:-${REPO_ROOT}/datasets}"
DATASET_ROOT="${DATASETS_ROOT}/${DATASET}"
SIZE="${CLOVER_ABLATION_SIZE:-100}"
SEED="${CLOVER_ABLATION_SEED:-20260619}"
MANIFEST_ROOT="${CLOVER_ABLATION_MANIFEST_ROOT:-${SCRIPT_DIR}/ablation_cases}"
MANIFEST="${CLOVER_ABLATION_MANIFEST:-${MANIFEST_ROOT}/${DATASET}_${SIZE}_seed${SEED}.jsonl}"
REGENERATE_MANIFEST="${CLOVER_ABLATION_REGENERATE_MANIFEST:-0}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUITE_ROOT="${CLOVER_ABLATION_OUTPUT_ROOT:-${REPO_ROOT}/benchmark/runs/${DATASET}_ablation_${TIMESTAMP}}"
PID_FILE="${SUITE_ROOT}/vllm.pid"
MAX_RETRIES="${CLOVER_ABLATION_MAX_RETRIES:-1}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Converted dataset not found: ${DATASET_ROOT}" >&2
  exit 1
fi
mkdir -p "${MANIFEST_ROOT}" "${SUITE_ROOT}"

if [[ ! -f "${MANIFEST}" || "${REGENERATE_MANIFEST}" == "1" ]]; then
  "${PYTHON_BIN}" -m benchmarks.ablation_subset \
    --dataset "${DATASET}" \
    --dataset-root "${DATASET_ROOT}" \
    --output "${MANIFEST}" \
    --size "${SIZE}" \
    --seed "${SEED}" \
    >"${SUITE_ROOT}/manifest_generation.json"
fi

CASE_ARGS=()
while IFS= read -r case_id; do
  [[ -n "${case_id}" ]] && CASE_ARGS+=(--case-id "${case_id}")
done < <(
  "${PYTHON_BIN}" -m benchmarks.ablation_subset \
    --dataset "${DATASET}" \
    --dataset-root "${DATASET_ROOT}" \
    --output "${MANIFEST}" \
    --print-case-ids
)
if [[ "${#CASE_ARGS[@]}" -ne "${SIZE}" ]]; then
  echo "Manifest must contain exactly ${SIZE} cases: ${MANIFEST}" >&2
  exit 1
fi

cp "${MANIFEST}" "${SUITE_ROOT}/cases.jsonl"
if [[ -f "${MANIFEST%.jsonl}.summary.json" ]]; then
  cp "${MANIFEST%.jsonl}.summary.json" "${SUITE_ROOT}/cases.summary.json"
fi

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

run_variant() {
  local variant="$1"
  local edge_agent="$2"
  local contract_gate="$3"
  local node_review="$4"
  local cloud_recovery="$5"
  local static_finalization="$6"
  local edge_review_mode="safe"
  if [[ "${edge_agent}" != "true" ]]; then
    edge_review_mode="off"
  fi

  echo "Running ${DATASET} ablation variant: ${variant}" >&2
  CLOVER_ABLATION_VARIANT="${variant}" \
  CLOVER_ENABLE_EDGE_AGENT="${edge_agent}" \
  CLOVER_ENABLE_CONTRACT_GATE="${contract_gate}" \
  CLOVER_ENABLE_NODE_REVIEW="${node_review}" \
  CLOVER_ENABLE_CLOUD_RECOVERY="${cloud_recovery}" \
  CLOVER_ENABLE_STATIC_FINALIZATION="${static_finalization}" \
  CLOVER_EDGE_REVIEW_MODE="${edge_review_mode}" \
  CLOVER_VLLM_PERSIST_SERVER=true \
  CLOVER_VLLM_PID_FILE="${PID_FILE}" \
  OUTPUT_ROOT="${SUITE_ROOT}" \
  RUN_NAME="${DATASET}_${variant}" \
  bash "${SCRIPT_DIR}/run_vllm_eval.sh" \
    "${DATASET}" \
    "${EDGE_MODEL_PATH}" \
    --validation-mode remote_supervisor \
    --max-retries "${MAX_RETRIES}" \
    --seed "${SEED}" \
    "${CASE_ARGS[@]}" \
    "${EXTRA_EVAL_ARGS[@]}"

  cp "${MANIFEST}" "${SUITE_ROOT}/${DATASET}_${variant}/ablation_cases.jsonl"
}

run_variant full true true true true true
run_variant static false true true true true
run_variant no_contract true false true true true
run_variant end_review false true false true true
run_variant one_shot true true true false true
run_variant cloud_finalize true true true true false

"${PYTHON_BIN}" -m benchmarks.check_ablation_suite \
  --suite-root "${SUITE_ROOT}" \
  --dataset "${DATASET}" \
  >"${SUITE_ROOT}/sanity_check_stdout.json"

echo "Ablation suite completed: ${SUITE_ROOT}" >&2
