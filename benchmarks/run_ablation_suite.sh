#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# User settings
# Edit this block, then run:
#   bash benchmarks/run_ablation_suite.sh
#
# Command-line arguments and environment variables still override these values.
# =============================================================================
USER_DATASET="wikitq"  # wikitq | tablebench
USER_PYTHON_BIN="python"  # Use the python from the active conda environment
USER_EDGE_MODEL_PATH="/path/to/Qwen2.5-3B-Instruct"
USER_DEEPSEEK_API_KEY=""

# vLLM settings for the local Edge model.
USER_VLLM_GPUS="0"                 # Examples: "0" or "0,1"
USER_VLLM_HOST="127.0.0.1"
USER_VLLM_PORT="8000"
USER_VLLM_TENSOR_PARALLEL_SIZE=""  # Empty: infer from USER_VLLM_GPUS
USER_VLLM_MAX_MODEL_LEN=""         # Example: "16384"; empty: model default
USER_VLLM_GPU_MEMORY_UTILIZATION="0.88"
USER_VLLM_SERVER_ARGS=""           # Example: "--max-num-seqs 32"
USER_VLLM_WARMUP="true"            # Warm up before each timed variant
USER_EDGE_REVIEW_PROACTIVE="true"  # Review bounded semantic risks before static finalization

# Cloud model configs and experiment output.
USER_REMOTE_LLM_CONFIG="model_config/deepseek_remote_llm_config.json"
USER_SYNTHESIZE_LLM_CONFIG="model_config/deepseek_remote_llm_config.json"
USER_OUTPUT_ROOT=""                # Empty: benchmark/runs/<dataset>_ablation_<time>
USER_ABLATION_SIZE="100"
USER_ABLATION_SEED="20260619"
USER_MAX_RETRIES="1"
USER_VALIDATE_ONLY="false"         # true: validate settings/cases without running
USER_VARIANT_ORDER=""              # Empty: deterministic shuffle using the seed
                                    # Or: full,no_edge,static,no_contract,end_review,one_shot,cloud_finalize

# Optional explicit case IDs. Leave this empty to use the bundled fixed
# 100-case manifest in benchmarks/ablation_cases.
#
# WikiTQ example:
#   "nu-2294"
# TableBench example:
#   "ee98550f2f9e19f521b3c953c7c476a2"
USER_CASE_IDS=(
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/run_ablation_suite.sh [DATASET] [EDGE_MODEL_PATH] [eval options...]

DATASET:
  tablebench | wikitq

This runs seven variants on one fixed 100-case manifest:
  full, no_edge, static, no_contract, end_review, one_shot, cloud_finalize

The internal names map to:
  no_edge        = w/o Edge Agent (all local Edge paths disabled)
  static         = w/o Edge Repair (terminal Edge review remains enabled)
  end_review     = end-only Edge review
  one_shot       = w/o Cloud Replan (Cloud final synthesis remains enabled)
  cloud_finalize = force final synthesis through Cloud

Examples:
  # Use the User settings block at the top of this script:
  bash benchmarks/run_ablation_suite.sh

  # Override dataset and model temporarily:
  bash benchmarks/run_ablation_suite.sh wikitq /models/Qwen2.5-3B-Instruct
  bash benchmarks/run_ablation_suite.sh tablebench /models/Qwen2.5-3B-Instruct

Useful environment variables:
  CLOVER_ABLATION_SIZE=100
  CLOVER_ABLATION_SEED=20260619
  CLOVER_ABLATION_REGENERATE_MANIFEST=1
  CLOVER_ABLATION_OUTPUT_ROOT=/path/to/output
  CLOVER_EDGE_MODEL_PATH=/path/to/model
  CLOVER_ABLATION_VARIANT_ORDER=full,no_edge,static,no_contract,end_review,one_shot,cloud_finalize
  CLOVER_VLLM_WARMUP=true
  CLOVER_EDGE_REVIEW_PROACTIVE=true
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DATASET="${CLOVER_ABLATION_DATASET:-${USER_DATASET}}"
if [[ "${1:-}" != "" && "${1:-}" != --* ]]; then
  DATASET="$1"
  shift
fi
DATASET="$(printf '%s' "${DATASET}" | tr '[:upper:]' '[:lower:]')"
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

EDGE_MODEL_PATH="${CLOVER_EDGE_MODEL_PATH:-${USER_EDGE_MODEL_PATH}}"
if [[ "${1:-}" != "" && "${1:-}" != --* ]]; then
  EDGE_MODEL_PATH="$1"
  shift
fi
if [[ -z "${EDGE_MODEL_PATH}" || "${EDGE_MODEL_PATH}" == /path/to/* ]]; then
  echo "Set USER_EDGE_MODEL_PATH at the top of this script." >&2
  echo "Alternatively pass EDGE_MODEL_PATH as the second argument." >&2
  exit 2
fi
EXTRA_EVAL_ARGS=("$@")

PYTHON_BIN="${PYTHON_BIN:-${USER_PYTHON_BIN}}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v "${PYTHON_BIN}" 2>/dev/null || true)"
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Activate the intended Python environment before running ablations." >&2
  exit 1
fi

DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-${USER_DEEPSEEK_API_KEY}}"
if [[ -z "${DEEPSEEK_API_KEY}" ]]; then
  echo "Set USER_DEEPSEEK_API_KEY at the top of this script." >&2
  echo "Alternatively export DEEPSEEK_API_KEY before running." >&2
  exit 2
fi
export DEEPSEEK_API_KEY

export CLOVER_VLLM_GPUS="${CLOVER_VLLM_GPUS:-${USER_VLLM_GPUS}}"
export CLOVER_VLLM_HOST="${CLOVER_VLLM_HOST:-${USER_VLLM_HOST}}"
export CLOVER_VLLM_PORT="${CLOVER_VLLM_PORT:-${USER_VLLM_PORT}}"
export CLOVER_VLLM_GPU_MEMORY_UTILIZATION="${CLOVER_VLLM_GPU_MEMORY_UTILIZATION:-${USER_VLLM_GPU_MEMORY_UTILIZATION}}"
export CLOVER_VLLM_SERVER_ARGS="${CLOVER_VLLM_SERVER_ARGS:-${USER_VLLM_SERVER_ARGS}}"
export CLOVER_VLLM_WARMUP="${CLOVER_VLLM_WARMUP:-${USER_VLLM_WARMUP}}"
export CLOVER_EDGE_REVIEW_PROACTIVE="${CLOVER_EDGE_REVIEW_PROACTIVE:-${USER_EDGE_REVIEW_PROACTIVE}}"
if [[ -n "${USER_VLLM_TENSOR_PARALLEL_SIZE}" \
    && -z "${CLOVER_VLLM_TENSOR_PARALLEL_SIZE:-}" ]]; then
  export CLOVER_VLLM_TENSOR_PARALLEL_SIZE="${USER_VLLM_TENSOR_PARALLEL_SIZE}"
fi
if [[ -n "${USER_VLLM_MAX_MODEL_LEN}" \
    && -z "${CLOVER_VLLM_MAX_MODEL_LEN:-}" ]]; then
  export CLOVER_VLLM_MAX_MODEL_LEN="${USER_VLLM_MAX_MODEL_LEN}"
fi

REMOTE_LLM_CONFIG="${CLOVER_REMOTE_LLM_CONFIG:-${USER_REMOTE_LLM_CONFIG}}"
SYNTHESIZE_LLM_CONFIG="${CLOVER_SYNTHESIZE_LLM_CONFIG:-${USER_SYNTHESIZE_LLM_CONFIG}}"
[[ "${REMOTE_LLM_CONFIG}" == /* ]] \
  || REMOTE_LLM_CONFIG="${REPO_ROOT}/${REMOTE_LLM_CONFIG}"
[[ "${SYNTHESIZE_LLM_CONFIG}" == /* ]] \
  || SYNTHESIZE_LLM_CONFIG="${REPO_ROOT}/${SYNTHESIZE_LLM_CONFIG}"
export CLOVER_REMOTE_LLM_CONFIG="${REMOTE_LLM_CONFIG}"
export CLOVER_SYNTHESIZE_LLM_CONFIG="${SYNTHESIZE_LLM_CONFIG}"

DATASETS_ROOT="${CLOVER_DATASETS_ROOT:-${REPO_ROOT}/datasets}"
DATASET_ROOT="${DATASETS_ROOT}/${DATASET}"
SIZE="${CLOVER_ABLATION_SIZE:-${USER_ABLATION_SIZE}}"
SEED="${CLOVER_ABLATION_SEED:-${USER_ABLATION_SEED}}"
MANIFEST_ROOT="${CLOVER_ABLATION_MANIFEST_ROOT:-${SCRIPT_DIR}/ablation_cases}"
MANIFEST="${CLOVER_ABLATION_MANIFEST:-${MANIFEST_ROOT}/${DATASET}_${SIZE}_seed${SEED}.jsonl}"
REGENERATE_MANIFEST="${CLOVER_ABLATION_REGENERATE_MANIFEST:-0}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/benchmark/runs/${DATASET}_ablation_${TIMESTAMP}"
SUITE_ROOT="${CLOVER_ABLATION_OUTPUT_ROOT:-${USER_OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}}"
PID_FILE="${SUITE_ROOT}/vllm.pid"
MAX_RETRIES="${CLOVER_ABLATION_MAX_RETRIES:-${USER_MAX_RETRIES}}"
VALIDATE_ONLY="${CLOVER_ABLATION_VALIDATE_ONLY:-${USER_VALIDATE_ONLY}}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Converted dataset not found: ${DATASET_ROOT}" >&2
  exit 1
fi
mkdir -p "${MANIFEST_ROOT}" "${SUITE_ROOT}"

ACTIVE_MANIFEST="${MANIFEST}"
if [[ "${#USER_CASE_IDS[@]}" -gt 0 ]]; then
  SIZE="${#USER_CASE_IDS[@]}"
  ACTIVE_MANIFEST="${SUITE_ROOT}/inline_cases.jsonl"
  "${PYTHON_BIN}" - "${DATASET_ROOT}" "${ACTIVE_MANIFEST}" \
    "${USER_CASE_IDS[@]}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

dataset_root = Path(sys.argv[1])
output_path = Path(sys.argv[2])
requested = sys.argv[3:]
requested_counts = Counter(requested)
if any(count != 1 for count in requested_counts.values()):
    raise SystemExit("USER_CASE_IDS contains duplicate ids")

found = {}
ambiguous = set()
for cases_path in sorted(dataset_root.glob("*/cases.jsonl")):
    with cases_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            case_id = str(record.get("case_id") or "")
            if case_id not in requested_counts:
                continue
            record.setdefault("dataset_id", cases_path.parent.name)
            if case_id in found:
                ambiguous.add(case_id)
            found[case_id] = record

missing = [case_id for case_id in requested if case_id not in found]
if missing:
    raise SystemExit(f"USER_CASE_IDS not found: {missing}")
if ambiguous:
    raise SystemExit(f"USER_CASE_IDS are ambiguous across tables: {sorted(ambiguous)}")

with output_path.open("w", encoding="utf-8") as handle:
    for case_id in requested:
        handle.write(json.dumps(found[case_id], ensure_ascii=False) + "\n")
PY
elif [[ ! -f "${MANIFEST}" || "${REGENERATE_MANIFEST}" == "1" ]]; then
  "${PYTHON_BIN}" -m benchmarks.ablation_subset \
    --dataset "${DATASET}" \
    --dataset-root "${DATASET_ROOT}" \
    --output "${MANIFEST}" \
    --size "${SIZE}" \
    --seed "${SEED}" \
    >"${SUITE_ROOT}/manifest_generation.json"
fi

CASE_ID_LIST="${SUITE_ROOT}/case_ids.txt"
load_manifest_case_ids() {
  "${PYTHON_BIN}" -m benchmarks.ablation_subset \
    --dataset "${DATASET}" \
    --dataset-root "${DATASET_ROOT}" \
    --output "${ACTIVE_MANIFEST}" \
    --print-case-ids \
    >"${CASE_ID_LIST}"
}

if ! load_manifest_case_ids; then
  if [[ "${ACTIVE_MANIFEST}" != "${MANIFEST}" ]]; then
    echo "Failed to read inline USER_CASE_IDS." >&2
    exit 1
  fi
  echo "Existing manifest is invalid; regenerating: ${MANIFEST}" >&2
  "${PYTHON_BIN}" -m benchmarks.ablation_subset \
    --dataset "${DATASET}" \
    --dataset-root "${DATASET_ROOT}" \
    --output "${MANIFEST}" \
    --size "${SIZE}" \
    --seed "${SEED}" \
    >"${SUITE_ROOT}/manifest_generation.json"
  load_manifest_case_ids
fi

SELECTED_CASE_IDS=()
while IFS= read -r case_id; do
  [[ -n "${case_id}" ]] && SELECTED_CASE_IDS+=("${case_id}")
done <"${CASE_ID_LIST}"
if [[ "${#SELECTED_CASE_IDS[@]}" -ne "${SIZE}" ]]; then
  if [[ "${ACTIVE_MANIFEST}" == "${MANIFEST}" ]]; then
    echo "Manifest has ${#SELECTED_CASE_IDS[@]} cases; regenerating ${SIZE} cases." >&2
    "${PYTHON_BIN}" -m benchmarks.ablation_subset \
      --dataset "${DATASET}" \
      --dataset-root "${DATASET_ROOT}" \
      --output "${MANIFEST}" \
      --size "${SIZE}" \
      --seed "${SEED}" \
      >"${SUITE_ROOT}/manifest_generation.json"
    load_manifest_case_ids
    SELECTED_CASE_IDS=()
    while IFS= read -r case_id; do
      [[ -n "${case_id}" ]] && SELECTED_CASE_IDS+=("${case_id}")
    done <"${CASE_ID_LIST}"
  fi
fi
if [[ "${#SELECTED_CASE_IDS[@]}" -ne "${SIZE}" ]]; then
  echo "Expected ${SIZE} case ids but read ${#SELECTED_CASE_IDS[@]}: ${ACTIVE_MANIFEST}" >&2
  exit 1
fi

CASE_ARGS=()
for case_id in "${SELECTED_CASE_IDS[@]}"; do
  CASE_ARGS+=(--case-id "${case_id}")
done

cp "${ACTIVE_MANIFEST}" "${SUITE_ROOT}/cases.jsonl"
if [[ "${ACTIVE_MANIFEST}" == "${MANIFEST}" \
    && -f "${MANIFEST%.jsonl}.summary.json" ]]; then
  cp "${MANIFEST%.jsonl}.summary.json" "${SUITE_ROOT}/cases.summary.json"
fi

case "$(printf '%s' "${VALIDATE_ONLY}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    echo "Ablation settings are valid." >&2
    echo "  dataset: ${DATASET}" >&2
    echo "  edge model: ${EDGE_MODEL_PATH}" >&2
    echo "  GPUs: ${CLOVER_VLLM_GPUS}" >&2
    echo "  vLLM endpoint: ${CLOVER_VLLM_HOST}:${CLOVER_VLLM_PORT}" >&2
    echo "  cases: ${#SELECTED_CASE_IDS[@]}" >&2
    echo "  manifest: ${ACTIVE_MANIFEST}" >&2
    echo "  output: ${SUITE_ROOT}" >&2
    exit 0
    ;;
  0|false|no|n|off) ;;
  *)
    echo "USER_VALIDATE_ONLY must be true or false: ${VALIDATE_ONLY}" >&2
    exit 2
    ;;
esac

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
  local edge_repair="$3"
  local terminal_edge_review="$4"
  local contract_gate="$5"
  local node_review="$6"
  local cloud_replan="$7"
  local cloud_synthesis="$8"
  local static_finalization="$9"
  local edge_review_mode="safe"
  if [[ "${edge_agent}" != "true" || "${terminal_edge_review}" != "true" ]]; then
    edge_review_mode="off"
  fi

  echo "Running ${DATASET} ablation variant: ${variant}" >&2
  CLOVER_ABLATION_VARIANT="${variant}" \
  CLOVER_ENABLE_EDGE_AGENT="${edge_agent}" \
  CLOVER_ENABLE_EDGE_REPAIR="${edge_repair}" \
  CLOVER_ENABLE_TERMINAL_EDGE_REVIEW="${terminal_edge_review}" \
  CLOVER_ENABLE_CONTRACT_GATE="${contract_gate}" \
  CLOVER_ENABLE_NODE_REVIEW="${node_review}" \
  CLOVER_ENABLE_CLOUD_RECOVERY=true \
  CLOVER_ENABLE_CLOUD_REPLAN="${cloud_replan}" \
  CLOVER_ENABLE_CLOUD_SYNTHESIS="${cloud_synthesis}" \
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

  cp "${ACTIVE_MANIFEST}" \
    "${SUITE_ROOT}/${DATASET}_${variant}/ablation_cases.jsonl"
}

run_named_variant() {
  case "$1" in
    full)
      run_variant full true true true true true true true true
      ;;
    no_edge)
      run_variant no_edge false false false true false true true true
      ;;
    static)
      run_variant static true false true true true true true true
      ;;
    no_contract)
      run_variant no_contract true true true false true true true true
      ;;
    end_review)
      run_variant end_review true false true true false true true true
      ;;
    one_shot)
      run_variant one_shot true true true true true false true true
      ;;
    cloud_finalize)
      run_variant cloud_finalize true true false true true true true false
      ;;
    *)
      echo "Unknown ablation variant: $1" >&2
      exit 2
      ;;
  esac
}

VARIANT_ORDER_RAW="${CLOVER_ABLATION_VARIANT_ORDER:-${USER_VARIANT_ORDER}}"
VARIANT_ORDER=()
if [[ -n "${VARIANT_ORDER_RAW}" ]]; then
  while IFS= read -r variant; do
    [[ -n "${variant}" ]] && VARIANT_ORDER+=("${variant}")
  done < <(
    printf '%s' "${VARIANT_ORDER_RAW}" \
      | tr ',' '\n' \
      | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
  )
else
  while IFS= read -r variant; do
    [[ -n "${variant}" ]] && VARIANT_ORDER+=("${variant}")
  done < <(
    "${PYTHON_BIN}" - "${SEED}" <<'PY'
import random
import sys

variants = [
    "full",
    "no_edge",
    "static",
    "no_contract",
    "end_review",
    "one_shot",
    "cloud_finalize",
]
random.Random(int(sys.argv[1])).shuffle(variants)
print("\n".join(variants))
PY
  )
fi

"${PYTHON_BIN}" - "${VARIANT_ORDER[@]}" <<'PY'
import sys

expected = {
    "full",
    "no_edge",
    "static",
    "no_contract",
    "end_review",
    "one_shot",
    "cloud_finalize",
}
actual = sys.argv[1:]
if len(actual) != len(expected) or set(actual) != expected:
    raise SystemExit(
        "Variant order must contain each variant exactly once: "
        + ",".join(sorted(expected))
    )
PY

printf '%s\n' "${VARIANT_ORDER[@]}" >"${SUITE_ROOT}/variant_order.txt"
echo "Ablation variant order: ${VARIANT_ORDER[*]}" >&2
for variant in "${VARIANT_ORDER[@]}"; do
  run_named_variant "${variant}"
done

"${PYTHON_BIN}" -m benchmarks.check_ablation_suite \
  --suite-root "${SUITE_ROOT}" \
  --dataset "${DATASET}" \
  >"${SUITE_ROOT}/sanity_check_stdout.json"

"${PYTHON_BIN}" -m benchmarks.summarize_ablation_suite \
  --suite-root "${SUITE_ROOT}" \
  --dataset "${DATASET}"

echo "Ablation suite completed: ${SUITE_ROOT}" >&2
echo "Summary: ${SUITE_ROOT}/ablation_summary.md" >&2
