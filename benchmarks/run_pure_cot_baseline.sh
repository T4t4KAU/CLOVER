#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Pure Chain-of-Thought (CoT) baseline for CLOVER table benchmarks.
#
# Sends the full table(s) + question to one LLM, asks it to reason step by step,
# parses the explicit "Final Answer:", and scores with the dataset-native metric.
# No tools, no SQL/Python execution, no edge model, no CLOVER runtime.
#
# Usage:
#   bash benchmarks/run_pure_cot_baseline.sh DATASET [MODEL_CONFIG] [options...]
#
# DATASET:
#   tablebench | wikitq | tablefact | mmqa
#   aliases: wikitablequestions -> wikitq, tabfact -> tablefact
#
# Common options:
#   --max-cases N
#   --sample-size N
#   --case-id ID              Repeatable or comma-separated
#   --dataset-id ID
#   --split SPLIT             WikiTQ/MMQA split override
#   --seed N
#   --max-workers N
#   --output-root PATH
#   --run-name NAME
#   --overwrite
#   --validate-only
#
# Dataset roots:
#   CLOVER_DATASETS_ROOT      Default <repo>/datasets
#   TABLEBENCH_ROOT
#   WIKITQ_ROOT
#   TABLEFACT_ROOT
#   MMQA_ROOT
#
# Examples:
#   bash benchmarks/run_pure_cot_baseline.sh mmqa model_config/local_llm.json --sample-size 100
#   MMQA_SPLIT=two_table bash benchmarks/run_pure_cot_baseline.sh mmqa model_config/local_llm.json
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASET="${1:-tablebench}"
if [[ "$#" -gt 0 ]]; then
  shift
fi
DATASET="$(printf '%s' "${DATASET}" | tr '[:upper:]' '[:lower:]')"
case "${DATASET}" in
  tablebench|wikitq|tablefact|mmqa) ;;
  wikitablequestions) DATASET="wikitq" ;;
  tabfact) DATASET="tablefact" ;;
  -h|--help)
    sed -n '4,38p' "$0"
    exit 0
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

MODEL_CONFIG="${CLOVER_COT_LLM_CONFIG:-${REPO_ROOT}/model_config/deepseek_remote_llm_config.json}"
if [[ "${1:-}" != "" && "${1:-}" != --* ]]; then
  MODEL_CONFIG="$1"
  shift
fi
[[ "${MODEL_CONFIG}" == /* ]] || MODEL_CONFIG="${REPO_ROOT}/${MODEL_CONFIG}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python 2>/dev/null || true)}"
DATASETS_ROOT="${CLOVER_DATASETS_ROOT:-${REPO_ROOT}/datasets}"
TABLEBENCH_ROOT="${TABLEBENCH_ROOT:-${DATASETS_ROOT}/tablebench}"
WIKITQ_ROOT="${WIKITQ_ROOT:-${DATASETS_ROOT}/wikitq}"
TABLEFACT_ROOT="${TABLEFACT_ROOT:-${DATASETS_ROOT}/tablefact}"
MMQA_ROOT="${MMQA_ROOT:-${DATASETS_ROOT}/mmqa}"
WIKITQ_SPLIT="${WIKITQ_SPLIT:-pristine-unseen-tables}"
TABLEFACT_SPLIT="${TABLEFACT_SPLIT:-test}"
TABLEFACT_SUBSET="${TABLEFACT_SUBSET:-small}"
MMQA_SPLIT="${MMQA_SPLIT:-}"

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Activate the intended Python environment before running this script." >&2
  exit 1
fi
if [[ ! -f "${MODEL_CONFIG}" ]]; then
  echo "Pure CoT model config not found: ${MODEL_CONFIG}" >&2
  exit 1
fi

declare -a DEFAULT_ARGS=()
case "${DATASET}" in
  tablebench)
    DATASET_ROOT="${TABLEBENCH_ROOT}"
    DEFAULT_ARGS=(--qtype FactChecking --qtype NumericalReasoning)
    ;;
  wikitq)
    DATASET_ROOT="${WIKITQ_ROOT}"
    DEFAULT_ARGS=(--split "${WIKITQ_SPLIT}")
    ;;
  tablefact)
    DATASET_ROOT="${TABLEFACT_ROOT}"
    DEFAULT_ARGS=(--split "${TABLEFACT_SPLIT}" --subset "${TABLEFACT_SUBSET}")
    ;;
  mmqa)
    DATASET_ROOT="${MMQA_ROOT}"
    DEFAULT_ARGS=()
    if [[ -n "${MMQA_SPLIT}" ]]; then
      DEFAULT_ARGS+=(--split "${MMQA_SPLIT}")
    fi
    ;;
esac

PYTHONWARNINGS="ignore" \
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
exec "${PYTHON_BIN}" -m benchmarks.table_cot_baseline \
  "${DATASET}" "${MODEL_CONFIG}" "${DATASET_ROOT}" \
  ${DEFAULT_ARGS[@]+"${DEFAULT_ARGS[@]}"} "$@"
