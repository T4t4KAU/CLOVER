#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS_ROOT="${CLOVER_DATASETS_ROOT:-${REPO_ROOT}/datasets}"
DATASETS="${CLOVER_DATASETS:-tablebench,wikitq,tablefact}"
MODELSCOPE_CACHE_DIR="${CLOVER_MODELSCOPE_CACHE_DIR:-${DATASETS_ROOT}/modelscope_cache}"
TABLEBENCH_REPO="${TABLEBENCH_MODELSCOPE_REPO:-Multilingual-Multimodal-NLP/TableBench}"
WIKITQ_REPO="${WIKITQ_MODELSCOPE_REPO:-stanfordnlp/wikitablequestions}"
WIKITQ_SUBSET="${WIKITQ_MODELSCOPE_SUBSET:-random-split-1}"
WIKITQ_SOURCE_SPLIT="${WIKITQ_MODELSCOPE_SPLIT:-test}"
WIKITQ_SPLIT="${WIKITQ_SPLIT:-pristine-unseen-tables}"
TABLEFACT_REPO="${TABLEFACT_MODELSCOPE_REPO:-ibm-research/tab_fact}"
TABLEFACT_SPLITS="${TABLEFACT_SPLITS:-test}"
OVERWRITE="${CLOVER_DATASET_OVERWRITE:-0}"
DOWNLOAD_OVERWRITE="${CLOVER_DOWNLOAD_OVERWRITE:-0}"

usage() {
  cat <<'EOF'
Usage: bash benchmarks/download_datasets_modelscope.sh [options]

Download and convert TableBench, WikiTableQuestions, and TableFact through
ModelScope.

Options:
  --dataset NAME       Dataset to prepare; repeatable:
                       tablebench, wikitq, tablefact, all
  --datasets-root DIR  Converted dataset root
  --cache-dir DIR      ModelScope download cache
  --overwrite          Replace converted datasets
  --download-overwrite Force ModelScope to redownload source data
  -h, --help           Show this help

Environment:
  PYTHON_BIN, CLOVER_DATASETS, CLOVER_DATASETS_ROOT
  CLOVER_MODELSCOPE_CACHE_DIR
  TABLEBENCH_MODELSCOPE_REPO
  WIKITQ_MODELSCOPE_REPO, WIKITQ_MODELSCOPE_SUBSET
  WIKITQ_MODELSCOPE_SPLIT, WIKITQ_SPLIT
  TABLEFACT_MODELSCOPE_REPO, TABLEFACT_SPLITS
EOF
}

truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

SELECTED=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dataset)
      [[ "$#" -ge 2 ]] || { echo "--dataset requires a value" >&2; exit 2; }
      SELECTED+=("$2")
      shift 2
      ;;
    --datasets-root)
      [[ "$#" -ge 2 ]] || { echo "--datasets-root requires a value" >&2; exit 2; }
      DATASETS_ROOT="$2"
      shift 2
      ;;
    --cache-dir)
      [[ "$#" -ge 2 ]] || { echo "--cache-dir requires a value" >&2; exit 2; }
      MODELSCOPE_CACHE_DIR="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --download-overwrite)
      DOWNLOAD_OVERWRITE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${#SELECTED[@]}" -eq 0 ]]; then
  IFS=',' read -r -a SELECTED <<< "${DATASETS}"
fi

normalize_dataset() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')" in
    tablebench) printf '%s\n' tablebench ;;
    wikitq|wikitablequestions) printf '%s\n' wikitq ;;
    tablefact|tabfact) printf '%s\n' tablefact ;;
    all) printf '%s\n' tablebench wikitq tablefact ;;
    "") ;;
    *) echo "Unsupported dataset: $1" >&2; exit 2 ;;
  esac
}

DATASET_LIST=()
for item in "${SELECTED[@]}"; do
  while IFS= read -r normalized; do
    [[ -n "${normalized}" ]] && DATASET_LIST+=("${normalized}")
  done < <(normalize_dataset "${item}")
done

contains_dataset() {
  local wanted="$1"
  local item
  for item in "${DATASET_LIST[@]}"; do
    [[ "${item}" == "${wanted}" ]] && return 0
  done
  return 1
}

converted_exists() {
  find "$1" -mindepth 2 -maxdepth 2 -name cases.jsonl \
    -print -quit 2>/dev/null | grep -q .
}

if [[ ! -x "${PYTHON_BIN}" ]] && ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi
"${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("modelscope") is None:
    raise SystemExit(
        f"ModelScope is not installed in {sys.executable}. "
        f"Run: {sys.executable} -m pip install -r requirements.txt"
    )
PY

mkdir -p "${DATASETS_ROOT}" "${MODELSCOPE_CACHE_DIR}"

run_python() {
  PYTHONWARNINGS="ignore" \
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" "$@"
}

if contains_dataset tablebench; then
  if converted_exists "${DATASETS_ROOT}/tablebench" && ! truthy "${OVERWRITE}"; then
    echo "TableBench is already converted: ${DATASETS_ROOT}/tablebench" >&2
  else
    echo "Downloading TableBench from ModelScope: ${TABLEBENCH_REPO}" >&2
    ARGS=(
      -m benchmarks.tablebench.download
      --repo-id "${TABLEBENCH_REPO}"
      --dataset-source modelscope
      --modelscope-cache-dir "${MODELSCOPE_CACHE_DIR}/tablebench"
      --source-root "${DATASETS_ROOT}/tablebench_modelscope_source"
      --output-root "${DATASETS_ROOT}/tablebench"
    )
    truthy "${OVERWRITE}" && ARGS+=(--overwrite)
    truthy "${DOWNLOAD_OVERWRITE}" && ARGS+=(--download-overwrite)
    run_python "${ARGS[@]}" \
      >"${DATASETS_ROOT}/tablebench_modelscope_summary.json"
  fi
fi

if contains_dataset wikitq; then
  if converted_exists "${DATASETS_ROOT}/wikitq" && ! truthy "${OVERWRITE}"; then
    echo "WikiTableQuestions is already converted: ${DATASETS_ROOT}/wikitq" >&2
  else
    echo "Downloading WikiTableQuestions from ModelScope: ${WIKITQ_REPO}" >&2
    ARGS=(
      -m benchmarks.modelscope_download
      --dataset wikitq
      --repo-id "${WIKITQ_REPO}"
      --subset-name "${WIKITQ_SUBSET}"
      --split "${WIKITQ_SOURCE_SPLIT}"
      --output-split "${WIKITQ_SPLIT}"
      --cache-dir "${MODELSCOPE_CACHE_DIR}/wikitq"
      --output-root "${DATASETS_ROOT}/wikitq"
    )
    truthy "${OVERWRITE}" && ARGS+=(--overwrite)
    truthy "${DOWNLOAD_OVERWRITE}" && ARGS+=(--force-redownload)
    run_python "${ARGS[@]}" \
      >"${DATASETS_ROOT}/wikitq_modelscope_summary.json"
  fi
fi

if contains_dataset tablefact; then
  if converted_exists "${DATASETS_ROOT}/tablefact" && ! truthy "${OVERWRITE}"; then
    echo "TableFact is already converted: ${DATASETS_ROOT}/tablefact" >&2
  else
    echo "Downloading TableFact from ModelScope: ${TABLEFACT_REPO}" >&2
    ARGS=(
      -m benchmarks.modelscope_download
      --dataset tablefact
      --repo-id "${TABLEFACT_REPO}"
      --cache-dir "${MODELSCOPE_CACHE_DIR}/tablefact"
      --output-root "${DATASETS_ROOT}/tablefact"
    )
    IFS=',' read -r -a SPLIT_ITEMS <<< "${TABLEFACT_SPLITS}"
    for split in "${SPLIT_ITEMS[@]}"; do
      split="${split#"${split%%[![:space:]]*}"}"
      split="${split%"${split##*[![:space:]]}"}"
      [[ -n "${split}" ]] && ARGS+=(--split "${split}")
    done
    truthy "${OVERWRITE}" && ARGS+=(--overwrite)
    truthy "${DOWNLOAD_OVERWRITE}" && ARGS+=(--force-redownload)
    run_python "${ARGS[@]}" \
      >"${DATASETS_ROOT}/tablefact_modelscope_summary.json"
  fi
fi

echo "ModelScope datasets are ready under: ${DATASETS_ROOT}" >&2
