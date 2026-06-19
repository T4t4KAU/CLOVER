#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS_ROOT="${CLOVER_DATASETS_ROOT:-${REPO_ROOT}/datasets}"
DATASETS="${CLOVER_DATASETS:-tablebench,wikitq,tablefact}"
TABLEBENCH_SOURCE="${CLOVER_TABLEBENCH_SOURCE:-huggingface}"
WIKITQ_REPO="${WIKITQ_REPO:-https://github.com/ppasupat/WikiTableQuestions.git}"
TABLEFACT_REPO="${TABLEFACT_REPO:-https://github.com/wenhuchen/Table-Fact-Checking.git}"
WIKITQ_SOURCE_ROOT="${WIKITQ_SOURCE_ROOT:-${DATASETS_ROOT}/WikiTableQuestions}"
TABLEFACT_SOURCE_ROOT="${TABLEFACT_SOURCE_ROOT:-${DATASETS_ROOT}/Table-Fact-Checking}"
WIKITQ_SPLIT="${WIKITQ_SPLIT:-pristine-unseen-tables}"
TABLEFACT_SPLITS="${TABLEFACT_SPLITS:-test}"
OVERWRITE="${CLOVER_DATASET_OVERWRITE:-0}"
DOWNLOAD_OVERWRITE="${CLOVER_DOWNLOAD_OVERWRITE:-0}"

usage() {
  cat <<'EOF'
Usage: bash benchmarks/download_datasets.sh [options]

Download and convert TableBench, WikiTableQuestions, and TableFact.

Options:
  --dataset NAME       Dataset to prepare; repeatable. Supported:
                       tablebench, wikitq, tablefact, all
  --datasets-root DIR  Raw and converted dataset root (default: datasets)
  --overwrite          Replace converted dataset directories
  --download-overwrite Redownload/reclone raw sources
  -h, --help           Show this help

Environment:
  PYTHON_BIN, CLOVER_DATASETS, CLOVER_DATASETS_ROOT
  WIKITQ_SOURCE_ROOT, TABLEFACT_SOURCE_ROOT
  WIKITQ_SPLIT, TABLEFACT_SPLITS
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
      WIKITQ_SOURCE_ROOT="${DATASETS_ROOT}/WikiTableQuestions"
      TABLEFACT_SOURCE_ROOT="${DATASETS_ROOT}/Table-Fact-Checking"
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
    *)
      echo "Unsupported dataset: $1" >&2
      exit 2
      ;;
  esac
}

DATASET_LIST=()
for item in "${SELECTED[@]}"; do
  while IFS= read -r normalized; do
    [[ -n "${normalized}" ]] && DATASET_LIST+=("${normalized}")
  done < <(normalize_dataset "${item}")
done

if [[ "${#DATASET_LIST[@]}" -eq 0 ]]; then
  echo "No datasets selected." >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]] && ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "git is required to download WikiTableQuestions and TableFact." >&2
  exit 1
fi

mkdir -p "${DATASETS_ROOT}"

clone_or_update() {
  local repo_url="$1"
  local target="$2"
  local label="$3"

  if [[ -d "${target}/.git" ]]; then
    if truthy "${DOWNLOAD_OVERWRITE}"; then
      echo "Updating ${label}: ${target}" >&2
      git -C "${target}" pull --ff-only
    else
      echo "Using existing ${label} source: ${target}" >&2
    fi
    return
  fi
  if [[ -e "${target}" ]]; then
    echo "Using existing ${label} source: ${target}" >&2
    return
  fi
  echo "Downloading ${label}: ${repo_url}" >&2
  git clone --depth 1 "${repo_url}" "${target}"
}

run_converter() {
  PYTHONWARNINGS="ignore" \
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" "$@"
}

contains_dataset() {
  local wanted="$1"
  local item
  for item in "${DATASET_LIST[@]}"; do
    [[ "${item}" == "${wanted}" ]] && return 0
  done
  return 1
}

if contains_dataset tablebench; then
  if find "${DATASETS_ROOT}/tablebench" -mindepth 2 -maxdepth 2 \
      -name cases.jsonl -print -quit 2>/dev/null | grep -q . \
      && ! truthy "${OVERWRITE}"; then
    echo "TableBench is already converted: ${DATASETS_ROOT}/tablebench" >&2
  else
    echo "Preparing TableBench" >&2
    TABLEBENCH_ARGS=(
      -m benchmarks.tablebench.download
      --output-root "${DATASETS_ROOT}/tablebench"
      --source-root "${DATASETS_ROOT}/tablebench_source"
      --dataset-source "${TABLEBENCH_SOURCE}"
    )
    truthy "${OVERWRITE}" && TABLEBENCH_ARGS+=(--overwrite)
    truthy "${DOWNLOAD_OVERWRITE}" && TABLEBENCH_ARGS+=(--download-overwrite)
    run_converter "${TABLEBENCH_ARGS[@]}" \
      >"${DATASETS_ROOT}/tablebench_download_summary.json"
  fi
fi

if contains_dataset wikitq; then
  clone_or_update "${WIKITQ_REPO}" "${WIKITQ_SOURCE_ROOT}" "WikiTableQuestions"
  if find "${DATASETS_ROOT}/wikitq" -mindepth 2 -maxdepth 2 \
      -name cases.jsonl -print -quit 2>/dev/null | grep -q . \
      && ! truthy "${OVERWRITE}"; then
    echo "WikiTableQuestions is already converted: ${DATASETS_ROOT}/wikitq" >&2
  else
    echo "Converting WikiTableQuestions (${WIKITQ_SPLIT})" >&2
    WIKITQ_ARGS=(
      -m benchmarks.wikitq.download
      --source-root "${WIKITQ_SOURCE_ROOT}"
      --output-root "${DATASETS_ROOT}/wikitq"
      --split "${WIKITQ_SPLIT}"
    )
    truthy "${OVERWRITE}" && WIKITQ_ARGS+=(--overwrite)
    run_converter "${WIKITQ_ARGS[@]}" \
      >"${DATASETS_ROOT}/wikitq_conversion_summary.json"
  fi
fi

if contains_dataset tablefact; then
  clone_or_update "${TABLEFACT_REPO}" "${TABLEFACT_SOURCE_ROOT}" "TableFact"
  if find "${DATASETS_ROOT}/tablefact" -mindepth 2 -maxdepth 2 \
      -name cases.jsonl -print -quit 2>/dev/null | grep -q . \
      && ! truthy "${OVERWRITE}"; then
    echo "TableFact is already converted: ${DATASETS_ROOT}/tablefact" >&2
  else
    echo "Converting TableFact (${TABLEFACT_SPLITS})" >&2
    TABLEFACT_ARGS=(
      -m benchmarks.tablefact.download
      --source-root "${TABLEFACT_SOURCE_ROOT}"
      --output-root "${DATASETS_ROOT}/tablefact"
    )
    IFS=',' read -r -a SPLIT_ITEMS <<< "${TABLEFACT_SPLITS}"
    for split in "${SPLIT_ITEMS[@]}"; do
      split="${split#"${split%%[![:space:]]*}"}"
      split="${split%"${split##*[![:space:]]}"}"
      [[ -n "${split}" ]] && TABLEFACT_ARGS+=(--split "${split}")
    done
    truthy "${OVERWRITE}" && TABLEFACT_ARGS+=(--overwrite)
    run_converter "${TABLEFACT_ARGS[@]}" \
      >"${DATASETS_ROOT}/tablefact_conversion_summary.json"
  fi
fi

echo "Datasets are ready under: ${DATASETS_ROOT}" >&2
