#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Final local-model test matrix for CLOVER.
#
# Runs each configured local model on TableBench, TabFact/TableFact, WikiTQ, and
# MMQA when an MMQA runner is available. This script is for coverage and result
# collection only: it does not enforce ACC thresholds.
#
# The main table datasets use benchmarks/run_vllm_eval_clover.sh. MMQA is kept
# pluggable because the MMQA evaluator lives on feat/mmqa-multitable-support:
# set CLOVER_MMQA_RUNNER=/path/to/runner.sh, or provide one of the default
# runner names below on that branch.
#
# Usage:
#   bash benchmarks/run_final_model_tests.sh
#
# Useful overrides:
#   CLOVER_FINAL_MODELS=/root/autodl-tmp/models/Qwen3-Coder-30B-A3B-Instruct:/root/autodl-tmp/models/Qwen3.6-27B:/root/autodl-tmp/models/Qwen3.6-35B-A3B:/root/autodl-tmp/models/North-Mini-Code-1.0
#   CLOVER_FINAL_DATASETS="tablebench tablefact wikitq mmqa"
#   CLOVER_FINAL_SAMPLE_SIZE=200   # optional smoke run; empty means full eval
#   CLOVER_FINAL_OUTPUT_ROOT=/root/autodl-tmp/CLOVER/benchmark/runs/final_tests
# =============================================================================

USER_PYTHON_BIN="/root/miniconda3/envs/clover/bin/python"
USER_MODELS="/root/autodl-tmp/models/Qwen2.5-7B-Instruct:/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct:/root/autodl-tmp/models/Qwen2.5-14B-Instruct:/root/autodl-tmp/models/Qwen2.5-Coder-14B-Instruct:/root/autodl-tmp/models/Qwen2.5-32B-Instruct:/root/autodl-tmp/models/Qwen2.5-Coder-32B-Instruct:/root/autodl-tmp/models/Qwen3-Coder-30B-A3B-Instruct:/root/autodl-tmp/models/Qwen3.6-27B:/root/autodl-tmp/models/Qwen3.6-35B-A3B:/root/autodl-tmp/models/North-Mini-Code-1.0"
USER_DATASETS="tablebench tablefact wikitq mmqa"
USER_GPUS="0"
USER_PORT="8000"
USER_MAX_MODEL_LEN="8192"
USER_GPU_MEM_UTIL="0.90"
USER_EVAL_CONCURRENCY="16"
USER_EDGE2_CONCURRENCY="8"
USER_EDGE2_BATCH_SIZE="1"
USER_MAX_RETRIES="3"
USER_OUTPUT_ROOT=""
USER_SAMPLE_SIZE=""
USER_MAX_CASES=""
USER_CONTINUE_ON_ERROR="true"
USER_RESTART_VLLM_PER_MODEL="true"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/run_final_model_tests.sh

Runs a local-model matrix over TableBench, TableFact, WikiTQ, and MMQA when an
MMQA runner is present. No ACC threshold is enforced.

Environment variables:
  CLOVER_FINAL_MODELS=path1:path2
  CLOVER_FINAL_DATASETS="tablebench tablefact wikitq mmqa"
  CLOVER_FINAL_SAMPLE_SIZE=200       # optional
  CLOVER_FINAL_MAX_CASES=100         # optional
  CLOVER_FINAL_OUTPUT_ROOT=/path
  CLOVER_MMQA_RUNNER=benchmarks/run_mmqa_eval_clover.sh
  CLOVER_FINAL_CONTINUE_ON_ERROR=true
  CLOVER_FINAL_RESTART_VLLM_PER_MODEL=true
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
  echo "Activate the intended Python environment, or set PYTHON_BIN." >&2
  exit 1
fi

MODELS="${CLOVER_FINAL_MODELS:-${USER_MODELS}}"
DATASETS="${CLOVER_FINAL_DATASETS:-${USER_DATASETS}}"
GPUS="${CLOVER_FINAL_GPUS:-${CLOVER_EDGE1_GPUS:-${USER_GPUS}}}"
PORT="${CLOVER_FINAL_PORT:-${CLOVER_EDGE1_PORT:-${USER_PORT}}}"
MAX_MODEL_LEN="${CLOVER_FINAL_MAX_MODEL_LEN:-${USER_MAX_MODEL_LEN}}"
GPU_MEM_UTIL="${CLOVER_FINAL_GPU_MEM_UTIL:-${USER_GPU_MEM_UTIL}}"
EVAL_CONCURRENCY="${CLOVER_FINAL_EVAL_CONCURRENCY:-${USER_EVAL_CONCURRENCY}}"
EDGE2_CONCURRENCY="${CLOVER_FINAL_EDGE2_CONCURRENCY:-${USER_EDGE2_CONCURRENCY}}"
EDGE2_BATCH_SIZE="${CLOVER_FINAL_EDGE2_BATCH_SIZE:-${USER_EDGE2_BATCH_SIZE}}"
MAX_RETRIES="${CLOVER_FINAL_MAX_RETRIES:-${USER_MAX_RETRIES}}"
SAMPLE_SIZE="${CLOVER_FINAL_SAMPLE_SIZE:-${USER_SAMPLE_SIZE}}"
MAX_CASES="${CLOVER_FINAL_MAX_CASES:-${USER_MAX_CASES}}"
CONTINUE_ON_ERROR="${CLOVER_FINAL_CONTINUE_ON_ERROR:-${USER_CONTINUE_ON_ERROR}}"
RESTART_VLLM_PER_MODEL="${CLOVER_FINAL_RESTART_VLLM_PER_MODEL:-${USER_RESTART_VLLM_PER_MODEL}}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/benchmark/runs/final_model_tests_${TIMESTAMP}"
OUTPUT_ROOT="${CLOVER_FINAL_OUTPUT_ROOT:-${USER_OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}}"
mkdir -p "${OUTPUT_ROOT}"

slugify() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import re, sys
text = sys.argv[1].strip().rstrip("/")
base = text.split("/")[-1] or text
print(re.sub(r"[^A-Za-z0-9_.-]+", "_", base))
PY
}

normalize_bool() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) printf 'true' ;;
    *) printf 'false' ;;
  esac
}

stop_vllm_port() {
  local port="$1"
  local pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)"
  fi
  if [[ -z "${pids}" ]]; then
    pids="$(pgrep -f "vllm serve .*--port ${port}" 2>/dev/null || true)"
  fi
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  echo "Stopping existing vLLM process(es) on port ${port}: ${pids}" >&2
  # shellcheck disable=SC2086
  kill ${pids} >/dev/null 2>&1 || true
  sleep 3
}

find_mmqa_runner() {
  if [[ -n "${CLOVER_MMQA_RUNNER:-}" && -f "${CLOVER_MMQA_RUNNER}" ]]; then
    printf '%s' "${CLOVER_MMQA_RUNNER}"
    return 0
  fi
  for candidate in \
    "${SCRIPT_DIR}/run_vllm_eval_mmqa.sh" \
    "${SCRIPT_DIR}/run_mmqa_eval_clover.sh" \
    "${SCRIPT_DIR}/run_mmqa_eval.sh"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s' "${candidate}"
      return 0
    fi
  done
  return 1
}

EXTRA_EVAL_ARGS=()
if [[ -n "${SAMPLE_SIZE}" ]]; then
  EXTRA_EVAL_ARGS+=(--sample-size "${SAMPLE_SIZE}")
fi
if [[ -n "${MAX_CASES}" ]]; then
  EXTRA_EVAL_ARGS+=(--max-cases "${MAX_CASES}")
fi

CONTINUE_ON_ERROR="$(normalize_bool "${CONTINUE_ON_ERROR}")"
RESTART_VLLM_PER_MODEL="$(normalize_bool "${RESTART_VLLM_PER_MODEL}")"

MODEL_PATHS=()
IFS=':' read -ra RAW_MODELS <<<"${MODELS}"
for model in "${RAW_MODELS[@]}"; do
  [[ -z "${model}" ]] && continue
  MODEL_PATHS+=("${model}")
done
if [[ "${#MODEL_PATHS[@]}" -eq 0 ]]; then
  echo "No models configured." >&2
  exit 2
fi

"${PYTHON_BIN}" - "${OUTPUT_ROOT}" "${MODELS}" "${DATASETS}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
metadata = {
    "models": [item for item in sys.argv[2].split(":") if item],
    "datasets": sys.argv[3].split(),
    "note": "Final model test matrix; no ACC threshold enforced.",
}
(root / "final_model_tests_metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

run_table_dataset() {
  local dataset="$1"
  local model_path="$2"
  local model_name="$3"
  local run_name="$4"

  env -u CLOVER_EDGE2_PORT \
    PYTHON_BIN="${PYTHON_BIN}" \
    CLOVER_EDGE1_MODEL_PATH="${model_path}" \
    CLOVER_EDGE2_MODEL_PATH="${model_path}" \
    CLOVER_EDGE1_GPUS="${GPUS}" \
    CLOVER_EDGE1_PORT="${PORT}" \
    CLOVER_EDGE1_MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    CLOVER_EDGE1_GPU_MEM_UTIL="${GPU_MEM_UTIL}" \
    CLOVER_EDGE2_GPU_MEM_UTIL="${GPU_MEM_UTIL}" \
    CLOVER_EVAL_CONCURRENCY="${EVAL_CONCURRENCY}" \
    CLOVER_EDGE2_CONCURRENCY="${EDGE2_CONCURRENCY}" \
    CLOVER_EDGE2_BATCH_SIZE="${EDGE2_BATCH_SIZE}" \
    CLOVER_EDGE2_MAX_RETRIES="${MAX_RETRIES}" \
    CLOVER_VLLM_PERSIST_SERVER=true \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    RUN_NAME="${run_name}" \
    bash "${SCRIPT_DIR}/run_vllm_eval_clover.sh" "${dataset}" "${EXTRA_EVAL_ARGS[@]}"
}

run_mmqa_dataset() {
  local model_path="$1"
  local model_name="$2"
  local run_name="$3"
  local runner=""
  if ! runner="$(find_mmqa_runner)"; then
    local skip_dir="${OUTPUT_ROOT}/${run_name}"
    mkdir -p "${skip_dir}"
    "${PYTHON_BIN}" - "${skip_dir}/run_summary.json" "${model_path}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "stage": "mmqa_eval",
    "skipped": True,
    "reason": "MMQA runner not found on this branch",
    "model": sys.argv[2],
    "brief_summary": {"benchmark": "mmqa_eval", "skipped": True},
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
    echo "Skipping MMQA for ${model_name}: no MMQA runner found on this branch." >&2
    return 0
  fi

  env -u CLOVER_EDGE2_PORT \
    PYTHON_BIN="${PYTHON_BIN}" \
    CLOVER_EDGE_MODEL_PATH="${model_path}" \
    CLOVER_EDGE1_MODEL_PATH="${model_path}" \
    CLOVER_EDGE2_MODEL_PATH="${model_path}" \
    CLOVER_EDGE1_GPUS="${GPUS}" \
    CLOVER_EDGE1_PORT="${PORT}" \
    CLOVER_EDGE1_MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    CLOVER_EDGE1_GPU_MEM_UTIL="${GPU_MEM_UTIL}" \
    CLOVER_EDGE2_GPU_MEM_UTIL="${GPU_MEM_UTIL}" \
    CLOVER_EVAL_CONCURRENCY="${EVAL_CONCURRENCY}" \
    CLOVER_EDGE2_CONCURRENCY="${EDGE2_CONCURRENCY}" \
    CLOVER_EDGE2_BATCH_SIZE="${EDGE2_BATCH_SIZE}" \
    CLOVER_EDGE2_MAX_RETRIES="${MAX_RETRIES}" \
    CLOVER_VLLM_PERSIST_SERVER=true \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    RUN_NAME="${run_name}" \
    bash "${runner}" "${EXTRA_EVAL_ARGS[@]}"
}

run_one() {
  local dataset="$1"
  local model_path="$2"
  local model_name="$3"
  local run_name="${dataset}_${model_name}"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  if [[ -f "${run_dir}/run_summary.json" ]]; then
    echo "Skipping existing run: ${run_name}" >&2
    return 0
  fi
  echo "Running final test: ${run_name}" >&2
  case "${dataset}" in
    tablebench|wikitq|tablefact)
      run_table_dataset "${dataset}" "${model_path}" "${model_name}" "${run_name}"
      ;;
    tabfact)
      run_table_dataset "tablefact" "${model_path}" "${model_name}" "${run_name}"
      ;;
    mmqa)
      run_mmqa_dataset "${model_path}" "${model_name}" "${run_name}"
      ;;
    *)
      echo "Unsupported final-test dataset: ${dataset}" >&2
      return 2
      ;;
  esac
}

for model_path in "${MODEL_PATHS[@]}"; do
  if [[ "${model_path}" == /* && ! -e "${model_path}" ]]; then
    echo "Model path does not exist, skipping: ${model_path}" >&2
    continue
  fi
  model_name="$(slugify "${model_path}")"
  if [[ "${RESTART_VLLM_PER_MODEL}" == "true" ]]; then
    stop_vllm_port "${PORT}"
  fi
  for dataset in ${DATASETS}; do
    if ! run_one "${dataset}" "${model_path}" "${model_name}"; then
      if [[ "${CONTINUE_ON_ERROR}" == "true" ]]; then
        echo "Continuing after failed run: ${dataset}_${model_name}" >&2
      else
        exit 1
      fi
    fi
  done
  if [[ "${RESTART_VLLM_PER_MODEL}" == "true" ]]; then
    stop_vllm_port "${PORT}"
  fi
done

"${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for summary_path in sorted(root.glob("*/run_summary.json")):
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        rows.append({"run": summary_path.parent.name, "error": str(exc)})
        continue
    brief = summary.get("brief_summary") or {}
    rows.append({
        "run": summary_path.parent.name,
        "stage": summary.get("stage") or brief.get("benchmark"),
        "model": (
            (summary.get("remote_llm") or {}).get("model")
            or (summary.get("cloud_model") or summary.get("model"))
            or ""
        ),
        "accuracy_percent": brief.get("Acc. (%)", brief.get("accuracy_percent", "")),
        "skipped": summary.get("skipped", False),
        "total_cases": summary.get("total_cases", ""),
        "calls_per_query": brief.get("Calls/Q", ""),
        "avg_total_tokens_per_query": brief.get("Avg Total Tok / Query", ""),
        "avg_max_context_tokens_per_query": summary.get("avg_max_context_tokens_per_query", ""),
    })

(root / "final_model_tests.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
if rows:
    fieldnames = list(rows[0])
    with (root / "final_model_tests.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(fieldnames) + " |", "|" + "|".join(["---"] * len(fieldnames)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(name, "")) for name in fieldnames) + " |")
    (root / "final_model_tests.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Final model test summary: {root / 'final_model_tests.md'}")
PY

echo "Final model tests completed: ${OUTPUT_ROOT}" >&2
