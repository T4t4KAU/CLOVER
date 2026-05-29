#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EXAMPLES_ROOT="${EXAMPLES_ROOT:-${SCRIPT_DIR}}"
GENERATED_ROOT="${CLOVER_EXAMPLE_DATABENCH_ROOT:-${SCRIPT_DIR}/.databench_eval_root}"
OUTPUT_ROOT="${CLOVER_EXAMPLE_OUTPUT_ROOT:-${SCRIPT_DIR}/runs}"
RUN_NAME="${CLOVER_EXAMPLE_RUN_NAME:-eval}"
REMOTE_BATCH_SIZE="${CLOVER_REMOTE_BATCH_SIZE:-8}"
LOCAL_BATCH_SIZE="${CLOVER_LOCAL_BATCH_SIZE:-8}"
MAX_WORKERS="${CLOVER_EVAL_MAX_WORKERS:-1}"
MAX_RETRIES="${CLOVER_MAX_RETRIES:-1}"
REMOTE_LLM_CONFIG="${CLOVER_REMOTE_LLM_CONFIG:-${CLOVER_LLM_CONFIG:-${LLM_CONFIG:-${REPO_ROOT}/config/doubao_remote_llm_config.json}}}"
LOCAL_SLM_CONFIG="${CLOVER_LOCAL_SLM_CONFIG:-${CLOVER_SLM_CONFIG:-${SLM_CONFIG:-${REPO_ROOT}/config/local_slm_config.json}}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"

cd "${REPO_ROOT}"

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_CMD[@]}" - "${EXAMPLES_ROOT}" "${GENERATED_ROOT}" <<'PY'
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

examples_root = Path(sys.argv[1]).expanduser().resolve()
generated_root = Path(sys.argv[2]).expanduser().resolve()

if generated_root.exists():
    shutil.rmtree(generated_root)
generated_root.mkdir(parents=True, exist_ok=True)

records = [
    json.loads(line)
    for line in (examples_root / "index.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
for record in records:
    example_id = record["example_id"]
    source_dir = examples_root / example_id
    dataset_dir = generated_root / example_id
    task_specs_dir = dataset_dir / "task_specs"
    task_specs_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_dir / "table.csv", dataset_dir / "table.csv")
    shutil.copy2(source_dir / "task.json", task_specs_dir / f"{example_id}.json")

    case_record = {
        "case_id": example_id,
        "dataset_id": example_id,
        "question": record["question"],
        "answer": record.get("expected_answer"),
        "type": record.get("answer_type"),
        "source_case_id": record.get("case_id"),
        "source_dataset_id": record.get("dataset_id"),
        "columns_used": record.get("columns_used"),
        "column_types": record.get("column_types"),
    }
    (dataset_dir / "cases.jsonl").write_text(
        json.dumps(case_record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
PY

PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}" \
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_CMD[@]}" eval.py \
    --eval \
    --databench-root "${GENERATED_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --run-name "${RUN_NAME}" \
    --remote-llm-config "${REMOTE_LLM_CONFIG}" \
    --local-slm-config "${LOCAL_SLM_CONFIG}" \
    --remote-batch-size "${REMOTE_BATCH_SIZE}" \
    --local-batch-size "${LOCAL_BATCH_SIZE}" \
    --max-workers "${MAX_WORKERS}" \
    --max-retries "${MAX_RETRIES}" \
    --overwrite \
    "$@"
