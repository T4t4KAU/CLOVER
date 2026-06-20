#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASET="${1:-tablebench}"
if [[ "$#" -gt 0 ]]; then
  shift
fi
DATASET="$(printf '%s' "${DATASET}" | tr '[:upper:]' '[:lower:]')"
case "${DATASET}" in
  tablebench|wikitq|tablefact) ;;
  wikitablequestions) DATASET="wikitq" ;;
  tabfact) DATASET="tablefact" ;;
  -h|--help)
    cat <<'EOF'
Usage:
  bash benchmarks/run_pure_cot_baseline.sh DATASET [MODEL_CONFIG] [options]

DATASET:
  tablebench | wikitq | tablefact

Official experiment scopes:
  TableBench: FactChecking + NumericalReasoning (493 cases)
  WikiTQ: pristine-unseen-tables (4344 cases)
  TableFact: TabFact small-test (1998 cases)

The baseline sends the full table and question to one LLM, asks it to reason
step by step, and scores the final answer with each dataset's native metric.
The model receives no tools and does not generate or execute SQL/Python.

Options:
  --max-cases N
  --sample-size N
  --case-id ID              Repeatable
  --seed N
  --max-workers N
  --output-root PATH
  --run-name NAME
  --overwrite
  --validate-only

Examples:
  bash benchmarks/run_pure_cot_baseline.sh tablebench
  bash benchmarks/run_pure_cot_baseline.sh tablefact
  bash benchmarks/run_pure_cot_baseline.sh wikitq \
    model_config/deepseek_remote_llm_config.json --sample-size 100
EOF
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
WIKITQ_SPLIT="${WIKITQ_SPLIT:-pristine-unseen-tables}"
TABLEFACT_SPLIT="test"
TABLEFACT_SUBSET="small"

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Activate the intended Python environment before running this script." >&2
  exit 1
fi
if [[ ! -f "${MODEL_CONFIG}" ]]; then
  echo "Pure CoT model config not found: ${MODEL_CONFIG}" >&2
  exit 1
fi

PYTHONWARNINGS="ignore" \
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
"${PYTHON_BIN}" - "${DATASET}" "${MODEL_CONFIG}" "${TABLEBENCH_ROOT}" \
  "${WIKITQ_ROOT}" "${TABLEFACT_ROOT}" "${WIKITQ_SPLIT}" \
  "${TABLEFACT_SPLIT}" "${TABLEFACT_SUBSET}" "$@" <<'PY'
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from benchmarks.costing import estimate_openai_text_cost
from benchmarks.tablebench.metrics import score_tablebench_answer
from benchmarks.wikitq.metrics import score_wikitq_answer
from clover.config import load_model_config
from clover.supervisor.client import (
    create_remote_llm_client,
    extract_token_usage,
    generate_remote_text,
)


dataset, config_path, tablebench_root, wikitq_root, tablefact_root = sys.argv[1:6]
wikitq_split, tablefact_split, tablefact_subset = sys.argv[6:9]

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--max-cases", type=int, default=None)
parser.add_argument("--sample-size", type=int, default=None)
parser.add_argument("--case-id", action="append", default=[])
parser.add_argument("--seed", type=int, default=20260528)
parser.add_argument("--max-workers", type=int, default=8)
parser.add_argument("--output-root", type=Path, default=Path("benchmark/runs"))
parser.add_argument("--run-name", default=None)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--validate-only", action="store_true")
args = parser.parse_args(sys.argv[9:])

if args.max_cases is not None and args.max_cases <= 0:
    raise SystemExit("--max-cases must be positive")
if args.sample_size is not None and args.sample_size <= 0:
    raise SystemExit("--sample-size must be positive")
if args.max_workers <= 0:
    raise SystemExit("--max-workers must be positive")

roots = {
    "tablebench": Path(tablebench_root).expanduser().resolve(),
    "wikitq": Path(wikitq_root).expanduser().resolve(),
    "tablefact": Path(tablefact_root).expanduser().resolve(),
}
dataset_root = roots[dataset]
if not dataset_root.is_dir():
    raise SystemExit(f"Converted dataset not found: {dataset_root}")


def select_cases(*, limited: bool) -> list[dict[str, Any]]:
    common = {
        "max_cases": args.max_cases if limited else None,
        "case_ids": set(args.case_id) if limited else set(),
        "dataset_id": None,
        "sample_size": args.sample_size if limited else None,
        "seed": args.seed,
    }
    if dataset == "tablebench":
        from benchmarks.tablebench.eval import select_tablebench_cases

        return select_tablebench_cases(
            tablebench_root=dataset_root,
            qtypes={"FactChecking", "NumericalReasoning"},
            qsubtypes=set(),
            include_visualization=False,
            **common,
        )
    if dataset == "wikitq":
        from benchmarks.wikitq.eval import select_wikitq_cases

        return select_wikitq_cases(
            wikitq_root=dataset_root,
            split=wikitq_split,
            **common,
        )
    from benchmarks.tablefact.eval import select_tablefact_cases

    return select_tablefact_cases(
        tablefact_root=dataset_root,
        split=tablefact_split,
        subset=tablefact_subset,
        **common,
    )


expected_counts = {"tablebench": 493, "wikitq": 4344, "tablefact": 1998}
available_cases = select_cases(limited=False)
expected_count = expected_counts[dataset]
if len(available_cases) != expected_count:
    raise SystemExit(
        f"Unexpected {dataset} scope: expected {expected_count}, "
        f"selected {len(available_cases)}"
    )
selected_cases = select_cases(limited=True)
if not selected_cases:
    raise SystemExit("No cases selected")
if args.validate_only:
    print(
        json.dumps(
            {
                "dataset": dataset,
                "method": "vanilla_chain_of_thought",
                "available_cases": len(available_cases),
                "selected_cases": len(selected_cases),
                "expected_cases": expected_count,
                "scope_valid": True,
            },
            indent=2,
        )
    )
    raise SystemExit(0)

run_name = args.run_name or f"{dataset}_pure_cot_{time.strftime('%Y%m%d_%H%M%S')}"
output_dir = (args.output_root / run_name).expanduser().resolve()
if output_dir.exists():
    if not args.overwrite:
        raise SystemExit(
            f"Output directory already exists: {output_dir}; pass --overwrite"
        )
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)

model_config = load_model_config(Path(config_path).expanduser().resolve())
client = create_remote_llm_client(model_config)


def load_case_payload(case: dict[str, Any]) -> dict[str, Any]:
    cases_path = dataset_root / str(case["dataset_id"]) / "cases.jsonl"
    with cases_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if str(payload.get("case_id")) == str(case["case_id"]):
                return payload
    raise ValueError(f"Case payload not found: {case['case_id']}")


def load_table_csv(case: dict[str, Any]) -> str:
    table_path = dataset_root / str(case["dataset_id"]) / "table.csv"
    with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    rendered = []
    for row in rows:
        rendered.append(
            ",".join(
                '"' + value.replace('"', '""') + '"'
                if any(char in value for char in ',\"\n')
                else value
                for value in row
            )
        )
    return "\n".join(rendered)


def render_prompt(case: dict[str, Any], payload: dict[str, Any]) -> str:
    context = str(payload.get("caption") or "").strip()
    context_block = f"\nTable context: {context}\n" if context else ""
    answer_type = str(case.get("answer_type") or payload.get("type") or "string")
    return f"""Answer the question using the table below.

Let's think step by step. Then give the answer on the final line in this format:
Final Answer: <answer>

Use true or false for a boolean answer. Separate list items with commas.
{context_block}
Answer type: {answer_type}
Question: {case.get("question") or payload.get("question")}

Table (CSV):
```csv
{load_table_csv(case)}
```
"""


FINAL_ANSWER_RE = re.compile(
    r"(?im)^\s*(?:\*\*)?final\s+answer(?:\*\*)?\s*:\s*(.*?)\s*$"
)


def parse_answer(text: str) -> tuple[str, str]:
    matches = [match.strip() for match in FINAL_ANSWER_RE.findall(text) if match.strip()]
    if matches:
        return matches[-1].strip("*").strip(), "explicit_final_answer"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "", "empty_response"
    return lines[-1].strip("*").strip(), "last_line_fallback"


def score_case(
    case: dict[str, Any],
    payload: dict[str, Any],
    answer: str,
) -> tuple[str, float, bool, str, str]:
    if dataset == "wikitq":
        score = score_wikitq_answer(
            expected=case.get("expected_answer"),
            expected_canon=case.get("expected_canon"),
            actual=answer,
        )
    else:
        score = score_tablebench_answer(
            expected=case.get("expected_answer"),
            actual=answer,
            qtype=(
                "FactChecking"
                if dataset == "tablefact"
                else case.get("qtype")
            ),
            qsubtype=(
                case.get("subset")
                if dataset == "tablefact"
                else case.get("qsubtype")
            ),
        )
    metric = "accuracy" if dataset == "tablefact" else score.metric
    return metric, score.score, score.correct, score.expected, score.actual


def evaluate(index: int, case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    payload = load_case_payload(case)
    prompt = render_prompt(case, payload)
    base = {
        "sample_index": index,
        "dataset": dataset,
        "dataset_id": case["dataset_id"],
        "case_id": case["case_id"],
        "question": case.get("question"),
        "answer_type": case.get("answer_type"),
        "qtype": case.get("qtype"),
        "qsubtype": case.get("qsubtype"),
        "subset": case.get("subset"),
        "expected": case.get("expected_answer"),
    }
    try:
        result = generate_remote_text(prompt, model_config, client=client)
        answer, parser_mode = parse_answer(result.text)
        metric, score, correct, expected_text, actual_text = score_case(
            case,
            payload,
            answer,
        )
        return {
            **base,
            "runtime_ok": True,
            "answer_correct": correct,
            "metric": metric,
            "score": score,
            "expected_standard_text": expected_text,
            "final_answer": answer,
            "final_answer_standard_text": actual_text,
            "parser_mode": parser_mode,
            "reasoning": result.text,
            "token_usage": extract_token_usage(result.response_payload),
            "error": None,
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as exc:  # noqa: BLE001 - preserve failures in baseline output.
        return {
            **base,
            "runtime_ok": False,
            "answer_correct": False,
            "metric": None,
            "score": 0.0,
            "expected_standard_text": str(case.get("expected_answer")),
            "final_answer": "",
            "final_answer_standard_text": "",
            "parser_mode": "error",
            "reasoning": "",
            "token_usage": {},
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "elapsed_seconds": time.perf_counter() - started,
        }


print(
    f"Pure CoT baseline: dataset={dataset}, cases={len(selected_cases)}, "
    f"model={model_config.get('model')}",
    file=sys.stderr,
)
started = time.perf_counter()
records = []
with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    futures = {
        executor.submit(evaluate, index, case): index
        for index, case in enumerate(selected_cases)
    }
    for completed, future in enumerate(as_completed(futures), start=1):
        records.append(future.result())
        if completed == 1 or completed % 10 == 0 or completed == len(futures):
            correct = sum(bool(record["answer_correct"]) for record in records)
            print(
                f"\r{completed}/{len(futures)} correct={correct} "
                f"acc={correct / completed:.3f}",
                file=sys.stderr,
                end="" if completed < len(futures) else "\n",
                flush=True,
            )
client.close()

records.sort(key=lambda record: record["sample_index"])
usage = Counter()
for record in records:
    usage.update(record.get("token_usage") or {})
usage_dict = {
    key: int(usage[key])
    for key in (
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
    )
}
correct = sum(bool(record["answer_correct"]) for record in records)
failures = sum(not bool(record["runtime_ok"]) for record in records)
error_types = Counter(
    str(record["error"]["type"])
    for record in records
    if isinstance(record.get("error"), dict)
)
parser_modes = Counter(str(record.get("parser_mode")) for record in records)
summary = {
    "stage": f"{dataset}_pure_cot_baseline",
    "dataset": dataset,
    "method": "vanilla_chain_of_thought",
    "baseline_contract": {
        "prompting": "zero_shot",
        "reasoning": "single_pass_step_by_step",
        "generations_per_case": 1,
        "uses_few_shot_examples": False,
        "uses_self_consistency": False,
        "uses_reflection": False,
        "uses_iterative_revision": False,
        "uses_tools": False,
        "uses_sql": False,
        "uses_python": False,
        "uses_edge_model": False,
        "uses_clover_runtime": False,
    },
    "scope": {
        "available_cases": len(available_cases),
        "selected_cases": len(selected_cases),
        "tablebench_qtypes": ["FactChecking", "NumericalReasoning"],
        "wikitq_split": wikitq_split,
        "tablefact_split": tablefact_split,
        "tablefact_subset": tablefact_subset,
    },
    "model": {
        "provider": model_config.get("provider"),
        "api_type": model_config.get("api_type"),
        "model": model_config.get("model"),
        "base_url": model_config.get("base_url"),
    },
    "total_cases": len(records),
    "correct": correct,
    "accuracy": correct / len(records),
    "runtime_failures": failures,
    "remote_calls": len(records),
    "remote_token_usage": usage_dict,
    "remote_cost_estimate": estimate_openai_text_cost(
        usage_dict,
        remote_config=model_config,
    ),
    "parser_modes": dict(sorted(parser_modes.items())),
    "error_types": dict(sorted(error_types.items())),
    "elapsed_seconds": time.perf_counter() - started,
    "max_workers": args.max_workers,
    "seed": args.seed,
}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


write_jsonl(output_dir / "cases_index.jsonl", records)
write_jsonl(
    output_dir / "answer_mismatch_cases.jsonl",
    [record for record in records if record["runtime_ok"] and not record["answer_correct"]],
)
write_jsonl(
    output_dir / "failure_cases.jsonl",
    [record for record in records if not record["runtime_ok"]],
)
(output_dir / "run_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
