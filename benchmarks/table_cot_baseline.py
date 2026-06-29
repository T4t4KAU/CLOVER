"""Pure Chain-of-Thought baseline for table benchmarks.

This baseline is intentionally simple: one model call per case, no SQL/Python,
no tools, no CLOVER runtime, and no self-consistency/retry.  It renders the
available table(s) as CSV, asks the model to reason step by step, parses the
last explicit ``Final Answer:``, and scores it with the dataset-native metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.costing import estimate_openai_text_cost
from benchmarks.mmqa.metrics import score_mmqa_answer
from benchmarks.tablebench.metrics import score_tablebench_answer
from benchmarks.utils import (
    compact_run_summary,
    display_path,
    format_brief_summary,
    json_ready,
    safe_divide,
    write_jsonl,
)
from benchmarks.wikitq.metrics import score_wikitq_answer
from clover.config import load_model_config
from clover.supervisor.client import (
    create_remote_llm_client,
    extract_token_usage,
    generate_remote_text,
)


SUPPORTED_COT_DATASETS = {"tablebench", "wikitq", "tablefact", "mmqa"}
DEFAULT_SEED = 20260528

FINAL_ANSWER_RE = re.compile(
    r"(?is)(?:\*\*)?\bfinal\s+answer\b(?:\*\*)?\s*:\s*(.*?)(?=\n|$)"
)


@dataclass(frozen=True)
class CaseScore:
    metric: str
    score: float
    correct: bool
    expected: str
    actual: str


def normalize_cot_dataset(dataset: str) -> str:
    normalized = str(dataset or "").strip().lower()
    aliases = {
        "wikitablequestions": "wikitq",
        "wiki_table_questions": "wikitq",
        "tabfact": "tablefact",
        "table_fact": "tablefact",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_COT_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset!r}. "
            f"Expected one of {sorted(SUPPORTED_COT_DATASETS)}."
        )
    return normalized


def render_pure_cot_prompt(
    *,
    dataset: str,
    table: dict[str, Any] | list[dict[str, Any]],
    question: str,
    answer_type: str | None = None,
    context: str | None = None,
    table_names: list[str] | None = None,
    primary_keys: list[Any] | None = None,
    foreign_keys: list[Any] | None = None,
) -> str:
    """Render a zero-shot CoT prompt for one table or multiple tables."""

    dataset = normalize_cot_dataset(dataset)
    tables = table if isinstance(table, list) else [table]
    names = table_names or []
    rendered_tables = []
    for index, item in enumerate(tables, start=1):
        name = names[index - 1] if index - 1 < len(names) and names[index - 1] else f"table_{index}"
        rendered_tables.append(
            f"Table {index}: {name}\n```csv\n{_table_dict_to_csv(item)}\n```"
        )

    context_parts = []
    if context:
        context_parts.append(f"Context: {str(context).strip()}")
    if dataset == "mmqa":
        if primary_keys:
            context_parts.append(f"Primary keys: {json.dumps(primary_keys, ensure_ascii=False)}")
        if foreign_keys:
            context_parts.append(f"Foreign keys: {json.dumps(foreign_keys, ensure_ascii=False)}")
    context_block = "\n".join(context_parts)
    if context_block:
        context_block = f"\n{context_block}\n"

    type_hint = str(answer_type or "string").strip() or "string"
    bool_hint = (
        "For a boolean answer, use exactly true or false. "
        "For multiple values, separate items with semicolons if values may contain commas; otherwise commas are fine."
    )
    final_example = "Final Answer: true" if "bool" in type_hint.lower() else "Final Answer: <answer>"
    table_label = "tables" if len(tables) > 1 else "table"

    return f"""You are evaluating a pure single-model Chain-of-Thought baseline.

Answer the question using only the provided {table_label}.
Reason step by step, but keep the reasoning concise.
Do not write or execute code, SQL, Python, or use tools.
Then put the final answer on the last line in exactly this format:
{final_example}

{bool_hint}
{context_block}
Dataset: {dataset}
Answer type: {type_hint}
Question: {question}

{chr(10).join(rendered_tables)}
"""


def parse_pure_cot_prediction(text: str) -> str:
    """Return the last explicit Final Answer value, or an empty string."""

    matches = [match.strip() for match in FINAL_ANSWER_RE.findall(str(text or ""))]
    matches = [match.strip("*").strip() for match in matches if match.strip()]
    if not matches:
        return ""
    return matches[-1]


def score_pure_cot_answer(
    *,
    dataset: str,
    case_payload: dict[str, Any],
    actual: Any,
) -> CaseScore:
    """Score a parsed CoT answer with the dataset-native metric."""

    dataset = normalize_cot_dataset(dataset)
    if dataset == "wikitq":
        score = score_wikitq_answer(
            expected=case_payload.get("answer"),
            expected_canon=case_payload.get("answer_canon"),
            actual=actual,
        )
        return CaseScore(score.metric, score.score, score.correct, score.expected, score.actual)
    if dataset == "mmqa":
        raw_score = score_mmqa_answer(
            expected=case_payload.get("answer_raw"),
            actual=actual,
            expected_answer_type=case_payload.get("type"),
        )
        flat_score = score_mmqa_answer(
            expected=case_payload.get("answer"),
            actual=actual,
            expected_answer_type=case_payload.get("type"),
        )
        score = flat_score if flat_score.correct and not raw_score.correct else raw_score
        return CaseScore(score.metric, score.score, score.correct, score.expected, score.actual)

    qtype = case_payload.get("qtype")
    qsubtype = case_payload.get("qsubtype")
    if dataset == "tablefact":
        qtype = "FactChecking"
        if not qsubtype:
            qsubtype = case_payload.get("subset")
        actual = _normalize_tablefact_actual(actual)
    score = score_tablebench_answer(
        expected=case_payload.get("answer"),
        actual=actual,
        qtype=qtype,
        qsubtype=qsubtype,
    )
    metric = "accuracy" if dataset == "tablefact" else score.metric
    return CaseScore(metric, score.score, score.correct, score.expected, score.actual)


def select_cot_cases(
    *,
    dataset: str,
    dataset_root: Path,
    max_cases: int | None,
    case_ids: set[str],
    dataset_id: str | None,
    split: str | None,
    subset: str | None,
    qtypes: set[str],
    qsubtypes: set[str],
    include_visualization: bool,
    sample_size: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Select benchmark cases using the same helpers as the main evaluators."""

    dataset = normalize_cot_dataset(dataset)
    dataset_root = Path(dataset_root)
    common = {
        "max_cases": max_cases,
        "case_ids": case_ids,
        "dataset_id": dataset_id,
        "sample_size": sample_size,
        "seed": seed,
    }
    if dataset == "tablebench":
        from benchmarks.tablebench.eval import select_tablebench_cases

        return select_tablebench_cases(
            tablebench_root=dataset_root,
            qtypes=qtypes or {"FactChecking", "NumericalReasoning"},
            qsubtypes=qsubtypes,
            include_visualization=include_visualization,
            **common,
        )
    if dataset == "wikitq":
        from benchmarks.wikitq.eval import select_wikitq_cases

        return select_wikitq_cases(
            wikitq_root=dataset_root,
            split=split or "pristine-unseen-tables",
            **common,
        )
    if dataset == "tablefact":
        from benchmarks.tablefact.eval import select_tablefact_cases

        return select_tablefact_cases(
            tablefact_root=dataset_root,
            split=split or "test",
            subset=subset,
            **common,
        )

    from benchmarks.mmqa.eval import select_mmqa_cases

    return select_mmqa_cases(
        mmqa_root=dataset_root,
        max_cases=max_cases,
        case_ids=case_ids,
        dataset_id=dataset_id,
        split=split,
        sample_size=sample_size,
        seed=seed,
    )


def run_table_cot_baseline(
    *,
    dataset: str,
    dataset_root: Path,
    output_dir: Path,
    remote_config: dict[str, Any],
    max_cases: int | None = None,
    case_ids: set[str] | None = None,
    dataset_id: str | None = None,
    split: str | None = None,
    subset: str | None = None,
    qtypes: set[str] | None = None,
    qsubtypes: set[str] | None = None,
    include_visualization: bool = False,
    sample_size: int | None = None,
    seed: int = DEFAULT_SEED,
    max_workers: int = 8,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the pure CoT baseline and write case records plus run_summary.json."""

    dataset = normalize_cot_dataset(dataset)
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Converted dataset not found: {dataset_root}")
    if max_cases is not None and max_cases <= 0:
        raise ValueError("max_cases must be positive")
    if sample_size is not None and sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_cases = select_cot_cases(
        dataset=dataset,
        dataset_root=dataset_root,
        max_cases=max_cases,
        case_ids=case_ids or set(),
        dataset_id=dataset_id,
        split=split,
        subset=subset,
        qtypes=qtypes or set(),
        qsubtypes=qsubtypes or set(),
        include_visualization=include_visualization,
        sample_size=sample_size,
        seed=seed,
    )
    if not selected_cases:
        raise ValueError("No cases selected")

    remote_config = _remote_config_with_default_api_key(remote_config)
    client = create_remote_llm_client(remote_config)
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_case,
                    index,
                    case,
                    dataset=dataset,
                    dataset_root=dataset_root,
                    remote_config=remote_config,
                    client=client,
                ): index
                for index, case in enumerate(selected_cases)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                records.append(future.result())
                if completed == 1 or completed % 10 == 0 or completed == len(futures):
                    correct = sum(bool(record.get("answer_correct")) for record in records)
                    print(
                        f"\r{completed}/{len(futures)} correct={correct} "
                        f"acc={correct / completed:.3f}",
                        end="" if completed < len(futures) else "\n",
                        flush=True,
                    )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    records.sort(key=lambda record: int(record.get("sample_index", 0)))
    write_jsonl(output_dir / "cases_index.jsonl", records)
    write_jsonl(
        output_dir / "answer_mismatch_cases.jsonl",
        [
            record
            for record in records
            if record.get("runtime_ok") and not record.get("answer_correct")
        ],
    )
    write_jsonl(
        output_dir / "failure_cases.jsonl",
        [record for record in records if not record.get("runtime_ok")],
    )

    summary = _build_cot_summary(
        dataset=dataset,
        records=records,
        output_dir=output_dir,
        remote_config=remote_config,
        elapsed_seconds=time.perf_counter() - started,
        seed=seed,
        sample_size=sample_size,
        split=split,
        subset=subset,
        max_workers=max_workers,
    )
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _evaluate_case(
    index: int,
    case: dict[str, Any],
    *,
    dataset: str,
    dataset_root: Path,
    remote_config: dict[str, Any],
    client: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    base = {
        "sample_index": index,
        "dataset": dataset,
        "dataset_id": case.get("dataset_id"),
        "case_id": case.get("case_id"),
        "question": case.get("question"),
        "answer_type": case.get("answer_type"),
        "qtype": case.get("qtype"),
        "qsubtype": case.get("qsubtype"),
        "subset": case.get("subset"),
        "split": case.get("split"),
        "table_count": case.get("table_count"),
        "expected": case.get("expected_answer"),
        "expected_raw": case.get("expected_raw"),
    }
    try:
        payload, tables = _load_case_payload_and_tables(
            dataset=dataset,
            dataset_root=dataset_root,
            case=case,
        )
        question = str(case.get("question") or payload.get("question") or payload.get("statement") or "")
        answer_type = str(case.get("answer_type") or payload.get("type") or "string")
        prompt = render_pure_cot_prompt(
            dataset=dataset,
            table=tables,
            question=question,
            answer_type=answer_type,
            context=_case_context(payload),
            table_names=list(payload.get("table_names") or case.get("table_names") or []),
            primary_keys=list(payload.get("primary_keys") or case.get("primary_keys") or []),
            foreign_keys=list(payload.get("foreign_keys") or case.get("foreign_keys") or []),
        )
        result = generate_remote_text(prompt, remote_config, client=client)
        final_answer = parse_pure_cot_prediction(result.text)
        score = score_pure_cot_answer(
            dataset=dataset,
            case_payload=payload,
            actual=final_answer,
        )
        return {
            **base,
            "runtime_ok": True,
            "answer_correct": bool(score.correct),
            "metric": score.metric,
            "score": score.score,
            "expected_standard_text": score.expected,
            "final_answer": final_answer,
            "final_answer_standard_text": score.actual,
            "parser_mode": "explicit_final_answer" if final_answer else "missing_final_answer",
            "reasoning": result.text,
            "token_usage": extract_token_usage(result.response_payload),
            "error": None,
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as exc:  # noqa: BLE001 - preserve baseline failure records.
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


def _build_cot_summary(
    *,
    dataset: str,
    records: list[dict[str, Any]],
    output_dir: Path,
    remote_config: dict[str, Any],
    elapsed_seconds: float,
    seed: int,
    sample_size: int | None,
    split: str | None,
    subset: str | None,
    max_workers: int,
) -> dict[str, Any]:
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
    total = len(records)
    correct = sum(bool(record.get("answer_correct")) for record in records)
    runtime_successes = sum(bool(record.get("runtime_ok")) for record in records)
    parser_modes = Counter(str(record.get("parser_mode")) for record in records)
    error_types = Counter(
        str(record["error"]["type"])
        for record in records
        if isinstance(record.get("error"), dict)
    )
    summary = {
        "run_name": output_dir.name,
        "stage": f"{dataset}_pure_cot_baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workflow": "pure_chain_of_thought",
        "sample_size": total,
        "requested_sample_size": sample_size,
        "seed": seed,
        "split": split,
        "subset": subset,
        "parallel_workers": max_workers,
        "max_retries": 0,
        "validation_mode": "none",
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
        "remote_llm": {
            "provider": remote_config.get("provider"),
            "api_type": remote_config.get("api_type"),
            "model": remote_config.get("model"),
            "base_url": remote_config.get("base_url"),
        },
        "total_cases": total,
        "runtime_successes": runtime_successes,
        "runtime_failures": total - runtime_successes,
        "correct": correct,
        "mismatches": sum(
            1
            for record in records
            if record.get("runtime_ok") and not record.get("answer_correct")
        ),
        "failures": total - runtime_successes,
        "accuracy_on_all_cases": safe_divide(correct, total),
        "accuracy_on_successes": safe_divide(correct, runtime_successes),
        "remote_calls": total,
        "local_slm_calls": 0,
        "tool_calls": 0,
        "remote_token_usage": usage_dict,
        "remote_cost_estimate": estimate_openai_text_cost(
            usage_dict,
            remote_config=remote_config,
        ),
        "parser_modes": dict(sorted(parser_modes.items())),
        "error_types": dict(sorted(error_types.items())),
        "elapsed_seconds": elapsed_seconds,
        "run_dir": display_path(output_dir),
        "cases_index": display_path(output_dir / "cases_index.jsonl"),
        "answer_mismatch_cases": display_path(output_dir / "answer_mismatch_cases.jsonl"),
        "failure_cases": display_path(output_dir / "failure_cases.jsonl"),
    }
    compact = compact_run_summary(summary)
    for key in (
        "baseline_contract",
        "tool_calls",
        "parser_modes",
        "error_types",
        "split",
        "subset",
    ):
        if key in summary:
            compact[key] = summary[key]
    return compact


def _load_case_payload_and_tables(
    *,
    dataset: str,
    dataset_root: Path,
    case: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_dir = _resolve_case_dataset_dir(dataset=dataset, dataset_root=dataset_root, case=case)
    payload = _load_case_payload(dataset_dir, case)
    if dataset == "mmqa":
        source_files = list(payload.get("source_files") or case.get("source_files") or [])
        if not source_files:
            table_count = int(payload.get("table_count") or case.get("table_count") or 0)
            source_files = [f"table_{index}.csv" for index in range(1, table_count + 1)]
        tables = [_csv_file_to_table(dataset_dir / source_file) for source_file in source_files]
    else:
        tables = [_csv_file_to_table(dataset_dir / "table.csv")]
    return payload, tables


def _resolve_case_dataset_dir(
    *,
    dataset: str,
    dataset_root: Path,
    case: dict[str, Any],
) -> Path:
    dataset_id = str(case["dataset_id"])
    direct = dataset_root / dataset_id
    if direct.is_dir():
        return direct
    if dataset == "mmqa":
        split = case.get("split")
        if split:
            candidate = dataset_root / str(split) / dataset_id
            if candidate.is_dir():
                return candidate
        for split_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
            candidate = split_dir / dataset_id
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError(f"Dataset directory not found for case: {dataset_id}")


def _load_case_payload(dataset_dir: Path, case: dict[str, Any]) -> dict[str, Any]:
    cases_path = dataset_dir / "cases.jsonl"
    case_id = str(case["case_id"])
    with cases_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if str(payload.get("case_id")) == case_id:
                return payload
    raise ValueError(f"Case payload not found: {case_id}")


def _csv_file_to_table(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return {"columns": [], "data": []}
    return {"columns": rows[0], "data": rows[1:]}


def _table_dict_to_csv(table: dict[str, Any]) -> str:
    rows = [list(table.get("columns") or [])]
    rows.extend(list(row) for row in table.get("data") or [])
    rendered = []
    for row in rows:
        rendered.append(",".join(_csv_cell(value) for value in row))
    return "\n".join(rendered)


def _csv_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    if any(char in text for char in ',\"\n\r'):
        return '"' + text.replace('"', '""') + '"'
    return text


def _case_context(payload: dict[str, Any]) -> str:
    parts = []
    for key in ("caption", "context", "statement"):
        value = payload.get(key)
        if value:
            parts.append(str(value).strip())
    return "\n".join(part for part in parts if part)


def _normalize_tablefact_actual(actual: Any) -> Any:
    text = str(actual).strip().lower()
    if text in {"entailed", "entailment", "supported", "yes"}:
        return "true"
    if text in {"refuted", "refute", "contradicted", "contradiction", "no"}:
        return "false"
    return actual


def _remote_config_with_default_api_key(remote_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(remote_config)
    if not config.get("api_key") and not config.get("api_key_env"):
        config["api_key"] = os.environ.get("OPENAI_API_KEY", "EMPTY")
    return config


def _expand_csv_values(values: list[str] | None) -> list[str]:
    expanded: list[str] = []
    for value in values or []:
        expanded.extend(part.strip() for part in str(value).split(",") if part.strip())
    return expanded


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a pure CoT table baseline.")
    parser.add_argument("dataset", choices=sorted(SUPPORTED_COT_DATASETS))
    parser.add_argument("model_config", type=Path)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--qtype", action="append", default=[])
    parser.add_argument("--qsubtype", action="append", default=[])
    parser.add_argument("--include-visualization", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--output-root", type=Path, default=Path("benchmark/runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    dataset = normalize_cot_dataset(args.dataset)
    dataset_root = args.dataset_root.expanduser().resolve()
    model_config = load_model_config(args.model_config.expanduser().resolve())

    selected_cases = select_cot_cases(
        dataset=dataset,
        dataset_root=dataset_root,
        max_cases=args.max_cases,
        case_ids=set(_expand_csv_values(args.case_id)),
        dataset_id=args.dataset_id,
        split=args.split,
        subset=args.subset,
        qtypes=set(_expand_csv_values(args.qtype)),
        qsubtypes=set(_expand_csv_values(args.qsubtype)),
        include_visualization=args.include_visualization,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    if args.validate_only:
        available_cases = select_cot_cases(
            dataset=dataset,
            dataset_root=dataset_root,
            max_cases=None,
            case_ids=set(),
            dataset_id=args.dataset_id,
            split=args.split,
            subset=args.subset,
            qtypes=set(_expand_csv_values(args.qtype)),
            qsubtypes=set(_expand_csv_values(args.qsubtype)),
            include_visualization=args.include_visualization,
            sample_size=None,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "method": "vanilla_chain_of_thought",
                    "available_cases": len(available_cases),
                    "selected_cases": len(selected_cases),
                    "split": args.split,
                    "subset": args.subset,
                    "scope_valid": bool(selected_cases),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    run_name = args.run_name or f"{dataset}_pure_cot_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = (args.output_root / run_name).expanduser().resolve()
    summary = run_table_cot_baseline(
        dataset=dataset,
        dataset_root=dataset_root,
        output_dir=output_dir,
        remote_config=model_config,
        max_cases=args.max_cases,
        case_ids=set(_expand_csv_values(args.case_id)),
        dataset_id=args.dataset_id,
        split=args.split,
        subset=args.subset,
        qtypes=set(_expand_csv_values(args.qtype)),
        qsubtypes=set(_expand_csv_values(args.qsubtype)),
        include_visualization=args.include_visualization,
        sample_size=args.sample_size,
        seed=args.seed,
        max_workers=args.max_workers,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n===== Brief Summary =====")
    print(format_brief_summary(summary.get("brief_summary", {})))
    print("=========================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
