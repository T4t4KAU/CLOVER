"""Pure chain-of-thought baselines for table reasoning benchmarks.

Each case is handled by exactly one model call over a serialized table and
question. The baseline does not use CLOVER planning, agents, SQL, Python,
Pandas, code execution, or external tools.
"""

from __future__ import annotations

import copy
import json
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.costing import estimate_openai_text_cost
from benchmarks.databench.static_tool_eval import (
    display_path,
    format_error,
    json_ready,
    preview,
    safe_divide,
    write_jsonl,
)
from benchmarks.tablebench.adapter import iter_tablebench_dataset_dirs, read_cases, write_json
from benchmarks.tablebench.metrics import score_tablebench_answer
from benchmarks.tablebench.remote_baseline import tablebench_table_from_csv
from benchmarks.warnings import suppress_benchmark_warnings
from benchmarks.wikitq.metrics import score_wikitq_answer
from clover.supervisor import extract_token_usage, generate_remote_text


SUPPORTED_COT_DATASETS = frozenset({"tablebench", "tablefact", "wikitq"})
TOKEN_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)


@dataclass(frozen=True)
class PureCotCase:
    sampled_case: dict[str, Any]
    sample_index: int
    case_dir: Path


@dataclass(frozen=True)
class PureCotScore:
    metric: str
    score: float
    correct: bool
    expected: str
    actual: str


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
    seed: int = 20260528,
    max_workers: int | None = 64,
    overwrite: bool = False,
    remote_cost_model: str | None = None,
    progress_factory: Callable[[int], Any] | None = None,
) -> dict[str, Any]:
    """Run one-call, no-tool CoT over a converted table benchmark."""

    with suppress_benchmark_warnings():
        dataset = normalize_cot_dataset(dataset)
        started = time.perf_counter()
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {output_dir}. "
                    "Use --overwrite to replace it."
                )
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
        worker_count = _resolve_worker_count(
            max_workers=max_workers,
            remote_config=remote_config,
            case_count=len(selected_cases),
        )
        progress_bar = progress_factory(len(selected_cases)) if progress_factory else None
        try:
            records = _run_cases(
                dataset=dataset,
                dataset_root=dataset_root,
                output_dir=output_dir,
                selected_cases=selected_cases,
                remote_config=remote_config,
                worker_count=worker_count,
                progress_bar=progress_bar,
            )
        finally:
            if progress_bar is not None:
                progress_bar.close()

        records.sort(key=lambda item: item["sample_index"])
        cases_index = output_dir / "cases_index.jsonl"
        predictions_path = output_dir / "predictions.jsonl"
        mismatch_cases = output_dir / "answer_mismatch_cases.jsonl"
        failure_cases = output_dir / "failure_cases.jsonl"
        write_jsonl(cases_index, [_case_index_record(record) for record in records])
        write_jsonl(predictions_path, [_prediction_record(record) for record in records])
        write_jsonl(
            mismatch_cases,
            [
                _mismatch_record(record)
                for record in records
                if record.get("runtime_ok") and not record.get("answer_correct")
            ],
        )
        write_jsonl(
            failure_cases,
            [record for record in records if not record.get("runtime_ok")],
        )

        summary = _build_summary(
            dataset=dataset,
            records=records,
            output_dir=output_dir,
            remote_config=remote_config,
            selected_cases=selected_cases,
            elapsed_seconds=time.perf_counter() - started,
            worker_count=worker_count,
            seed=seed,
            sample_size=sample_size,
            split=split,
            subset=subset,
            include_visualization=include_visualization,
            cases_index=cases_index,
            predictions_path=predictions_path,
            mismatch_cases=mismatch_cases,
            failure_cases=failure_cases,
            remote_cost_model=remote_cost_model,
        )
        write_json(output_dir / "run_summary.json", summary)
        return summary


def _run_cases(
    *,
    dataset: str,
    dataset_root: Path,
    output_dir: Path,
    selected_cases: list[dict[str, Any]],
    remote_config: dict[str, Any],
    worker_count: int,
    progress_bar: Any | None,
) -> list[dict[str, Any]]:
    if not selected_cases:
        return []
    records = []
    max_workers = max(1, min(worker_count, len(selected_cases)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for sample_index, sampled_case in enumerate(selected_cases):
            case = PureCotCase(
                sampled_case=sampled_case,
                sample_index=sample_index,
                case_dir=output_dir / "cases" / sampled_case["case_id"],
            )
            future = executor.submit(
                run_pure_cot_case,
                dataset=dataset,
                dataset_root=dataset_root,
                case=case,
                remote_config=copy.deepcopy(remote_config),
            )
            futures[future] = case
        for future in as_completed(futures):
            case = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001 - isolate benchmark cases.
                record = _failed_case_record(
                    dataset=dataset,
                    sampled_case=case.sampled_case,
                    sample_index=case.sample_index,
                    case_dir=case.case_dir,
                    exc=exc,
                )
            records.append(record)
            if progress_bar is not None:
                progress_bar.update(records)
    return records


def run_pure_cot_case(
    *,
    dataset: str,
    dataset_root: Path,
    case: PureCotCase,
    remote_config: dict[str, Any],
) -> dict[str, Any]:
    """Run one table case with a single model call and no tools."""

    case.case_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    dataset_dir = dataset_root / case.sampled_case["dataset_id"]
    case_payload = _load_case_payload(dataset_dir, case.sampled_case["case_id"])
    table = tablebench_table_from_csv(dataset_dir / "table.csv")
    prompt = render_pure_cot_prompt(
        dataset=dataset,
        table=table,
        question=_case_question(dataset, case_payload),
        answer_type=str(case_payload.get("type") or "string"),
        context=_case_context(dataset, case_payload),
    )

    remote_started = time.perf_counter()
    llm_result = generate_remote_text(prompt=prompt, remote_config=remote_config)
    remote_elapsed = time.perf_counter() - remote_started
    remote_output = llm_result.text
    usage = extract_token_usage(llm_result.response_payload)
    parsed_answer = parse_pure_cot_prediction(remote_output)
    parse_ok = bool(parsed_answer)
    score = score_pure_cot_answer(
        dataset=dataset,
        case_payload=case_payload,
        actual=parsed_answer,
    )
    error = None
    if not parse_ok:
        error = {
            "type": "PredictionParseError",
            "message": "Missing final line in the form 'Final Answer: ...'",
        }

    write_json(case.case_dir / "case.json", case_payload)
    write_json(case.case_dir / "table.json", table)
    (case.case_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    write_json(case.case_dir / "remote_response.json", llm_result.response_payload)
    (case.case_dir / "remote_output.txt").write_text(
        remote_output + "\n",
        encoding="utf-8",
    )
    if error is not None:
        write_json(case.case_dir / "error.json", error)

    record = {
        "sample_index": case.sample_index,
        "dataset": dataset,
        "dataset_id": case.sampled_case["dataset_id"],
        "case_id": case.sampled_case["case_id"],
        "case_index": case.sampled_case.get("case_index"),
        "answer_type": case_payload.get("type"),
        "qtype": case_payload.get("qtype"),
        "qsubtype": case_payload.get("qsubtype"),
        "split": case_payload.get("split"),
        "subset": _case_subset(dataset, case_payload),
        "label_text": case_payload.get("label_text"),
        "question": _case_question(dataset, case_payload),
        "expected_raw": json_ready(case_payload.get("answer")),
        "expected_canon": json_ready(case_payload.get("answer_canon")),
        "expected_standard_text": score.expected,
        "response_id": llm_result.response_id,
        "response_status": llm_result.response_status,
        "api_type": llm_result.api_type,
        "model_name": str(remote_config.get("model") or "remote"),
        "method": "pure_cot",
        "uses_tools": False,
        "model_calls": 1,
        "instruction": prompt,
        "table": table,
        "remote_output": remote_output,
        "parse_ok": parse_ok,
        "runtime_ok": parse_ok,
        "answer_correct": bool(parse_ok and score.correct),
        "metric": score.metric,
        "score": score.score if parse_ok else 0.0,
        "final_answer": parsed_answer or None,
        "final_answer_preview": preview(parsed_answer),
        "final_answer_standard_text": score.actual,
        "remote_elapsed_seconds": remote_elapsed,
        "elapsed_seconds": time.perf_counter() - started,
        "remote_token_usage": usage,
        "error": error,
        "case_dir": display_path(case.case_dir),
        "files": {
            "case": display_path(case.case_dir / "case.json"),
            "table": display_path(case.case_dir / "table.json"),
            "prompt": display_path(case.case_dir / "prompt.txt"),
            "remote_output": display_path(case.case_dir / "remote_output.txt"),
            "remote_response": display_path(case.case_dir / "remote_response.json"),
            "error": display_path(case.case_dir / "error.json") if error else None,
        },
    }
    write_json(case.case_dir / "case_result.json", record)
    return record


def render_pure_cot_prompt(
    *,
    dataset: str,
    table: dict[str, Any],
    question: str,
    answer_type: str,
    context: str | None = None,
) -> str:
    """Render the one-call, no-tool CoT prompt shared by all datasets."""

    dataset = normalize_cot_dataset(dataset)
    final_rule = _final_answer_rule(dataset, answer_type=answer_type)
    context_section = ""
    if context:
        context_section = (
            "\nTable context:\n"
            f"{context.strip()}\n"
        )
    table_text = json.dumps(table, ensure_ascii=False, separators=(",", ":"))
    return (
        "You are a table reasoning expert. Answer the question using only the "
        "provided table and table context.\n"
        "Reason step by step in natural language. Do not write or execute code, "
        "SQL, Python, or use tools. Keep the reasoning concise.\n"
        f"{final_rule}\n"
        f"{context_section}\n"
        "Table (JSON):\n"
        f"{table_text}\n\n"
        f"Question: {question.strip()}\n\n"
        f"After the reasoning, {final_rule[0].lower() + final_rule[1:]}\n"
    )


def parse_pure_cot_prediction(prediction: Any) -> str:
    """Return the value after the last explicit Final Answer marker."""

    text = str(prediction or "").replace("**", "").replace("__", "")
    matches = re.findall(
        r"Final\s+Answer\s*:\s*([^\r\n]+)",
        text,
        flags=re.I,
    )
    return matches[-1].strip() if matches else ""


def score_pure_cot_answer(
    *,
    dataset: str,
    case_payload: dict[str, Any],
    actual: Any,
) -> PureCotScore:
    dataset = normalize_cot_dataset(dataset)
    if dataset == "wikitq":
        result = score_wikitq_answer(
            expected=case_payload.get("answer"),
            expected_canon=case_payload.get("answer_canon"),
            actual=actual,
        )
        return PureCotScore(
            metric=result.metric,
            score=result.score,
            correct=result.correct,
            expected=result.expected,
            actual=result.actual,
        )
    if dataset == "tablefact":
        expected_bool = _tablefact_boolean(case_payload.get("answer"))
        actual_bool = _tablefact_boolean(actual)
        correct = expected_bool is not None and expected_bool == actual_bool
        return PureCotScore(
            metric="accuracy",
            score=1.0 if correct else 0.0,
            correct=correct,
            expected=(
                "true"
                if expected_bool is True
                else "false"
                if expected_bool is False
                else str(case_payload.get("answer") or "").strip()
            ),
            actual=(
                "true"
                if actual_bool is True
                else "false"
                if actual_bool is False
                else str(actual or "").strip()
            ),
        )
    result = score_tablebench_answer(
        expected=case_payload.get("answer"),
        actual=actual,
        qtype=case_payload.get("qtype") or "FactChecking",
        qsubtype=case_payload.get("qsubtype"),
    )
    return PureCotScore(
        metric=result.metric,
        score=result.score,
        correct=result.correct,
        expected=result.expected,
        actual=result.actual,
    )


def _tablefact_boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    first_token = text.split(maxsplit=1)[0] if text else ""
    truthy = {"true", "entailed", "entailment", "supported", "support", "yes", "1"}
    falsy = {"false", "refuted", "refutation", "unsupported", "no", "0"}
    if text in truthy or first_token in truthy:
        return True
    if text in falsy or first_token in falsy:
        return False
    return None


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
    dataset = normalize_cot_dataset(dataset)
    if max_cases == 0:
        return []
    cases = list_cot_cases(dataset=dataset, dataset_root=dataset_root)
    if dataset_id is not None:
        cases = [case for case in cases if case["dataset_id"] == dataset_id]
    if case_ids:
        cases = [case for case in cases if case["case_id"] in case_ids]
    if split is not None:
        cases = [case for case in cases if case.get("split") == split]
    if dataset == "tablebench":
        if not include_visualization:
            cases = [case for case in cases if case.get("qtype") != "Visualization"]
        if qtypes:
            cases = [case for case in cases if str(case.get("qtype") or "") in qtypes]
        if qsubtypes:
            cases = [
                case for case in cases if str(case.get("qsubtype") or "") in qsubtypes
            ]
    if dataset == "tablefact" and subset is not None:
        normalized_subset = str(subset).strip().lower()
        if normalized_subset not in {"simple", "complex", "small"}:
            raise ValueError(f"Unsupported TableFact subset: {subset!r}")
        if normalized_subset == "small":
            cases = [case for case in cases if case.get("is_small_test")]
        else:
            cases = [case for case in cases if case.get("subset") == normalized_subset]
    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        cases = random.Random(seed).sample(cases, min(sample_size, len(cases)))
    if max_cases is not None:
        cases = cases[:max_cases]
    return cases


def list_cot_cases(*, dataset: str, dataset_root: Path) -> list[dict[str, Any]]:
    dataset = normalize_cot_dataset(dataset)
    cases = []
    for dataset_dir in iter_tablebench_dataset_dirs(dataset_root):
        cases_path = dataset_dir / "cases.jsonl"
        if not cases_path.is_file():
            continue
        for case_index, case in enumerate(read_cases(cases_path)):
            cases.append(
                {
                    "dataset": dataset,
                    "dataset_id": dataset_dir.name,
                    "case_id": case["case_id"],
                    "case_index": case_index,
                    "answer_type": case.get("type"),
                    "qtype": case.get("qtype"),
                    "qsubtype": case.get("qsubtype"),
                    "split": case.get("split"),
                    "subset": _case_subset(dataset, case),
                    "is_small_test": bool(case.get("is_small_test")),
                }
            )
    return cases


def normalize_cot_dataset(dataset: str) -> str:
    normalized = str(dataset or "").strip().lower()
    if normalized == "tabfact":
        normalized = "tablefact"
    if normalized not in SUPPORTED_COT_DATASETS:
        choices = ", ".join(sorted(SUPPORTED_COT_DATASETS))
        raise ValueError(f"Unsupported pure CoT dataset {dataset!r}; use {choices}")
    return normalized


def _final_answer_rule(dataset: str, *, answer_type: str) -> str:
    if dataset == "tablefact":
        return (
            "End with exactly one final line: Final Answer: true or "
            "Final Answer: false."
        )
    if dataset == "wikitq":
        return (
            "End with exactly one final line: Final Answer: <answer>. "
            "For multiple answers, separate values with comma and a space. "
            "Do not add explanation to the final line."
        )
    type_hint = str(answer_type or "answer").strip()
    return (
        "End with exactly one final line: Final Answer: <answer>. "
        f"The expected answer type is {type_hint}. Keep the final answer as short "
        "as possible and do not add explanation to that line."
    )


def _case_question(dataset: str, case_payload: dict[str, Any]) -> str:
    if dataset == "tablefact":
        statement = str(case_payload.get("statement") or "").strip()
        if statement:
            return f"Is the following statement entailed by the table? {statement}"
    return str(case_payload.get("question") or "").strip()


def _case_context(dataset: str, case_payload: dict[str, Any]) -> str | None:
    if dataset == "tablefact":
        caption = str(case_payload.get("caption") or "").strip()
        return caption or None
    return None


def _case_subset(dataset: str, case_payload: dict[str, Any]) -> str | None:
    if dataset != "tablefact":
        return None
    return str(case_payload.get("qsubtype") or "").strip() or None


def _load_case_payload(dataset_dir: Path, case_id: str) -> dict[str, Any]:
    for case in read_cases(dataset_dir / "cases.jsonl"):
        if case.get("case_id") == case_id:
            return case
    raise ValueError(f"Case id not found in {dataset_dir}: {case_id}")


def _build_summary(
    *,
    dataset: str,
    records: list[dict[str, Any]],
    output_dir: Path,
    remote_config: dict[str, Any],
    selected_cases: list[dict[str, Any]],
    elapsed_seconds: float,
    worker_count: int,
    seed: int,
    sample_size: int | None,
    split: str | None,
    subset: str | None,
    include_visualization: bool,
    cases_index: Path,
    predictions_path: Path,
    mismatch_cases: Path,
    failure_cases: Path,
    remote_cost_model: str | None,
) -> dict[str, Any]:
    total = len(records)
    runtime_successes = sum(1 for record in records if record.get("runtime_ok"))
    correct = sum(1 for record in records if record.get("answer_correct"))
    score_sum = sum(float(record.get("score") or 0.0) for record in records)
    token_usage = _sum_token_usage(records)
    remote_elapsed = sum(
        float(record.get("remote_elapsed_seconds", 0.0) or 0.0)
        for record in records
    )
    summary = {
        "run_name": output_dir.name,
        "stage": f"{dataset}_pure_cot_baseline",
        "workflow": f"pure_cot_{dataset}",
        "method": "pure_chain_of_thought",
        "prompt_mode": "single_model_table_cot",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_contract": {
            "model_calls_per_case": 1,
            "uses_clover_agents": False,
            "uses_tools": False,
            "uses_sql": False,
            "uses_code_execution": False,
            "uses_pandas": False,
            "input": "serialized full table plus question",
            "output_parser": "last explicit Final Answer line",
        },
        "dataset": dataset,
        "split": split,
        "subset": subset,
        "visualization_included": include_visualization if dataset == "tablebench" else None,
        "sample_size": len(selected_cases),
        "requested_sample_size": sample_size,
        "seed": seed,
        "parallel_workers": worker_count,
        "max_retries": 0,
        "validation_mode": "single_model_no_tools",
        "remote_concurrency": worker_count,
        "remote_llm": _config_summary(remote_config),
        "local_slm": None,
        "total_cases": total,
        "remote_calls": total,
        "local_slm_calls": 0,
        "tool_calls": 0,
        "parse_successes": sum(1 for record in records if record.get("parse_ok")),
        "parse_failures": sum(1 for record in records if not record.get("parse_ok")),
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
        "standard_score_average": safe_divide(score_sum, total),
        "scores_by_metric": _score_groups(records, "metric"),
        "scores_by_answer_type": _score_groups(records, "answer_type"),
        "answer_types": _string_counter(records, "answer_type"),
        "error_types": dict(
            sorted(
                Counter(
                    record["error"]["type"]
                    for record in records
                    if isinstance(record.get("error"), dict) and record.get("error")
                ).items()
            )
        ),
        "remote_token_usage": token_usage,
        "local_slm_token_usage": _empty_usage(),
        "remote_cost_estimate": estimate_openai_text_cost(
            token_usage,
            remote_config=remote_config,
            pricing_model=remote_cost_model,
        ),
        "system_profile": _baseline_system_profile(
            remote_calls=total,
            remote_token_usage=token_usage,
        ),
        "remote_elapsed_seconds_sum": remote_elapsed,
        "remote_elapsed_seconds_avg": safe_divide(remote_elapsed, total),
        "elapsed_seconds": elapsed_seconds,
        "run_dir": display_path(output_dir),
        "cases_index": display_path(cases_index),
        "predictions": display_path(predictions_path),
        "answer_mismatch_cases": display_path(mismatch_cases),
        "failure_cases": display_path(failure_cases),
    }
    if dataset == "tablebench":
        summary["scores_by_qtype"] = _score_groups(records, "qtype")
        summary["scores_by_qsubtype"] = _score_groups(records, "qsubtype")
        summary["qtypes"] = _string_counter(records, "qtype")
        summary["qsubtypes"] = _string_counter(records, "qsubtype")
    elif dataset == "wikitq":
        summary["scores_by_split"] = _score_groups(records, "split")
        summary["splits"] = _string_counter(records, "split")
    else:
        summary["scores_by_subset"] = _score_groups(records, "subset")
        summary["subsets"] = _string_counter(records, "subset")
        summary["labels"] = _string_counter(records, "label_text")
    return summary


def _score_groups(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(key) or "unknown")].append(record)
    result = {}
    for group_key, group in sorted(groups.items()):
        total = len(group)
        runtime_successes = sum(1 for record in group if record.get("runtime_ok"))
        correct = sum(1 for record in group if record.get("answer_correct"))
        score_sum = sum(float(record.get("score") or 0.0) for record in group)
        result[group_key] = {
            "total": total,
            "runtime_successes": runtime_successes,
            "correct": correct,
            "accuracy": safe_divide(correct, total),
            "score": safe_divide(score_sum, total),
        }
    return result


def _string_counter(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(
        sorted(Counter(str(record.get(key) or "unknown") for record in records).items())
    )


def _sum_token_usage(records: list[dict[str, Any]]) -> dict[str, int]:
    totals = Counter()
    for record in records:
        usage = record.get("remote_token_usage")
        if not isinstance(usage, dict):
            continue
        for key in TOKEN_KEYS:
            totals[key] += int(usage.get(key, 0) or 0)
    return {key: int(totals.get(key, 0)) for key in TOKEN_KEYS}


def _prediction_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": record.get("case_id"),
        "dataset_id": record.get("dataset_id"),
        "dataset": record.get("dataset"),
        "question": record.get("question"),
        "answer": record.get("expected_raw"),
        "model_name": record.get("model_name"),
        "method": "pure_cot",
        "prediction": record.get("remote_output"),
        "parsed_prediction": record.get("final_answer_standard_text"),
        "parse_ok": record.get("parse_ok"),
        "metric": record.get("metric"),
        "score": record.get("score"),
    }


def _case_index_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"table", "instruction", "remote_output"}
    }


def _mismatch_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_index": record["sample_index"],
        "dataset": record["dataset"],
        "dataset_id": record["dataset_id"],
        "case_id": record["case_id"],
        "metric": record.get("metric"),
        "score": record.get("score"),
        "question": record.get("question"),
        "expected": record.get("expected_standard_text"),
        "actual": record.get("final_answer_standard_text"),
        "remote_output": record.get("files", {}).get("remote_output"),
    }


def _failed_case_record(
    *,
    dataset: str,
    sampled_case: dict[str, Any],
    sample_index: int,
    case_dir: Path,
    exc: Exception,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    error = format_error(exc)
    write_json(case_dir / "case_error.json", error)
    return {
        "sample_index": sample_index,
        "dataset": dataset,
        "dataset_id": sampled_case["dataset_id"],
        "case_id": sampled_case["case_id"],
        "case_index": sampled_case.get("case_index"),
        "answer_type": sampled_case.get("answer_type"),
        "qtype": sampled_case.get("qtype"),
        "qsubtype": sampled_case.get("qsubtype"),
        "split": sampled_case.get("split"),
        "subset": sampled_case.get("subset"),
        "method": "pure_cot",
        "uses_tools": False,
        "model_calls": 0,
        "metric": _default_metric(dataset),
        "score": 0.0,
        "question": None,
        "expected_raw": None,
        "expected_standard_text": None,
        "parse_ok": False,
        "runtime_ok": False,
        "answer_correct": False,
        "final_answer": None,
        "final_answer_preview": None,
        "final_answer_standard_text": None,
        "remote_elapsed_seconds": 0.0,
        "elapsed_seconds": 0.0,
        "remote_token_usage": _empty_usage(),
        "error": error,
        "case_dir": display_path(case_dir),
    }


def _default_metric(dataset: str) -> str:
    if dataset == "wikitq":
        return "denotation_em"
    if dataset == "tablefact":
        return "accuracy"
    return "EM"


def _empty_usage() -> dict[str, int]:
    return {key: 0 for key in TOKEN_KEYS}


def _baseline_system_profile(
    *,
    remote_calls: int,
    remote_token_usage: dict[str, int],
) -> dict[str, Any]:
    return {
        "stages": {},
        "counters": {},
        "summary": {
            "remote_calls": remote_calls,
            "local_slm_calls": 0,
            "tool_calls": 0,
            "merged_plan_count": 0,
            "reused_nodes": 0,
            "parallel_system_instances": 0,
            "remote_token_usage": remote_token_usage,
            "local_slm_token_usage": _empty_usage(),
            "validation_mode": "single_model_no_tools",
        },
    }


def _resolve_worker_count(
    *,
    max_workers: int | None,
    remote_config: dict[str, Any],
    case_count: int,
) -> int:
    if case_count <= 0:
        return 1
    configured_workers = max_workers or remote_config.get("parallel_workers") or 5
    worker_count = int(configured_workers)
    if worker_count <= 0:
        raise ValueError("parallel worker count must be positive")
    return min(worker_count, case_count)


def _config_summary(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": config.get("provider"),
        "api_type": config.get("api_type"),
        "base_url": config.get("base_url"),
        "model": config.get("model"),
    }
