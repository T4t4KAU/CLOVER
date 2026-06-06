"""Remote-only FinanceBench baseline following the public eval modes."""

from __future__ import annotations

import copy
import re
import shutil
import time
from collections import Counter
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
    preview,
    safe_divide,
    write_json,
    write_jsonl,
)
from benchmarks.financebench.adapter import FinanceBenchTask, load_financebench_task, select_cases
from benchmarks.warnings import suppress_benchmark_warnings
from clover.supervisor import extract_token_usage, generate_remote_text
from clover.resource import ResourceCache
from clover.resource.preprocess.pdf_schema import materialize_pdf_text


EVAL_MODE_CLOSED_BOOK = "closedBook"
EVAL_MODE_IN_CONTEXT = "inContext"
EVAL_MODE_IN_CONTEXT_REVERSE = "inContext_reverse"
EVAL_MODE_ORACLE = "oracle"
EVAL_MODE_ORACLE_REVERSE = "oracle_reverse"
SUPPORTED_EVAL_MODES = frozenset(
    {
        EVAL_MODE_CLOSED_BOOK,
        EVAL_MODE_IN_CONTEXT,
        EVAL_MODE_IN_CONTEXT_REVERSE,
        EVAL_MODE_ORACLE,
        EVAL_MODE_ORACLE_REVERSE,
    }
)
DEFAULT_MAX_CONTEXT_CHARS = 500_000
PROMPT_MODE_PREFIX = "financebench_public_eval"
TOKEN_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)


@dataclass(frozen=True)
class FinanceBenchBaselineCase:
    """One selected FinanceBench case plus output location."""

    sampled_case: dict[str, Any]
    sample_index: int
    case_dir: Path


def run_financebench_remote_only_baseline(
    *,
    financebench_root: Path,
    output_dir: Path,
    remote_config: dict[str, Any],
    max_cases: int | None = None,
    case_ids: set[str] | None = None,
    sample_size: int | None = None,
    seed: int = 20260529,
    question_reasoning: str | None = None,
    eval_mode: str = EVAL_MODE_IN_CONTEXT,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_workers: int | None = None,
    overwrite: bool = False,
    remote_cost_model: str | None = None,
    progress_factory: Callable[[int], Any] | None = None,
) -> dict[str, Any]:
    """Run FinanceBench through one Remote LLM answer call per case."""

    with suppress_benchmark_warnings():
        _validate_eval_mode(eval_mode)
        started = time.perf_counter()
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {output_dir}. "
                    "Use --overwrite to replace it."
                )
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        selected_cases = select_cases(
            financebench_root=financebench_root,
            max_cases=max_cases,
            case_ids=case_ids or set(),
            sample_size=sample_size,
            seed=seed,
            question_reasoning=question_reasoning,
        )
        worker_count = _resolve_worker_count(
            max_workers=max_workers,
            remote_config=remote_config,
            case_count=len(selected_cases),
        )
        progress_bar = progress_factory(len(selected_cases)) if progress_factory else None
        try:
            records = _run_cases(
                financebench_root=financebench_root,
                output_dir=output_dir,
                selected_cases=selected_cases,
                remote_config=remote_config,
                worker_count=worker_count,
                eval_mode=eval_mode,
                max_context_chars=max_context_chars,
                progress_bar=progress_bar,
            )
        finally:
            if progress_bar is not None:
                progress_bar.close()
        records.sort(key=lambda item: item["sample_index"])

        cases_index = output_dir / "cases_index.jsonl"
        mismatch_cases = output_dir / "answer_mismatch_cases.jsonl"
        failure_cases = output_dir / "failure_cases.jsonl"
        write_jsonl(cases_index, records)
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
            records=records,
            output_dir=output_dir,
            remote_config=remote_config,
            selected_cases=selected_cases,
            elapsed_seconds=time.perf_counter() - started,
            worker_count=worker_count,
            seed=seed,
            sample_size=sample_size,
            question_reasoning=question_reasoning,
            eval_mode=eval_mode,
            max_context_chars=max_context_chars,
            cases_index=cases_index,
            mismatch_cases=mismatch_cases,
            failure_cases=failure_cases,
            remote_cost_model=remote_cost_model,
        )
        write_json(output_dir / "run_summary.json", summary)
        return summary


def _run_cases(
    *,
    financebench_root: Path,
    output_dir: Path,
    selected_cases: list[dict[str, Any]],
    remote_config: dict[str, Any],
    worker_count: int,
    eval_mode: str,
    max_context_chars: int,
    progress_bar: Any | None,
) -> list[dict[str, Any]]:
    if not selected_cases:
        return []
    records: list[dict[str, Any]] = []
    max_workers = max(1, min(worker_count, len(selected_cases)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_financebench_remote_baseline_case,
                financebench_root=financebench_root,
                case=FinanceBenchBaselineCase(
                    sampled_case=sampled_case,
                    sample_index=sample_index,
                    case_dir=output_dir / "cases" / sampled_case["case_id"],
                ),
                remote_config=copy.deepcopy(remote_config),
                eval_mode=eval_mode,
                max_context_chars=max_context_chars,
            ): sampled_case
            for sample_index, sampled_case in enumerate(selected_cases)
        }
        for future in as_completed(futures):
            sampled_case = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001 - isolate benchmark cases.
                record = _failed_case_record(
                    sampled_case=sampled_case,
                    sample_index=selected_cases.index(sampled_case),
                    case_dir=output_dir / "cases" / sampled_case["case_id"],
                    exc=exc,
                )
            records.append(record)
            if progress_bar is not None:
                progress_bar.update(records)
    return records


def run_financebench_remote_baseline_case(
    *,
    financebench_root: Path,
    case: FinanceBenchBaselineCase,
    remote_config: dict[str, Any],
    eval_mode: str = EVAL_MODE_IN_CONTEXT,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> dict[str, Any]:
    """Run one FinanceBench case with one Remote LLM answer call."""

    case.case_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    task = load_financebench_task(
        financebench_root=financebench_root,
        case_id=case.sampled_case["case_id"],
    )
    expected_raw = task.metadata.get("expected_answer")
    context_payload = build_context_payload(
        task,
        eval_mode=eval_mode,
        max_context_chars=max_context_chars,
    )
    prompt = render_financebench_prompt(
        question=task.task_dsl["question"],
        context=context_payload["context"],
        eval_mode=eval_mode,
    )

    remote_started = time.perf_counter()
    llm_result = generate_remote_text(prompt=prompt, remote_config=remote_config)
    remote_elapsed = time.perf_counter() - remote_started
    usage = extract_token_usage(llm_result.response_payload)
    final_answer = llm_result.text.strip()
    eval_result = financebench_answer_correct(expected_raw, final_answer)

    write_json(case.case_dir / "task_dsl.json", task.task_dsl)
    write_json(case.case_dir / "metadata.json", task.metadata)
    write_json(case.case_dir / "context_stats.json", context_payload["stats"])
    (case.case_dir / "prompt.md").write_text(prompt + "\n", encoding="utf-8")
    (case.case_dir / "remote_output.txt").write_text(final_answer + "\n", encoding="utf-8")
    write_json(case.case_dir / "remote_response.json", llm_result.response_payload)

    record = {
        "sample_index": case.sample_index,
        "dataset_id": "financebench",
        "case_id": case.sampled_case["case_id"],
        "case_index": case.sampled_case.get("case_index"),
        "answer_type": "string",
        "question_reasoning": case.sampled_case.get("question_reasoning"),
        "question_type": case.sampled_case.get("question_type"),
        "doc_name": case.sampled_case.get("doc_name"),
        "question": task.task_dsl["question"],
        "expected_raw": expected_raw,
        "expected_normalized": eval_result["expected_normalized"],
        "response_id": llm_result.response_id,
        "response_status": llm_result.response_status,
        "api_type": llm_result.api_type,
        "parse_ok": bool(final_answer),
        "runtime_ok": bool(final_answer),
        "answer_correct": bool(final_answer and eval_result["correct"]),
        "final_answer": final_answer,
        "final_answer_preview": preview(final_answer),
        "final_answer_normalized": eval_result["actual_normalized"],
        "financebench_eval": eval_result,
        "remote_elapsed_seconds": remote_elapsed,
        "elapsed_seconds": time.perf_counter() - started,
        "remote_token_usage": usage,
        "error": None
        if final_answer
        else {"type": "EmptyAnswer", "message": "Remote output was empty"},
        "case_dir": display_path(case.case_dir),
        "files": {
            "task_dsl": display_path(case.case_dir / "task_dsl.json"),
            "metadata": display_path(case.case_dir / "metadata.json"),
            "context_stats": display_path(case.case_dir / "context_stats.json"),
            "prompt": display_path(case.case_dir / "prompt.md"),
            "remote_output": display_path(case.case_dir / "remote_output.txt"),
            "remote_response": display_path(case.case_dir / "remote_response.json"),
        },
    }
    write_json(case.case_dir / "case_result.json", record)
    return record


def build_context_payload(
    task: FinanceBenchTask,
    *,
    eval_mode: str,
    max_context_chars: int,
) -> dict[str, Any]:
    """Return the context string used by the public FinanceBench eval modes."""

    _validate_eval_mode(eval_mode)
    if eval_mode == EVAL_MODE_CLOSED_BOOK:
        context = ""
        original_length = 0
        truncated = False
        source = "none"
    elif eval_mode in {EVAL_MODE_ORACLE, EVAL_MODE_ORACLE_REVERSE}:
        context = _oracle_context(task.metadata["case"])
        original_length = len(context)
        truncated = False
        source = "evidence_text_full_page"
    else:
        pdf_path = Path(task.metadata["pdf_path"])
        cache_entry = materialize_pdf_text(pdf_path, cache=ResourceCache())
        text_path = cache_entry.artifact_path("text")
        document_text = text_path.read_text(encoding="utf-8")
        original_length = len(document_text)
        context = document_text[:max_context_chars]
        truncated = len(context) < original_length
        source = "pdf_text"
    return {
        "context": context,
        "stats": {
            "eval_mode": eval_mode,
            "source": source,
            "context_char_count": len(context),
            "original_context_char_count": original_length,
            "truncated": truncated,
            "max_context_chars": max_context_chars,
        },
    }


def render_financebench_prompt(
    *,
    question: str,
    context: str,
    eval_mode: str,
) -> str:
    """Render prompts matching the public FinanceBench evaluation notebook."""

    if eval_mode == EVAL_MODE_CLOSED_BOOK:
        return f"Answer this question: {question}"
    if eval_mode == EVAL_MODE_ORACLE:
        return (
            f"Answer this question: {question} \n"
            "Here is the relevant evidence that you need to answer the question:\n"
            f"[START OF FILING] {context} [END OF FILING]"
        )
    if eval_mode == EVAL_MODE_ORACLE_REVERSE:
        return (
            f"Context:\n[START OF FILING] {context} [END OF FILING]\n\n "
            f"Answer this question: {question} \n"
        )
    if eval_mode == EVAL_MODE_IN_CONTEXT:
        return (
            f"Answer this question: {question} \n"
            "Here is the relevant filing that you need to answer the question:\n"
            f"[START OF FILING] {context} [END OF FILING]"
        )
    if eval_mode == EVAL_MODE_IN_CONTEXT_REVERSE:
        return (
            f"Context:\n[START OF FILING] {context} [END OF FILING]\n\n "
            f"Answer this question: {question}\n"
        )
    raise ValueError(f"Unsupported FinanceBench eval_mode: {eval_mode}")


def financebench_answer_correct(expected: Any, actual: Any) -> dict[str, Any]:
    """Numerical-focused automatic scoring for FinanceBench smoke tests."""

    expected_text = "" if expected is None else str(expected)
    actual_text = "" if actual is None else str(actual)
    expected_mentions = _important_numeric_mentions(expected_text)
    actual_mentions = _numeric_mentions(actual_text)
    missing_numbers = [
        mention["raw"]
        for mention in expected_mentions
        if not _mention_matched(mention, actual_mentions)
    ]
    expected_yes_no = _yes_no_value(expected_text)
    actual_yes_no = _yes_no_value(actual_text)
    yes_no_ok = expected_yes_no is None or expected_yes_no == actual_yes_no
    expected_activity = _cash_flow_activity(expected_text)
    actual_activity = _cash_flow_activity(actual_text)
    activity_ok = expected_activity is None or expected_activity == actual_activity

    if expected_mentions:
        correct = not missing_numbers and yes_no_ok and activity_ok
    else:
        correct = (
            _normalize_text(expected_text) in _normalize_text(actual_text)
            or _normalize_text(actual_text) in _normalize_text(expected_text)
        )
        correct = correct and yes_no_ok and activity_ok

    return {
        "correct": bool(correct),
        "expected_normalized": expected_text.strip(),
        "actual_normalized": actual_text.strip(),
        "expected_numbers": [mention["raw"] for mention in expected_mentions],
        "actual_numbers": [mention["raw"] for mention in actual_mentions],
        "missing_numbers": missing_numbers,
        "expected_yes_no": expected_yes_no,
        "actual_yes_no": actual_yes_no,
        "expected_cash_flow_activity": expected_activity,
        "actual_cash_flow_activity": actual_activity,
    }


def _oracle_context(row: dict[str, Any]) -> str:
    evidence_items = row.get("evidence") or []
    pages = []
    for evidence in evidence_items:
        if not isinstance(evidence, dict):
            continue
        text = evidence.get("evidence_text_full_page") or evidence.get("text")
        if isinstance(text, str) and text.strip():
            pages.append(text.strip())
    return "\n\n".join(pages)


def _important_numeric_mentions(text: str) -> list[dict[str, Any]]:
    return [mention for mention in _numeric_mentions(text) if not _looks_like_year(mention)]


def _numeric_mentions(text: str) -> list[dict[str, Any]]:
    mentions = []
    pattern = re.compile(r"(?<![A-Za-z])-?\d+(?:,\d{3})*(?:\.\d+)?\s*%?")
    for match in pattern.finditer(text):
        token = match.group(0).strip()
        raw_text = token.rstrip("%").strip()
        try:
            raw = float(raw_text.replace(",", ""))
        except ValueError:
            continue
        suffix = text[match.end() : match.end() + 18].lower()
        percent = token.endswith("%")
        candidates = {raw}
        if percent:
            candidates.add(raw / 100.0)
        if re.match(r"\s*(bn|billion)\b", suffix):
            candidates.add(raw * 1000.0)
        if re.match(r"\s*(mm|m|million)\b", suffix):
            candidates.add(raw)
        mentions.append(
            {
                "raw": raw,
                "text": token,
                "percent": percent,
                "candidates": sorted(candidates),
            }
        )
    return mentions


def _looks_like_year(mention: dict[str, Any]) -> bool:
    value = mention["raw"]
    return float(value).is_integer() and 1900 <= int(value) <= 2100


def _mention_matched(
    expected: dict[str, Any],
    actual_mentions: list[dict[str, Any]],
) -> bool:
    for left in expected["candidates"]:
        for actual in actual_mentions:
            for right in actual["candidates"]:
                if _numbers_close(left, right):
                    return True
        if _derived_difference_matched(left, actual_mentions):
            return True
    return False


def _numbers_close(left: float, right: float) -> bool:
    magnitude = max(abs(left), abs(right))
    tolerance = 0.05 if magnitude < 100 else max(1.0, magnitude * 0.02)
    return abs(left - right) <= tolerance


def _derived_difference_matched(
    expected_value: float,
    actual_mentions: list[dict[str, Any]],
) -> bool:
    candidates = [
        candidate
        for mention in actual_mentions
        if not _looks_like_year(mention)
        for candidate in mention["candidates"]
    ]
    if len(candidates) < 2:
        return False
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            if _numbers_close(abs(left - right), expected_value):
                return True
    return False


def _yes_no_value(text: str) -> bool | None:
    normalized = text.strip().lower()[:160]
    if re.match(r"^(yes|yeah|true)\b", normalized):
        return True
    if re.match(r"^(no|false)\b", normalized):
        return False
    match = re.search(r"\b(answer\s*:\s*)?(yes|no)\b", normalized)
    if match:
        return match.group(2) == "yes"
    return None


def _cash_flow_activity(text: str) -> str | None:
    normalized = text.lower()
    for activity in ("operating activities", "investing activities", "financing activities"):
        if activity in normalized:
            return activity
    return None


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _failed_case_record(
    *,
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
        "dataset_id": "financebench",
        "case_id": sampled_case["case_id"],
        "case_index": sampled_case.get("case_index"),
        "answer_type": "string",
        "question_reasoning": sampled_case.get("question_reasoning"),
        "question_type": sampled_case.get("question_type"),
        "doc_name": sampled_case.get("doc_name"),
        "question": None,
        "expected_raw": None,
        "expected_normalized": None,
        "parse_ok": False,
        "runtime_ok": False,
        "answer_correct": False,
        "final_answer": None,
        "final_answer_preview": None,
        "final_answer_normalized": None,
        "remote_elapsed_seconds": 0.0,
        "elapsed_seconds": 0.0,
        "remote_token_usage": {key: 0 for key in TOKEN_KEYS},
        "error": error,
        "case_dir": display_path(case_dir),
    }


def _build_summary(
    *,
    records: list[dict[str, Any]],
    output_dir: Path,
    remote_config: dict[str, Any],
    selected_cases: list[dict[str, Any]],
    elapsed_seconds: float,
    worker_count: int,
    seed: int,
    sample_size: int | None,
    question_reasoning: str | None,
    eval_mode: str,
    max_context_chars: int,
    cases_index: Path,
    mismatch_cases: Path,
    failure_cases: Path,
    remote_cost_model: str | None,
) -> dict[str, Any]:
    total = len(records)
    runtime_successes = sum(1 for record in records if record.get("runtime_ok"))
    correct = sum(1 for record in records if record.get("answer_correct"))
    mismatches = sum(
        1
        for record in records
        if record.get("runtime_ok") and not record.get("answer_correct")
    )
    error_types = Counter(
        record["error"]["type"]
        for record in records
        if isinstance(record.get("error"), dict) and record.get("error")
    )
    answer_types = Counter(record.get("answer_type") for record in records)
    mismatches_by_type = Counter(
        record.get("answer_type")
        for record in records
        if record.get("runtime_ok") and not record.get("answer_correct")
    )
    token_usage = _sum_token_usage(records)
    local_slm_token_usage = _empty_usage()
    remote_cost_estimate = estimate_openai_text_cost(
        token_usage,
        remote_config=remote_config,
        pricing_model=remote_cost_model,
    )
    remote_elapsed = sum(
        float(record.get("remote_elapsed_seconds", 0.0) or 0.0)
        for record in records
    )
    return {
        "run_name": output_dir.name,
        "stage": "financebench_remote_only_baseline",
        "workflow": "remote_only_financebench",
        "prompt_mode": f"{PROMPT_MODE_PREFIX}_{eval_mode}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(selected_cases),
        "requested_sample_size": sample_size,
        "seed": seed,
        "question_reasoning": question_reasoning,
        "eval_mode": eval_mode,
        "max_context_chars": max_context_chars,
        "parallel_workers": worker_count,
        "max_retries": 0,
        "validation_mode": "remote_only",
        "remote_batch_size": None,
        "remote_concurrency": worker_count,
        "max_parallel_execution_units": None,
        "eval_batch_size": None,
        "profile_baseline": None,
        "remote_llm": _config_summary(remote_config),
        "local_slm": None,
        "total_cases": total,
        "remote_calls": total,
        "local_slm_calls": 0,
        "parse_successes": sum(1 for record in records if record.get("parse_ok")),
        "parse_failures": sum(1 for record in records if not record.get("parse_ok")),
        "runtime_successes": runtime_successes,
        "runtime_failures": total - runtime_successes,
        "correct": correct,
        "mismatches": mismatches,
        "failures": total - runtime_successes,
        "accuracy_on_successes": safe_divide(correct, runtime_successes),
        "accuracy_on_all_cases": safe_divide(correct, total),
        "retry_cases": 0,
        "retry_case_ids": [],
        "total_retry_rounds": 0,
        "sql_repair_cases": 0,
        "sql_repair_case_ids": [],
        "initial_execution_failures": 0,
        "initial_execution_failure_case_ids": [],
        "answer_types": _counter_as_strings(answer_types),
        "mismatches_by_type": _counter_as_strings(mismatches_by_type),
        "error_types": dict(sorted(error_types.items())),
        "remote_token_usage": token_usage,
        "local_slm_token_usage": local_slm_token_usage,
        "remote_cost_estimate": remote_cost_estimate,
        "system_profile": _baseline_system_profile(
            remote_calls=total,
            remote_token_usage=token_usage,
            local_slm_token_usage=local_slm_token_usage,
        ),
        "remote_elapsed_seconds_sum": remote_elapsed,
        "remote_elapsed_seconds_avg": safe_divide(remote_elapsed, total),
        "elapsed_seconds": elapsed_seconds,
        "run_dir": display_path(output_dir),
        "cases_index": display_path(cases_index),
        "answer_mismatch_cases": display_path(mismatch_cases),
        "failure_cases": display_path(failure_cases),
    }


def _sum_token_usage(records: list[dict[str, Any]]) -> dict[str, int]:
    totals = Counter()
    for record in records:
        usage = record.get("remote_token_usage")
        if not isinstance(usage, dict):
            continue
        for key in TOKEN_KEYS:
            totals[key] += int(usage.get(key, 0) or 0)
    return {key: int(totals.get(key, 0)) for key in TOKEN_KEYS}


def _empty_usage() -> dict[str, int]:
    return {key: 0 for key in TOKEN_KEYS}


def _counter_as_strings(counter: Counter[Any]) -> dict[str, int]:
    return dict(sorted((str(key or "unknown"), int(value)) for key, value in counter.items()))


def _baseline_system_profile(
    *,
    remote_calls: int,
    remote_token_usage: dict[str, int],
    local_slm_token_usage: dict[str, int],
) -> dict[str, Any]:
    return {
        "stages": {},
        "counters": {},
        "summary": {
            "remote_calls": remote_calls,
            "local_slm_calls": 0,
            "merged_plan_count": 0,
            "reused_nodes": 0,
            "parallel_system_instances": 0,
            "remote_token_usage": remote_token_usage,
            "local_slm_token_usage": local_slm_token_usage,
            "validation_mode": "remote_only",
        },
    }


def _config_summary(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        "provider": config.get("provider"),
        "api_type": config.get("api_type"),
        "base_url": config.get("base_url"),
        "model": config.get("model"),
    }


def _mismatch_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_index": record["sample_index"],
        "dataset_id": record["dataset_id"],
        "case_id": record["case_id"],
        "question_reasoning": record.get("question_reasoning"),
        "question": record.get("question"),
        "expected": record.get("expected_normalized"),
        "actual": record.get("final_answer_normalized"),
        "financebench_eval": record.get("financebench_eval"),
        "remote_output": record.get("files", {}).get("remote_output"),
    }


def _resolve_worker_count(
    *,
    max_workers: int | None,
    remote_config: dict[str, Any],
    case_count: int,
) -> int:
    if case_count <= 0:
        return 1
    worker_count = int(max_workers or remote_config.get("parallel_workers") or 3)
    if worker_count <= 0:
        raise ValueError("parallel worker count must be positive")
    return min(worker_count, case_count)


def _validate_eval_mode(eval_mode: str) -> None:
    if eval_mode not in SUPPORTED_EVAL_MODES:
        available = ", ".join(sorted(SUPPORTED_EVAL_MODES))
        raise ValueError(f"Unsupported FinanceBench eval_mode: {eval_mode}. Available: {available}")
