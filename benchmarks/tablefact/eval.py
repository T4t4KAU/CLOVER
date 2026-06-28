"""TableFact evaluation using the shared TableBench table runtime."""

from __future__ import annotations

import csv
import io
import json
import random
import re
import shutil
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from benchmarks.costing import estimate_openai_text_cost, normalize_remote_token_usage
from benchmarks.utils import (
    build_brief_summary,
    compact_run_summary,
    format_error,
    json_ready,
    preview,
    safe_divide,
    write_jsonl,
)
from benchmarks.tablebench.adapter import iter_tablebench_dataset_dirs, read_cases, write_json
from benchmarks.tablebench.eval import (
    TOKEN_KEYS,
    _ctx_stats,
    _runtime_case_id,
    base_case_record,
    build_summary,
    mismatch_record,
    run_tablebench_eval,
)
from benchmarks.tablebench.metrics import score_tablebench_answer
from clover.supervisor.client import extract_token_usage, generate_remote_text


TABLEFACT_SUBSETS = frozenset({"simple", "complex", "small"})
DEFAULT_DIRECT_MAX_TOKENS = 1024
DEFAULT_DIRECT_TABLE_CHAR_LIMIT = 24000
DEFAULT_SECOND_PASS_MAX_TOKENS = 1024


def run_tablefact_eval(
    *,
    tablefact_root: Path,
    output_dir: Path,
    remote_config: dict[str, Any],
    synthesize_config: dict[str, Any] | None = None,
    local_slm_config: dict[str, Any] | None = None,
    max_cases: int | None = None,
    case_ids: set[str] | None = None,
    dataset_id: str | None = None,
    split: str | None = "test",
    subset: str | None = None,
    sample_size: int | None = None,
    seed: int = 20260528,
    max_workers: int | None = 64,
    max_retries: int = 1,
    validation_mode: str = "remote_supervisor",
    remote_batch_size: int = 64,
    remote_concurrency: int = 64,
    max_parallel_execution_units: int = 64,
    max_parallel_slm_node_jobs: int = 64,
    max_parallel_slm_sequences: int = 64,
    max_pending_slm_sequences: int = 1024,
    eval_batch_size: int | None = None,
    profile_baseline: bool = False,
    remote_cost_model: str | None = None,
    overwrite: bool = False,
    progress_factory: Callable[[int], Any] | None = None,
) -> dict[str, Any]:
    """Run converted TableFact cases and report official classification accuracy."""

    selected_cases = select_tablefact_cases(
        tablefact_root=tablefact_root,
        max_cases=max_cases,
        case_ids=case_ids or set(),
        dataset_id=dataset_id,
        split=split,
        subset=subset,
        sample_size=sample_size,
        seed=seed,
    )
    selected_ids = {case["case_id"] for case in selected_cases}
    if not selected_ids:
        selected_ids = {"__no_tablefact_cases_selected__"}
    if _tablefact_direct_enabled(local_slm_config):
        summary = _run_tablefact_direct_eval(
            tablefact_root=tablefact_root,
            output_dir=output_dir,
            selected_cases=selected_cases,
            remote_config=remote_config,
            synthesize_config=synthesize_config,
            local_slm_config=local_slm_config,
            max_workers=max_workers,
            max_retries=max_retries,
            validation_mode=validation_mode,
            remote_batch_size=remote_batch_size,
            remote_concurrency=remote_concurrency,
            max_parallel_execution_units=max_parallel_execution_units,
            max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
            max_parallel_slm_sequences=max_parallel_slm_sequences,
            max_pending_slm_sequences=max_pending_slm_sequences,
            eval_batch_size=eval_batch_size,
            profile_baseline=profile_baseline,
            remote_cost_model=remote_cost_model,
            overwrite=overwrite,
            progress_factory=progress_factory,
            seed=seed,
            sample_size=sample_size,
        )
        return _normalize_tablefact_outputs(
            summary=summary,
            output_dir=output_dir,
            split=split,
            subset=subset,
            requested_sample_size=sample_size,
        )
    summary = run_tablebench_eval(
        tablebench_root=tablefact_root,
        output_dir=output_dir,
        remote_config=remote_config,
        synthesize_config=synthesize_config,
        local_slm_config=local_slm_config,
        max_cases=None,
        case_ids=selected_ids,
        dataset_id=None,
        qtypes={"FactChecking"},
        qsubtypes=set(),
        include_visualization=False,
        sample_size=None,
        seed=seed,
        max_workers=max_workers,
        max_retries=max_retries,
        validation_mode=validation_mode,
        remote_batch_size=remote_batch_size,
        remote_concurrency=remote_concurrency,
        max_parallel_execution_units=max_parallel_execution_units,
        max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
        max_parallel_slm_sequences=max_parallel_slm_sequences,
        max_pending_slm_sequences=max_pending_slm_sequences,
        eval_batch_size=eval_batch_size,
        profile_baseline=profile_baseline,
        remote_cost_model=remote_cost_model,
        overwrite=overwrite,
        progress_factory=progress_factory,
    )
    return _normalize_tablefact_outputs(
        summary=summary,
        output_dir=output_dir,
        split=split,
        subset=subset,
        requested_sample_size=sample_size,
    )


def select_tablefact_cases(
    *,
    tablefact_root: Path,
    max_cases: int | None,
    case_ids: set[str],
    dataset_id: str | None,
    split: str | None,
    subset: str | None,
    sample_size: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if max_cases == 0:
        return []
    normalized_subset = str(subset or "").strip().lower() or None
    if normalized_subset not in TABLEFACT_SUBSETS | {None}:
        raise ValueError(f"Unsupported TableFact subset: {subset!r}")

    cases = list_tablefact_cases(tablefact_root)
    if dataset_id is not None:
        cases = [case for case in cases if case["dataset_id"] == dataset_id]
    if split is not None:
        cases = [case for case in cases if case.get("split") == split]
    if normalized_subset == "small":
        cases = [case for case in cases if case.get("is_small_test")]
    elif normalized_subset is not None:
        cases = [case for case in cases if case.get("subset") == normalized_subset]
    if case_ids:
        cases = [case for case in cases if case["case_id"] in case_ids]
    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        cases = random.Random(seed).sample(cases, min(sample_size, len(cases)))
    if max_cases is not None:
        cases = cases[:max_cases]
    return cases


def list_tablefact_cases(tablefact_root: Path) -> list[dict[str, Any]]:
    cases = []
    for dataset_dir in iter_tablebench_dataset_dirs(tablefact_root):
        cases_path = dataset_dir / "cases.jsonl"
        if not cases_path.is_file():
            continue
        for case_index, case in enumerate(read_cases(cases_path)):
            cases.append(
                {
                    "dataset_id": dataset_dir.name,
                    "case_id": case["case_id"],
                    "case_index": case_index,
                    "question": case.get("question"),
                    "statement": case.get("statement"),
                    "expected_answer": case.get("answer"),
                    "answer_type": case.get("type"),
                    "split": case.get("split"),
                    "subset": case.get("qsubtype"),
                    "is_small_test": bool(case.get("is_small_test")),
                    "label_text": case.get("label_text"),
                    "caption": case.get("caption"),
                    "source_table": case.get("source_table"),
                    "hints": case.get("hints"),
                }
            )
    return cases


def _tablefact_direct_enabled(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return False
    for key in (
        "enable_tablefact_direct_verifier",
        "tablefact_direct_verifier",
        "tablefact_direct_verify",
    ):
        if key in config:
            return bool(config.get(key))
    return False


def _tablefact_second_pass_enabled(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return True
    for key in (
        "enable_tablefact_second_pass_verifier",
        "tablefact_second_pass_verifier",
    ):
        if key in config:
            return bool(config.get(key))
    return True


def _run_tablefact_direct_eval(
    *,
    tablefact_root: Path,
    output_dir: Path,
    selected_cases: list[dict[str, Any]],
    remote_config: dict[str, Any],
    synthesize_config: dict[str, Any] | None,
    local_slm_config: dict[str, Any] | None,
    max_workers: int | None,
    max_retries: int,
    validation_mode: str,
    remote_batch_size: int,
    remote_concurrency: int,
    max_parallel_execution_units: int,
    max_parallel_slm_node_jobs: int,
    max_parallel_slm_sequences: int,
    max_pending_slm_sequences: int,
    eval_batch_size: int | None,
    profile_baseline: bool,
    remote_cost_model: str | None,
    overwrite: bool,
    progress_factory: Callable[[int], Any] | None,
    seed: int,
    sample_size: int | None,
) -> dict[str, Any]:
    """Run TableFact through a compact direct verifier.

    TabFact is binary fact verification, and the transformed SQL pathway often
    loses same-row evidence. The direct verifier keeps the whole statement/table
    relation in one local vLLM call, while still emitting the same benchmark
    accounting fields as the shared TableBench evaluator.
    """

    started = time.perf_counter()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    direct_config = _tablefact_direct_config(
        synthesize_config or remote_config,
        local_slm_config=local_slm_config,
    )
    worker_count = max(1, int(max_workers or 1))
    request_workers = max(1, min(worker_count, int(remote_concurrency or worker_count)))
    progress_bar = progress_factory(len(selected_cases)) if progress_factory else None
    completed_records: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=request_workers) as pool:
            futures = {
                pool.submit(
                    _run_tablefact_direct_case,
                    tablefact_root=tablefact_root,
                    output_dir=output_dir,
                    sampled_case=sampled_case,
                    sample_index=sample_index,
                    remote_config=direct_config,
                    local_slm_config=local_slm_config,
                ): sample_index
                for sample_index, sampled_case in enumerate(selected_cases)
            }
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                completed_records.append(record)
                if progress_bar is not None:
                    progress_bar.update(completed_records)
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
            mismatch_record(record)
            for record in records
            if record.get("runtime_ok") and not record.get("answer_correct")
        ],
    )
    write_jsonl(failure_cases, [record for record in records if not record.get("runtime_ok")])

    system_profile = _direct_system_profile(
        records=records,
        validation_mode=validation_mode,
    )
    summary = build_summary(
        records=records,
        output_dir=output_dir,
        remote_config=direct_config,
        synthesize_config=synthesize_config,
        local_slm_config=local_slm_config,
        selected_cases=selected_cases,
        elapsed_seconds=time.perf_counter() - started,
        worker_count=worker_count,
        max_retries=max_retries,
        validation_mode=validation_mode,
        include_visualization=False,
        remote_batch_size=remote_batch_size,
        remote_concurrency=remote_concurrency,
        max_parallel_execution_units=max_parallel_execution_units,
        max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
        max_parallel_slm_sequences=max_parallel_slm_sequences,
        max_pending_slm_sequences=max_pending_slm_sequences,
        eval_batch_size=eval_batch_size,
        profile_baseline=profile_baseline,
        system_profile=system_profile,
        remote_cost_model=remote_cost_model,
        seed=seed,
        sample_size=sample_size,
        cases_index=cases_index,
        mismatch_cases=mismatch_cases,
        failure_cases=failure_cases,
    )
    _attach_direct_context_stats(summary, records)
    summary["tablefact_direct_verifier"] = {
        "enabled": True,
        "max_tokens": direct_config.get("max_tokens"),
        "table_char_limit": _direct_table_char_limit(local_slm_config),
        "second_pass_enabled": _tablefact_second_pass_enabled(local_slm_config),
        "second_pass_max_tokens": _tablefact_second_pass_max_tokens(local_slm_config),
        "request_workers": request_workers,
        "avg_case_seconds": safe_divide(
            sum(float(record.get("elapsed_seconds") or 0.0) for record in records),
            len(records),
        ),
    }
    summary["remote_cost_estimate"] = estimate_openai_text_cost(
        normalize_remote_token_usage(summary.get("remote_token_usage", {})),
        remote_config=direct_config,
        pricing_model=remote_cost_model,
    )
    summary["brief_summary"] = build_brief_summary(summary)
    summary = compact_run_summary(summary)
    write_json(output_dir / "run_summary.json", summary)
    return summary


def _run_tablefact_direct_case(
    *,
    tablefact_root: Path,
    output_dir: Path,
    sampled_case: dict[str, Any],
    sample_index: int,
    remote_config: dict[str, Any],
    local_slm_config: dict[str, Any] | None,
) -> dict[str, Any]:
    runtime_case_id = _runtime_case_id(sampled_case, sample_index)
    case_dir = output_dir / "cases" / runtime_case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    record = base_case_record(
        sampled_case={
            **sampled_case,
            "qtype": "FactChecking",
            "qsubtype": sampled_case.get("subset"),
        },
        sample_index=sample_index,
        runtime_case_id=runtime_case_id,
        case_dir=case_dir,
    )
    record.update(
        {
            "task_answer_type": "boolean",
            "dsl_builder_mode": "tablefact_direct_verifier",
            "qtype": "FactChecking",
            "qsubtype": sampled_case.get("subset"),
            "question": sampled_case.get("statement") or sampled_case.get("question"),
        }
    )

    case_started = time.perf_counter()
    trace: dict[str, Any] = {
        "mode": "tablefact_direct_verifier",
        "prompt": None,
        "raw_output": None,
        "parsed_answer": None,
        "second_pass": None,
        "true_guard": None,
        "token_usage": {key: 0 for key in TOKEN_KEYS},
        "elapsed_seconds": None,
    }
    try:
        table_csv = _read_table_csv_for_case(
            tablefact_root=tablefact_root,
            sampled_case=sampled_case,
            limit=_direct_table_char_limit(local_slm_config),
        )
        statement = str(sampled_case.get("statement") or sampled_case.get("question") or "")
        evidence_hints = _tablefact_evidence_hints(table_csv=table_csv, statement=statement)
        prompt = _tablefact_direct_prompt(
            table_csv=table_csv,
            statement=statement,
            caption=_case_caption(sampled_case),
        )
        trace["prompt"] = prompt
        result = generate_remote_text(prompt=prompt, remote_config=remote_config)
        usage = extract_token_usage(result.response_payload)
        answer = _parse_direct_boolean(result.text)
        combined_usage = _sum_token_usage(usage)
        max_context_tokens = int(usage.get("input_tokens", 0) or 0) if isinstance(usage, dict) else 0
        calls = 1
        trace.update(
            {
                "raw_output": result.text,
                "parsed_answer": answer,
                "token_usage": combined_usage,
                "response_id": result.response_id,
                "response_status": result.response_status,
                "api_type": result.api_type,
            }
        )
        second_pass_trace: dict[str, Any] | None = None
        second_pass_record: dict[str, Any] | None = None
        true_guard_trace: dict[str, Any] | None = None
        true_guard_record: dict[str, Any] | None = None
        if answer is False and _tablefact_second_pass_enabled(local_slm_config):
            second_pass_config = _tablefact_second_pass_config(
                remote_config,
                local_slm_config=local_slm_config,
            )
            second_prompt = _tablefact_second_pass_prompt(
                table_csv=table_csv,
                statement=statement,
                caption=_case_caption(sampled_case),
                first_output=result.text,
                evidence_hints=evidence_hints,
            )
            second_result = generate_remote_text(
                prompt=second_prompt,
                remote_config=second_pass_config,
            )
            second_usage = extract_token_usage(second_result.response_payload)
            combined_usage = _sum_token_usage(combined_usage, second_usage)
            calls += 1
            second_context = (
                int(second_usage.get("input_tokens", 0) or 0)
                if isinstance(second_usage, dict)
                else 0
            )
            max_context_tokens = max(max_context_tokens, second_context)
            second_answer, second_confidence = _parse_second_pass_result(second_result.text)
            second_pass_trace = {
                "prompt": second_prompt,
                "raw_output": second_result.text,
                "parsed_answer": second_answer,
                "confidence": second_confidence,
                "accepted": bool(second_answer is True and second_confidence == "high"),
                "token_usage": _sum_token_usage(second_usage),
                "response_id": second_result.response_id,
                "response_status": second_result.response_status,
                "api_type": second_result.api_type,
            }
            second_pass_record = {
                "parsed_answer": second_answer,
                "confidence": second_confidence,
                "accepted": second_pass_trace["accepted"],
                "token_usage": _sum_token_usage(second_usage),
                "raw_output_preview": preview(second_result.text, max_length=1000),
            }
            trace["second_pass"] = second_pass_trace
            trace["token_usage"] = combined_usage
            if second_pass_trace["accepted"]:
                answer = True
                trace["parsed_answer"] = answer
        elif (
            answer is True
            and _tablefact_true_guard_enabled(local_slm_config)
            and _tablefact_true_guard_needed(statement)
        ):
            true_guard_config = _tablefact_second_pass_config(
                remote_config,
                local_slm_config=local_slm_config,
            )
            true_guard_prompt = _tablefact_true_guard_prompt(
                table_csv=table_csv,
                statement=statement,
                caption=_case_caption(sampled_case),
                first_output=result.text,
            )
            true_guard_result = generate_remote_text(
                prompt=true_guard_prompt,
                remote_config=true_guard_config,
            )
            true_guard_usage = extract_token_usage(true_guard_result.response_payload)
            combined_usage = _sum_token_usage(combined_usage, true_guard_usage)
            calls += 1
            true_guard_context = (
                int(true_guard_usage.get("input_tokens", 0) or 0)
                if isinstance(true_guard_usage, dict)
                else 0
            )
            max_context_tokens = max(max_context_tokens, true_guard_context)
            true_guard_answer, true_guard_confidence = _parse_second_pass_result(
                true_guard_result.text
            )
            true_guard_trace = {
                "prompt": true_guard_prompt,
                "raw_output": true_guard_result.text,
                "parsed_answer": true_guard_answer,
                "confidence": true_guard_confidence,
                "accepted": bool(
                    true_guard_answer is False and true_guard_confidence == "high"
                ),
                "token_usage": _sum_token_usage(true_guard_usage),
                "response_id": true_guard_result.response_id,
                "response_status": true_guard_result.response_status,
                "api_type": true_guard_result.api_type,
            }
            true_guard_record = {
                "parsed_answer": true_guard_answer,
                "confidence": true_guard_confidence,
                "accepted": true_guard_trace["accepted"],
                "token_usage": _sum_token_usage(true_guard_usage),
                "raw_output_preview": preview(true_guard_result.text, max_length=1000),
            }
            trace["true_guard"] = true_guard_trace
            trace["token_usage"] = combined_usage
            if true_guard_trace["accepted"]:
                answer = False
                trace["parsed_answer"] = answer
        score = score_tablebench_answer(
            expected=record.get("expected_raw"),
            actual=answer,
            qtype="FactChecking",
            qsubtype=sampled_case.get("subset"),
        )
        parse_ok = answer is not None
        record.update(
            {
                "parse_ok": parse_ok,
                "runtime_ok": parse_ok,
                "final_answer": json_ready(answer),
                "final_answer_preview": preview(answer),
                "final_answer_standard_text": score.actual,
                "answer_correct": bool(parse_ok and score.correct),
                "tablebench_metric": score.metric,
                "tablebench_score": score.score if parse_ok else 0.0,
                "round_count": 1,
                "retry_count": 0,
                "retry_exhausted": False,
                "error": None
                if parse_ok
                else {
                    "type": "DirectVerifierParseError",
                    "message": "Could not parse a boolean answer from the verifier output.",
                },
                "direct_verifier": {
                    "calls": calls,
                    "token_usage": combined_usage,
                    "max_context_tokens": max_context_tokens,
                    "first_pass": {
                        "parsed_answer": _parse_direct_boolean(result.text),
                        "raw_output_preview": preview(result.text, max_length=1000),
                        "token_usage": _sum_token_usage(usage),
                    },
                    "second_pass": second_pass_record,
                    "true_guard": true_guard_record,
                    "elapsed_seconds": None,
                    "raw_output_preview": preview(result.text, max_length=1000),
                },
            }
        )
    except Exception as exc:  # noqa: BLE001
        record.update(
            {
                "parse_ok": False,
                "runtime_ok": False,
                "tablebench_score": 0.0,
                "error": format_error(exc),
                "round_count": 1,
                "retry_count": 0,
                "direct_verifier": trace,
            }
        )
    elapsed = time.perf_counter() - case_started
    record["elapsed_seconds"] = elapsed
    trace["elapsed_seconds"] = elapsed
    if isinstance(record.get("direct_verifier"), dict):
        record["direct_verifier"]["elapsed_seconds"] = elapsed
    write_json(case_dir / "direct_verifier.json", json_ready(trace))
    write_json(case_dir / "case_result.json", record)
    return record


def _tablefact_direct_config(
    config: dict[str, Any],
    *,
    local_slm_config: dict[str, Any] | None,
) -> dict[str, Any]:
    selected = dict(config)
    selected["temperature"] = _direct_float(
        local_slm_config,
        "tablefact_direct_temperature",
        default=0.0,
    )
    selected["top_p"] = _direct_float(
        local_slm_config,
        "tablefact_direct_top_p",
        default=1.0,
    )
    max_tokens = _direct_int(
        local_slm_config,
        "tablefact_direct_max_tokens",
        default=DEFAULT_DIRECT_MAX_TOKENS,
    )
    if selected.get("api_type", "chat_completions") == "responses":
        selected["max_output_tokens"] = max_tokens
    else:
        selected["max_tokens"] = max_tokens
    selected["connection_retry_attempts"] = max(
        1,
        _direct_int(
            local_slm_config,
            "tablefact_direct_connection_retry_attempts",
            default=1,
        ),
    )
    return selected


def _tablefact_second_pass_config(
    config: dict[str, Any],
    *,
    local_slm_config: dict[str, Any] | None,
) -> dict[str, Any]:
    selected = dict(config)
    selected["temperature"] = 0.0
    selected["top_p"] = 1.0
    max_tokens = _tablefact_second_pass_max_tokens(local_slm_config)
    if selected.get("api_type", "chat_completions") == "responses":
        selected["max_output_tokens"] = max_tokens
    else:
        selected["max_tokens"] = max_tokens
    return selected


def _direct_int(
    config: dict[str, Any] | None,
    key: str,
    *,
    default: int,
) -> int:
    if not isinstance(config, dict):
        return default
    try:
        return int(config.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _direct_float(
    config: dict[str, Any] | None,
    key: str,
    *,
    default: float,
) -> float:
    if not isinstance(config, dict):
        return default
    try:
        return float(config.get(key, default))
    except (TypeError, ValueError):
        return default


def _direct_table_char_limit(config: dict[str, Any] | None) -> int:
    return max(
        1000,
        _direct_int(
            config,
            "tablefact_direct_table_char_limit",
            default=DEFAULT_DIRECT_TABLE_CHAR_LIMIT,
        ),
    )


def _tablefact_second_pass_max_tokens(config: dict[str, Any] | None) -> int:
    return max(
        128,
        _direct_int(
            config,
            "tablefact_second_pass_max_tokens",
            default=DEFAULT_SECOND_PASS_MAX_TOKENS,
        ),
    )


def _tablefact_true_guard_enabled(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return False
    return bool(config.get("enable_tablefact_true_guard", False))


def _read_table_csv_for_case(
    *,
    tablefact_root: Path,
    sampled_case: dict[str, Any],
    limit: int,
) -> str:
    table_path = tablefact_root / sampled_case["dataset_id"] / "table.csv"
    text = table_path.read_text(encoding="utf-8", errors="replace")
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[table truncated]"


def _case_caption(sampled_case: dict[str, Any]) -> str:
    if sampled_case.get("caption"):
        return str(sampled_case["caption"])
    hints = sampled_case.get("hints")
    if isinstance(hints, dict) and hints.get("source_context"):
        return str(hints["source_context"])
    return ""


_TABLEFACT_EVIDENCE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "for",
        "from",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "there",
        "this",
        "to",
        "with",
    }
)


def _tablefact_evidence_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\w]+(?:\.[\w]+)?", text.lower(), flags=re.UNICODE)
        if len(token) > 1 and token not in _TABLEFACT_EVIDENCE_STOPWORDS
    }


def _tablefact_evidence_hints(
    *,
    table_csv: str,
    statement: str,
    max_rows: int = 12,
    max_chars: int = 3600,
) -> str:
    """Return compact row hints for the direct verifier.

    The full table remains authoritative. These hints simply put likely evidence
    rows in the model's attention window so retry/second-pass verification has
    concrete row-level anchors instead of re-discovering them from a long CSV.
    """

    statement_norm = " ".join(re.findall(r"[\w]+(?:\.[\w]+)?", statement.lower(), flags=re.UNICODE))
    statement_tokens = _tablefact_evidence_tokens(statement)
    if not statement_tokens:
        return ""

    try:
        reader = csv.DictReader(io.StringIO(table_csv))
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    except csv.Error:
        return ""
    if not headers or not rows:
        return ""

    numeric_tokens = {token for token in statement_tokens if any(ch.isdigit() for ch in token)}
    header_token_map = {header: _tablefact_evidence_tokens(header) for header in headers}
    scored: list[tuple[float, int, dict[str, str]]] = []
    for row_index, row in enumerate(rows, start=1):
        score = 0.0
        for header in headers:
            value = str(row.get(header) or "").strip()
            if not value:
                continue
            header_tokens = header_token_map.get(header, set())
            value_tokens = _tablefact_evidence_tokens(value)
            header_overlap = len(header_tokens & statement_tokens)
            value_overlap = len(value_tokens & statement_tokens)
            value_norm = " ".join(
                re.findall(r"[\w]+(?:\.[\w]+)?", value.lower(), flags=re.UNICODE)
            ).strip()
            exact_value_hit = len(value_norm) > 1 and value_norm in statement_norm
            numeric_overlap = len(value_tokens & numeric_tokens)

            if header_overlap:
                score += min(header_overlap, 3) * 1.5
            if exact_value_hit:
                score += 6.0 + min(len(value_tokens), 4)
            if value_overlap:
                score += min(value_overlap, 4) * 2.0
            if numeric_overlap:
                score += min(numeric_overlap, 3) * 1.0
            if header_overlap and (exact_value_hit or value_overlap):
                score += 2.0
        if score > 0:
            scored.append((score, row_index, row))

    if not scored:
        return ""

    scored.sort(key=lambda item: (-item[0], item[1]))
    lines = [
        "Potential evidence rows (heuristic, not exhaustive; verify against the full CSV, especially for counts/ranks/only):"
    ]
    for score, row_index, row in scored[:max_rows]:
        parts: list[str] = []
        for header in headers:
            value = str(row.get(header) or "").strip()
            if not value:
                continue
            cell = f"{header}={value}"
            if len(cell) > 120:
                cell = cell[:117].rstrip() + "..."
            parts.append(cell)
        row_text = " | ".join(parts)
        if len(row_text) > 620:
            row_text = row_text[:617].rstrip() + "..."
        lines.append(f"- row {row_index} (score {score:.1f}): {row_text}")
        if sum(len(line) + 1 for line in lines) >= max_chars:
            break
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n...[evidence hints truncated]"
    return text


def _tablefact_true_guard_needed(statement: str) -> bool:
    """Return true for statements where false positives are common.

    A clean row lookup usually does not need another call. Counts, ranks,
    comparisons, negation, and multi-condition sports/schedule claims are where
    TabFact-style generated text most often fools the permissive first pass.
    """

    text = statement.lower()
    return bool(
        re.search(
            r"\b("
            r"all but|at least|at most|average|count|different|fewest|first|"
            r"higher|highest|last|lead|least|less|lower|lowest|more|most|"
            r"no|not|only|over|rank|second|total|under"
            r")\b",
            text,
        )
        or re.search(r"\b\d+(?:\.\d+)?\b", text)
        or re.search(
            r"\b(game|games|episode|episodes|event|events|instance|instances|"
            r"occasion|occasions|player|players|time|times|venue|venues)\b",
            text,
        )
    )


def _tablefact_direct_prompt(
    *,
    table_csv: str,
    statement: str,
    caption: str,
    evidence_hints: str = "",
) -> str:
    evidence_section = f"\n{evidence_hints}\n" if evidence_hints else ""
    return (
        "You are a TabFact table fact checker.\n"
        "Decide if the statement is ENTAILED by the table under TabFact-style "
        "semantics: evidence must come from the table, but generated statements "
        "may use loose wording.\n"
        "Rules:\n"
        "- Verify every entity/value/comparison/count/rank/date/arithmetic relation; all conjunctive parts must hold.\n"
        "- Normalize capitalization, punctuation, accents, hyphens, spaces, ordinal/cardinal forms, dates, and units.\n"
        "- Allow unambiguous aliases, initials, prefixes, truncated names, and descriptor words from column headers "
        "(for example a school/club/team column value 'Fordham' can support 'Fordham club/team').\n"
        "- Superlatives with ties are satisfied by any tied maximum/minimum unless the statement says only/unique/single.\n"
        "- If all compared values are equal, that value is still both the highest and the lowest.\n"
        "- Use table-local, label-style semantics: the table is the universe of discourse. "
        "Do not reject a statement because the table is not a complete external history or because a row value is not a global total.\n"
        "- For awkward generated grammar, prefer row-level support: if the mentioned entity/row has the mentioned property/value in the table, it is entailed unless another required part is directly contradicted.\n"
        "- In sports leader columns such as high points/high rebounds/high assists/top scorer, each listed name is already a leader or tied leader for that row; count the row as supported when the target name appears in that leader cell.\n"
        "- In election tables, 'first elected in YEAR' can support wording such as winning or being the official in YEAR; do not object that the person was not an incumbent before that year.\n"
        "- Phrases like 'county/team/party/district has VALUE' can be supported by a concrete row under that county/team/party/district; require totals/averages only when explicitly requested.\n"
        "- Interpret 'run opposed' or 'run oppose' as having an opponent; 'unopposed' means no opponent.\n"
        "- Numeric counts, 'more than', 'all but N', date ordering, and explicit arithmetic remain strict.\n"
        "- Do not average, sum, or aggregate across multiple rows unless the statement explicitly asks for an average/total/count.\n"
        "- For category comparisons, look for a concrete row or pair that supports the stated relation unless an aggregate is explicit.\n"
        "- False only when a required part is contradicted or no unambiguous supporting evidence exists.\n"
        "Caption is context only: "
        f"{caption}\n"
        f"{evidence_section}"
        'Think briefly. On the LAST line output exactly one JSON object: {"answer": true} or {"answer": false}.\n\n'
        f"Table CSV:\n{table_csv}\n\n"
        f"Statement: {statement}"
    )


def _tablefact_second_pass_prompt(
    *,
    table_csv: str,
    statement: str,
    caption: str,
    first_output: str,
    evidence_hints: str = "",
) -> str:
    evidence_section = f"\n{evidence_hints}\n" if evidence_hints else ""
    return (
        "You are a second-pass TabFact verifier focused on recovering false negatives.\n"
        "The first verifier answered false. Re-check whether the statement is "
        "actually ENTAILED by the table under TabFact-style loose wording.\n"
        "Common false-negative traps to correct:\n"
        "- A tied maximum/minimum still satisfies highest/lowest/most/least unless uniqueness is explicit.\n"
        "- If all values are equal, the common value is still the highest/lowest.\n"
        "- Column-header descriptors such as school/club/team, player, region, or object type need not appear literally in the cell.\n"
        "- Unambiguous aliases, initials, prefixes, and truncated names can match the full table value.\n"
        "- The table is the local universe. Do not reject because external/global completeness is unknown.\n"
        "- Prefer row-level entailment for awkward generated grammar: a row containing the mentioned entity and value usually supports the statement.\n"
        "- In sports leader columns such as high points/high rebounds/high assists/top scorer, listed names already mark row leaders or tied leaders; count a mentioned player as leading/tied when their name appears in that cell.\n"
        "- In election tables, first-elected years can support generated wording about winning/being official in that year.\n"
        "- 'run opposed'/'run oppose' means having an opponent; 'unopposed' means no opponent.\n"
        "- Keep explicit counts, 'more than', 'all but N', date ordering, and arithmetic strict.\n"
        "- Do not average/sum/group unless the statement explicitly says average/total/count; prefer a concrete supporting row or pair.\n"
        "- Treat date formats, ordinal/cardinal forms, units, punctuation, and hyphens as equivalent when unambiguous.\n"
        "Output true with high confidence when specific table evidence supports "
        "every required part. Keep false when a required part is contradicted or "
        "there is no unambiguous support. Caption is context only: "
        f"{caption}\n"
        f"{evidence_section}\n"
        "First verifier output:\n"
        f"{preview(first_output, max_length=1200)}\n\n"
        "Return exactly one JSON object on the last line with this schema:\n"
        '{"answer": true|false, "confidence": "high"|"medium"|"low", "evidence": "brief table evidence"}\n'
        "The answer may be true only when confidence is high.\n\n"
        f"Table CSV:\n{table_csv}\n\n"
        f"Statement: {statement}"
    )


def _tablefact_true_guard_prompt(
    *,
    table_csv: str,
    statement: str,
    caption: str,
    first_output: str,
) -> str:
    return (
        "You are a strict TabFact false-positive guard.\n"
        "The first verifier answered true. Re-check whether the statement is "
        "actually contradicted or unsupported by the table.\n"
        "Only change the answer to false when the contradiction is high-confidence.\n"
        "False-positive traps to inspect carefully:\n"
        "- For counts, totals, averages, 'only', 'N of', 'out of N', 'different', "
        "'times/instances/occasions', and 'more/less/over/under', count all relevant rows exactly; one matching row is not enough.\n"
        "- For ranks and superlatives, compare all relevant values; ties satisfy highest/lowest unless unique/only is stated.\n"
        "- For negation ('not', 'no', 'did not'), verify the negated relation exactly; do not treat a nearby row as enough.\n"
        "- For multi-part statements, every entity, value, date, score, venue, and comparison must hold in the same required scope.\n"
        "- Keep true if the table gives clear support under loose TabFact wording; output false only for a concrete contradiction or missing required support.\n"
        "Caption is context only: "
        f"{caption}\n\n"
        "First verifier output:\n"
        f"{preview(first_output, max_length=1200)}\n\n"
        "Return exactly one JSON object on the last line with this schema:\n"
        '{"answer": true|false, "confidence": "high"|"medium"|"low", "evidence": "brief table evidence or contradiction"}\n'
        "Use answer=false only when confidence is high.\n\n"
        f"Table CSV:\n{table_csv}\n\n"
        f"Statement: {statement}"
    )


def _parse_direct_boolean(text: str) -> bool | None:
    matches = re.findall(r"\{[^{}]*\"answer\"\s*:\s*(true|false|\"true\"|\"false\"|1|0)[^{}]*\}", text, re.I)
    if matches:
        value = matches[-1].strip().strip('"').lower()
        if value in {"true", "1"}:
            return True
        if value in {"false", "0"}:
            return False
    value_matches = re.findall(
        r"\b(true|false|entailed|refuted|supported|unsupported|yes|no)\b",
        text,
        re.I,
    )
    if not value_matches:
        return None
    value = value_matches[-1].lower()
    if value in {"true", "entailed", "supported", "yes"}:
        return True
    if value in {"false", "refuted", "unsupported", "no"}:
        return False
    return None


def _parse_second_pass_result(text: str) -> tuple[bool | None, str | None]:
    answer = _parse_direct_boolean(text)
    confidence: str | None = None
    json_objects = re.findall(r"\{[^{}]*\}", text, re.S)
    for raw in reversed(json_objects):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            if answer is None:
                parsed_answer = payload.get("answer")
                if isinstance(parsed_answer, bool):
                    answer = parsed_answer
                elif isinstance(parsed_answer, str):
                    lowered = parsed_answer.strip().lower()
                    if lowered in {"true", "entailed", "supported", "yes"}:
                        answer = True
                    elif lowered in {"false", "refuted", "unsupported", "no"}:
                        answer = False
            raw_confidence = payload.get("confidence")
            if raw_confidence is not None:
                confidence = str(raw_confidence).strip().lower()
            break
    if confidence is None:
        match = re.search(r"\bconfidence\b[^A-Za-z0-9_]*(high|medium|low)\b", text, re.I)
        if match:
            confidence = match.group(1).lower()
    return answer, confidence


def _sum_token_usage(*items: dict[str, Any] | None) -> dict[str, int]:
    total = {key: 0 for key in TOKEN_KEYS}
    for usage in items:
        if not isinstance(usage, dict):
            continue
        for key in TOKEN_KEYS:
            total[key] += int(usage.get(key, 0) or 0)
    return total


def _direct_system_profile(
    *,
    records: list[dict[str, Any]],
    validation_mode: str,
) -> dict[str, Any]:
    token_usage = {key: 0 for key in TOKEN_KEYS}
    total_seconds = 0.0
    calls = 0
    for record in records:
        verifier = record.get("direct_verifier")
        if not isinstance(verifier, dict):
            continue
        usage = verifier.get("token_usage")
        if isinstance(usage, dict):
            calls += max(1, int(verifier.get("calls", 1) or 1))
            for key in TOKEN_KEYS:
                token_usage[key] += int(usage.get(key, 0) or 0)
        total_seconds += float(verifier.get("elapsed_seconds") or 0.0)
    counters = Counter()
    counters["supervisor_synthesis_calls"] = calls
    for key, value in token_usage.items():
        counters[f"remote_{key}"] = value
        counters[f"supervisor_synthesis_{key}"] = value
    return {
        "stages": {
            "tablefact_direct_verify": {
                "calls": calls,
                "items": calls,
                "total_seconds": total_seconds,
                "average_seconds": total_seconds / calls if calls else 0.0,
            }
        },
        "counters": dict(sorted(counters.items())),
        "summary": {
            "remote_calls": calls,
            "local_slm_calls": 0,
            "edge_local_review_calls": 0,
            "edge_local_review_hits": 0,
            "edge_local_review_escalations": 0,
            "merged_plan_count": 0,
            "reused_nodes": 0,
            "parallel_system_instances": 1 if calls else 0,
            "remote_token_usage": token_usage,
            "supervisor_decompose_token_usage": {key: 0 for key in TOKEN_KEYS},
            "supervisor_synthesis_token_usage": token_usage,
            "local_slm_token_usage": {key: 0 for key in TOKEN_KEYS},
            "validation_mode": validation_mode,
        },
    }


def _attach_direct_context_stats(
    summary: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    remote_ctx = []
    for record in records:
        verifier = record.get("direct_verifier") or {}
        if isinstance(verifier, dict) and verifier.get("max_context_tokens") is not None:
            remote_ctx.append(int(verifier.get("max_context_tokens", 0) or 0))
            continue
        usage = verifier.get("token_usage") if isinstance(verifier, dict) else None
        remote_ctx.append(int(usage.get("input_tokens", 0) or 0) if isinstance(usage, dict) else 0)
    local_ctx = [0 for _ in remote_ctx]
    combined = [max(remote, local) for remote, local in zip(remote_ctx, local_ctx)]
    summary["max_context_tokens_per_case"] = {
        "remote": remote_ctx,
        "local": local_ctx,
        "combined": combined,
    }
    summary["max_context_tokens_stats"] = {
        "remote": _ctx_stats(remote_ctx),
        "local": _ctx_stats(local_ctx),
        "combined": _ctx_stats(combined),
    }
    summary["avg_max_context_tokens_per_query"] = (
        sum(combined) / len(combined) if combined else 0.0
    )


def _normalize_tablefact_outputs(
    *,
    summary: dict[str, Any],
    output_dir: Path,
    split: str | None,
    subset: str | None,
    requested_sample_size: int | None,
) -> dict[str, Any]:
    cases_index = output_dir / "cases_index.jsonl"
    records = _read_jsonl(cases_index)
    converted_records = [_tablefact_record(record) for record in records]
    write_jsonl(cases_index, converted_records)

    mismatch_path = output_dir / "answer_mismatch_cases.jsonl"
    mismatch_records = [
        _tablefact_record(record) for record in _read_jsonl(mismatch_path)
    ]
    write_jsonl(mismatch_path, mismatch_records)

    for record in converted_records:
        case_dir = output_dir / "cases" / str(record.get("runtime_case_id") or "")
        case_result_path = case_dir / "case_result.json"
        if case_result_path.is_file():
            write_json(case_result_path, _tablefact_record(_read_json(case_result_path)))

    normalized = dict(summary)
    normalized["stage"] = "tablefact_eval"
    normalized["tablefact_standard"] = {
        "official_name": "TabFact",
        "task": "binary table fact verification",
        "labels": {"true": "entailed", "false": "refuted"},
        "metric": "accuracy",
        "split": split,
        "subset": subset,
    }
    normalized.pop("tablebench_standard", None)
    normalized["requested_sample_size"] = requested_sample_size
    normalized["scores_by_subset"] = normalized.pop("scores_by_qsubtype", {})
    normalized["subsets"] = normalized.pop("qsubtypes", {})
    normalized.pop("scores_by_qtype", None)
    normalized.pop("qtypes", None)
    normalized["scores_by_metric"] = {
        "accuracy": normalized.get("scores_by_metric", {}).get("EM", {})
    }
    label_counts = Counter(
        "entailed" if bool(record.get("expected_raw")) else "refuted"
        for record in converted_records
    )
    label_correct = Counter(
        "entailed" if bool(record.get("expected_raw")) else "refuted"
        for record in converted_records
        if record.get("answer_correct")
    )
    normalized["labels"] = dict(sorted(label_counts.items()))
    normalized["accuracy_by_label"] = {
        label: {
            "total": count,
            "correct": label_correct[label],
            "accuracy": safe_divide(label_correct[label], count),
        }
        for label, count in sorted(label_counts.items())
    }
    normalized["brief_summary"] = build_brief_summary(normalized)
    normalized = compact_run_summary(normalized)
    write_json(output_dir / "run_summary.json", normalized)
    return normalized


def _tablefact_record(record: dict[str, Any]) -> dict[str, Any]:
    converted = dict(record)
    if "tablebench_metric" in converted:
        converted["tablefact_metric"] = "accuracy"
        converted.pop("tablebench_metric", None)
    if "tablebench_score" in converted:
        converted["tablefact_score"] = converted.pop("tablebench_score")
    if converted.get("metric") == "EM":
        converted["metric"] = "accuracy"
    if "qsubtype" in converted:
        converted["subset"] = converted.pop("qsubtype")
    converted.pop("qtype", None)
    return converted


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    records.append(payload)
    return records
