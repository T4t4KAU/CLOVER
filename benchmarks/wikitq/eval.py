"""WikiTableQuestions evaluation for CLOVER table reasoning."""

from __future__ import annotations

import random
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from benchmarks.costing import estimate_openai_text_cost, normalize_remote_token_usage
from benchmarks.utils import (
    build_brief_summary,
    display_path,
    json_ready,
    preview,
    safe_divide,
    write_jsonl,
)
from benchmarks.tablebench.eval import (
    _config_summary,
    _ctx_stats,
    _dsl_builder_record_summary,
    _dsl_builder_total_tokens,
    _merge_system_profiles,
    _per_case_max_context_tokens,
    _run_system_groups,
    _slm_scheduler_summary,
    _sum_dsl_builder_token_usage,
    _update_record_from_task_item,
    _write_case_trace_artifacts,
    _write_runtime_task_artifacts,
)
from benchmarks.warnings import suppress_benchmark_warnings
from benchmarks.wikitq.adapter import (
    WIKITQ_DSL_MODE_BUILDER_AGENT,
    iter_wikitq_dataset_dirs,
    read_cases,
    wikitq_source_context_path,
    wikitq_source_root,
    write_json,
)
from benchmarks.wikitq.metrics import score_wikitq_answer
from clover.runtime import CaseResult, TableReasoningCaseSpec


def run_wikitq_eval(
    *,
    wikitq_root: Path,
    output_dir: Path,
    remote_config: dict[str, Any],
    synthesize_config: dict[str, Any] | None = None,
    local_slm_config: dict[str, Any] | None = None,
    max_cases: int | None = None,
    case_ids: set[str] | None = None,
    dataset_id: str | None = None,
    split: str | None = None,
    sample_size: int | None = None,
    seed: int = 20260528,
    max_workers: int | None = 64,
    max_retries: int = 1,
    validation_mode: str = "none",
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
    """Run converted WikiTQ cases through CLOVER and score denotation EM."""

    with suppress_benchmark_warnings():
        started = time.perf_counter()
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {output_dir}. "
                    "Use --overwrite to replace it."
                )
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        selected_cases = select_wikitq_cases(
            wikitq_root=wikitq_root,
            max_cases=max_cases,
            case_ids=case_ids or set(),
            dataset_id=dataset_id,
            split=split,
            sample_size=sample_size,
            seed=seed,
        )
        _validate_parallel_args(
            remote_batch_size=remote_batch_size,
            remote_concurrency=remote_concurrency,
            max_parallel_execution_units=max_parallel_execution_units,
            max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
            max_parallel_slm_sequences=max_parallel_slm_sequences,
            max_pending_slm_sequences=max_pending_slm_sequences,
            eval_batch_size=eval_batch_size,
        )
        if local_slm_config is None:
            raise ValueError(
                "WikiTQ eval requires local_slm_config for local SLM repair/synthesis"
            )
        validation_mode = str(validation_mode or "none").strip().lower()
        progress_bar = progress_factory(len(selected_cases)) if progress_factory else None
        try:
            records, system_profile = _run_wikitq_cases(
                wikitq_root=wikitq_root,
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
                profile_baseline=profile_baseline,
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
                mismatch_record(record)
                for record in records
                if record.get("runtime_ok") and not record.get("answer_correct")
            ],
        )
        write_jsonl(
            failure_cases,
            [record for record in records if not record.get("runtime_ok")],
        )

        summary = build_summary(
            records=records,
            output_dir=output_dir,
            remote_config=remote_config,
            synthesize_config=synthesize_config,
            local_slm_config=local_slm_config,
            selected_cases=selected_cases,
            elapsed_seconds=time.perf_counter() - started,
            worker_count=max(1, int(max_workers or 1)),
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
            system_profile=system_profile,
            remote_cost_model=remote_cost_model,
            seed=seed,
            sample_size=sample_size,
            split=split,
            cases_index=cases_index,
            mismatch_cases=mismatch_cases,
            failure_cases=failure_cases,
        )
        write_json(output_dir / "run_summary.json", summary)
        return summary


def _run_wikitq_cases(
    *,
    wikitq_root: Path,
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
    profile_baseline: bool,
    progress_bar: Any | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records_by_case: dict[str, dict[str, Any]] = {}
    records_by_answer: dict[str, dict[str, Any]] = {}
    completed_records: list[dict[str, Any]] = []
    started_by_case: dict[str, float] = {}
    case_specs: list[TableReasoningCaseSpec] = []
    progress_lock = Lock()
    startup_started = time.perf_counter()
    startup_profile: dict[str, Any] = {
        "workers": max(1, int(max_workers or 1)),
        "selected_cases": len(selected_cases),
        "loaded_cases": 0,
        "preprocess_failed_cases": 0,
        "startup_seconds": 0.0,
    }

    for sample_index, sampled_case in enumerate(selected_cases):
        runtime_case_id = _runtime_case_id(sampled_case, sample_index)
        case_dir = output_dir / "cases" / runtime_case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        started_by_case[runtime_case_id] = time.perf_counter()
        record = base_case_record(
            sampled_case=sampled_case,
            sample_index=sample_index,
            runtime_case_id=runtime_case_id,
            case_dir=case_dir,
        )
        records_by_case[runtime_case_id] = record
        startup_profile["loaded_cases"] += 1
        case_specs.append(
            _builder_case_spec(
                wikitq_root=wikitq_root,
                sampled_case=sampled_case,
                sample_index=sample_index,
                runtime_case_id=runtime_case_id,
            )
        )

    def on_case_result(case_result: CaseResult) -> None:
        with progress_lock:
            record = records_by_case[case_result.case_id]
            records_by_answer[case_result.answer_key] = record
            _update_record_from_case_result(record, case_result)
            record["elapsed_seconds"] = (
                time.perf_counter() - started_by_case[case_result.case_id]
            )
            case_dir = output_dir / "cases" / case_result.case_id
            write_json(case_dir / "case_result.json", record)
            _write_case_trace_artifacts(case_dir, case_result)
            completed_records.append(record)
            if progress_bar is not None:
                progress_bar.update(completed_records)

    system_results = _run_system_groups(
        spec_groups=[case_specs],
        remote_config=remote_config,
        synthesize_config=synthesize_config,
        local_slm_config=local_slm_config,
        remote_batch_size=remote_batch_size,
        remote_concurrency=remote_concurrency,
        max_parallel_execution_units=max_parallel_execution_units,
        max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
        max_parallel_slm_sequences=max_parallel_slm_sequences,
        max_pending_slm_sequences=max_pending_slm_sequences,
        max_retries=max_retries,
        validation_mode=validation_mode,
        max_workers=max_workers,
        case_result_callback=on_case_result,
        profile_baseline=profile_baseline,
        records_by_case=records_by_case,
        completed_records=completed_records,
        started_by_case=started_by_case,
        output_dir=output_dir,
        progress_bar=progress_bar,
        progress_lock=progress_lock,
    )
    for system_result in system_results:
        for case_result in system_result.case_results:
            record = records_by_case.get(case_result.case_id)
            if record is None or record.get("answer_key"):
                continue
            records_by_answer[case_result.answer_key] = record
            _update_record_from_case_result(record, case_result)
            record["elapsed_seconds"] = (
                time.perf_counter() - started_by_case[case_result.case_id]
            )
            write_json(
                output_dir / "cases" / case_result.case_id / "case_result.json",
                record,
            )
        for answer_key, task_item in system_result.task_items.items():
            record = records_by_answer.get(answer_key) or records_by_case.get(task_item.case_id)
            if record is None:
                continue
            _update_record_from_task_item(record, task_item)
            _write_runtime_task_artifacts(
                output_dir / "cases" / task_item.case_id,
                task_item,
            )
            record["answer_key"] = answer_key
            record["current_sql"] = task_item.current_sql
            record.setdefault("initial_sql", None)
            record["retry_count"] = task_item.retry_count
            record["task_status"] = task_item.status
            write_json(
                output_dir / "cases" / task_item.case_id / "case_result.json",
                record,
            )
    system_profile = _merge_system_profiles([item.profile for item in system_results])
    startup_profile["startup_seconds"] = time.perf_counter() - startup_started
    system_profile["eval_startup"] = startup_profile
    return list(records_by_case.values()), system_profile


def _builder_case_spec(
    *,
    wikitq_root: Path,
    sampled_case: dict[str, Any],
    sample_index: int,
    runtime_case_id: str,
) -> TableReasoningCaseSpec:
    dataset_dir = wikitq_root / sampled_case["dataset_id"]
    builder = {
        "kind": WIKITQ_DSL_MODE_BUILDER_AGENT,
        "question": sampled_case["question"],
        "table_path": "table.csv",
        "source_file": "table.csv",
        "source_context_path": sampled_case["source_context_path"],
        "answer_type": sampled_case.get("answer_type"),
        "task_type": "table_reasoning.analyze",
        "source_id": 0,
        "hints": {"benchmark": "wikitq"},
    }
    metadata = {
        "sample_index": sample_index,
        "dataset": "wikitq",
        "dataset_id": sampled_case["dataset_id"],
        "case_id": sampled_case["case_id"],
        "case_index": sampled_case.get("case_index"),
        "answer_type": sampled_case.get("answer_type"),
        "expected_answer": sampled_case.get("expected_answer"),
        "expected_canon": sampled_case.get("expected_canon"),
        "answer_canon_type": sampled_case.get("answer_canon_type"),
        "question": sampled_case.get("question"),
        "split": sampled_case.get("split"),
        "context": sampled_case.get("context"),
        "source_context_path": sampled_case.get("source_context_path"),
        "dsl_builder_mode": WIKITQ_DSL_MODE_BUILDER_AGENT,
        "builder": builder,
    }
    return TableReasoningCaseSpec(
        case_id=runtime_case_id,
        task_dsl={},
        base_dir=dataset_dir,
        metadata=metadata,
        answer_key=f"answer_{sample_index + 1}",
        builder=builder,
    )


def _update_record_from_case_result(
    record: dict[str, Any],
    case_result: CaseResult,
) -> None:
    metadata = case_result.metadata if isinstance(case_result.metadata, dict) else {}
    _update_record_from_result_metadata(record, metadata)
    score = score_wikitq_answer(
        expected=record.get("expected_raw"),
        expected_canon=record.get("expected_canon"),
        actual=case_result.answer,
    )
    record.update(
        {
            "answer_key": case_result.answer_key,
            "runtime_ok": case_result.ok,
            "final_answer": json_ready(case_result.answer),
            "final_answer_preview": preview(case_result.answer),
            "final_answer_standard_text": score.actual,
            "answer_correct": bool(case_result.ok and score.correct),
            "wikitq_metric": score.metric,
            "wikitq_score": score.score if case_result.ok else 0.0,
            "round_count": case_result.retry_count + 1,
            "retry_count": case_result.retry_count,
            "retry_exhausted": (
                not case_result.ok
                and isinstance(case_result.error, dict)
                and case_result.error.get("type") == "RepairBudgetExhausted"
            ),
            "error": case_result.error,
            "rounds": [],
        }
    )


def _update_record_from_result_metadata(
    record: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    for source_key, record_key in (
        ("answer_type", "answer_type"),
        ("question", "question"),
        ("expected_answer", "expected_raw"),
        ("expected_canon", "expected_canon"),
        ("answer_canon_type", "answer_canon_type"),
        ("split", "split"),
        ("context", "context"),
        ("task_answer_type", "task_answer_type"),
        ("final_answer_source", "final_answer_source"),
        ("table_diagnostics", "table_diagnostics"),
    ):
        value = metadata.get(source_key)
        if value is not None:
            record[record_key] = value
    if record.get("expected_raw") is not None:
        record["expected_standard_text"] = json_ready(record["expected_raw"])
    dsl_builder = metadata.get("dsl_builder")
    if isinstance(dsl_builder, dict):
        record["dsl_builder"] = json_ready(_dsl_builder_record_summary(dsl_builder))
        record["dsl_builder_mode"] = dsl_builder.get("mode") or record.get(
            "dsl_builder_mode"
        )
        record["parse_ok"] = True
        if dsl_builder.get("task_answer_type") is not None:
            record["task_answer_type"] = dsl_builder.get("task_answer_type")


def build_summary(
    *,
    records: list[dict[str, Any]],
    output_dir: Path,
    remote_config: dict[str, Any],
    synthesize_config: dict[str, Any] | None,
    local_slm_config: dict[str, Any] | None,
    selected_cases: list[dict[str, Any]],
    elapsed_seconds: float,
    worker_count: int,
    max_retries: int,
    validation_mode: str,
    seed: int,
    sample_size: int | None,
    split: str | None,
    cases_index: Path,
    mismatch_cases: Path,
    failure_cases: Path,
    remote_batch_size: int | None,
    remote_concurrency: int | None,
    max_parallel_execution_units: int | None,
    max_parallel_slm_node_jobs: int | None,
    max_parallel_slm_sequences: int | None,
    max_pending_slm_sequences: int | None,
    eval_batch_size: int | None,
    profile_baseline: bool,
    system_profile: dict[str, Any],
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
    failures = total - runtime_successes
    retry_case_ids = [
        record.get("runtime_case_id") or record["case_id"]
        for record in records
        if record.get("retry_count", 0) > 0
    ]
    score_sum = sum(float(record.get("wikitq_score") or 0.0) for record in records)
    error_types = Counter(
        record["error"]["type"]
        for record in records
        if isinstance(record.get("error"), dict) and record.get("error")
    )
    answer_types = Counter(record.get("answer_type") for record in records)
    task_answer_types = Counter(record.get("task_answer_type") for record in records)
    task_answer_type_matches = sum(
        1
        for record in records
        if record.get("answer_type") == record.get("task_answer_type")
    )
    dsl_builder_modes = Counter(record.get("dsl_builder_mode") for record in records)
    builder_agent_token_usage = _sum_dsl_builder_token_usage(records)
    builder_agent_calls = sum(
        1
        for record in records
        if _dsl_builder_total_tokens(record) > 0
    )
    mismatches_by_type = Counter(
        record.get("answer_type")
        for record in records
        if record.get("runtime_ok") and not record.get("answer_correct")
    )
    summary_profile = system_profile.get("summary", {})
    supervisor_decompose_token_usage = normalize_remote_token_usage(
        summary_profile.get("supervisor_decompose_token_usage", {})
    )
    supervisor_synthesis_token_usage = normalize_remote_token_usage(
        summary_profile.get("supervisor_synthesis_token_usage", {})
    )
    remote_token_summary = normalize_remote_token_usage(
        summary_profile.get("remote_token_usage", {})
    )
    local_slm_token_summary = normalize_remote_token_usage(
        summary_profile.get("local_slm_token_usage", {})
    )
    remote_cost_estimate = estimate_openai_text_cost(
        remote_token_summary,
        remote_config=remote_config,
        pricing_model=remote_cost_model,
    )
    summary = {
        "run_name": output_dir.name,
        "stage": "wikitq_eval",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workflow": "table_reasoning.analyze",
        "wikitq_standard": {
            "split": split,
            "metric": "denotation_em",
            "uses_target_canon": True,
        },
        "sample_size": len(selected_cases),
        "requested_sample_size": sample_size,
        "seed": seed,
        "parallel_workers": worker_count,
        "max_retries": max_retries,
        "validation_mode": validation_mode,
        "remote_batch_size": remote_batch_size,
        "remote_concurrency": remote_concurrency,
        "max_parallel_execution_units": max_parallel_execution_units,
        "max_parallel_slm_node_jobs": max_parallel_slm_node_jobs,
        "max_parallel_slm_sequences": max_parallel_slm_sequences,
        "max_pending_slm_sequences": max_pending_slm_sequences,
        "slm_scheduler": _slm_scheduler_summary(local_slm_config),
        "eval_batch_size": eval_batch_size,
        "profile_baseline": profile_baseline,
        "dsl_builder_mode": WIKITQ_DSL_MODE_BUILDER_AGENT,
        "remote_llm": _config_summary(remote_config),
        "synthesize_llm": _config_summary(synthesize_config),
        "local_slm": _config_summary(local_slm_config),
        "supervisor_decompose_token_usage": supervisor_decompose_token_usage,
        "supervisor_synthesis_token_usage": supervisor_synthesis_token_usage,
        "remote_calls": summary_profile.get("remote_calls", 0),
        "local_slm_calls": summary_profile.get("local_slm_calls", 0),
        "edge_local_review_calls": summary_profile.get(
            "edge_local_review_calls",
            0,
        ),
        "edge_local_review_hits": summary_profile.get(
            "edge_local_review_hits",
            0,
        ),
        "edge_local_review_escalations": summary_profile.get(
            "edge_local_review_escalations",
            0,
        ),
        "total_cases": total,
        "parse_successes": sum(1 for record in records if record.get("parse_ok")),
        "parse_failures": sum(1 for record in records if not record.get("parse_ok")),
        "runtime_successes": runtime_successes,
        "runtime_failures": failures,
        "correct": correct,
        "mismatches": mismatches,
        "failures": failures,
        "accuracy_on_all_cases": safe_divide(correct, total),
        "accuracy_on_successes": safe_divide(correct, runtime_successes),
        "retry_cases": len(retry_case_ids),
        "retry_case_ids": retry_case_ids,
        "total_retry_rounds": sum(record.get("retry_count", 0) for record in records),
        "standard_score_average": safe_divide(score_sum, total),
        "scores_by_metric": _score_groups(records, "wikitq_metric"),
        "scores_by_split": _score_groups(records, "split"),
        "scores_by_answer_type": _score_groups(records, "answer_type"),
        "answer_types": _counter_as_strings(answer_types),
        "task_answer_types": _counter_as_strings(task_answer_types),
        "task_answer_type_matches": task_answer_type_matches,
        "task_answer_type_accuracy": safe_divide(task_answer_type_matches, total),
        "dsl_builder_modes": _counter_as_strings(dsl_builder_modes),
        "builder_agent_calls": builder_agent_calls,
        "builder_agent_token_usage": builder_agent_token_usage,
        "mismatches_by_type": _counter_as_strings(mismatches_by_type),
        "splits": _string_counter(records, "split"),
        "answer_canon_types": _string_counter(records, "answer_canon_type"),
        "error_types": dict(sorted(error_types.items())),
        "remote_token_usage": remote_token_summary,
        "local_slm_token_usage": local_slm_token_summary,
        "remote_cost_estimate": remote_cost_estimate,
        "system_profile": system_profile,
        "elapsed_seconds": elapsed_seconds,
        "run_dir": display_path(output_dir),
        "cases_index": display_path(cases_index),
        "answer_mismatch_cases": display_path(mismatch_cases),
        "failure_cases": display_path(failure_cases),
    }
    summary["brief_summary"] = build_brief_summary(summary)
    ctx_tokens = _per_case_max_context_tokens(records, output_dir)
    summary["max_context_tokens_per_case"] = ctx_tokens
    summary["max_context_tokens_stats"] = {
        "remote": _ctx_stats(ctx_tokens["remote"]),
        "local": _ctx_stats(ctx_tokens["local"]),
        "combined": _ctx_stats(ctx_tokens["combined"]),
    }
    summary["avg_max_context_tokens_per_query"] = (
        sum(ctx_tokens["combined"]) / len(ctx_tokens["combined"])
        if ctx_tokens["combined"]
        else 0.0
    )
    return summary


def mismatch_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_index": record["sample_index"],
        "runtime_case_id": record.get("runtime_case_id"),
        "dataset_id": record["dataset_id"],
        "case_id": record["case_id"],
        "split": record.get("split"),
        "context": record.get("context"),
        "answer_type": record.get("answer_type"),
        "task_answer_type": record.get("task_answer_type"),
        "dsl_builder_mode": record.get("dsl_builder_mode"),
        "metric": record.get("wikitq_metric"),
        "score": record.get("wikitq_score"),
        "question": record.get("question"),
        "expected": record.get("expected_standard_text"),
        "expected_canon": record.get("expected_canon"),
        "actual": record.get("final_answer_standard_text"),
        "initial_sql": record.get("initial_sql"),
        "current_sql": record.get("current_sql"),
        "retry_count": record.get("retry_count"),
    }


def base_case_record(
    *,
    sampled_case: dict[str, Any],
    sample_index: int,
    runtime_case_id: str,
    case_dir: Path,
) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "runtime_case_id": runtime_case_id,
        "dataset_id": sampled_case["dataset_id"],
        "case_id": sampled_case["case_id"],
        "case_index": sampled_case.get("case_index"),
        "answer_type": sampled_case.get("answer_type"),
        "task_answer_type": None,
        "dsl_builder_mode": None,
        "split": sampled_case.get("split"),
        "context": sampled_case.get("context"),
        "source_context_path": sampled_case.get("source_context_path"),
        "answer_canon_type": sampled_case.get("answer_canon_type"),
        "wikitq_metric": "denotation_em",
        "wikitq_score": 0.0,
        "question": sampled_case.get("question"),
        "parse_ok": False,
        "runtime_ok": False,
        "answer_correct": False,
        "expected_raw": sampled_case.get("expected_answer"),
        "expected_canon": sampled_case.get("expected_canon"),
        "expected_standard_text": json_ready(sampled_case.get("expected_answer")),
        "final_answer": None,
        "final_answer_preview": None,
        "final_answer_standard_text": None,
        "round_count": 0,
        "retry_count": 0,
        "error": None,
        "elapsed_seconds": None,
        "case_dir": display_path(case_dir),
    }


def _runtime_case_id(sampled_case: dict[str, Any], sample_index: int) -> str:
    dataset_id = str(sampled_case["dataset_id"])
    case_id = str(sampled_case["case_id"])
    return f"sample_{sample_index:05d}__{dataset_id}__{case_id}"


def select_wikitq_cases(
    *,
    wikitq_root: Path,
    max_cases: int | None,
    case_ids: set[str],
    dataset_id: str | None,
    split: str | None,
    sample_size: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if max_cases == 0:
        return []
    cases = list_wikitq_cases(wikitq_root)
    if dataset_id is not None:
        cases = [case for case in cases if case["dataset_id"] == dataset_id]
    if split is not None:
        cases = [case for case in cases if case.get("split") == split]
    if case_ids:
        cases = [case for case in cases if case["case_id"] in case_ids]
    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        rng = random.Random(seed)
        cases = rng.sample(cases, min(sample_size, len(cases)))
    if max_cases is not None:
        cases = cases[:max_cases]
    return cases


def list_wikitq_cases(wikitq_root: Path) -> list[dict[str, Any]]:
    cases = []
    source_root = wikitq_source_root(wikitq_root)
    for dataset_dir in iter_wikitq_dataset_dirs(wikitq_root):
        cases_path = dataset_dir / "cases.jsonl"
        if not cases_path.is_file():
            continue
        for case_index, case in enumerate(read_cases(cases_path)):
            source_context_path = wikitq_source_context_path(
                wikitq_root,
                context=case.get("context"),
                source_root=source_root,
            )
            cases.append(
                {
                    "dataset_id": dataset_dir.name,
                    "case_id": case["case_id"],
                    "case_index": case_index,
                    "question": case.get("question"),
                    "expected_answer": case.get("answer"),
                    "expected_canon": case.get("answer_canon"),
                    "answer_canon_type": case.get("answer_canon_type"),
                    "answer_type": case.get("type"),
                    "split": case.get("split"),
                    "context": case.get("context"),
                    "source_context_path": str(source_context_path),
                }
            )
    return cases


def _score_groups(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(key) or "unknown")].append(record)
    result = {}
    for group_key, group in sorted(groups.items()):
        total = len(group)
        runtime_successes = sum(1 for record in group if record.get("runtime_ok"))
        correct = sum(1 for record in group if record.get("answer_correct"))
        score_sum = sum(float(record.get("wikitq_score") or 0.0) for record in group)
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


def _counter_as_strings(counter: Counter[Any]) -> dict[str, int]:
    return dict(sorted((str(key or "unknown"), value) for key, value in counter.items()))


def _validate_parallel_args(
    *,
    remote_batch_size: int,
    remote_concurrency: int,
    max_parallel_execution_units: int,
    max_parallel_slm_node_jobs: int,
    max_parallel_slm_sequences: int,
    max_pending_slm_sequences: int,
    eval_batch_size: int | None,
) -> None:
    values = {
        "remote_batch_size": remote_batch_size,
        "remote_concurrency": remote_concurrency,
        "max_parallel_execution_units": max_parallel_execution_units,
        "max_parallel_slm_node_jobs": max_parallel_slm_node_jobs,
        "max_parallel_slm_sequences": max_parallel_slm_sequences,
        "max_pending_slm_sequences": max_pending_slm_sequences,
    }
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if eval_batch_size is not None and eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")
