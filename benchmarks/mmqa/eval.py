"""MMQA multi-table evaluation for CLOVER table reasoning."""

from __future__ import annotations

import random
import shutil
import time
import gc
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from benchmarks.costing import estimate_openai_text_cost, normalize_remote_token_usage
from benchmarks.mmqa.adapter import (
    MMQA_DSL_MODE_BUILDER_AGENT,
    iter_mmqa_dataset_dirs,
    load_mmqa_task,
    read_cases,
    write_json,
)
from benchmarks.mmqa.metrics import score_mmqa_answer
from benchmarks.tablebench.eval import (
    _config_summary,
    _dsl_builder_record_summary,
    _dsl_builder_total_tokens,
    _merge_system_profiles,
    _run_system_groups,
    _slm_scheduler_summary,
    _sum_dsl_builder_token_usage,
    _update_record_from_task_item,
    _write_case_trace_artifacts,
)
from benchmarks.utils import (
    build_brief_summary,
    compact_run_summary,
    display_path,
    json_ready,
    preview,
    safe_divide,
    write_jsonl,
)
from benchmarks.warnings import suppress_benchmark_warnings
from clover.runtime import CaseResult, TableReasoningCaseSpec

DEFAULT_MMQA_EVAL_BATCH_SIZE = 50


def run_mmqa_eval(
    *,
    mmqa_root: Path,
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
    """Run converted MMQA multi-table cases through CLOVER and score denotation EM."""

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

        selected_cases = select_mmqa_cases(
            mmqa_root=mmqa_root,
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
                "MMQA eval requires local_slm_config for local SLM repair/synthesis"
            )
        validation_mode = str(validation_mode or "none").strip().lower()
        if eval_batch_size is None and len(selected_cases) > DEFAULT_MMQA_EVAL_BATCH_SIZE:
            eval_batch_size = DEFAULT_MMQA_EVAL_BATCH_SIZE
        progress_bar = progress_factory(len(selected_cases)) if progress_factory else None
        try:
            if eval_batch_size is not None and eval_batch_size < len(selected_cases):
                records = []
                profiles = []
                for start_index in range(0, len(selected_cases), eval_batch_size):
                    chunk = selected_cases[start_index : start_index + eval_batch_size]
                    chunk_records, chunk_profile = _run_mmqa_cases(
                        mmqa_root=mmqa_root,
                        output_dir=output_dir,
                        selected_cases=chunk,
                        sample_offset=start_index,
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
                        progress_bar=None,
                    )
                    records.extend(_compact_record_for_summary(record) for record in chunk_records)
                    chunk_records.clear()
                    profiles.append(_compact_system_profile_for_merge(chunk_profile))
                    if progress_bar is not None:
                        progress_bar.update(records)
                    gc.collect()
                system_profile = _merge_system_profiles(profiles)
            else:
                records, system_profile = _run_mmqa_cases(
                    mmqa_root=mmqa_root,
                    output_dir=output_dir,
                    selected_cases=selected_cases,
                    sample_offset=0,
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
                system_profile = _merge_system_profiles(
                    [_compact_system_profile_for_merge(system_profile)]
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


def _compact_record_for_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Drop bulky per-case traces after they have been written to case files."""

    compact = dict(record)
    compact.pop("table_diagnostics", None)
    compact.pop("rounds", None)
    return compact


def _compact_system_profile_for_merge(profile: dict[str, Any]) -> dict[str, Any]:
    """Keep only profile fields required for aggregate counters/token usage."""

    if not isinstance(profile, dict):
        return {}
    summary = profile.get("summary") if isinstance(profile.get("summary"), dict) else {}
    compact_summary = {}
    if summary.get("validation_mode"):
        compact_summary["validation_mode"] = summary.get("validation_mode")
    return {
        "stages": dict(profile.get("stages") or {}),
        "counters": dict(profile.get("counters") or {}),
        "summary": compact_summary,
    }


def _run_mmqa_cases(
    *,
    mmqa_root: Path,
    output_dir: Path,
    selected_cases: list[dict[str, Any]],
    sample_offset: int,
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

    for local_index, sampled_case in enumerate(selected_cases):
        sample_index = sample_offset + local_index
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
        try:
            case_specs.append(
                _materialized_case_spec(
                    mmqa_root=mmqa_root,
                    sampled_case=sampled_case,
                    sample_index=sample_index,
                    runtime_case_id=runtime_case_id,
                )
            )
            startup_profile["loaded_cases"] += 1
        except Exception:  # noqa: BLE001 - record preprocess failure and skip
            startup_profile["preprocess_failed_cases"] += 1
            record["runtime_ok"] = False
            record["error"] = {"type": "DslBuildFailed", "message": "task_dsl build failed"}
            record["elapsed_seconds"] = time.perf_counter() - started_by_case[runtime_case_id]
            write_json(case_dir / "case_result.json", record)

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
            _drop_record_heavy_fields(record)
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
    profiles = [item.profile for item in system_results]
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
        system_result.case_results.clear()
        for answer_key, task_item in system_result.task_items.items():
            record = records_by_answer.get(answer_key) or records_by_case.get(task_item.case_id)
            if record is None:
                _drop_task_item_heavy_fields(task_item)
                continue
            current_sql = task_item.current_sql
            _update_record_from_task_item(record, task_item)
            _write_light_runtime_task_artifact(
                output_dir / "cases" / task_item.case_id,
                task_item,
                current_sql=current_sql,
            )
            record["answer_key"] = answer_key
            record["current_sql"] = current_sql
            record.setdefault("initial_sql", None)
            record["retry_count"] = task_item.retry_count
            record["task_status"] = task_item.status
            _drop_task_item_heavy_fields(task_item)
            _drop_record_heavy_fields(record)
            write_json(
                output_dir / "cases" / task_item.case_id / "case_result.json",
                record,
            )
        system_result.task_items.clear()
    system_profile = _merge_system_profiles(profiles)
    startup_profile["startup_seconds"] = time.perf_counter() - startup_started
    system_profile["eval_startup"] = startup_profile
    case_specs.clear()
    records_by_answer.clear()
    started_by_case.clear()
    completed_records.clear()
    system_results.clear()
    gc.collect()
    return list(records_by_case.values()), system_profile


def _drop_record_heavy_fields(record: dict[str, Any]) -> None:
    record.pop("table_diagnostics", None)
    record.pop("rounds", None)


def _write_light_runtime_task_artifact(
    case_dir: Path,
    task_item: Any,
    *,
    current_sql: str | None,
) -> None:
    """Write a compact task snapshot for MMQA.

    The generic TableBench artifact writer stores full task/local/remote DSLs and
    context. For MMQA these objects include multi-table schemas and value
    profiles, and copying them for every case can dominate memory after the
    cases have already finished. Case-level decompose, synthesis, and execution
    traces are already written from ``CaseResult`` in real time, so this compact
    record keeps only the fields needed to audit final routing and SQL.
    """

    write_json(
        case_dir / "runtime_task.json",
        {
            "answer_key": task_item.answer_key,
            "status": task_item.status,
            "retry_count": task_item.retry_count,
            "answer_type": task_item.answer_type,
            "current_sql": current_sql,
            "last_error": task_item.last_error,
        },
    )


def _drop_task_item_heavy_fields(task_item: Any) -> None:
    """Release bulky runtime fields once their compact summary is recorded."""

    for attr in ("task_dsl", "local_dsl", "remote_dsl", "context"):
        try:
            setattr(task_item, attr, {})
        except Exception:  # noqa: BLE001 - best-effort memory pruning
            pass
    metadata = getattr(task_item, "metadata", None)
    if isinstance(metadata, dict):
        for key in (
            "table_diagnostics",
            "decompose_trace",
            "synthesis_trace",
            "edge_review_trace",
            "agent_loop_trace",
            "dsl_builder",
        ):
            metadata.pop(key, None)
    memory = getattr(task_item, "memory", None)
    if isinstance(memory, list):
        memory.clear()
    try:
        task_item.current_command = None
    except Exception:  # noqa: BLE001
        pass
    try:
        task_item.result_callback = None
    except Exception:  # noqa: BLE001
        pass


def _materialized_case_spec(
    *,
    mmqa_root: Path,
    sampled_case: dict[str, Any],
    sample_index: int,
    runtime_case_id: str,
) -> TableReasoningCaseSpec:
    """Build a multi-table task DSL up front and pass it as a ready spec.

    The runtime's builder stage only supports single-table builders, so MMQA
    cases are materialized here via ``load_mmqa_task`` and submitted with a
    non-empty ``task_dsl`` and no ``builder`` payload. The runtime then
    preprocesses and executes the multi-table DSL directly.
    """

    dataset_id = sampled_case["dataset_id"]
    case_id = sampled_case["case_id"]
    task = load_mmqa_task(
        mmqa_root=mmqa_root,
        dataset_id=dataset_id,
        case_id=case_id,
    )
    dataset_dir = mmqa_root
    for split_dir in sorted(path for path in mmqa_root.iterdir() if path.is_dir()):
        candidate = split_dir / dataset_id
        if candidate.is_dir():
            dataset_dir = candidate
            break
    metadata = {
        "sample_index": sample_index,
        "dataset": "mmqa",
        "dataset_id": dataset_id,
        "case_id": case_id,
        "case_index": sampled_case.get("case_index"),
        "answer_type": sampled_case.get("answer_type"),
        "expected_answer": sampled_case.get("expected_answer"),
        "expected_raw": sampled_case.get("expected_raw"),
        "question": sampled_case.get("question"),
        "split": sampled_case.get("split"),
        "table_names": sampled_case.get("table_names"),
        "foreign_keys": sampled_case.get("foreign_keys"),
        "primary_keys": sampled_case.get("primary_keys"),
        "table_count": sampled_case.get("table_count"),
        "dsl_builder_mode": MMQA_DSL_MODE_BUILDER_AGENT,
        "dsl_builder": task.metadata.get("dsl_builder"),
    }
    return TableReasoningCaseSpec(
        case_id=runtime_case_id,
        task_dsl=task.task_dsl,
        base_dir=dataset_dir,
        metadata=metadata,
        answer_key=f"answer_{sample_index + 1}",
        builder=None,
    )


def _update_record_from_case_result(
    record: dict[str, Any],
    case_result: CaseResult,
) -> None:
    metadata = case_result.metadata if isinstance(case_result.metadata, dict) else {}
    _update_record_from_result_metadata(record, metadata)
    score = _score_mmqa_record(record, case_result.answer)
    record.update(
        {
            "answer_key": case_result.answer_key,
            "runtime_ok": case_result.ok,
            "final_answer": json_ready(case_result.answer),
            "final_answer_preview": preview(case_result.answer),
            "final_answer_standard_text": score.actual,
            "answer_correct": bool(case_result.ok and score.correct),
            "mmqa_metric": score.metric,
            "mmqa_score": score.score if case_result.ok else 0.0,
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
        ("expected_answer", "expected_answer"),
        ("expected_raw", "expected_raw"),
        ("split", "split"),
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


def _score_mmqa_record(record: dict[str, Any], actual: Any) -> Any:
    raw_score = score_mmqa_answer(
        expected=record.get("expected_raw"),
        actual=actual,
        expected_answer_type=record.get("answer_type"),
    )
    expected_answer = record.get("expected_answer")
    if expected_answer is None:
        return raw_score
    answer_score = score_mmqa_answer(
        expected=expected_answer,
        actual=actual,
        expected_answer_type=record.get("answer_type"),
    )
    if answer_score.correct and not raw_score.correct:
        return answer_score
    return raw_score


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
    score_sum = sum(float(record.get("mmqa_score") or 0.0) for record in records)
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
        "stage": "mmqa_eval",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workflow": "table_reasoning.query",
        "mmqa_standard": {
            "split": split,
            "metric": "denotation_em",
            "multitable": True,
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
        "dsl_builder_mode": MMQA_DSL_MODE_BUILDER_AGENT,
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
        "scores_by_metric": _score_groups(records, "mmqa_metric"),
        "scores_by_split": _score_groups(records, "split"),
        "scores_by_answer_type": _score_groups(records, "answer_type"),
        "scores_by_table_count": _score_groups(records, "table_count"),
        "answer_types": _counter_as_strings(answer_types),
        "task_answer_types": _counter_as_strings(task_answer_types),
        "task_answer_type_matches": task_answer_type_matches,
        "task_answer_type_accuracy": safe_divide(task_answer_type_matches, total),
        "dsl_builder_modes": _counter_as_strings(dsl_builder_modes),
        "builder_agent_calls": builder_agent_calls,
        "builder_agent_token_usage": builder_agent_token_usage,
        "mismatches_by_type": _counter_as_strings(mismatches_by_type),
        "splits": _string_counter(records, "split"),
        "table_counts": _string_counter(records, "table_count"),
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
    return compact_run_summary(summary)


def mismatch_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_index": record["sample_index"],
        "runtime_case_id": record.get("runtime_case_id"),
        "dataset_id": record["dataset_id"],
        "case_id": record["case_id"],
        "split": record.get("split"),
        "table_count": record.get("table_count"),
        "answer_type": record.get("answer_type"),
        "task_answer_type": record.get("task_answer_type"),
        "dsl_builder_mode": record.get("dsl_builder_mode"),
        "metric": record.get("mmqa_metric"),
        "score": record.get("mmqa_score"),
        "question": record.get("question"),
        "expected": record.get("expected_standard_text"),
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
        "table_count": sampled_case.get("table_count"),
        "table_names": sampled_case.get("table_names"),
        "foreign_keys": sampled_case.get("foreign_keys"),
        "primary_keys": sampled_case.get("primary_keys"),
        "mmqa_metric": "denotation_em",
        "mmqa_score": 0.0,
        "question": sampled_case.get("question"),
        "parse_ok": False,
        "runtime_ok": False,
        "answer_correct": False,
        "expected_answer": sampled_case.get("expected_answer"),
        "expected_raw": sampled_case.get("expected_raw"),
        "expected_standard_text": json_ready(sampled_case.get("expected_raw")),
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


def select_mmqa_cases(
    *,
    mmqa_root: Path,
    max_cases: int | None,
    case_ids: set[str],
    dataset_id: str | None,
    split: str | None,
    sample_size: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if max_cases == 0:
        return []
    cases = list_mmqa_cases(mmqa_root)
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


def list_mmqa_cases(mmqa_root: Path) -> list[dict[str, Any]]:
    cases = []
    for dataset_dir in iter_mmqa_dataset_dirs(mmqa_root):
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
                    "expected_answer": case.get("answer"),
                    "expected_raw": case.get("answer_raw"),
                    "answer_type": case.get("type"),
                    "split": case.get("split"),
                    "table_names": case.get("table_names"),
                    "foreign_keys": case.get("foreign_keys"),
                    "primary_keys": case.get("primary_keys"),
                    "table_count": case.get("table_count"),
                    "source_files": case.get("source_files"),
                    "gold_sql": case.get("gold_sql"),
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
        score_sum = sum(float(record.get("mmqa_score") or 0.0) for record in group)
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
