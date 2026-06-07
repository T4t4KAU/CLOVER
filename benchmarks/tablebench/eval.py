"""TableBench non-visual evaluation for CLOVER table reasoning."""

from __future__ import annotations

import random
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from benchmarks.costing import estimate_openai_text_cost, normalize_remote_token_usage
from benchmarks.databench.static_tool_eval import (
    display_path,
    format_error,
    json_ready,
    preview,
    safe_divide,
    write_jsonl,
)
from benchmarks.tablebench.adapter import (
    TABLEBENCH_DSL_MODE_BUILDER_AGENT,
    TablebenchTask,
    iter_tablebench_dataset_dirs,
    load_tablebench_task,
    read_cases,
    write_json,
)
from benchmarks.tablebench.metrics import score_tablebench_answer, tablebench_metric_name
from benchmarks.warnings import suppress_benchmark_warnings
from clover.resource import preprocess_task_dsl
from clover.runtime import (
    CaseResult,
    TableReasoningCaseSpec,
    run_table_reasoning_system,
)


TOKEN_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)


@dataclass(frozen=True)
class _PreparedTableBenchCase:
    sampled_case: dict[str, Any]
    sample_index: int
    runtime_case_id: str
    case_id: str
    case_dir: Path
    task: TablebenchTask
    preprocess_result: dict[str, Any]
    expected_raw: Any
    answer_type: str | None
    qtype: str | None
    qsubtype: str | None
    source_file: str
    dsl_builder_mode: str
    task_answer_type: str | None
    dsl_builder: dict[str, Any]


def run_tablebench_eval(
    *,
    tablebench_root: Path,
    output_dir: Path,
    remote_config: dict[str, Any],
    local_slm_config: dict[str, Any] | None = None,
    max_cases: int | None = None,
    case_ids: set[str] | None = None,
    dataset_id: str | None = None,
    qtypes: set[str] | None = None,
    qsubtypes: set[str] | None = None,
    include_visualization: bool = False,
    sample_size: int | None = None,
    seed: int = 20260528,
    max_workers: int | None = None,
    max_retries: int = 1,
    validation_mode: str = "none",
    remote_batch_size: int = 16,
    remote_concurrency: int = 2,
    max_parallel_execution_units: int = 32,
    max_parallel_slm_node_jobs: int = 1,
    max_parallel_slm_sequences: int = 8,
    max_pending_slm_sequences: int = 1024,
    eval_batch_size: int | None = None,
    profile_baseline: bool = False,
    remote_cost_model: str | None = None,
    overwrite: bool = False,
    progress_factory: Callable[[int], Any] | None = None,
) -> dict[str, Any]:
    """Run converted TableBench cases through CLOVER and score non-visual metrics."""

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

        selected_cases = select_tablebench_cases(
            tablebench_root=tablebench_root,
            max_cases=max_cases,
            case_ids=case_ids or set(),
            dataset_id=dataset_id,
            qtypes=qtypes or set(),
            qsubtypes=qsubtypes or set(),
            include_visualization=include_visualization,
            sample_size=sample_size,
            seed=seed,
        )
        if remote_batch_size <= 0:
            raise ValueError("remote_batch_size must be positive")
        if remote_concurrency <= 0:
            raise ValueError("remote_concurrency must be positive")
        if max_parallel_execution_units <= 0:
            raise ValueError("max_parallel_execution_units must be positive")
        if max_parallel_slm_node_jobs <= 0:
            raise ValueError("max_parallel_slm_node_jobs must be positive")
        if max_parallel_slm_sequences <= 0:
            raise ValueError("max_parallel_slm_sequences must be positive")
        if max_pending_slm_sequences <= 0:
            raise ValueError("max_pending_slm_sequences must be positive")
        if eval_batch_size is not None and eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")
        if local_slm_config is None:
            raise ValueError(
                "TableBench eval requires local_slm_config for BuilderAgent DSL construction"
            )
        validation_mode = str(validation_mode or "none").strip().lower()
        progress_bar = progress_factory(len(selected_cases)) if progress_factory else None
        try:
            records, system_profile = _run_tablebench_cases(
                tablebench_root=tablebench_root,
                output_dir=output_dir,
                selected_cases=selected_cases,
                remote_config=remote_config,
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
            local_slm_config=local_slm_config,
            selected_cases=selected_cases,
            elapsed_seconds=time.perf_counter() - started,
            worker_count=max(1, int(max_workers or 1)),
            max_retries=0 if validation_mode == "none" else max_retries,
            validation_mode=validation_mode,
            include_visualization=include_visualization,
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
        write_json(output_dir / "run_summary.json", summary)
        return summary


def _run_tablebench_cases(
    *,
    tablebench_root: Path,
    output_dir: Path,
    selected_cases: list[dict[str, Any]],
    remote_config: dict[str, Any],
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
    specs_by_source_file: dict[str, list[TableReasoningCaseSpec]] = defaultdict(list)
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
        try:
            prepared = _prepare_case(
                tablebench_root=tablebench_root,
                sampled_case=sampled_case,
                sample_index=sample_index,
                runtime_case_id=runtime_case_id,
                case_dir=case_dir,
                dsl_builder_slm_config=local_slm_config,
            )
        except Exception as exc:  # noqa: BLE001 - isolate startup failures.
            _mark_case_failure(
                record=record,
                runtime_case_id=runtime_case_id,
                case_dir=case_dir,
                exc=exc,
                started_by_case=started_by_case,
                completed_records=completed_records,
                progress_bar=progress_bar,
                progress_lock=progress_lock,
            )
            startup_profile["preprocess_failed_cases"] += 1
            continue

        _update_record_from_prepared_case(record, prepared)
        _write_startup_artifacts(prepared)
        startup_profile["loaded_cases"] += 1
        spec = TableReasoningCaseSpec(
            case_id=prepared.runtime_case_id,
            task_dsl=prepared.task.task_dsl,
            base_dir=prepared.task.base_dir,
            preprocess_result=prepared.preprocess_result,
            answer_key=f"answer_{prepared.sample_index + 1}",
            metadata={
                "sample_index": prepared.sample_index,
                "dataset_id": prepared.sampled_case["dataset_id"],
                "case_id": prepared.case_id,
                "case_index": prepared.sampled_case.get("case_index"),
                "answer_type": prepared.answer_type,
                "qtype": prepared.qtype,
                "qsubtype": prepared.qsubtype,
                "dsl_builder_mode": prepared.dsl_builder_mode,
                "task_answer_type": prepared.task_answer_type,
            },
        )
        specs_by_source_file[prepared.source_file].append(spec)

    def on_case_result(case_result: CaseResult) -> None:
        with progress_lock:
            record = records_by_case[case_result.case_id]
            records_by_answer[case_result.answer_key] = record
            _update_record_from_case_result(record, case_result)
            record["elapsed_seconds"] = (
                time.perf_counter() - started_by_case[case_result.case_id]
            )
            write_json(
                output_dir / "cases" / case_result.case_id / "case_result.json",
                record,
            )
            completed_records.append(record)
            if progress_bar is not None:
                progress_bar.update(completed_records)

    system_results = _run_system_groups(
        spec_groups=list(specs_by_source_file.values()),
        remote_config=remote_config,
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
        for answer_key, task_item in system_result.task_items.items():
            record = records_by_answer.get(answer_key) or records_by_case.get(task_item.case_id)
            if record is None:
                continue
            record["answer_key"] = answer_key
            record["current_sql"] = task_item.current_sql
            record.setdefault("initial_sql", None)
            record["retry_count"] = task_item.retry_count
            record["task_status"] = task_item.status
            write_json(
                output_dir / "cases" / task_item.case_id / "case_result.json",
                record,
            )
    system_profile = _merge_system_profiles(
        [item.profile for item in system_results]
    )
    startup_profile["startup_seconds"] = time.perf_counter() - startup_started
    system_profile["eval_startup"] = startup_profile
    return list(records_by_case.values()), system_profile


def _prepare_case(
    *,
    tablebench_root: Path,
    sampled_case: dict[str, Any],
    sample_index: int,
    runtime_case_id: str,
    case_dir: Path,
    dsl_builder_slm_config: dict[str, Any] | None,
) -> _PreparedTableBenchCase:
    task = load_tablebench_task(
        tablebench_root=tablebench_root,
        dataset_id=sampled_case["dataset_id"],
        case_id=sampled_case["case_id"],
        dsl_builder_slm_config=dsl_builder_slm_config,
    )
    preprocess_result = preprocess_task_dsl(task.task_dsl, base_dir=task.base_dir)
    source_file = _source_file_from_task(task)
    return _PreparedTableBenchCase(
        sampled_case=sampled_case,
        sample_index=sample_index,
        runtime_case_id=runtime_case_id,
        case_id=sampled_case["case_id"],
        case_dir=case_dir,
        task=task,
        preprocess_result=preprocess_result,
        expected_raw=task.metadata.get("expected_answer"),
        answer_type=task.metadata["case"].get("type"),
        qtype=task.metadata.get("qtype"),
        qsubtype=task.metadata.get("qsubtype"),
        source_file=source_file,
        dsl_builder_mode=str(
            task.metadata.get("dsl_builder", {}).get("mode")
            or TABLEBENCH_DSL_MODE_BUILDER_AGENT
        ),
        task_answer_type=task.task_dsl.get("answer", {}).get("type"),
        dsl_builder=dict(task.metadata.get("dsl_builder", {})),
    )


def _run_system_groups(
    *,
    spec_groups: list[list[TableReasoningCaseSpec]],
    remote_config: dict[str, Any],
    local_slm_config: dict[str, Any] | None,
    remote_batch_size: int,
    remote_concurrency: int,
    max_parallel_execution_units: int,
    max_parallel_slm_node_jobs: int,
    max_parallel_slm_sequences: int,
    max_pending_slm_sequences: int,
    max_retries: int,
    validation_mode: str,
    max_workers: int | None,
    case_result_callback: Callable[[CaseResult], None],
    profile_baseline: bool,
    records_by_case: dict[str, dict[str, Any]],
    completed_records: list[dict[str, Any]],
    started_by_case: dict[str, float],
    output_dir: Path,
    progress_bar: Any | None,
    progress_lock: Lock,
) -> list[Any]:
    all_specs = [spec for group in spec_groups for spec in group]
    if not all_specs:
        return []

    def run_all() -> Any:
        return run_table_reasoning_system(
            case_specs=all_specs,
            remote_config=remote_config,
            local_slm_config=local_slm_config,
            remote_batch_size=remote_batch_size,
            remote_concurrency=remote_concurrency,
            max_parallel_execution_units=max_parallel_execution_units,
            max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
            max_parallel_slm_sequences=max_parallel_slm_sequences,
            max_pending_slm_sequences=max_pending_slm_sequences,
            max_retries=max_retries,
            validation_mode=validation_mode,
            case_result_callback=case_result_callback,
            profile_baseline=profile_baseline,
        )

    del max_workers
    try:
        return [run_all()]
    except Exception as exc:  # noqa: BLE001
        _fail_group(
            all_specs,
            exc=exc,
            records_by_case=records_by_case,
            completed_records=completed_records,
            started_by_case=started_by_case,
            output_dir=output_dir,
            progress_bar=progress_bar,
            progress_lock=progress_lock,
        )
        return []


def _merge_system_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    stages: dict[str, dict[str, Any]] = {}
    counters: Counter[str] = Counter()
    for profile in profiles:
        for name, stage in profile.get("stages", {}).items():
            merged = stages.setdefault(
                name,
                {"calls": 0, "items": 0, "total_seconds": 0.0},
            )
            merged["calls"] += int(stage.get("calls", 0) or 0)
            merged["items"] += int(stage.get("items", 0) or 0)
            merged["total_seconds"] += float(stage.get("total_seconds", 0.0) or 0.0)
        counters.update(profile.get("counters", {}))
    for stage in stages.values():
        calls = stage["calls"]
        stage["average_seconds"] = stage["total_seconds"] / calls if calls else 0.0
    validation_modes = sorted(
        {
            str(profile.get("summary", {}).get("validation_mode"))
            for profile in profiles
            if profile.get("summary", {}).get("validation_mode")
        }
    )
    summary = {
        "remote_calls": counters.get("supervisor_decompose_calls", 0)
        + counters.get("supervisor_synthesis_calls", 0)
        + counters.get("supervisor_repair_calls", 0),
        "local_slm_calls": counters.get("local_slm_calls", 0)
        or counters.get("executor_local_slm_steps", 0),
        "merged_plan_count": counters.get("merged_plan_count", 0),
        "reused_nodes": counters.get("reused_nodes", 0),
        "parallel_system_instances": len(profiles),
        "remote_token_usage": _token_usage_from_counters(counters, "remote"),
        "local_slm_token_usage": _token_usage_from_counters(counters, "local_slm"),
    }
    if validation_modes:
        summary["validation_mode"] = (
            validation_modes[0] if len(validation_modes) == 1 else validation_modes
        )
    baseline_executor = stages.get("baseline_executor", {}).get("total_seconds", 0.0)
    executor_seconds = stages.get("executor", {}).get("total_seconds", 0.0)
    if baseline_executor:
        summary["local_executor_speedup"] = (
            baseline_executor / executor_seconds if executor_seconds else None
        )
    return {
        "stages": dict(sorted(stages.items())),
        "counters": dict(sorted(counters.items())),
        "summary": summary,
    }


def _token_usage_from_counters(counters: Counter[str], prefix: str) -> dict[str, int]:
    return {
        key: int(counters.get(f"{prefix}_{key}", 0) or 0)
        for key in TOKEN_KEYS
    }


def _update_record_from_prepared_case(
    record: dict[str, Any],
    prepared: _PreparedTableBenchCase,
) -> None:
    metric = tablebench_metric_name(prepared.qtype, prepared.qsubtype)
    record.update(
        {
            "answer_type": prepared.answer_type,
            "task_answer_type": prepared.task_answer_type,
            "qtype": prepared.qtype,
            "qsubtype": prepared.qsubtype,
            "dsl_builder_mode": prepared.dsl_builder_mode,
            "dsl_builder": json_ready(_dsl_builder_record_summary(prepared.dsl_builder)),
            "question": prepared.task.task_dsl["question"],
            "expected_raw": prepared.expected_raw,
            "expected_standard_text": json_ready(prepared.expected_raw),
            "tablebench_metric": metric,
            "parse_ok": True,
        }
    )


def _update_record_from_case_result(
    record: dict[str, Any],
    case_result: CaseResult,
) -> None:
    score = score_tablebench_answer(
        expected=record.get("expected_raw"),
        actual=case_result.answer,
        qtype=record.get("qtype"),
        qsubtype=record.get("qsubtype"),
    )
    record.update(
        {
            "answer_key": case_result.answer_key,
            "runtime_ok": case_result.ok,
            "final_answer": json_ready(case_result.answer),
            "final_answer_preview": preview(case_result.answer),
            "final_answer_standard_text": score.actual,
            "answer_correct": bool(case_result.ok and score.correct),
            "tablebench_metric": score.metric,
            "tablebench_score": score.score if case_result.ok else 0.0,
            "round_count": case_result.retry_count + 1,
            "retry_count": case_result.retry_count,
            "retry_exhausted": (
                not case_result.ok
                and isinstance(case_result.error, dict)
                and case_result.error.get("type") == "RetryLimitExceeded"
            ),
            "error": case_result.error,
            "rounds": [],
        }
    )


def _mark_case_failure(
    *,
    record: dict[str, Any],
    runtime_case_id: str,
    case_dir: Path,
    exc: Exception,
    started_by_case: dict[str, float],
    completed_records: list[dict[str, Any]],
    progress_bar: Any | None,
    progress_lock: Lock,
) -> None:
    record["error"] = format_error(exc)
    record["elapsed_seconds"] = time.perf_counter() - started_by_case[runtime_case_id]
    write_json(case_dir / "case_error.json", record["error"])
    write_json(case_dir / "case_result.json", record)
    with progress_lock:
        completed_records.append(record)
        if progress_bar is not None:
            progress_bar.update(completed_records)


def _fail_group(
    group: list[TableReasoningCaseSpec],
    *,
    exc: Exception,
    records_by_case: dict[str, dict[str, Any]],
    completed_records: list[dict[str, Any]],
    started_by_case: dict[str, float],
    output_dir: Path,
    progress_bar: Any | None,
    progress_lock: Lock,
) -> None:
    error = format_error(exc)
    with progress_lock:
        for spec in group:
            record = records_by_case.get(spec.case_id)
            if record is None or record.get("answer_key"):
                continue
            record["runtime_ok"] = False
            record["tablebench_score"] = 0.0
            record["error"] = error
            record["elapsed_seconds"] = time.perf_counter() - started_by_case[spec.case_id]
            write_json(output_dir / "cases" / spec.case_id / "case_error.json", error)
            write_json(output_dir / "cases" / spec.case_id / "case_result.json", record)
            completed_records.append(record)
        if progress_bar is not None:
            progress_bar.update(completed_records)


def _write_startup_artifacts(prepared: _PreparedTableBenchCase) -> None:
    write_json(prepared.case_dir / "task_dsl.json", prepared.task.task_dsl)
    write_json(prepared.case_dir / "local_dsl.json", prepared.preprocess_result["local_dsl"])
    write_json(prepared.case_dir / "remote_dsl.json", prepared.preprocess_result["remote_dsl"])
    write_json(prepared.case_dir / "context.json", prepared.preprocess_result["context"])
    write_json(prepared.case_dir / "dsl_builder.json", json_ready(prepared.dsl_builder))


def _dsl_builder_record_summary(dsl_builder: dict[str, Any]) -> dict[str, Any]:
    diagnostics = dsl_builder.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    tool_call = dsl_builder.get("tool_call")
    if not isinstance(tool_call, dict):
        tool_call = {}
    return {
        "mode": dsl_builder.get("mode"),
        "source": dsl_builder.get("source"),
        "tool": tool_call.get("tool") or diagnostics.get("tool"),
        "task_answer_type": dsl_builder.get("task_answer_type"),
        "target_column": diagnostics.get("target_column"),
        "token_usage": dsl_builder.get("token_usage"),
    }


def _sum_dsl_builder_token_usage(records: list[dict[str, Any]]) -> dict[str, int]:
    total = {key: 0 for key in TOKEN_KEYS}
    for record in records:
        usage = (record.get("dsl_builder") or {}).get("token_usage")
        if not isinstance(usage, dict):
            continue
        for key in TOKEN_KEYS:
            total[key] += int(usage.get(key, 0) or 0)
    return total


def build_summary(
    *,
    records: list[dict[str, Any]],
    output_dir: Path,
    remote_config: dict[str, Any],
    local_slm_config: dict[str, Any] | None,
    selected_cases: list[dict[str, Any]],
    elapsed_seconds: float,
    worker_count: int,
    max_retries: int,
    validation_mode: str,
    include_visualization: bool,
    seed: int,
    sample_size: int | None,
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
    sql_repair_case_ids = [
        record.get("runtime_case_id") or record["case_id"]
        for record in records
        if any(round_item.get("prompt_has_sql_repair") for round_item in record.get("rounds", []))
    ]
    initial_execution_failure_ids = [
        record.get("runtime_case_id") or record["case_id"]
        for record in records
        if record.get("rounds") and not record["rounds"][0].get("execution_ok")
    ]
    score_sum = sum(float(record.get("tablebench_score") or 0.0) for record in records)
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
        if (record.get("dsl_builder") or {}).get("mode")
        == TABLEBENCH_DSL_MODE_BUILDER_AGENT
    )
    mismatches_by_type = Counter(
        record.get("answer_type")
        for record in records
        if record.get("runtime_ok") and not record.get("answer_correct")
    )
    remote_token_usage = system_profile.get("summary", {}).get("remote_token_usage", {})
    summary_profile = system_profile.get("summary", {})
    remote_token_summary = normalize_remote_token_usage(remote_token_usage)
    local_slm_token_summary = normalize_remote_token_usage(
        summary_profile.get("local_slm_token_usage", {})
    )
    remote_cost_estimate = estimate_openai_text_cost(
        remote_token_summary,
        remote_config=remote_config,
        pricing_model=remote_cost_model,
    )
    return {
        "run_name": output_dir.name,
        "stage": "tablebench_eval",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workflow": "table_reasoning.analyze",
        "tablebench_standard": {
            "visualization_included": include_visualization,
            "visualization_default": "skipped",
            "metrics": {
                "FactChecking": "EM",
                "NumericalReasoning": "EM",
                "DataAnalysis.CorrelationAnalysis": "EM_with_error_10",
                "DataAnalysis.TrendForecasting": "EM_with_error_10",
                "DataAnalysis.StatisticalAnalysis": "EM_with_error_10",
                "DataAnalysis.ImpactAnalysis": "EM",
                "DataAnalysis.other": "ROUGE-L",
            },
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
        "dsl_builder_mode": TABLEBENCH_DSL_MODE_BUILDER_AGENT,
        "remote_llm": _config_summary(remote_config),
        "local_slm": _config_summary(local_slm_config),
        "remote_calls": summary_profile.get("remote_calls", 0),
        "local_slm_calls": summary_profile.get("local_slm_calls", 0),
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
        "sql_repair_cases": len(sql_repair_case_ids),
        "sql_repair_case_ids": sql_repair_case_ids,
        "initial_execution_failures": len(initial_execution_failure_ids),
        "initial_execution_failure_case_ids": initial_execution_failure_ids,
        "standard_score_average": safe_divide(score_sum, total),
        "scores_by_metric": _score_groups(records, "tablebench_metric"),
        "scores_by_qtype": _score_groups(records, "qtype"),
        "scores_by_qsubtype": _score_groups(records, "qsubtype"),
        "answer_types": _counter_as_strings(answer_types),
        "task_answer_types": _counter_as_strings(task_answer_types),
        "task_answer_type_matches": task_answer_type_matches,
        "task_answer_type_accuracy": safe_divide(task_answer_type_matches, total),
        "dsl_builder_modes": _counter_as_strings(dsl_builder_modes),
        "builder_agent_calls": builder_agent_calls,
        "builder_agent_token_usage": builder_agent_token_usage,
        "mismatches_by_type": _counter_as_strings(mismatches_by_type),
        "qtypes": _string_counter(records, "qtype"),
        "qsubtypes": _string_counter(records, "qsubtype"),
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


def _score_groups(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(key) or "unknown")].append(record)
    result = {}
    for group_key, group in sorted(groups.items()):
        total = len(group)
        runtime_successes = sum(1 for record in group if record.get("runtime_ok"))
        correct = sum(1 for record in group if record.get("answer_correct"))
        score_sum = sum(float(record.get("tablebench_score") or 0.0) for record in group)
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


def mismatch_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_index": record["sample_index"],
        "runtime_case_id": record.get("runtime_case_id"),
        "dataset_id": record["dataset_id"],
        "case_id": record["case_id"],
        "qtype": record.get("qtype"),
        "qsubtype": record.get("qsubtype"),
        "answer_type": record.get("answer_type"),
        "task_answer_type": record.get("task_answer_type"),
        "dsl_builder_mode": record.get("dsl_builder_mode"),
        "metric": record.get("tablebench_metric"),
        "score": record.get("tablebench_score"),
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
        "qtype": sampled_case.get("qtype"),
        "qsubtype": sampled_case.get("qsubtype"),
        "tablebench_metric": tablebench_metric_name(
            sampled_case.get("qtype"),
            sampled_case.get("qsubtype"),
        ),
        "tablebench_score": 0.0,
        "question": None,
        "parse_ok": False,
        "runtime_ok": False,
        "answer_correct": False,
        "expected_raw": None,
        "expected_standard_text": None,
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
    """Return an evaluation-unique case id while preserving the original id."""

    dataset_id = str(sampled_case["dataset_id"])
    case_id = str(sampled_case["case_id"])
    return f"sample_{sample_index:05d}__{dataset_id}__{case_id}"


def select_tablebench_cases(
    *,
    tablebench_root: Path,
    max_cases: int | None,
    case_ids: set[str],
    dataset_id: str | None,
    qtypes: set[str],
    qsubtypes: set[str],
    include_visualization: bool,
    sample_size: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if max_cases == 0:
        return []
    cases = list_tablebench_cases(tablebench_root)
    if not include_visualization:
        cases = [case for case in cases if case.get("qtype") != "Visualization"]
    if dataset_id is not None:
        cases = [case for case in cases if case["dataset_id"] == dataset_id]
    if case_ids:
        cases = [case for case in cases if case["case_id"] in case_ids]
    if qtypes:
        cases = [case for case in cases if str(case.get("qtype") or "") in qtypes]
    if qsubtypes:
        cases = [case for case in cases if str(case.get("qsubtype") or "") in qsubtypes]
    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        rng = random.Random(seed)
        cases = rng.sample(cases, min(sample_size, len(cases)))
    if max_cases is not None:
        cases = cases[:max_cases]
    return cases


def list_tablebench_cases(tablebench_root: Path) -> list[dict[str, Any]]:
    cases = []
    for dataset_dir in iter_tablebench_dataset_dirs(tablebench_root):
        cases_path = dataset_dir / "cases.jsonl"
        if not cases_path.is_file():
            continue
        for case_index, case in enumerate(read_cases(cases_path)):
            cases.append(
                {
                    "dataset_id": dataset_dir.name,
                    "case_id": case["case_id"],
                    "case_index": case_index,
                    "answer_type": case.get("type"),
                    "qtype": case.get("qtype"),
                    "qsubtype": case.get("qsubtype"),
                }
            )
    return cases


def _source_file_from_task(task: TablebenchTask) -> str:
    source = _single_task_source(task)
    path = Path(source["file"]).expanduser()
    if not path.is_absolute():
        path = task.base_dir / path
    return str(path.resolve())


def _single_task_source(task: TablebenchTask) -> dict[str, Any]:
    sources = task.task_dsl.get("sources", [])
    if len(sources) != 1:
        raise ValueError("TableBench table reasoning eval requires one table source")
    return sources[0]


def _config_summary(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        "provider": config.get("provider"),
        "api_type": config.get("api_type"),
        "base_url": config.get("base_url"),
        "model": config.get("model"),
        "agent_loop_max_iterations": config.get("agent_loop_max_iterations"),
        "disable_agent_loop": config.get("disable_agent_loop"),
        "slm_scheduler": config.get("slm_scheduler", "tptt"),
    }


def _slm_scheduler_summary(config: dict[str, Any] | None) -> str:
    if not isinstance(config, dict):
        return "tptt"
    return str(config.get("slm_scheduler") or "tptt")
