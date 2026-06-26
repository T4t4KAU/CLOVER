"""TableBench non-visual evaluation for CLOVER table reasoning."""

from __future__ import annotations

import json
import random
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd

from benchmarks.costing import estimate_openai_text_cost, normalize_remote_token_usage
from benchmarks.tablebench.adapter import (
    TABLEBENCH_DSL_MODE_BUILDER_AGENT,
    iter_tablebench_dataset_dirs,
    read_cases,
    write_json,
)
from benchmarks.tablebench.download import TABLEBENCH_REASONING_QTYPES
from benchmarks.tablebench.metrics import score_tablebench_answer, tablebench_metric_name
from benchmarks.utils import (
    build_brief_summary,
    display_path,
    format_error,
    json_ready,
    preview,
    safe_divide,
    write_jsonl,
)
from benchmarks.warnings import suppress_benchmark_warnings
from clover.config import runtime_feature_flags
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


def run_tablebench_eval(
    *,
    tablebench_root: Path,
    output_dir: Path,
    remote_config: dict[str, Any],
    synthesize_config: dict[str, Any] | None = None,
    local_slm_config: dict[str, Any] | None = None,
    max_cases: int | None = None,
    case_ids: set[str] | None = None,
    dataset_id: str | None = None,
    qtypes: set[str] | None = None,
    qsubtypes: set[str] | None = None,
    include_visualization: bool = False,
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
                "TableBench eval requires local_slm_config for local SLM repair/synthesis"
            )
        validation_mode = str(validation_mode or "none").strip().lower()
        progress_bar = progress_factory(len(selected_cases)) if progress_factory else None
        try:
            records, system_profile = _run_tablebench_cases(
                tablebench_root=tablebench_root,
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
                tablebench_root=tablebench_root,
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
    system_profile = _merge_system_profiles(
        [item.profile for item in system_results]
    )
    startup_profile["startup_seconds"] = time.perf_counter() - startup_started
    system_profile["eval_startup"] = startup_profile
    return list(records_by_case.values()), system_profile


def _builder_case_spec(
    *,
    tablebench_root: Path,
    sampled_case: dict[str, Any],
    sample_index: int,
    runtime_case_id: str,
) -> TableReasoningCaseSpec:
    dataset_dir = tablebench_root / sampled_case["dataset_id"]
    hints = _tablebench_hints(sampled_case)
    builder = {
        "kind": TABLEBENCH_DSL_MODE_BUILDER_AGENT,
        "question": sampled_case["question"],
        "table_path": "table.csv",
        "source_file": "table.csv",
        "answer_type": sampled_case.get("answer_type"),
        "task_type": "table_reasoning.analyze",
        "source_id": 0,
        "hints": hints,
    }
    metadata = {
        "sample_index": sample_index,
        "dataset": "tablebench",
        "dataset_id": sampled_case["dataset_id"],
        "case_id": sampled_case["case_id"],
        "case_index": sampled_case.get("case_index"),
        "answer_type": sampled_case.get("answer_type"),
        "qtype": sampled_case.get("qtype"),
        "qsubtype": sampled_case.get("qsubtype"),
        "expected_answer": sampled_case.get("expected_answer"),
        "question": sampled_case.get("question"),
        "dsl_builder_mode": TABLEBENCH_DSL_MODE_BUILDER_AGENT,
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


def _resolve_table_paths(case_specs: list[TableReasoningCaseSpec]) -> list[Path]:
    """Resolve unique absolute table paths from case specs for pre-warming."""

    paths: set[Path] = set()
    for spec in case_specs:
        base_dir = Path(spec.base_dir).expanduser()
        source_file = "table.csv"
        builder = spec.builder or spec.metadata.get("builder")
        if isinstance(builder, dict):
            source_file = str(builder.get("source_file") or "table.csv")
            table_path = str(builder.get("table_path") or source_file)
        else:
            table_path = source_file
        path = Path(table_path).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        paths.add(path.resolve())
    return list(paths)


def _warm_table_cache(case_specs: list[TableReasoningCaseSpec]) -> dict[str, pd.DataFrame]:
    """Pre-load all unique table CSV files into a shared cache to avoid
    repeated disk I/O across batches."""

    table_cache: dict[str, pd.DataFrame] = {}
    for path in _resolve_table_paths(case_specs):
        if not path.is_file():
            continue
        try:
            table_cache[str(path)] = pd.read_csv(path, low_memory=False)
        except Exception:
            pass
    return table_cache


def _run_system_groups(
    *,
    spec_groups: list[list[TableReasoningCaseSpec]],
    remote_config: dict[str, Any],
    synthesize_config: dict[str, Any] | None,
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
        table_cache = _warm_table_cache(all_specs) if all_specs else {}
        return run_table_reasoning_system(
            case_specs=all_specs,
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
            case_result_callback=case_result_callback,
            profile_baseline=profile_baseline,
            table_cache=table_cache,
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
        "edge_local_review_calls": counters.get("edge_local_review_calls", 0),
        "edge_local_review_hits": counters.get("edge_local_review_hits", 0),
        "edge_local_review_escalations": counters.get(
            "edge_local_review_escalations",
            0,
        ),
        "merged_plan_count": counters.get("merged_plan_count", 0),
        "reused_nodes": counters.get("reused_nodes", 0),
        "parallel_system_instances": len(profiles),
        "remote_token_usage": _token_usage_from_counters(counters, "remote"),
        "supervisor_decompose_token_usage": _token_usage_from_counters(
            counters,
            "supervisor_decompose",
        ),
        "supervisor_synthesis_token_usage": _token_usage_from_counters(
            counters,
            "supervisor_synthesis",
        ),
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


def _update_record_from_task_item(
    record: dict[str, Any],
    task_item: Any,
) -> None:
    metadata = task_item.metadata if isinstance(task_item.metadata, dict) else {}
    dsl_builder = metadata.get("dsl_builder")
    if isinstance(dsl_builder, dict):
        record["dsl_builder"] = json_ready(_dsl_builder_record_summary(dsl_builder))
        record["dsl_builder_mode"] = dsl_builder.get("mode") or record.get(
            "dsl_builder_mode"
        )
    record.update(
        {
            "task_answer_type": task_item.answer_type,
            "question": task_item.question,
            "parse_ok": True,
        }
    )


def _update_record_from_case_result(
    record: dict[str, Any],
    case_result: CaseResult,
) -> None:
    metadata = case_result.metadata if isinstance(case_result.metadata, dict) else {}
    _update_record_from_result_metadata(record, metadata)
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
        ("qtype", "qtype"),
        ("qsubtype", "qsubtype"),
        ("question", "question"),
        ("expected_answer", "expected_raw"),
        ("task_answer_type", "task_answer_type"),
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


def _write_case_trace_artifacts(case_dir: Path, case_result: Any) -> None:
    """Write trace artifacts from case_result.metadata in real-time.

    Called from on_case_result callback so traces are available immediately
    after each case finishes, not only after the full run completes.
    """
    metadata = getattr(case_result, "metadata", None) or {}
    decompose_trace = metadata.get("decompose_trace")
    if isinstance(decompose_trace, dict):
        write_json(case_dir / "decompose.json", json_ready(decompose_trace))
    synthesis_trace = metadata.get("synthesis_trace")
    if isinstance(synthesis_trace, dict):
        write_json(case_dir / "synthesis.json", json_ready(synthesis_trace))
    agent_loop_trace = _extract_agent_loop_trace(metadata)
    if agent_loop_trace:
        write_json(case_dir / "agent_loop.json", json_ready(agent_loop_trace))
    dsl_builder = metadata.get("dsl_builder")
    if isinstance(dsl_builder, dict):
        write_json(case_dir / "dsl_builder.json", json_ready(dsl_builder))
    table_diagnostics = metadata.get("table_diagnostics")
    if isinstance(table_diagnostics, list) and table_diagnostics:
        write_json(case_dir / "execution.json", json_ready(table_diagnostics))


def _write_runtime_task_artifacts(case_dir: Path, task_item: Any) -> None:
    write_json(case_dir / "task_dsl.json", json_ready(task_item.task_dsl))
    write_json(case_dir / "local_dsl.json", json_ready(task_item.local_dsl))
    write_json(case_dir / "remote_dsl.json", json_ready(task_item.remote_dsl))
    write_json(case_dir / "context.json", json_ready(task_item.context))
    dsl_builder = task_item.metadata.get("dsl_builder")
    if isinstance(dsl_builder, dict):
        write_json(case_dir / "dsl_builder.json", json_ready(dsl_builder))
    metadata = getattr(task_item, "metadata", None) or {}
    decompose_trace = metadata.get("decompose_trace")
    if isinstance(decompose_trace, dict):
        write_json(case_dir / "decompose.json", json_ready(decompose_trace))
    synthesis_trace = metadata.get("synthesis_trace")
    if isinstance(synthesis_trace, dict):
        write_json(case_dir / "synthesis.json", json_ready(synthesis_trace))
    agent_loop_trace = _extract_agent_loop_trace(metadata)
    if agent_loop_trace:
        write_json(case_dir / "agent_loop.json", json_ready(agent_loop_trace))


def _extract_agent_loop_trace(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract agent_loop steps (with prompt/response) from table_diagnostics traces."""
    diagnostics = metadata.get("table_diagnostics")
    if not isinstance(diagnostics, list):
        return []
    traces: list[dict[str, Any]] = []
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        execution_result = diagnostic.get("execution_result")
        if not isinstance(execution_result, dict):
            continue
        for trace in execution_result.get("traces", []) or []:
            if not isinstance(trace, dict):
                continue
            agent_loop = trace.get("agent_loop")
            if isinstance(agent_loop, dict):
                traces.append(agent_loop)
    return traces


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


def _dsl_builder_total_tokens(record: dict[str, Any]) -> int:
    usage = (record.get("dsl_builder") or {}).get("token_usage")
    if not isinstance(usage, dict):
        return 0
    return int(usage.get("total_tokens", 0) or 0)


def _max_input_tokens_in_obj(obj: Any) -> int:
    """Recursively find the max input_tokens in any token_usage dict."""
    result = 0
    if isinstance(obj, dict):
        tu = obj.get("token_usage")
        if isinstance(tu, dict):
            inp = int(tu.get("input_tokens") or 0)
            if inp > result:
                result = inp
        for value in obj.values():
            inner = _max_input_tokens_in_obj(value)
            if inner > result:
                result = inner
    elif isinstance(obj, list):
        for value in obj:
            inner = _max_input_tokens_in_obj(value)
            if inner > result:
                result = inner
    return result


def _ctx_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": 0.0, "max": 0, "min": 0, "sum": 0}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "max": max(values),
        "min": min(values),
        "sum": sum(values),
    }


def _per_case_max_context_tokens(
    records: list[dict[str, Any]], output_dir: Path
) -> dict[str, list[int]]:
    """Extract per-case max input_tokens for remote and local models.

    Reads decompose.json/synthesis.json for remote (supervisor) tokens and
    execution.json for local SLM (edge agent) tokens. ``combined`` is the
    element-wise max of remote and local per case.
    """
    remote_max: list[int] = []
    local_max: list[int] = []
    for record in records:
        runtime_case_id = record.get("runtime_case_id")
        if not runtime_case_id:
            remote_max.append(0)
            local_max.append(0)
            continue
        case_dir = output_dir / "cases" / runtime_case_id
        r_max = 0
        l_max = 0
        for trace_file in ("decompose.json", "synthesis.json"):
            trace_path = case_dir / trace_file
            if not trace_path.exists():
                continue
            try:
                trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for round_item in trace_data.get("rounds", []) or []:
                tu = round_item.get("token_usage") or {}
                inp = int(tu.get("input_tokens") or 0)
                if inp > r_max:
                    r_max = inp
        exec_path = case_dir / "execution.json"
        if exec_path.exists():
            try:
                exec_data = json.loads(exec_path.read_text(encoding="utf-8"))
                l_max = _max_input_tokens_in_obj(exec_data)
            except Exception:
                pass
        remote_max.append(r_max)
        local_max.append(l_max)
    combined = [max(r, l) for r, l in zip(remote_max, local_max)]
    return {"remote": remote_max, "local": local_max, "combined": combined}


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
        if _dsl_builder_total_tokens(record) > 0
    )
    mismatches_by_type = Counter(
        record.get("answer_type")
        for record in records
        if record.get("runtime_ok") and not record.get("answer_correct")
    )
    remote_token_usage = system_profile.get("summary", {}).get("remote_token_usage", {})
    summary_profile = system_profile.get("summary", {})
    supervisor_decompose_token_usage = normalize_remote_token_usage(
        summary_profile.get("supervisor_decompose_token_usage", {})
    )
    supervisor_synthesis_token_usage = normalize_remote_token_usage(
        summary_profile.get("supervisor_synthesis_token_usage", {})
    )
    remote_token_summary = normalize_remote_token_usage(remote_token_usage)
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
        "question": sampled_case.get("question"),
        "parse_ok": False,
        "runtime_ok": False,
        "answer_correct": False,
        "expected_raw": sampled_case.get("expected_answer"),
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
    del include_visualization
    cases = [
        case
        for case in cases
        if str(case.get("qtype") or "") in TABLEBENCH_REASONING_QTYPES
    ]
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
                    "question": case.get("question"),
                    "expected_answer": case.get("answer"),
                    "answer_type": case.get("type"),
                    "qtype": case.get("qtype"),
                    "qsubtype": case.get("qsubtype"),
                    "hints": case.get("hints"),
                }
            )
    return cases


def _tablebench_hints(sampled_case: dict[str, Any]) -> dict[str, Any]:
    hints = (
        dict(sampled_case.get("hints") or {})
        if isinstance(sampled_case.get("hints"), dict)
        else {}
    )
    if sampled_case.get("qtype") is not None:
        hints["category"] = sampled_case["qtype"]
    if sampled_case.get("qsubtype") is not None:
        hints["subcategory"] = sampled_case["qsubtype"]
    if sampled_case.get("chart_type") is not None:
        hints["chart_type"] = sampled_case["chart_type"]
    return hints


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
        "ablation_variant": config.get("ablation_variant", "full"),
        "edge_review_mode": config.get("edge_review_mode", "off"),
        "edge_review_proactive": config.get("edge_review_proactive", True),
        "runtime_features": runtime_feature_flags(config),
        "slm_scheduler": config.get("slm_scheduler", "tptt"),
    }


def _slm_scheduler_summary(config: dict[str, Any] | None) -> str:
    if not isinstance(config, dict):
        return "tptt"
    return str(config.get("slm_scheduler") or "tptt")
