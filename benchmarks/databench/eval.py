"""Databench evaluation from initial task DSL through Supervisor synthesis."""

from __future__ import annotations

import copy
import random
import shutil
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from benchmarks.costing import estimate_openai_text_cost, normalize_remote_token_usage
from clover.reasoning_profiles import (
    HINTS_KEY,
    PROFILE_KEY,
)
from clover.resource import preprocess_task_dsl
from clover.runtime import (
    CaseResult,
    TableReasoningCaseSpec,
    run_table_reasoning_system,
)
from benchmarks.databench.adapter import (
    DatabenchTask,
    iter_databench_dataset_dirs,
    load_databench_task,
    read_cases,
    write_json,
)
from benchmarks.databench.static_tool_eval import (
    answers_equal_relaxed,
    display_path,
    format_error,
    json_ready,
    normalize_answer,
    preview,
    safe_divide,
    write_jsonl,
)
from benchmarks.warnings import suppress_benchmark_warnings


TOKEN_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)


def run_databench_eval(
    *,
    databench_root: Path,
    output_dir: Path,
    remote_config: dict[str, Any],
    local_slm_config: dict[str, Any] | None = None,
    max_cases: int | None = None,
    case_ids: set[str] | None = None,
    dataset_id: str | None = None,
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
    preprocess_progress_factory: Callable[[int], Any] | None = None,
) -> dict[str, Any]:
    """Run Databench cases through the full CLOVER workflow."""

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

        selected_cases = select_cases(
            databench_root=databench_root,
            max_cases=max_cases,
            case_ids=case_ids or set(),
            dataset_id=dataset_id,
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
        validation_mode = str(validation_mode or "none").strip().lower()
        worker_count = max(1, int(max_workers or 1))
        progress_bar = progress_factory(len(selected_cases)) if progress_factory else None
        records, system_profile = run_cases_table_pipeline(
            databench_root=databench_root,
            output_dir=output_dir,
            selected_cases=selected_cases,
            remote_config=remote_config,
            local_slm_config=local_slm_config,
            max_retries=max_retries,
            validation_mode=validation_mode,
            remote_batch_size=remote_batch_size,
            remote_concurrency=remote_concurrency,
            max_parallel_execution_units=max_parallel_execution_units,
            max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
            max_parallel_slm_sequences=max_parallel_slm_sequences,
            max_pending_slm_sequences=max_pending_slm_sequences,
            profile_baseline=profile_baseline,
            system_worker_count=worker_count,
            progress_bar=progress_bar,
            preprocess_progress_factory=preprocess_progress_factory,
        )
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
            worker_count=worker_count,
            max_retries=max_retries,
            validation_mode=validation_mode,
            workflow="table_reasoning.query",
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


def run_cases_table_pipeline(
    *,
    databench_root: Path,
    output_dir: Path,
    selected_cases: list[dict[str, Any]],
    remote_config: dict[str, Any],
    local_slm_config: dict[str, Any] | None,
    max_retries: int,
    validation_mode: str,
    remote_batch_size: int,
    remote_concurrency: int,
    max_parallel_execution_units: int,
    max_parallel_slm_node_jobs: int,
    max_parallel_slm_sequences: int,
    max_pending_slm_sequences: int,
    profile_baseline: bool,
    system_worker_count: int,
    progress_bar: Any | None,
    preprocess_progress_factory: Callable[[int], Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run selected Databench cases through the latest table reasoning system."""

    records_by_answer: dict[str, dict[str, Any]] = {}
    records_by_case: dict[str, dict[str, Any]] = {}
    case_specs: list[TableReasoningCaseSpec] = []
    specs_by_source_file: dict[str, list[TableReasoningCaseSpec]] = {}
    completed_records: list[dict[str, Any]] = []
    started_by_case: dict[str, float] = {}
    progress_lock = Lock()
    startup_started = time.perf_counter()
    startup_worker_count = max(1, int(system_worker_count or 1))
    startup_profile: dict[str, Any] = {
        "workers": startup_worker_count,
        "selected_cases": len(selected_cases),
        "loaded_cases": 0,
        "preprocess_cache_hits": 0,
        "preprocess_cache_misses": 0,
        "preprocess_failed_cases": 0,
        "artifact_write_failed_cases": 0,
        "case_load_seconds": 0.0,
        "preprocess_seconds": 0.0,
        "artifact_write_seconds": 0.0,
        "startup_seconds": 0.0,
    }

    try:
        for sample_index, sampled_case in enumerate(selected_cases):
            case_dir = output_dir / "cases" / sampled_case["case_id"]
            case_dir.mkdir(parents=True, exist_ok=True)
            started_by_case[sampled_case["case_id"]] = time.perf_counter()
            record = base_case_record(
                sampled_case=sampled_case,
                sample_index=sample_index,
                case_dir=case_dir,
            )
            records_by_case[sampled_case["case_id"]] = record

        load_started = time.perf_counter()
        loaded_cases = _load_table_cases_parallel(
            databench_root=databench_root,
            selected_cases=selected_cases,
            output_dir=output_dir,
            worker_count=startup_worker_count,
            records_by_case=records_by_case,
            started_by_case=started_by_case,
            completed_records=completed_records,
            progress_bar=progress_bar,
            progress_lock=progress_lock,
        )
        startup_profile["case_load_seconds"] = time.perf_counter() - load_started
        startup_profile["loaded_cases"] = len(loaded_cases)

        preprocess_started = time.perf_counter()
        prepared_cases = _preprocess_table_cases_parallel(
            loaded_cases=loaded_cases,
            worker_count=startup_worker_count,
            records_by_case=records_by_case,
            started_by_case=started_by_case,
            completed_records=completed_records,
            progress_bar=progress_bar,
            progress_lock=progress_lock,
            startup_profile=startup_profile,
            preprocess_progress_factory=preprocess_progress_factory,
        )
        startup_profile["preprocess_seconds"] = time.perf_counter() - preprocess_started

        write_started = time.perf_counter()
        writable_cases = _write_table_startup_artifacts_parallel(
            prepared_cases=prepared_cases,
            worker_count=startup_worker_count,
            records_by_case=records_by_case,
            started_by_case=started_by_case,
            completed_records=completed_records,
            progress_bar=progress_bar,
            progress_lock=progress_lock,
            startup_profile=startup_profile,
        )
        startup_profile["artifact_write_seconds"] = time.perf_counter() - write_started

        for prepared_case in sorted(writable_cases, key=lambda item: item.sample_index):
            record = records_by_case[prepared_case.case_id]
            record.update(
                {
                    "answer_type": prepared_case.answer_type,
                    "question": prepared_case.task.task_dsl["question"],
                    "expected_raw": prepared_case.expected_raw,
                    "expected_normalized": json_ready(prepared_case.expected_normalized),
                    "parse_ok": True,
                }
            )
            case_specs.append(
                spec := TableReasoningCaseSpec(
                    case_id=prepared_case.case_id,
                    task_dsl=prepared_case.task.task_dsl,
                    base_dir=prepared_case.task.base_dir,
                    preprocess_result=prepared_case.preprocess_result,
                    answer_key=f"answer_{prepared_case.sample_index + 1}",
                    metadata={
                        "sample_index": prepared_case.sample_index,
                        "dataset_id": prepared_case.sampled_case["dataset_id"],
                        "case_index": prepared_case.sampled_case.get("case_index"),
                        "answer_type": prepared_case.answer_type,
                    },
                )
            )
            specs_by_source_file.setdefault(prepared_case.source_file, []).append(spec)
        startup_profile["startup_seconds"] = time.perf_counter() - startup_started
        if progress_bar is not None:
            progress_bar.update(completed_records)

        def on_case_result(case_result: CaseResult) -> None:
            with progress_lock:
                record = records_by_case[case_result.case_id]
                records_by_answer[case_result.answer_key] = record
                _update_record_from_table_case_result(record, case_result)
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

        system_results = _run_table_system_groups(
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
            system_worker_count=system_worker_count,
            case_result_callback=on_case_result,
            profile_baseline=profile_baseline,
            records_by_case=records_by_case,
            final_records=completed_records,
            started_by_case=started_by_case,
            output_dir=output_dir,
            progress_bar=progress_bar,
            progress_lock=progress_lock,
        )

        for system_result in system_results:
            for answer_key, task_item in system_result.task_items.items():
                record = (
                    records_by_answer.get(answer_key)
                    or records_by_case.get(task_item.case_id)
                )
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
        system_profile = _merge_system_profiles([item.profile for item in system_results])
        system_profile["eval_startup"] = startup_profile
        return list(records_by_case.values()), system_profile
    finally:
        if progress_bar is not None:
            progress_bar.close()


@dataclass(frozen=True)
class _LoadedTableCase:
    sampled_case: dict[str, Any]
    sample_index: int
    case_id: str
    case_dir: Path
    task: DatabenchTask
    answer_type: str
    expected_raw: Any
    expected_normalized: Any
    source_file: str


@dataclass(frozen=True)
class _PreparedTableCase:
    sampled_case: dict[str, Any]
    sample_index: int
    case_id: str
    case_dir: Path
    task: DatabenchTask
    answer_type: str
    expected_raw: Any
    expected_normalized: Any
    source_file: str
    preprocess_result: dict[str, Any]


def _load_table_cases_parallel(
    *,
    databench_root: Path,
    selected_cases: list[dict[str, Any]],
    output_dir: Path,
    worker_count: int,
    records_by_case: dict[str, dict[str, Any]],
    started_by_case: dict[str, float],
    completed_records: list[dict[str, Any]],
    progress_bar: Any | None,
    progress_lock: Lock,
) -> list[_LoadedTableCase]:
    if not selected_cases:
        return []

    loaded_cases: list[_LoadedTableCase] = []
    max_workers = max(1, min(worker_count, len(selected_cases)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _load_table_case,
                databench_root=databench_root,
                sampled_case=sampled_case,
                sample_index=sample_index,
                case_dir=output_dir / "cases" / sampled_case["case_id"],
            ): sampled_case
            for sample_index, sampled_case in enumerate(selected_cases)
        }
        for future in as_completed(futures):
            sampled_case = futures[future]
            case_id = sampled_case["case_id"]
            case_dir = output_dir / "cases" / case_id
            record = records_by_case[case_id]
            try:
                loaded_case = future.result()
            except Exception as exc:  # noqa: BLE001 - isolate case startup failures.
                _mark_table_startup_failure(
                    record=record,
                    case_id=case_id,
                    case_dir=case_dir,
                    exc=exc,
                    started_by_case=started_by_case,
                    completed_records=completed_records,
                    progress_bar=progress_bar,
                    progress_lock=progress_lock,
                )
                continue
            _update_record_from_loaded_table_case(record, loaded_case, parse_ok=False)
            loaded_cases.append(loaded_case)
    return loaded_cases


def _load_table_case(
    *,
    databench_root: Path,
    sampled_case: dict[str, Any],
    sample_index: int,
    case_dir: Path,
) -> _LoadedTableCase:
    task = load_databench_task(
        databench_root=databench_root,
        dataset_id=sampled_case["dataset_id"],
        case_id=sampled_case["case_id"],
    )
    answer_type = task.metadata["case"].get("type") or task.task_dsl["answer"]["type"]
    expected_raw = task.metadata.get("expected_answer")
    return _LoadedTableCase(
        sampled_case=sampled_case,
        sample_index=sample_index,
        case_id=sampled_case["case_id"],
        case_dir=case_dir,
        task=task,
        answer_type=answer_type,
        expected_raw=expected_raw,
        expected_normalized=normalize_answer(expected_raw, answer_type),
        source_file=_source_file_from_task(task),
    )


def _preprocess_table_cases_parallel(
    *,
    loaded_cases: list[_LoadedTableCase],
    worker_count: int,
    records_by_case: dict[str, dict[str, Any]],
    started_by_case: dict[str, float],
    completed_records: list[dict[str, Any]],
    progress_bar: Any | None,
    progress_lock: Lock,
    startup_profile: dict[str, Any],
    preprocess_progress_factory: Callable[[int], Any] | None = None,
) -> list[_PreparedTableCase]:
    if not loaded_cases:
        return []

    groups_by_source: dict[str, list[_LoadedTableCase]] = {}
    for loaded_case in loaded_cases:
        groups_by_source.setdefault(loaded_case.source_file, []).append(loaded_case)
    startup_profile["preprocess_cache_misses"] = len(groups_by_source)
    startup_profile["preprocess_cache_hits"] = len(loaded_cases) - len(groups_by_source)
    startup_profile["unique_source_files"] = len(groups_by_source)

    prepared_cases: list[_PreparedTableCase] = []
    completed_sources = 0
    failed_cases = 0
    preprocess_progress = (
        preprocess_progress_factory(len(groups_by_source))
        if preprocess_progress_factory
        else None
    )
    if preprocess_progress is not None:
        preprocess_progress.update(0)

    max_workers = max(1, min(worker_count, len(groups_by_source)))
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_preprocess_table_source_group, group): (source_file, group)
                for source_file, group in groups_by_source.items()
            }
            for future in as_completed(futures):
                completed_sources += 1
                _source_file, group = futures[future]
                try:
                    prepared_cases.extend(future.result())
                except Exception as exc:  # noqa: BLE001 - fail all cases sharing that source.
                    startup_profile["preprocess_failed_cases"] += len(group)
                    failed_cases += len(group)
                    for loaded_case in group:
                        _mark_table_startup_failure(
                            record=records_by_case[loaded_case.case_id],
                            case_id=loaded_case.case_id,
                            case_dir=loaded_case.case_dir,
                            exc=exc,
                            started_by_case=started_by_case,
                            completed_records=completed_records,
                            progress_bar=None,
                            progress_lock=progress_lock,
                        )
                if preprocess_progress is not None:
                    preprocess_progress.update(
                        completed_sources,
                        prepared_cases=len(prepared_cases),
                        failed_cases=failed_cases,
                    )
    finally:
        if preprocess_progress is not None:
            preprocess_progress.close()
    return prepared_cases


def _preprocess_table_source_group(group: list[_LoadedTableCase]) -> list[_PreparedTableCase]:
    representative = group[0]
    # DataBench has many questions per table. Extract schema once per table and
    # project the cached local/remote source metadata onto each question.
    cached_preprocess = preprocess_task_dsl(
        representative.task.task_dsl,
        base_dir=representative.task.base_dir,
    )
    return [
        _PreparedTableCase(
            sampled_case=loaded_case.sampled_case,
            sample_index=loaded_case.sample_index,
            case_id=loaded_case.case_id,
            case_dir=loaded_case.case_dir,
            task=loaded_case.task,
            answer_type=loaded_case.answer_type,
            expected_raw=loaded_case.expected_raw,
            expected_normalized=loaded_case.expected_normalized,
            source_file=loaded_case.source_file,
            preprocess_result=_preprocess_result_for_table_case(
                cached_preprocess,
                loaded_case.task,
            ),
        )
        for loaded_case in group
    ]


def _preprocess_result_for_table_case(
    cached_preprocess: dict[str, Any],
    task: DatabenchTask,
) -> dict[str, Any]:
    preprocess_result = copy.deepcopy(cached_preprocess)
    question = task.task_dsl["question"]
    answer = copy.deepcopy(task.task_dsl["answer"])
    for dsl_name in ("local_dsl", "remote_dsl"):
        preprocess_result[dsl_name]["question"] = question
        preprocess_result[dsl_name]["answer"] = copy.deepcopy(answer)
        _copy_optional_task_fields(task.task_dsl, preprocess_result[dsl_name])

    source = _single_task_source(task)
    local_sources = preprocess_result["local_dsl"].get("sources", [])
    if local_sources:
        local_source = local_sources[0]
        local_source["file"] = source["file"]
        local_source["original_id"] = source.get("id")
        source_map = preprocess_result["context"].get("source_map", {})
        mapped_source = source_map.get(local_source["id"])
        if isinstance(mapped_source, dict):
            mapped_source["file"] = source["file"]
            mapped_source["original_id"] = source.get("id")
    return preprocess_result


def _copy_optional_task_fields(source: dict[str, Any], target: dict[str, Any]) -> None:
    for key in (PROFILE_KEY, HINTS_KEY):
        if key in source:
            target[key] = copy.deepcopy(source[key])


def _write_table_startup_artifacts_parallel(
    *,
    prepared_cases: list[_PreparedTableCase],
    worker_count: int,
    records_by_case: dict[str, dict[str, Any]],
    started_by_case: dict[str, float],
    completed_records: list[dict[str, Any]],
    progress_bar: Any | None,
    progress_lock: Lock,
    startup_profile: dict[str, Any],
) -> list[_PreparedTableCase]:
    if not prepared_cases:
        return []

    writable_cases: list[_PreparedTableCase] = []
    max_workers = max(1, min(worker_count, len(prepared_cases)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_write_table_startup_artifacts, prepared_case): prepared_case
            for prepared_case in prepared_cases
        }
        for future in as_completed(futures):
            prepared_case = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - isolate artifact write failures.
                startup_profile["artifact_write_failed_cases"] += 1
                _mark_table_startup_failure(
                    record=records_by_case[prepared_case.case_id],
                    case_id=prepared_case.case_id,
                    case_dir=prepared_case.case_dir,
                    exc=exc,
                    started_by_case=started_by_case,
                    completed_records=completed_records,
                    progress_bar=progress_bar,
                    progress_lock=progress_lock,
                )
                continue
            writable_cases.append(prepared_case)
    return writable_cases


def _write_table_startup_artifacts(prepared_case: _PreparedTableCase) -> None:
    write_json(prepared_case.case_dir / "task_dsl.json", prepared_case.task.task_dsl)
    write_json(
        prepared_case.case_dir / "local_dsl.json",
        prepared_case.preprocess_result["local_dsl"],
    )
    write_json(
        prepared_case.case_dir / "remote_dsl.json",
        prepared_case.preprocess_result["remote_dsl"],
    )
    write_json(
        prepared_case.case_dir / "context.json",
        prepared_case.preprocess_result["context"],
    )


def _update_record_from_loaded_table_case(
    record: dict[str, Any],
    loaded_case: _LoadedTableCase | _PreparedTableCase,
    *,
    parse_ok: bool,
) -> None:
    record.update(
        {
            "answer_type": loaded_case.answer_type,
            "question": loaded_case.task.task_dsl["question"],
            "expected_raw": loaded_case.expected_raw,
            "expected_normalized": json_ready(loaded_case.expected_normalized),
            "parse_ok": parse_ok,
        }
    )


def _mark_table_startup_failure(
    *,
    record: dict[str, Any],
    case_id: str,
    case_dir: Path,
    exc: Exception,
    started_by_case: dict[str, float],
    completed_records: list[dict[str, Any]],
    progress_bar: Any | None,
    progress_lock: Lock,
) -> None:
    record["error"] = format_error(exc)
    record["elapsed_seconds"] = time.perf_counter() - started_by_case[case_id]
    write_json(case_dir / "case_error.json", record["error"])
    write_json(case_dir / "case_result.json", record)
    with progress_lock:
        completed_records.append(record)
        if progress_bar is not None:
            progress_bar.update(completed_records)


def _source_file_from_task(task: DatabenchTask) -> str:
    source = _single_task_source(task)
    path = Path(source["file"]).expanduser()
    if not path.is_absolute():
        path = task.base_dir / path
    return str(path.resolve())


def _single_task_source(task: DatabenchTask) -> dict[str, Any]:
    sources = task.task_dsl.get("sources", [])
    if len(sources) != 1:
        raise ValueError("Databench table reasoning eval requires one table source")
    return sources[0]


def _run_table_system_groups(
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
    system_worker_count: int,
    case_result_callback: Callable[[CaseResult], None],
    profile_baseline: bool,
    records_by_case: dict[str, dict[str, Any]],
    final_records: list[dict[str, Any]],
    started_by_case: dict[str, float],
    output_dir: Path,
    progress_bar: Any | None,
    progress_lock: Lock,
) -> list[Any]:
    """Run all table cases through one shared table system instance."""

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

    del system_worker_count
    try:
        return [run_all()]
    except Exception as exc:  # noqa: BLE001 - isolate system-level failure.
        _fail_table_group(
            all_specs,
            exc=exc,
            records_by_case=records_by_case,
            final_records=final_records,
            started_by_case=started_by_case,
            output_dir=output_dir,
            progress_bar=progress_bar,
            progress_lock=progress_lock,
        )
        return []


def _fail_table_group(
    group: list[TableReasoningCaseSpec],
    *,
    exc: Exception,
    records_by_case: dict[str, dict[str, Any]],
    final_records: list[dict[str, Any]],
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
            record["error"] = error
            record["elapsed_seconds"] = time.perf_counter() - started_by_case[spec.case_id]
            write_json(output_dir / "cases" / spec.case_id / "case_error.json", error)
            write_json(output_dir / "cases" / spec.case_id / "case_result.json", record)
            final_records.append(record)
        if progress_bar is not None:
            progress_bar.update(final_records)


def _source_file_from_preprocess(preprocess_result: dict[str, Any]) -> str:
    source = preprocess_result["local_dsl"]["sources"][0]
    return str(Path(source["path"]).expanduser().resolve())


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


def _update_record_from_table_case_result(
    record: dict[str, Any],
    case_result: CaseResult,
) -> None:
    answer_type = record.get("answer_type")
    actual_normalized = normalize_answer(case_result.answer, answer_type)
    record.update(
        {
            "answer_key": case_result.answer_key,
            "runtime_ok": case_result.ok,
            "final_answer": json_ready(case_result.answer),
            "final_answer_preview": preview(case_result.answer),
            "final_answer_normalized": json_ready(actual_normalized),
            "answer_correct": bool(
                case_result.ok
                and answers_equal_relaxed(
                    record.get("expected_normalized"),
                    actual_normalized,
                    answer_type,
                )
            ),
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
    if isinstance(case_result.error, dict) and case_result.error.get("type") == "SqlParseError":
        record["parse_ok"] = False


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
    seed: int,
    sample_size: int | None,
    cases_index: Path,
    mismatch_cases: Path,
    failure_cases: Path,
    workflow: str = "table_reasoning.query",
    remote_batch_size: int | None = None,
    remote_concurrency: int | None = None,
    max_parallel_execution_units: int | None = None,
    max_parallel_slm_node_jobs: int | None = None,
    max_parallel_slm_sequences: int | None = None,
    max_pending_slm_sequences: int | None = None,
    eval_batch_size: int | None = None,
    profile_baseline: bool | None = None,
    system_profile: dict[str, Any] | None = None,
    remote_cost_model: str | None = None,
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
        record["case_id"] for record in records if record.get("retry_count", 0) > 0
    ]
    sql_repair_case_ids = [
        record["case_id"]
        for record in records
        if any(round_item.get("prompt_has_sql_repair") for round_item in record.get("rounds", []))
    ]
    initial_execution_failure_ids = [
        record["case_id"]
        for record in records
        if record.get("rounds") and not record["rounds"][0].get("execution_ok")
    ]
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
    remote_token_usage = (
        system_profile.get("summary", {}).get("remote_token_usage", {})
        if isinstance(system_profile, dict)
        else {}
    )
    summary_profile = (
        system_profile.get("summary", {})
        if isinstance(system_profile, dict)
        else {}
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
    return {
        "run_name": output_dir.name,
        "stage": "databench_eval",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(selected_cases),
        "requested_sample_size": sample_size,
        "seed": seed,
        "workflow": workflow,
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
        "accuracy_on_successes": safe_divide(correct, runtime_successes),
        "accuracy_on_all_cases": safe_divide(correct, total),
        "retry_cases": len(retry_case_ids),
        "retry_case_ids": retry_case_ids,
        "total_retry_rounds": sum(record.get("retry_count", 0) for record in records),
        "sql_repair_cases": len(sql_repair_case_ids),
        "sql_repair_case_ids": sql_repair_case_ids,
        "initial_execution_failures": len(initial_execution_failure_ids),
        "initial_execution_failure_case_ids": initial_execution_failure_ids,
        "answer_types": _counter_as_strings(answer_types),
        "mismatches_by_type": _counter_as_strings(mismatches_by_type),
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


def _counter_as_strings(counter: Counter[Any]) -> dict[str, int]:
    return dict(sorted((str(key or "unknown"), value) for key, value in counter.items()))


def mismatch_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_index": record["sample_index"],
        "dataset_id": record["dataset_id"],
        "case_id": record["case_id"],
        "answer_type": record.get("answer_type"),
        "question": record.get("question"),
        "expected": record.get("expected_normalized"),
        "actual": record.get("final_answer_normalized"),
        "initial_sql": record.get("initial_sql"),
        "current_sql": record.get("current_sql"),
        "round_count": record.get("round_count"),
        "retry_count": record.get("retry_count"),
        "rounds": record.get("rounds", []),
    }


def base_case_record(
    *,
    sampled_case: dict[str, Any],
    sample_index: int,
    case_dir: Path,
) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "dataset_id": sampled_case["dataset_id"],
        "case_id": sampled_case["case_id"],
        "case_index": sampled_case.get("case_index"),
        "answer_type": sampled_case.get("answer_type"),
        "question": None,
        "parse_ok": False,
        "runtime_ok": False,
        "answer_correct": False,
        "expected_raw": None,
        "expected_normalized": None,
        "final_answer": None,
        "final_answer_preview": None,
        "final_answer_normalized": None,
        "round_count": 0,
        "retry_count": 0,
        "retry_exhausted": False,
        "error": None,
        "rounds": [],
        "elapsed_seconds": None,
        "case_dir": display_path(case_dir),
    }


def select_cases(
    *,
    databench_root: Path,
    max_cases: int | None,
    case_ids: set[str],
    dataset_id: str | None,
    sample_size: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if max_cases == 0:
        return []
    cases = list_databench_cases(databench_root)
    if dataset_id is not None:
        cases = [case for case in cases if case["dataset_id"] == dataset_id]
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


def list_databench_cases(databench_root: Path) -> list[dict[str, Any]]:
    cases = []
    for dataset_dir in iter_databench_dataset_dirs(databench_root):
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
                }
            )
    return cases
