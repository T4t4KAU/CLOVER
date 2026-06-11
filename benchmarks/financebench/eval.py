"""FinanceBench evaluation for CLOVER document numerical reasoning."""

from __future__ import annotations

import copy
import json
import random
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

from benchmarks.costing import estimate_openai_text_cost, normalize_remote_token_usage
from benchmarks.databench.static_tool_eval import (
    display_path,
    format_error,
    json_ready,
    preview,
    safe_divide,
    write_json,
    write_jsonl,
)
from benchmarks.financebench.remote_baseline import financebench_answer_correct
from benchmarks.warnings import suppress_benchmark_warnings
from clover.runtime import (
    CaseResult,
    DocumentReasoningCaseSpec,
    RoundLoopResult,
    RoundLoopStep,
    run_document_reasoning_system,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXAMPLES_ROOT = REPO_ROOT / "datasets" / "financebench"
TOKEN_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)


@dataclass(frozen=True)
class FinanceBenchDocumentExample:
    """One local FinanceBench example prepared for document reasoning eval."""

    sample_index: int
    example_id: str
    case_id: str
    case_dir: Path
    task_dsl: dict[str, Any]
    metadata: dict[str, Any]
    base_dir: Path
    expected_raw: Any
    answer_type: str


def run_financebench_document_eval(
    *,
    examples_root: Path = DEFAULT_EXAMPLES_ROOT,
    output_dir: Path,
    remote_config: dict[str, Any],
    local_slm_config: dict[str, Any] | None = None,
    max_cases: int | None = None,
    case_ids: set[str] | None = None,
    sample_size: int | None = None,
    seed: int = 20260528,
    max_retries: int = 1,
    max_parallel_execution_units: int = 64,
    max_parallel_slm_node_jobs: int = 64,
    max_parallel_slm_sequences: int = 64,
    max_pending_slm_sequences: int = 1024,
    max_workers: int | None = 64,
    node_timeout_seconds: float | None = None,
    overwrite: bool = False,
    remote_cost_model: str | None = None,
    progress_factory: Callable[[int], Any] | None = None,
) -> dict[str, Any]:
    """Run local FinanceBench examples through CLOVER document reasoning."""

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

        examples = load_financebench_document_examples(
            examples_root=examples_root,
            case_ids=case_ids or set(),
            max_cases=max_cases,
            sample_size=sample_size,
            seed=seed,
        )
        records_by_case = {
            example.case_id: _base_record(example, output_dir=output_dir)
            for example in examples
        }
        completed_records: list[dict[str, Any]] = []
        started_by_case = {example.case_id: time.perf_counter() for example in examples}
        progress_bar = progress_factory(len(examples)) if progress_factory else None

        try:
            case_outputs = _run_clover_cases(
                examples=examples,
                records_by_case=records_by_case,
                output_dir=output_dir,
                remote_config=remote_config,
                local_slm_config=local_slm_config,
                max_retries=max_retries,
                max_parallel_execution_units=max_parallel_execution_units,
                max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
                max_parallel_slm_sequences=max_parallel_slm_sequences,
                max_pending_slm_sequences=max_pending_slm_sequences,
                max_workers=max_workers,
                node_timeout_seconds=node_timeout_seconds,
                started_by_case=started_by_case,
                completed_records=completed_records,
                progress_bar=progress_bar,
            )
        finally:
            if progress_bar is not None:
                progress_bar.close()

        round_results_by_case = case_outputs["round_results_by_case"]
        for example in examples:
            record = records_by_case[example.case_id]
            round_result = round_results_by_case.get(example.case_id)
            if round_result is not None:
                record["rounds"] = _round_summaries(round_result)
                record["retry_exhausted"] = bool(round_result.retry_exhausted)
            _write_case_startup_artifacts(
                output_dir=output_dir,
                example=example,
                record=record,
            )
            write_json(
                output_dir / "cases" / example.example_id / "case_result.json",
                record,
            )

        records = [records_by_case[example.case_id] for example in examples]
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
        system_profile = case_outputs["profile"]
        system_profile["eval_startup"] = {
            "workers": _resolved_worker_count(max_workers, len(examples)),
            "selected_cases": len(examples),
            "loaded_cases": len(examples),
            "preprocess_failed_cases": 0,
            "startup_seconds": 0.0,
        }
        summary = build_summary(
            records=records,
            output_dir=output_dir,
            remote_config=remote_config,
            local_slm_config=local_slm_config,
            selected_cases=examples,
            elapsed_seconds=time.perf_counter() - started,
            max_retries=max_retries,
            seed=seed,
            sample_size=sample_size,
            max_parallel_execution_units=max_parallel_execution_units,
            max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
            max_parallel_slm_sequences=max_parallel_slm_sequences,
            max_pending_slm_sequences=max_pending_slm_sequences,
            max_workers=_resolved_worker_count(max_workers, len(examples)),
            node_timeout_seconds=node_timeout_seconds,
            system_profile=system_profile,
            remote_cost_model=remote_cost_model,
            cases_index=cases_index,
            mismatch_cases=mismatch_cases,
            failure_cases=failure_cases,
        )
        write_json(output_dir / "run_summary.json", summary)
        return summary


def _run_clover_cases(
    *,
    examples: list[FinanceBenchDocumentExample],
    records_by_case: dict[str, dict[str, Any]],
    output_dir: Path,
    remote_config: dict[str, Any],
    local_slm_config: dict[str, Any] | None,
    max_retries: int,
    max_parallel_execution_units: int,
    max_parallel_slm_node_jobs: int,
    max_parallel_slm_sequences: int,
    max_pending_slm_sequences: int,
    max_workers: int | None,
    node_timeout_seconds: float | None,
    started_by_case: dict[str, float],
    completed_records: list[dict[str, Any]],
    progress_bar: Any | None,
) -> dict[str, Any]:
    worker_count = _resolved_worker_count(max_workers, len(examples))
    profiles: list[dict[str, Any]] = []
    round_results_by_case: dict[str, RoundLoopResult] = {}

    def complete_case(example: FinanceBenchDocumentExample, payload: dict[str, Any]) -> None:
        profiles.append(payload["profile"])
        round_result = payload.get("round_result")
        if round_result is not None:
            round_results_by_case[example.case_id] = round_result
        case_result = payload.get("case_result")
        if isinstance(case_result, CaseResult):
            record = records_by_case[example.case_id]
            _update_record_from_case_result(record, case_result)
        else:
            record = records_by_case[example.case_id]
            record.update(
                {
                    "runtime_ok": False,
                    "answer_correct": False,
                    "error": payload.get("error")
                    or {"type": "RuntimeError", "message": "case produced no result"},
                }
            )
        record["elapsed_seconds"] = time.perf_counter() - started_by_case[example.case_id]
        write_json(output_dir / "cases" / record["example_id"] / "case_result.json", record)
        completed_records.append(record)
        if progress_bar is not None:
            progress_bar.update(completed_records)

    if worker_count <= 1:
        for example in examples:
            complete_case(
                example,
                _run_single_clover_case(
                    example=example,
                    remote_config=remote_config,
                    local_slm_config=local_slm_config,
                    max_retries=max_retries,
                    max_parallel_execution_units=max_parallel_execution_units,
                    max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
                    max_parallel_slm_sequences=max_parallel_slm_sequences,
                    max_pending_slm_sequences=max_pending_slm_sequences,
                    node_timeout_seconds=node_timeout_seconds,
                ),
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _run_single_clover_case,
                    example=example,
                    remote_config=copy.deepcopy(remote_config),
                    local_slm_config=copy.deepcopy(local_slm_config),
                    max_retries=max_retries,
                    max_parallel_execution_units=max_parallel_execution_units,
                    max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
                    max_parallel_slm_sequences=max_parallel_slm_sequences,
                    max_pending_slm_sequences=max_pending_slm_sequences,
                    node_timeout_seconds=node_timeout_seconds,
                ): example
                for example in examples
            }
            for future in as_completed(futures):
                example = futures[future]
                try:
                    payload = future.result()
                except Exception as exc:  # noqa: BLE001 - isolate benchmark cases.
                    payload = {
                        "profile": {},
                        "case_result": None,
                        "round_result": None,
                        "error": format_error(exc),
                    }
                complete_case(example, payload)

    return {
        "profile": _merge_system_profiles(profiles),
        "round_results_by_case": round_results_by_case,
    }


def _run_single_clover_case(
    *,
    example: FinanceBenchDocumentExample,
    remote_config: dict[str, Any],
    local_slm_config: dict[str, Any] | None,
    max_retries: int,
    max_parallel_execution_units: int,
    max_parallel_slm_node_jobs: int,
    max_parallel_slm_sequences: int,
    max_pending_slm_sequences: int,
    node_timeout_seconds: float | None,
) -> dict[str, Any]:
    system_result = run_document_reasoning_system(
        case_specs=[
            DocumentReasoningCaseSpec(
                case_id=example.case_id,
                task_dsl=example.task_dsl,
                base_dir=example.base_dir,
                metadata={
                    **copy.deepcopy(example.metadata),
                    "sample_index": example.sample_index,
                    "example_id": example.example_id,
                    "answer_type": example.answer_type,
                },
                answer_key=f"answer_{example.sample_index + 1}",
            )
        ],
        remote_config=remote_config,
        local_slm_config=local_slm_config,
        max_retries=max_retries,
        max_parallel_execution_units=max_parallel_execution_units,
        max_parallel_slm_node_jobs=max_parallel_slm_node_jobs,
        max_parallel_slm_sequences=max_parallel_slm_sequences,
        max_pending_slm_sequences=max_pending_slm_sequences,
        node_timeout_seconds=node_timeout_seconds,
    )
    answer_key = f"answer_{example.sample_index + 1}"
    return {
        "profile": system_result.profile,
        "case_result": system_result.case_results[0]
        if system_result.case_results
        else None,
        "round_result": system_result.round_results.get(answer_key),
    }


def _resolved_worker_count(max_workers: int | None, case_count: int) -> int:
    if case_count <= 0:
        return max(1, int(max_workers or 1))
    if max_workers is None:
        return 1
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    return max(1, min(max_workers, case_count))


def _merge_system_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    stages: dict[str, dict[str, Any]] = {}
    counters: Counter[str] = Counter()
    for profile in profiles:
        for name, stage in profile.get("stages", {}).items():
            target = stages.setdefault(
                name,
                {
                    "calls": 0,
                    "items": 0,
                    "total_seconds": 0.0,
                    "average_seconds": 0.0,
                },
            )
            target["calls"] += int(stage.get("calls", 0) or 0)
            target["items"] += int(stage.get("items", 0) or 0)
            target["total_seconds"] += float(stage.get("total_seconds", 0.0) or 0.0)
        counters.update(
            {
                str(key): int(value or 0)
                for key, value in profile.get("counters", {}).items()
            }
        )
    for stage in stages.values():
        calls = stage["calls"]
        stage["average_seconds"] = stage["total_seconds"] / calls if calls else 0.0
    return {
        "stages": dict(sorted(stages.items())),
        "counters": dict(sorted(counters.items())),
        "summary": {
            "remote_calls": counters.get("supervisor_decompose_calls", 0)
            + counters.get("supervisor_synthesis_calls", 0)
            + counters.get("supervisor_repair_calls", 0),
            "merged_plan_count": counters.get("merged_plan_count", 0),
            "reused_nodes": counters.get("reused_nodes", 0),
            "parallel_system_instances": len(profiles),
            "remote_token_usage": _token_usage_from_counters(counters, "remote"),
            "local_slm_calls": counters.get("local_slm_calls", 0),
            "local_slm_token_usage": _token_usage_from_counters(counters, "local_slm"),
        },
    }


def _token_usage_from_counters(counters: Counter[str], prefix: str) -> dict[str, int]:
    return {
        key: int(counters.get(f"{prefix}_{key}", 0) or 0)
        for key in TOKEN_KEYS
    }


def load_financebench_document_examples(
    *,
    examples_root: Path = DEFAULT_EXAMPLES_ROOT,
    case_ids: set[str] | None = None,
    max_cases: int | None = None,
    sample_size: int | None = None,
    seed: int = 20260528,
) -> list[FinanceBenchDocumentExample]:
    """Load local FinanceBench document examples in their index order."""

    if max_cases == 0:
        return []
    root = examples_root.expanduser().resolve()
    case_id_filter = case_ids or set()
    rows = _example_index_rows(root)
    examples: list[FinanceBenchDocumentExample] = []
    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        rng = random.Random(seed)
        rows = rng.sample(rows, min(sample_size, len(rows)))
    if max_cases is not None:
        rows = rows[:max_cases]

    for sample_index, row in enumerate(rows):
        example_dir = _example_dir(root, row)
        task_path = example_dir / "task.json"
        metadata_path = example_dir / "metadata.json"
        if not task_path.is_file() or not metadata_path.is_file():
            continue
        metadata = _read_json(metadata_path)
        case_id = str(metadata.get("case_id") or row.get("case_id") or example_dir.name)
        example_id = str(metadata.get("example_id") or row.get("example_id") or example_dir.name)
        if case_id_filter and case_id not in case_id_filter and example_id not in case_id_filter:
            continue
        task_dsl = _task_dsl_with_source_pdf(_read_json(task_path), metadata)
        base_dir = _base_dir_for_task(task_dsl, example_dir)
        examples.append(
            FinanceBenchDocumentExample(
                sample_index=len(examples),
                example_id=example_id,
                case_id=case_id,
                case_dir=example_dir,
                task_dsl=task_dsl,
                metadata=metadata,
                base_dir=base_dir,
                expected_raw=metadata.get("expected_answer"),
                answer_type=str(metadata.get("answer_type") or "string"),
            )
        )
    return examples


def build_summary(
    *,
    records: list[dict[str, Any]],
    output_dir: Path,
    remote_config: dict[str, Any],
    local_slm_config: dict[str, Any] | None,
    selected_cases: list[FinanceBenchDocumentExample],
    elapsed_seconds: float,
    max_retries: int,
    seed: int,
    sample_size: int | None,
    max_parallel_execution_units: int,
    max_parallel_slm_node_jobs: int,
    max_parallel_slm_sequences: int,
    max_pending_slm_sequences: int,
    max_workers: int,
    node_timeout_seconds: float | None,
    system_profile: dict[str, Any],
    remote_cost_model: str | None,
    cases_index: Path,
    mismatch_cases: Path,
    failure_cases: Path,
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
    retry_case_ids = [
        record["case_id"] for record in records if record.get("retry_count", 0) > 0
    ]
    initial_execution_failure_ids = [
        record["case_id"]
        for record in records
        if record.get("rounds") and not record["rounds"][0].get("execution_ok")
    ]
    answer_types = Counter(record.get("answer_type") for record in records)
    mismatches_by_type = Counter(
        record.get("answer_type")
        for record in records
        if record.get("runtime_ok") and not record.get("answer_correct")
    )
    summary_profile = system_profile.get("summary", {})
    remote_token_usage = normalize_remote_token_usage(
        summary_profile.get("remote_token_usage", {})
    )
    local_slm_token_usage = normalize_remote_token_usage(
        summary_profile.get("local_slm_token_usage", {})
    )
    remote_cost_estimate = estimate_openai_text_cost(
        remote_token_usage,
        remote_config=remote_config,
        pricing_model=remote_cost_model,
    )
    return {
        "run_name": output_dir.name,
        "stage": "financebench_document_eval",
        "workflow": "document_reasoning",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": total,
        "sample_size": len(selected_cases),
        "requested_sample_size": sample_size,
        "seed": seed,
        "validation_mode": "round_loop",
        "remote_batch_size": None,
        "remote_concurrency": None,
        "eval_batch_size": None,
        "profile_baseline": None,
        "max_retries": max_retries,
        "max_parallel_execution_units": max_parallel_execution_units,
        "max_parallel_slm_node_jobs": max_parallel_slm_node_jobs,
        "max_parallel_slm_sequences": max_parallel_slm_sequences,
        "max_pending_slm_sequences": max_pending_slm_sequences,
        "slm_scheduler": _slm_scheduler_summary(local_slm_config),
        "parallel_workers": max_workers,
        "node_timeout_seconds": node_timeout_seconds,
        "remote_llm": _config_summary(remote_config),
        "local_slm": _config_summary(local_slm_config),
        "remote_calls": summary_profile.get("remote_calls", 0),
        "local_slm_calls": summary_profile.get("local_slm_calls", 0),
        "parse_successes": sum(1 for record in records if record.get("parse_ok")),
        "parse_failures": sum(1 for record in records if not record.get("parse_ok")),
        "runtime_successes": runtime_successes,
        "runtime_failures": total - runtime_successes,
        "correct": correct,
        "mismatches": mismatches,
        "failures": total - runtime_successes,
        "accuracy_on_successes": safe_divide(correct, runtime_successes),
        "accuracy_on_all_cases": safe_divide(correct, total),
        "retry_cases": len(retry_case_ids),
        "retry_case_ids": retry_case_ids,
        "total_retry_rounds": sum(record.get("retry_count", 0) for record in records),
        "sql_repair_cases": 0,
        "sql_repair_case_ids": [],
        "initial_execution_failures": len(initial_execution_failure_ids),
        "initial_execution_failure_case_ids": initial_execution_failure_ids,
        "answer_types": _counter_as_strings(answer_types),
        "mismatches_by_type": _counter_as_strings(mismatches_by_type),
        "error_types": dict(sorted(error_types.items())),
        "remote_token_usage": remote_token_usage,
        "local_slm_token_usage": local_slm_token_usage,
        "remote_cost_estimate": remote_cost_estimate,
        "system_profile": system_profile,
        "elapsed_seconds": elapsed_seconds,
        "run_dir": display_path(output_dir),
        "cases_index": display_path(cases_index),
        "answer_mismatch_cases": display_path(mismatch_cases),
        "failure_cases": display_path(failure_cases),
    }


def _counter_as_strings(counter: Counter[Any]) -> dict[str, int]:
    return dict(sorted((str(key or "unknown"), value) for key, value in counter.items()))


def _base_record(
    example: FinanceBenchDocumentExample,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    eval_result = financebench_answer_correct(example.expected_raw, None)
    return {
        "sample_index": example.sample_index,
        "dataset_id": "financebench",
        "case_id": example.case_id,
        "example_id": example.example_id,
        "case_index": example.metadata.get("sampling", {}).get("sample_index"),
        "answer_type": example.answer_type,
        "question_reasoning": example.metadata.get("question_reasoning"),
        "question_type": example.metadata.get("question_type"),
        "doc_name": example.metadata.get("doc_name"),
        "question": example.task_dsl.get("question"),
        "expected_raw": example.expected_raw,
        "expected_normalized": eval_result["expected_normalized"],
        "parse_ok": True,
        "runtime_ok": False,
        "answer_correct": False,
        "final_answer": None,
        "final_answer_preview": None,
        "final_answer_normalized": None,
        "financebench_eval": None,
        "round_count": 0,
        "retry_count": 0,
        "retry_exhausted": False,
        "error": None,
        "rounds": [],
        "elapsed_seconds": None,
        "case_dir": display_path(output_dir / "cases" / example.example_id),
    }


def _update_record_from_case_result(
    record: dict[str, Any],
    case_result: CaseResult,
) -> None:
    eval_result = financebench_answer_correct(record.get("expected_raw"), case_result.answer)
    record.update(
        {
            "runtime_ok": case_result.ok,
            "final_answer": json_ready(case_result.answer),
            "final_answer_preview": preview(case_result.answer),
            "final_answer_normalized": eval_result["actual_normalized"],
            "answer_correct": bool(case_result.ok and eval_result["correct"]),
            "financebench_eval": eval_result,
            "round_count": case_result.retry_count + 1,
            "retry_count": case_result.retry_count,
            "error": case_result.error,
        }
    )
    if isinstance(case_result.error, dict) and _looks_like_parse_error(case_result.error):
        record["parse_ok"] = False


def _round_summaries(round_result: RoundLoopResult) -> list[dict[str, Any]]:
    return [_round_summary(step) for step in round_result.rounds]


def _result_get(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _collector_outputs(result: Any) -> dict[str, Any]:
    outputs = _result_get(result, "collector_outputs", {})
    return outputs if isinstance(outputs, dict) else {}


def _supervisor_decision(supervisor_result: Any) -> Any:
    if isinstance(supervisor_result, dict):
        return supervisor_result.get("decision")
    return getattr(supervisor_result, "decision", None)


def _round_summary(step: RoundLoopStep) -> dict[str, Any]:
    execution_result = step.execution_result
    collector = _collector_outputs(execution_result).get("document_evidence", {})
    if not isinstance(collector, dict):
        collector = {}
    decision = _supervisor_decision(step.supervisor_result)
    decision_payload = {}
    if decision is not None:
        decision_payload = {
            "answer": json_ready(_result_get(decision, "answer")),
            "answer_preview": preview(_result_get(decision, "answer")),
            "sufficient": _result_get(decision, "sufficient"),
            "retry": _result_get(decision, "retry"),
            "explanation_preview": preview(_result_get(decision, "explanation")),
            "feedback_preview": preview(_result_get(decision, "feedback")),
            "next_python_code_chars": len(_result_get(decision, "next_python_code", "") or ""),
        }
    return {
        "round_index": step.index,
        "command_chars": len(step.command_output or ""),
        "execution_ok": bool(_result_get(execution_result, "ok")),
        "execution_elapsed_ms": _result_get(execution_result, "elapsed_ms"),
        "worker_count": collector.get("worker_count"),
        "included_count": collector.get("included_count"),
        "fallback_used": collector.get("fallback_used"),
        "evidence_summary_preview": preview(collector.get("evidence_summary")),
        "supervisor_decision": decision_payload,
        "execution_error": _result_get(execution_result, "error"),
    }


def _write_case_startup_artifacts(
    *,
    output_dir: Path,
    example: FinanceBenchDocumentExample,
    record: dict[str, Any],
) -> None:
    case_output_dir = output_dir / "cases" / example.example_id
    write_json(case_output_dir / "task_dsl.json", example.task_dsl)
    write_json(case_output_dir / "metadata.json", example.metadata)
    files = {
        "task_dsl": display_path(case_output_dir / "task_dsl.json"),
        "metadata": display_path(case_output_dir / "metadata.json"),
        "case_result": display_path(case_output_dir / "case_result.json"),
    }
    record["files"] = files


def _mismatch_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_index": record["sample_index"],
        "dataset_id": record["dataset_id"],
        "case_id": record["case_id"],
        "example_id": record.get("example_id"),
        "question": record.get("question"),
        "expected": record.get("expected_normalized"),
        "actual": record.get("final_answer_normalized"),
        "financebench_eval": record.get("financebench_eval"),
        "round_count": record.get("round_count"),
        "retry_count": record.get("retry_count"),
        "case_result": record.get("files", {}).get("case_result"),
    }


def _example_index_rows(root: Path) -> list[dict[str, Any]]:
    index_path = root / "index.jsonl"
    if index_path.is_file():
        rows = []
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    return [{"example_dir": str(path)} for path in _sorted_case_dirs(root)]


def _sorted_case_dirs(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("case_")],
        key=lambda path: _case_dir_sort_key(path.name),
    )


def _case_dir_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)) if match else 10**9, name)


def _looks_like_parse_error(error: dict[str, Any]) -> bool:
    error_type = str(error.get("type") or "")
    return "Parse" in error_type or "Syntax" in error_type


def _example_dir(root: Path, row: dict[str, Any]) -> Path:
    raw_dir = row.get("example_dir") or row.get("example_id")
    if not isinstance(raw_dir, str) or not raw_dir:
        raise ValueError(f"FinanceBench example row missing example_dir: {row}")
    path = Path(raw_dir)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _task_dsl_with_source_pdf(
    task_dsl: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    task = copy.deepcopy(task_dsl)
    sources = task.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Document example task requires at least one source")
    source_pdf = metadata.get("source_pdf")
    if isinstance(source_pdf, str) and source_pdf:
        sources[0]["file"] = source_pdf
    return task


def _base_dir_for_task(task_dsl: dict[str, Any], example_dir: Path) -> Path:
    source = task_dsl["sources"][0]
    file_name = Path(str(source["file"])).expanduser()
    if file_name.is_absolute():
        return file_name.parent
    return example_dir


def _config_summary(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        "provider": config.get("provider"),
        "api_type": config.get("api_type"),
        "base_url": config.get("base_url"),
        "model": config.get("model"),
        "slm_scheduler": config.get("slm_scheduler", "tptt"),
    }


def _slm_scheduler_summary(config: dict[str, Any] | None) -> str:
    if not isinstance(config, dict):
        return "tptt"
    return str(config.get("slm_scheduler") or "tptt")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload
