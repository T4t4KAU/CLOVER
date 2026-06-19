"""TableFact evaluation using the shared TableBench table runtime."""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmarks.utils import safe_divide, write_jsonl
from benchmarks.tablebench.adapter import iter_tablebench_dataset_dirs, read_cases, write_json
from benchmarks.tablebench.eval import run_tablebench_eval


TABLEFACT_SUBSETS = frozenset({"simple", "complex", "small"})


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
                }
            )
    return cases


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
