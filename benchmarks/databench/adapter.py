"""Databench evaluation runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clover.resource import preprocess_task_dsl


@dataclass(frozen=True)
class DatabenchTask:
    task_dsl: dict[str, Any]
    base_dir: Path
    metadata: dict[str, Any]


def run_all_databench_tables(
    databench_root: Path,
    case_index: int,
    output_root: Path,
    run_name: str,
) -> dict[str, Any]:
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    index_path = run_dir / "tables_index.jsonl"

    total = 0
    completed = 0
    skipped: list[dict[str, str]] = []

    with index_path.open("w", encoding="utf-8") as index_file:
        for dataset_dir in iter_databench_dataset_dirs(databench_root):
            dataset_id = dataset_dir.name
            if not (dataset_dir / "table.csv").is_file() or not (
                dataset_dir / "cases.jsonl"
            ).is_file():
                skipped.append(
                    {
                        "dataset_id": dataset_id,
                        "reason": "missing table.csv or cases.jsonl",
                    }
                )
                continue

            total += 1
            case_record = run_databench_case(
                databench_root=databench_root,
                dataset_id=dataset_id,
                case_id=None,
                case_index=case_index,
                output_root=output_root,
                run_name=run_name,
                write_run_summary=False,
            )
            index_file.write(
                json.dumps(
                    {
                        "dataset_id": case_record["dataset_id"],
                        "case_id": case_record["case_id"],
                        "case_dir": case_record["case_dir"],
                        "remote_dsl": case_record["files"]["remote_dsl"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            completed += 1

    summary = {
        "run_name": run_name,
        "stage": "dsl_preprocess",
        "dataset": "databench",
        "mode": "all_tables",
        "case_index": case_index,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "processed_tables": completed,
        "total_tables": total,
        "skipped": skipped,
        "run_dir": str(run_dir),
        "tables_index": str(index_path),
    }
    write_json(run_dir / "run_summary.json", summary)
    return summary


def run_databench_case(
    databench_root: Path,
    dataset_id: str,
    case_id: str | None,
    case_index: int,
    output_root: Path,
    run_name: str,
    write_run_summary: bool = True,
) -> dict[str, Any]:
    task = load_databench_task(
        databench_root=databench_root,
        dataset_id=dataset_id,
        case_id=case_id,
        case_index=case_index,
    )
    preprocess_result = preprocess_task_dsl(task.task_dsl, base_dir=task.base_dir)

    run_dir = output_root / run_name
    selected_case_id = task.metadata["case_id"]
    case_dir = run_dir / "cases" / selected_case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    write_json(case_dir / "task_dsl.json", task.task_dsl)
    write_json(case_dir / "local_dsl.json", preprocess_result["local_dsl"])
    write_json(case_dir / "remote_dsl.json", preprocess_result["remote_dsl"])
    write_json(case_dir / "context.json", preprocess_result["context"])

    run_summary = {
        "run_name": run_name,
        "stage": "dsl_preprocess",
        "dataset": "databench",
        "dataset_id": dataset_id,
        "case_id": selected_case_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_dir": str(case_dir),
        "files": {
            "task_dsl": str(case_dir / "task_dsl.json"),
            "local_dsl": str(case_dir / "local_dsl.json"),
            "remote_dsl": str(case_dir / "remote_dsl.json"),
            "context": str(case_dir / "context.json"),
        },
    }
    if write_run_summary:
        write_json(run_dir / "run_summary.json", run_summary)
    return run_summary


def load_databench_task(
    databench_root: str | Path,
    dataset_id: str,
    case_id: str | None = None,
    case_index: int | None = None,
) -> DatabenchTask:
    """Load one Databench task as standard CLOVER task DSL."""

    dataset_dir = Path(databench_root).expanduser().resolve() / dataset_id
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Databench dataset not found: {dataset_dir}")

    case = select_case(dataset_dir, case_id=case_id, case_index=case_index)
    selected_case_id = case["case_id"]
    task_spec_path = dataset_dir / "task_specs" / f"{selected_case_id}.json"

    if task_spec_path.is_file():
        task_dsl = read_json(task_spec_path)
    else:
        task_dsl = task_dsl_from_case(case)
    task_dsl = _normalize_databench_task_dsl(task_dsl)

    metadata = {
        "dataset": "databench",
        "dataset_id": dataset_id,
        "case_id": selected_case_id,
        "expected_answer": case.get("answer"),
        "case": case,
        "task_spec_path": str(task_spec_path) if task_spec_path.is_file() else None,
    }
    return DatabenchTask(task_dsl=task_dsl, base_dir=dataset_dir, metadata=metadata)


def first_databench_dataset(databench_root: Path) -> str:
    dataset_dirs = iter_databench_dataset_dirs(databench_root)
    if not dataset_dirs:
        raise FileNotFoundError(f"No Databench datasets found in {databench_root}")
    return dataset_dirs[0].name


def iter_databench_dataset_dirs(databench_root: Path) -> list[Path]:
    return sorted(path for path in databench_root.iterdir() if path.is_dir())


def select_case(
    dataset_dir: Path,
    case_id: str | None,
    case_index: int | None,
) -> dict[str, Any]:
    cases = read_cases(dataset_dir / "cases.jsonl")
    if case_id is not None:
        for case in cases:
            if case.get("case_id") == case_id:
                return case
        raise ValueError(f"Case id not found in {dataset_dir}: {case_id}")

    if case_index is None:
        case_index = 0
    try:
        return cases[case_index]
    except IndexError as exc:
        raise ValueError(f"Case index out of range: {case_index}") from exc


def task_dsl_from_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_type": "table_reasoning.query",
        "question": case["question"],
        "sources": [
            {
                "id": 0,
                "type": "table",
                "file": "table.csv",
            }
        ],
        "answer": {
            "name": "answer",
            "type": case["type"],
        },
    }


def _normalize_databench_task_dsl(task_dsl: dict[str, Any]) -> dict[str, Any]:
    updated = dict(task_dsl)
    updated["task_type"] = "table_reasoning.query"
    updated.pop("profile", None)
    updated.pop("reasoning_profile", None)
    updated.pop("reasoning_context", None)
    return updated


def read_cases(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Databench cases file not found: {path}")
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                cases.append(json.loads(stripped))
    return cases


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
