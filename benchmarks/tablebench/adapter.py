"""TableBench local-layout adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clover.resource import preprocess_task_dsl


@dataclass(frozen=True)
class TablebenchTask:
    task_dsl: dict[str, Any]
    base_dir: Path
    metadata: dict[str, Any]


def run_tablebench_case(
    tablebench_root: Path,
    dataset_id: str,
    case_id: str | None,
    case_index: int,
    output_root: Path,
    run_name: str,
    write_run_summary: bool = True,
) -> dict[str, Any]:
    task = load_tablebench_task(
        tablebench_root=tablebench_root,
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
        "dataset": "tablebench",
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


def load_tablebench_task(
    tablebench_root: str | Path,
    dataset_id: str,
    case_id: str | None = None,
    case_index: int | None = None,
) -> TablebenchTask:
    """Load one converted TableBench case as standard CLOVER task DSL."""

    dataset_dir = Path(tablebench_root).expanduser().resolve() / dataset_id
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"TableBench dataset not found: {dataset_dir}")

    case = select_case(dataset_dir, case_id=case_id, case_index=case_index)
    selected_case_id = case["case_id"]
    task_spec_path = dataset_dir / "task_specs" / f"{selected_case_id}.json"

    if task_spec_path.is_file():
        task_dsl = read_json(task_spec_path)
    else:
        task_dsl = task_dsl_from_case(case)
    task_dsl = _normalize_tablebench_task_dsl(task_dsl)

    metadata = {
        "dataset": "tablebench",
        "dataset_id": dataset_id,
        "case_id": selected_case_id,
        "expected_answer": case.get("answer"),
        "qtype": case.get("qtype"),
        "qsubtype": case.get("qsubtype"),
        "case": case,
        "task_spec_path": str(task_spec_path) if task_spec_path.is_file() else None,
    }
    return TablebenchTask(task_dsl=task_dsl, base_dir=dataset_dir, metadata=metadata)


def first_tablebench_dataset(tablebench_root: Path) -> str:
    dataset_dirs = iter_tablebench_dataset_dirs(tablebench_root)
    if not dataset_dirs:
        raise FileNotFoundError(f"No TableBench datasets found in {tablebench_root}")
    return dataset_dirs[0].name


def iter_tablebench_dataset_dirs(tablebench_root: Path) -> list[Path]:
    return sorted(path for path in tablebench_root.iterdir() if path.is_dir())


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
    task = {
        "task_type": "table_reasoning.analyze",
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
    metadata = {
        key: case[key]
        for key in ("qtype", "qsubtype", "chart_type", "original_id")
        if case.get(key) is not None
    }
    if metadata:
        task["hints"] = _reasoning_context(metadata)
    return task


def _normalize_tablebench_task_dsl(task_dsl: dict[str, Any]) -> dict[str, Any]:
    updated = dict(task_dsl)
    updated["task_type"] = "table_reasoning.analyze"
    updated.pop("profile", None)
    metadata = updated.pop("metadata", None)
    if "hints" not in updated and isinstance(metadata, dict):
        updated["hints"] = _reasoning_context(metadata)
    updated.pop("reasoning_profile", None)
    updated.pop("reasoning_context", None)
    return updated


def _reasoning_context(metadata: dict[str, Any]) -> dict[str, Any]:
    context = {}
    if metadata.get("qtype") is not None:
        context["category"] = metadata["qtype"]
    if metadata.get("qsubtype") is not None:
        context["subcategory"] = metadata["qsubtype"]
    if metadata.get("chart_type") is not None:
        context["chart_type"] = metadata["chart_type"]
    return context


def read_cases(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"TableBench cases file not found: {path}")
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
