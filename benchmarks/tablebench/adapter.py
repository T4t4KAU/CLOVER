"""TableBench local-layout adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clover.resource import (
    build_table_task_dsl_with_builder_agent,
    preprocess_task_dsl,
)
from clover.supervisor.client import extract_token_usage


TABLEBENCH_DSL_MODE_BUILDER_AGENT = "build_table_dsl_tool"


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
    dsl_builder_slm_config: dict[str, Any],
    write_run_summary: bool = True,
) -> dict[str, Any]:
    task = load_tablebench_task(
        tablebench_root=tablebench_root,
        dataset_id=dataset_id,
        case_id=case_id,
        case_index=case_index,
        dsl_builder_slm_config=dsl_builder_slm_config,
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
    write_json(case_dir / "dsl_builder.json", task.metadata["dsl_builder"])

    run_summary = {
        "run_name": run_name,
        "stage": "dsl_preprocess",
        "dataset": "tablebench",
        "dataset_id": dataset_id,
        "case_id": selected_case_id,
        "dsl_builder_mode": task.metadata["dsl_builder"]["mode"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_dir": str(case_dir),
        "files": {
            "task_dsl": str(case_dir / "task_dsl.json"),
            "local_dsl": str(case_dir / "local_dsl.json"),
            "remote_dsl": str(case_dir / "remote_dsl.json"),
            "context": str(case_dir / "context.json"),
            "dsl_builder": str(case_dir / "dsl_builder.json"),
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
    dsl_builder_slm_config: dict[str, Any] | None = None,
    dsl_builder_client: Any | None = None,
) -> TablebenchTask:
    """Load one converted TableBench case and build task DSL."""

    dataset_dir = Path(tablebench_root).expanduser().resolve() / dataset_id
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"TableBench dataset not found: {dataset_dir}")

    case = select_case(dataset_dir, case_id=case_id, case_index=case_index)
    selected_case_id = case["case_id"]
    builder_result = build_table_task_dsl_with_builder_agent(
        question=case["question"],
        table_path=dataset_dir / "table.csv",
        source_file="table.csv",
        answer_type=case.get("type"),
        slm_config=dsl_builder_slm_config or {},
        client=dsl_builder_client,
    )
    task_dsl = _with_legacy_tablebench_hints(builder_result.task_dsl, case)
    dsl_builder_metadata: dict[str, Any] = {
        "mode": builder_result.builder_mode,
        "tool_call": builder_result.tool_call,
        "diagnostics": builder_result.diagnostics,
        "raw_output": builder_result.raw_output,
        "parsed_output": builder_result.parsed_output,
        "prompt_chars": len(builder_result.prompt),
        "token_usage": extract_token_usage(builder_result.response_payload),
        "task_answer_type": task_dsl["answer"]["type"],
    }
    task_dsl = _normalize_tablebench_task_dsl(task_dsl)
    dsl_builder_metadata["task_answer_type"] = task_dsl.get("answer", {}).get("type")

    metadata = {
        "dataset": "tablebench",
        "dataset_id": dataset_id,
        "case_id": selected_case_id,
        "expected_answer": case.get("answer"),
        "qtype": case.get("qtype"),
        "qsubtype": case.get("qsubtype"),
        "case": case,
        "dsl_builder": dsl_builder_metadata,
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


def _normalize_tablebench_task_dsl(task_dsl: dict[str, Any]) -> dict[str, Any]:
    updated = dict(task_dsl)
    updated["task_type"] = "table_reasoning.analyze"
    updated.pop("profile", None)
    updated.pop("metadata", None)
    updated.pop("reasoning_profile", None)
    updated.pop("reasoning_context", None)
    return updated


def _with_legacy_tablebench_hints(
    task_dsl: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(task_dsl)
    hints = {}
    if case.get("qtype") is not None:
        hints["category"] = case["qtype"]
    if case.get("qsubtype") is not None:
        hints["subcategory"] = case["qsubtype"]
    if case.get("chart_type") is not None:
        hints["chart_type"] = case["chart_type"]
    if hints:
        updated["hints"] = hints
    else:
        updated.pop("hints", None)
    return updated


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
