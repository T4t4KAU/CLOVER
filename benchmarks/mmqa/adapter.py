"""MMQA multi-table local-layout adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clover.resource import (
    build_multitable_task_dsl_with_builder_agent,
    preprocess_task_dsl,
)
from clover.supervisor.client import extract_token_usage


MMQA_DSL_MODE_BUILDER_AGENT = "build_multitable_dsl_tool"


@dataclass(frozen=True)
class MMQATask:
    task_dsl: dict[str, Any]
    base_dir: Path
    metadata: dict[str, Any]


def run_mmqa_case(
    mmqa_root: Path,
    dataset_id: str,
    case_id: str | None,
    case_index: int,
    output_root: Path,
    run_name: str,
    dsl_builder_slm_config: dict[str, Any],
    write_run_summary: bool = True,
) -> dict[str, Any]:
    task = load_mmqa_task(
        mmqa_root=mmqa_root,
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
        "dataset": "mmqa",
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


def load_mmqa_task(
    mmqa_root: str | Path,
    dataset_id: str,
    case_id: str | None = None,
    case_index: int | None = None,
    dsl_builder_slm_config: dict[str, Any] | None = None,
    dsl_builder_client: Any | None = None,
) -> MMQATask:
    """Load one converted MMQA case and build a multi-table task DSL."""

    mmqa_root_path = Path(mmqa_root).expanduser().resolve()
    dataset_dir = _resolve_dataset_dir(mmqa_root_path, dataset_id)

    case = select_case(dataset_dir, case_id=case_id, case_index=case_index)
    selected_case_id = case["case_id"]

    source_files = list(case.get("source_files") or [])
    table_count = int(case.get("table_count") or len(source_files))
    if not source_files:
        source_files = [f"table_{index}.csv" for index in range(1, table_count + 1)]
    table_paths = [dataset_dir / source_file for source_file in source_files]

    table_names_list = list(case.get("table_names") or [])
    table_names = {
        f"table_{index}": name
        for index, name in enumerate(table_names_list, start=1)
    }

    builder_result = build_multitable_task_dsl_with_builder_agent(
        question=case["question"],
        table_paths=table_paths,
        source_files=source_files,
        answer_type=case.get("type"),
        task_type="table_reasoning.query",
        table_names=table_names or None,
        foreign_keys=list(case.get("foreign_keys") or []) or None,
        primary_keys=list(case.get("primary_keys") or []) or None,
        slm_config=dsl_builder_slm_config or {},
        client=dsl_builder_client,
    )
    task_dsl = _normalize_mmqa_task_dsl(builder_result.task_dsl)
    dsl_builder_metadata: dict[str, Any] = {
        "mode": builder_result.builder_mode,
        "tool_call": builder_result.tool_call,
        "diagnostics": builder_result.diagnostics,
        "raw_output": builder_result.raw_output,
        "parsed_output": builder_result.parsed_output,
        "prompt_chars": len(builder_result.prompt),
        "token_usage": extract_token_usage(builder_result.response_payload),
        "task_answer_type": task_dsl.get("answer", {}).get("type"),
    }

    metadata = {
        "dataset": "mmqa",
        "dataset_id": dataset_id,
        "case_id": selected_case_id,
        "expected_answer": case.get("answer"),
        "expected_raw": case.get("answer_raw"),
        "answer_type": case.get("type"),
        "split": case.get("split"),
        "table_names": table_names_list,
        "foreign_keys": list(case.get("foreign_keys") or []),
        "primary_keys": list(case.get("primary_keys") or []),
        "gold_sql": case.get("gold_sql"),
        "source_files": source_files,
        "table_count": table_count,
        "case": case,
        "dsl_builder": dsl_builder_metadata,
    }
    return MMQATask(task_dsl=task_dsl, base_dir=dataset_dir, metadata=metadata)


def first_mmqa_dataset(mmqa_root: Path) -> str:
    dataset_dirs = iter_mmqa_dataset_dirs(mmqa_root)
    if not dataset_dirs:
        raise FileNotFoundError(f"No MMQA datasets found in {mmqa_root}")
    return dataset_dirs[0].name


def iter_mmqa_dataset_dirs(mmqa_root: Path) -> list[Path]:
    root = Path(mmqa_root).expanduser().resolve()
    dataset_dirs: list[Path] = []
    for split_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        dataset_dirs.extend(
            sorted(path for path in split_dir.iterdir() if path.is_dir())
        )
    return dataset_dirs


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


def _normalize_mmqa_task_dsl(task_dsl: dict[str, Any]) -> dict[str, Any]:
    updated = dict(task_dsl)
    updated["task_type"] = "table_reasoning.query"
    updated.pop("profile", None)
    updated.pop("metadata", None)
    updated.pop("reasoning_profile", None)
    updated.pop("reasoning_context", None)
    return updated


def _resolve_dataset_dir(mmqa_root: Path, dataset_id: str) -> Path:
    direct = mmqa_root / dataset_id
    if direct.is_dir():
        return direct
    for split_dir in sorted(path for path in mmqa_root.iterdir() if path.is_dir()):
        candidate = split_dir / dataset_id
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"MMQA dataset not found: {dataset_id}")


def read_cases(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"MMQA cases file not found: {path}")
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
