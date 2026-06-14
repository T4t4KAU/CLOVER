"""WikiTableQuestions local-layout adapter."""

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


WIKITQ_DSL_MODE_BUILDER_AGENT = "build_table_dsl_tool"


@dataclass(frozen=True)
class WikiTQTask:
    task_dsl: dict[str, Any]
    base_dir: Path
    metadata: dict[str, Any]


def run_wikitq_case(
    wikitq_root: Path,
    dataset_id: str,
    case_id: str | None,
    case_index: int,
    output_root: Path,
    run_name: str,
    dsl_builder_slm_config: dict[str, Any],
    write_run_summary: bool = True,
) -> dict[str, Any]:
    task = load_wikitq_task(
        wikitq_root=wikitq_root,
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
        "dataset": "wikitq",
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


def load_wikitq_task(
    wikitq_root: str | Path,
    dataset_id: str,
    case_id: str | None = None,
    case_index: int | None = None,
    dsl_builder_slm_config: dict[str, Any] | None = None,
    dsl_builder_client: Any | None = None,
) -> WikiTQTask:
    """Load one converted WikiTQ case and build task DSL."""

    wikitq_root_path = Path(wikitq_root).expanduser().resolve()
    dataset_dir = wikitq_root_path / dataset_id
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"WikiTQ dataset not found: {dataset_dir}")

    case = select_case(dataset_dir, case_id=case_id, case_index=case_index)
    selected_case_id = case["case_id"]
    source_context_path = wikitq_source_context_path(
        wikitq_root_path,
        context=case.get("context"),
    )
    builder_result = build_table_task_dsl_with_builder_agent(
        question=case["question"],
        table_path=dataset_dir / "table.csv",
        source_file="table.csv",
        source_context_path=source_context_path,
        answer_type=case.get("type"),
        slm_config=dsl_builder_slm_config or {},
        client=dsl_builder_client,
    )
    task_dsl = _normalize_wikitq_task_dsl(builder_result.task_dsl)
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
        "dataset": "wikitq",
        "dataset_id": dataset_id,
        "case_id": selected_case_id,
        "expected_answer": case.get("answer"),
        "expected_canon": case.get("answer_canon"),
        "answer_canon_type": case.get("answer_canon_type"),
        "split": case.get("split"),
        "context": case.get("context"),
        "case": case,
        "dsl_builder": dsl_builder_metadata,
    }
    return WikiTQTask(task_dsl=task_dsl, base_dir=dataset_dir, metadata=metadata)


def first_wikitq_dataset(wikitq_root: Path) -> str:
    dataset_dirs = iter_wikitq_dataset_dirs(wikitq_root)
    if not dataset_dirs:
        raise FileNotFoundError(f"No WikiTQ datasets found in {wikitq_root}")
    return dataset_dirs[0].name


def iter_wikitq_dataset_dirs(wikitq_root: Path) -> list[Path]:
    return sorted(path for path in wikitq_root.iterdir() if path.is_dir())


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


def _normalize_wikitq_task_dsl(task_dsl: dict[str, Any]) -> dict[str, Any]:
    updated = dict(task_dsl)
    updated["task_type"] = "table_reasoning.analyze"
    updated.pop("profile", None)
    updated.pop("metadata", None)
    updated.pop("reasoning_profile", None)
    updated.pop("reasoning_context", None)
    return updated


def wikitq_source_root(wikitq_root: str | Path) -> Path:
    root = Path(wikitq_root).expanduser().resolve()
    summary_path = root / "conversion_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"WikiTQ conversion summary not found: {summary_path}"
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid WikiTQ conversion summary: {summary_path}") from exc
    source_root_value = summary.get("source_root") if isinstance(summary, dict) else None
    if not isinstance(source_root_value, str) or not source_root_value.strip():
        raise ValueError(f"WikiTQ conversion summary missing source_root: {summary_path}")
    source_root = Path(source_root_value).expanduser()
    if not source_root.is_absolute():
        source_root = (Path.cwd() / source_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"WikiTQ source root not found: {source_root}")
    return source_root.resolve()


def wikitq_source_context_path(
    wikitq_root: str | Path,
    *,
    context: Any,
    source_root: Path | None = None,
) -> Path:
    context_text = str(context or "").strip()
    if not context_text:
        raise ValueError("WikiTQ case missing context source path")
    source_root_path = source_root or wikitq_source_root(wikitq_root)
    path = (source_root_path / context_text).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"WikiTQ source context file not found: {path}")
    return path


def read_cases(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"WikiTQ cases file not found: {path}")
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
