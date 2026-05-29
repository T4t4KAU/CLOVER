"""Generic DSL preprocessing."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from clover.preprocess.csv_schema import extract_csv_schema


SUPPORTED_SOURCE_TYPES = {"table"}
SUPPORTED_ANSWER_TYPES = {
    "boolean",
    "category",
    "date",
    "list",
    "list[category]",
    "list[number]",
    "list[string]",
    "number",
    "string",
    "table",
}


def preprocess_task_dsl(
    task_dsl: dict[str, Any],
    base_dir: str | Path,
) -> dict[str, Any]:
    """Bind local resource details and produce local/remote DSL variants.

    This step is deterministic and local-only: it validates the task, resolves
    source files, extracts structural schemas, and does not call any model.
    """

    _validate_task_dsl(task_dsl)

    base_path = Path(base_dir).expanduser().resolve()

    # The local DSL is for local execution and keeps absolute resource paths.
    local_dsl = {
        "task_type": task_dsl["task_type"],
        "question": task_dsl["question"],
        "sources": [],
        "answer": copy.deepcopy(task_dsl["answer"]),
    }

    # The remote DSL is the payload intended for remote planning. It includes
    # structural metadata only, not local filesystem paths.
    remote_dsl = {
        "task_type": task_dsl["task_type"],
        "question": task_dsl["question"],
        "sources": [],
        "answer": copy.deepcopy(task_dsl["answer"]),
    }

    # Context is local runtime state used to map remote-visible source ids back
    # to concrete resources during later planning/execution stages.
    context: dict[str, Any] = {
        "task_type": task_dsl["task_type"],
        "base_dir": str(base_path),
        "source_map": {},
    }

    for source_index, source in enumerate(task_dsl["sources"], start=1):
        local_source, remote_source, source_map = _preprocess_source(
            source=source,
            source_index=source_index,
            base_dir=base_path,
        )
        local_dsl["sources"].append(local_source)
        remote_dsl["sources"].append(remote_source)
        source_id = local_source["id"]
        context["source_map"][source_id] = source_map

    return {
        "local_dsl": local_dsl,
        "remote_dsl": remote_dsl,
        "context": context,
    }


def _validate_task_dsl(task_dsl: dict[str, Any]) -> None:
    required_fields = ("task_type", "question", "sources", "answer")
    missing = [field for field in required_fields if field not in task_dsl]
    if missing:
        raise ValueError(f"Task DSL missing required fields: {missing}")

    if not isinstance(task_dsl["sources"], list) or not task_dsl["sources"]:
        raise ValueError("Task DSL requires at least one source")

    answer = task_dsl["answer"]
    if not isinstance(answer, dict) or "type" not in answer:
        raise ValueError("Task DSL answer must include a type")
    if answer["type"] not in SUPPORTED_ANSWER_TYPES:
        raise ValueError(f"Unsupported answer type: {answer['type']}")

    for source in task_dsl["sources"]:
        source_type = source.get("type")
        if source_type not in SUPPORTED_SOURCE_TYPES:
            raise ValueError(f"Unsupported source type: {source_type}")
        if "file" not in source:
            raise ValueError("Source must include a file path")


def _preprocess_source(
    source: dict[str, Any],
    source_index: int,
    base_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_type = source["type"]
    source_id = _normal_source_id(source_type, source_index)
    resource_path = _resolve_source_path(source["file"], base_dir)

    if source_type == "table":
        return _preprocess_table_source(source, source_id, resource_path)

    raise ValueError(f"Unsupported source type: {source_type}")


def _preprocess_table_source(
    source: dict[str, Any],
    source_id: str,
    resource_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if resource_path.suffix.lower() != ".csv":
        raise ValueError(f"Only CSV table sources are supported now: {resource_path}")

    # CSV is structured data, but its cells are text. The schema therefore
    # records shape and column names only; it does not infer logical types.
    schema = extract_csv_schema(resource_path)

    local_source = {
        "id": source_id,
        "original_id": source.get("id"),
        "type": "table",
        "file": source["file"],
        "path": str(resource_path),
        "format": "csv",
        "schema": schema,
    }

    remote_source = {
        "id": source_id,
        "type": "table",
        "format": "csv",
        "schema": copy.deepcopy(schema),
    }

    source_map = {
        "original_id": source.get("id"),
        "type": "table",
        "file": source["file"],
        "path": str(resource_path),
        "format": "csv",
    }
    return local_source, remote_source, source_map


def _normal_source_id(source_type: str, source_index: int) -> str:
    return f"{source_type}_{source_index}"


def _resolve_source_path(file_name: str, base_dir: Path) -> Path:
    # Task DSL paths are resolved relative to the dataset/case directory.
    path = Path(file_name).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Source file not found: {resolved}")
    return resolved
