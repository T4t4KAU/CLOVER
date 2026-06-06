"""Generic DSL preprocessing."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from clover.reasoning_profiles import (
    HINTS_KEY,
    PROFILE_KEY,
    table_reasoning_profile_from_dsl,
)
from clover.resource.preprocess.csv_schema import extract_csv_schema
from clover.resource.preprocess.pdf_schema import extract_pdf_schema
from clover.task_types import is_table_task_type, task_type_spec


SUPPORTED_SOURCE_TYPES = {"pdf", "table"}
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
    task_type = task_dsl["task_type"]

    base_path = Path(base_dir).expanduser().resolve()

    # The local DSL is for local execution and keeps absolute resource paths.
    local_dsl = {
        "task_type": task_type,
        "question": task_dsl["question"],
        "sources": [],
        "answer": copy.deepcopy(task_dsl["answer"]),
    }

    # The remote DSL is the payload intended for remote planning. It includes
    # structural metadata only, not local filesystem paths.
    remote_dsl = {
        "task_type": task_type,
        "question": task_dsl["question"],
        "sources": [],
        "answer": copy.deepcopy(task_dsl["answer"]),
    }
    _copy_optional_task_fields(task_dsl, local_dsl, remote_dsl)

    # Context is local runtime state used to map remote-visible source ids back
    # to concrete resources during later planning/execution stages.
    context: dict[str, Any] = {
        "task_type": task_type,
        "base_dir": str(base_path),
        "source_map": {},
    }

    for source_index, source in enumerate(task_dsl["sources"], start=1):
        local_source, remote_source, source_map, source_context = _preprocess_source(
            source=source,
            source_index=source_index,
            base_dir=base_path,
        )
        local_dsl["sources"].append(local_source)
        remote_dsl["sources"].append(remote_source)
        source_id = local_source["id"]
        context["source_map"][source_id] = source_map
        _merge_source_context(context, source_id, source_context)

    return {
        "local_dsl": local_dsl,
        "remote_dsl": remote_dsl,
        "context": context,
    }


def _copy_optional_task_fields(
    task_dsl: dict[str, Any],
    local_dsl: dict[str, Any],
    remote_dsl: dict[str, Any],
) -> None:
    for key in (PROFILE_KEY, HINTS_KEY):
        if key in task_dsl:
            local_dsl[key] = copy.deepcopy(task_dsl[key])
            remote_dsl[key] = copy.deepcopy(task_dsl[key])
    if PROFILE_KEY not in task_dsl and is_table_task_type(task_dsl.get("task_type")):
        profile = table_reasoning_profile_from_dsl(task_dsl)
        local_dsl[PROFILE_KEY] = profile
        remote_dsl[PROFILE_KEY] = profile


def _validate_task_dsl(task_dsl: dict[str, Any]) -> None:
    required_fields = ("task_type", "question", "sources", "answer")
    missing = [field for field in required_fields if field not in task_dsl]
    if missing:
        raise ValueError(f"Task DSL missing required fields: {missing}")

    if task_type_spec(task_dsl["task_type"]) is None:
        raise ValueError(f"Unsupported task_type: {task_dsl['task_type']}")

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
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_type = source["type"]
    resource_path = _resolve_source_path(source["file"], base_dir)

    if source_type == "table":
        return _preprocess_table_source(
            source,
            _normal_source_id(source_type, source_index),
            resource_path,
        )
    if source_type == "pdf":
        return _preprocess_pdf_source(
            source=source,
            source_id=_normal_source_id("document", source_index),
            resource_path=resource_path,
        )

    raise ValueError(f"Unsupported source type: {source_type}")


def _preprocess_table_source(
    source: dict[str, Any],
    source_id: str,
    resource_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
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
    return local_source, remote_source, source_map, {}


def _preprocess_pdf_source(
    *,
    source: dict[str, Any],
    source_id: str,
    resource_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if resource_path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF document sources are supported now: {resource_path}")

    metadata = _source_metadata(source)
    extracted = extract_pdf_schema(resource_path)
    schema = extracted["schema"]

    local_source = {
        "id": source_id,
        "original_id": source.get("id"),
        "type": "document",
        "source_type": "pdf",
        "file": source["file"],
        "path": str(resource_path),
        "format": "pdf",
        "schema": schema,
        **metadata,
    }

    remote_source = {
        "id": source_id,
        "type": "document",
        "source_type": "pdf",
        "format": "pdf",
        "schema": copy.deepcopy(schema),
        **metadata,
    }

    source_map = {
        "original_id": source.get("id"),
        "type": "document",
        "source_type": "pdf",
        "file": source["file"],
        "path": str(resource_path),
        "format": "pdf",
        "resource_cache": copy.deepcopy(extracted["resource_cache"]),
        **metadata,
    }
    source_context = {
        "chunk_map": extracted["chunk_map"],
        "resource_cache": extracted["resource_cache"],
    }
    return local_source, remote_source, source_map, source_context


def _merge_source_context(
    context: dict[str, Any],
    source_id: str,
    source_context: dict[str, Any],
) -> None:
    if "chunk_map" in source_context:
        context.setdefault("chunk_map", {})[source_id] = source_context["chunk_map"]
    if "resource_cache" in source_context:
        context.setdefault("resource_cache", {})[source_id] = source_context[
            "resource_cache"
        ]


def _source_metadata(source: dict[str, Any]) -> dict[str, Any]:
    reserved_keys = {
        "file",
        "format",
        "id",
        "original_id",
        "path",
        "schema",
        "source_type",
        "type",
    }
    return {
        key: copy.deepcopy(value)
        for key, value in source.items()
        if key not in reserved_keys and _is_prompt_safe_scalar(value)
    }


def _is_prompt_safe_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


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
