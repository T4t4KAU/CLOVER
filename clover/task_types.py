"""Public task type registry.

Task types name CLOVER's user-facing reasoning semantics. Agent and tool names
remain implementation capabilities and may be shared across task types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TABLE_REASONING_QUERY_TASK_TYPE = "table_reasoning.query"
TABLE_REASONING_ANALYZE_TASK_TYPE = "table_reasoning.analyze"
DOCUMENT_REASONING_TASK_TYPE = "document_reasoning"


@dataclass(frozen=True)
class TaskTypeSpec:
    """Routing metadata for one public task type."""

    task_type: str
    family: str
    capability: str
    agent_kind: str
    supervisor_route: tuple[str, ...]


TASK_TYPE_SPECS: dict[str, TaskTypeSpec] = {
    TABLE_REASONING_QUERY_TASK_TYPE: TaskTypeSpec(
        task_type=TABLE_REASONING_QUERY_TASK_TYPE,
        family="table",
        capability="query",
        agent_kind="table_reasoning",
        supervisor_route=("table_reasoning", "query"),
    ),
    TABLE_REASONING_ANALYZE_TASK_TYPE: TaskTypeSpec(
        task_type=TABLE_REASONING_ANALYZE_TASK_TYPE,
        family="table",
        capability="analyze",
        agent_kind="table_reasoning",
        supervisor_route=("table_reasoning", "query"),
    ),
    DOCUMENT_REASONING_TASK_TYPE: TaskTypeSpec(
        task_type=DOCUMENT_REASONING_TASK_TYPE,
        family="document",
        capability="reasoning",
        agent_kind="document_reasoning",
        supervisor_route=("document_reasoning",),
    ),
}


def public_task_types() -> tuple[str, ...]:
    """Return supported public task types."""

    return tuple(TASK_TYPE_SPECS)


def task_type_spec(task_type: Any) -> TaskTypeSpec | None:
    """Return registry metadata for a public task type, if known."""

    if not isinstance(task_type, str):
        return None
    return TASK_TYPE_SPECS.get(task_type)


def require_task_type_spec(task_type: str) -> TaskTypeSpec:
    """Return registry metadata or raise for an unsupported task type."""

    spec = task_type_spec(task_type)
    if spec is None:
        available = ", ".join(public_task_types())
        raise ValueError(f"Unsupported task_type: {task_type!r}. Available: {available}")
    return spec


def agent_kind_for_task_type(task_type: str) -> str:
    """Return the implementation agent kind for one public task type."""

    return require_task_type_spec(task_type).agent_kind


def is_table_task_type(task_type: Any) -> bool:
    """Return whether a task type belongs to table reasoning capabilities."""

    spec = task_type_spec(task_type)
    return spec is not None and spec.family == "table"


def is_document_task_type(task_type: Any) -> bool:
    """Return whether a task type belongs to document reasoning capabilities."""

    spec = task_type_spec(task_type)
    return spec is not None and spec.family == "document"
