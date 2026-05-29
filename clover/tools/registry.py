"""Static tool routing by task type."""

from __future__ import annotations

from typing import Any

from .table_reasoning import TABLE_REASONING_STATIC_TOOLS
from .table_reasoning.static_tools import StaticToolError


TASK_STATIC_TOOL_REGISTRY = {
    "table_reasoning": TABLE_REASONING_STATIC_TOOLS,
    "table_reasoning_v1": TABLE_REASONING_STATIC_TOOLS,
    "table_reasoning_v2": TABLE_REASONING_STATIC_TOOLS,
}


def static_tool_registry_for_task(task_type: str) -> dict[str, Any]:
    """Return the static tool registry for one task type."""

    try:
        return TASK_STATIC_TOOL_REGISTRY[task_type]
    except KeyError as exc:
        raise StaticToolError(f"Unsupported task_type for static tools: {task_type}") from exc


def get_static_tool(task_type: str, op: str) -> Any:
    """Return the static tool for a task type and Logic DAG op."""

    registry = static_tool_registry_for_task(task_type)
    try:
        return registry[op]
    except KeyError as exc:
        raise StaticToolError(
            f"Unsupported static tool op for {task_type}: {op}"
        ) from exc


def build_static_tool_call(
    task_type: str,
    node: dict[str, Any],
    *,
    resources: dict[str, Any] | None = None,
    upstream_outputs: dict[str, Any] | None = None,
    external_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a static tool call for one Logic DAG node."""

    tool = get_static_tool(task_type, node.get("op", ""))
    call = tool.build_call(
        node=node,
        resources=resources or {},
        upstream_outputs=upstream_outputs or {},
        external_params=external_params or {},
    )
    call["task_type"] = task_type
    return call


def build_static_tool_calls(
    logic_dag: dict[str, Any],
    *,
    resources: dict[str, Any] | None = None,
    upstream_outputs: dict[str, Any] | None = None,
    external_params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build static tool calls for every node in a Logic DAG."""

    task_type = logic_dag.get("task_type")
    if not task_type:
        raise StaticToolError("Logic DAG missing task_type")
    return [
        build_static_tool_call(
            task_type,
            node,
            resources=resources,
            upstream_outputs=upstream_outputs,
            external_params=external_params,
        )
        for node in logic_dag.get("nodes", [])
    ]
