"""NodeAgent registry keyed by task type."""

from __future__ import annotations

from typing import Any

from clover.executor.agents.base import BaseNodeAgent
from clover.executor.agents.table_reasoning import TableReasoningNodeAgent
from clover.executor.context import NodeExecutionContext
from clover.executor.errors import UnsupportedTaskExecutionError
from clover.executor.sandbox import build_agent_sandbox


NODE_AGENT_REGISTRY: dict[str, type[BaseNodeAgent]] = {
    "table_reasoning": TableReasoningNodeAgent,
    "table_reasoning_v1": TableReasoningNodeAgent,
    "table_reasoning_v2": TableReasoningNodeAgent,
}


def node_agent_class_for_task(task_type: str) -> type[BaseNodeAgent]:
    try:
        return NODE_AGENT_REGISTRY[task_type]
    except KeyError as exc:
        raise UnsupportedTaskExecutionError(
            f"No NodeAgent registered for task_type: {task_type}"
        ) from exc


def build_node_agent(context: NodeExecutionContext) -> BaseNodeAgent:
    agent_class = node_agent_class_for_task(context.task_type)
    return agent_class(context, sandbox=build_agent_sandbox(context))
