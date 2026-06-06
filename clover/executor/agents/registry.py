"""NodeAgent registry keyed by task type."""

from __future__ import annotations

from clover.executor.agents.base import BaseNodeAgent
from clover.executor.agents.document_reasoning import DocumentReasoningNodeAgent
from clover.executor.agents.table_reasoning import TableReasoningNodeAgent
from clover.executor.context import NodeExecutionContext
from clover.executor.errors import UnsupportedTaskExecutionError
from clover.executor.sandbox import build_agent_sandbox
from clover.task_types import agent_kind_for_task_type


NODE_AGENT_REGISTRY_BY_KIND: dict[str, type[BaseNodeAgent]] = {
    "document_reasoning": DocumentReasoningNodeAgent,
    "table_reasoning": TableReasoningNodeAgent,
}


def node_agent_class_for_task(task_type: str) -> type[BaseNodeAgent]:
    agent_kind = agent_kind_for_task_type(task_type)
    try:
        return NODE_AGENT_REGISTRY_BY_KIND[agent_kind]
    except KeyError as exc:
        raise UnsupportedTaskExecutionError(
            f"No NodeAgent registered for task_type: {task_type}"
        ) from exc


def build_node_agent(context: NodeExecutionContext) -> BaseNodeAgent:
    agent_class = node_agent_class_for_task(context.task_type)
    return agent_class(context, sandbox=build_agent_sandbox(context))
