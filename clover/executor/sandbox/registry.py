"""Sandbox registry keyed by task type."""

from __future__ import annotations

from clover.executor.context import NodeExecutionContext
from clover.executor.sandbox.core import AgentSandbox
from clover.executor.sandbox.table_reasoning import TableReasoningSandboxPolicy


def build_agent_sandbox(context: NodeExecutionContext) -> AgentSandbox | None:
    """Return a task-specific Agent sandbox, if the task supports one."""

    if context.task_type in {
        "table_reasoning",
        "table_reasoning_v1",
        "table_reasoning_v2",
    }:
        return AgentSandbox(context, TableReasoningSandboxPolicy())
    return None
