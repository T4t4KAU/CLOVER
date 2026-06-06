"""Sandbox registry keyed by task type."""

from __future__ import annotations

from clover.executor.context import NodeExecutionContext
from clover.executor.sandbox.core import AgentSandbox, SandboxPolicy
from clover.executor.sandbox.document_reasoning import DocumentReasoningSandboxPolicy
from clover.executor.sandbox.table_reasoning import TableReasoningSandboxPolicy
from clover.task_types import agent_kind_for_task_type


SANDBOX_POLICY_REGISTRY_BY_KIND: dict[str, type[SandboxPolicy]] = {
    "document_reasoning": DocumentReasoningSandboxPolicy,
    "table_reasoning": TableReasoningSandboxPolicy,
}


def build_agent_sandbox(context: NodeExecutionContext) -> AgentSandbox | None:
    """Return a task-specific Agent sandbox, if the task supports one."""

    agent_kind = agent_kind_for_task_type(context.task_type)
    policy_class = SANDBOX_POLICY_REGISTRY_BY_KIND.get(agent_kind)
    if policy_class is None:
        return None
    return AgentSandbox(context, policy_class())
