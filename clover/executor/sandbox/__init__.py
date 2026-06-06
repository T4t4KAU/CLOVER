"""Agent sandbox support for Executor-owned NodeAgents."""

from clover.executor.sandbox.core import (
    AgentSandbox,
    SandboxActionResult,
    SandboxError,
    SandboxPolicy,
)
from clover.executor.sandbox.document_reasoning import DocumentReasoningSandboxPolicy
from clover.executor.sandbox.registry import build_agent_sandbox
from clover.executor.sandbox.table_reasoning import TableReasoningSandboxPolicy

__all__ = [
    "AgentSandbox",
    "DocumentReasoningSandboxPolicy",
    "SandboxActionResult",
    "SandboxError",
    "SandboxPolicy",
    "TableReasoningSandboxPolicy",
    "build_agent_sandbox",
]
