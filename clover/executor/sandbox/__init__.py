"""Agent sandbox support for Executor-owned NodeAgents."""

from clover.executor.sandbox.core import (
    AgentSandbox,
    SandboxActionResult,
    SandboxError,
)
from clover.executor.sandbox.registry import build_agent_sandbox

__all__ = [
    "AgentSandbox",
    "SandboxActionResult",
    "SandboxError",
    "build_agent_sandbox",
]
