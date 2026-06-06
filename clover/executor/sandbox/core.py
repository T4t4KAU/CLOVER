"""Agent sandbox primitives for local NodeAgent loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from clover.executor.context import NodeExecutionContext
from clover.executor.node_views import NodeView


class SandboxError(RuntimeError):
    """Raised when an Agent sandbox action cannot be completed."""


@dataclass
class SandboxActionResult:
    """Result returned by a sandbox action."""

    ok: bool
    observation: dict[str, Any] | None = None
    output: Any = None
    accepted: bool = False
    terminal: bool = False
    error: dict[str, Any] | None = None


class SandboxPolicy(Protocol):
    """Task-specific workspace policy.

    This is a lifecycle boundary for a NodeAgent workspace, not an operating
    system security boundary.
    """

    def start(
        self,
        context: NodeExecutionContext,
        *,
        decision: Any,
        trigger: str,
        error: Exception | None,
    ) -> Any:
        """Create task-specific sandbox state for one Agent Loop."""

    def view(self, state: Any, observations: list[dict[str, Any]]) -> NodeView:
        """Return the model-facing view visible to the NodeAgent."""

    def run_action(self, state: Any, action: dict[str, Any]) -> SandboxActionResult:
        """Execute one action inside the sandbox."""

    def close(self, state: Any) -> None:
        """Release task-specific workspace state after one Agent Loop."""


class AgentSandbox:
    """Small-world wrapper around one NodeAgent invocation."""

    def __init__(
        self,
        context: NodeExecutionContext,
        policy: SandboxPolicy,
    ) -> None:
        self.context = context
        self.policy = policy
        self._state: Any | None = None

    def start(
        self,
        *,
        decision: Any,
        trigger: str,
        error: Exception | None = None,
    ) -> None:
        self._state = self.policy.start(
            self.context,
            decision=decision,
            trigger=trigger,
            error=error,
        )

    def view(self, observations: list[dict[str, Any]]) -> NodeView:
        return self.policy.view(self._require_state(), observations)

    def run_action(self, action: dict[str, Any]) -> SandboxActionResult:
        return self.policy.run_action(self._require_state(), action)

    def close(self) -> None:
        """Release the current workspace state."""

        if self._state is None:
            return
        try:
            self.policy.close(self._state)
        finally:
            self._state = None

    def _require_state(self) -> Any:
        if self._state is None:
            raise SandboxError("Agent sandbox has not been started")
        return self._state
