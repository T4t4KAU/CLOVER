"""Execution errors for CLOVER physical DAGs."""

from __future__ import annotations

from typing import Any


class ExecutionError(RuntimeError):
    """Base error raised by the physical DAG executor."""


class PlanValidationError(ExecutionError):
    """Raised when a physical plan is malformed or cannot be scheduled."""


class UnsupportedTaskExecutionError(ExecutionError):
    """Raised when no NodeAgent is registered for a task type."""


class AgentLoopNotImplementedError(ExecutionError):
    """Raised when a node misses Fast Path and no Agent Loop exists yet."""


class NodeExecutionError(ExecutionError):
    """Raised when a single node cannot be executed."""

    def __init__(
        self,
        message: str,
        *,
        node: dict[str, Any] | None = None,
        original: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.node = node
        self.original = original


class NodeTimeoutError(NodeExecutionError):
    """Raised when one execution unit exceeds its node timeout."""

    def __init__(
        self,
        message: str,
        *,
        node: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
        elapsed_ms: float | None = None,
    ) -> None:
        super().__init__(message, node=node)
        self.timeout_seconds = timeout_seconds
        self.elapsed_ms = elapsed_ms


class CollectorExecutionError(ExecutionError):
    """Raised when a post-node result collector cannot run."""

    def __init__(
        self,
        message: str,
        *,
        collector: dict[str, Any] | None = None,
        original: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.collector = collector
        self.original = original
