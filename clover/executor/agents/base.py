"""Base NodeAgent abstractions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from clover.executor.context import NodeExecutionContext
from clover.executor.errors import AgentLoopNotImplementedError
from clover.executor.result import (
    NodeExecutionRecord,
    error_payload,
    summarize_output,
)


@dataclass(frozen=True)
class FastPathDecision:
    """Decision returned by a NodeAgent Fast Path check."""

    hit: bool
    call: dict[str, Any] | None = None
    tool: str | None = None
    backend: str | None = None
    miss_reason: str | None = None
    miss_detail: str | None = None


class BaseNodeAgent:
    """Base class for task-specific node agents."""

    backend_name = "unknown"

    def __init__(self, context: NodeExecutionContext, *, sandbox: Any | None = None) -> None:
        self.context = context
        self.node = context.node
        self.sandbox = sandbox
        self.agent_loop_trace: dict[str, Any] | None = None

    def run(self) -> NodeExecutionRecord:
        """Run one node through Fast Path first, then Agent Loop if needed."""

        started = time.perf_counter()
        decision = self.try_fast_path()
        trace = self._base_trace(decision)
        try:
            if decision.hit:
                # Fast Path decisions are pure capability checks; execution is
                # kept separate so traces can record why a miss happened.
                try:
                    output = self.execute_fast_path(decision)
                except Exception as exc:
                    if not self.should_run_agent_loop(
                        decision,
                        trigger="fast_path_execution_error",
                        error=exc,
                    ):
                        raise
                    trace["execution_path"] = "agent_loop"
                    trace["agent_loop_trigger"] = "fast_path_execution_error"
                    trace["fast_path_error"] = error_payload(exc)
                    output = self.run_agent_loop(
                        decision,
                        trigger="fast_path_execution_error",
                        error=exc,
                    )
                elapsed_ms = _elapsed_ms(started)
                trace.update(
                    {
                        "status": "ok",
                        "elapsed_ms": elapsed_ms,
                        "output_summary": summarize_output(output),
                    }
                )
                if self.agent_loop_trace is not None:
                    trace["agent_loop"] = self.agent_loop_trace
                return NodeExecutionRecord(
                    ok=True,
                    node_id=self.node.get("id"),
                    op=self.node.get("op"),
                    output_name=self.node.get("output"),
                    output=output,
                    trace=trace,
                )

            trace["agent_loop_trigger"] = "fast_path_miss"
            output = self.run_agent_loop(decision)
            elapsed_ms = _elapsed_ms(started)
            trace.update(
                {
                    "status": "ok",
                    "elapsed_ms": elapsed_ms,
                    "output_summary": summarize_output(output),
                }
            )
            if self.agent_loop_trace is not None:
                trace["agent_loop"] = self.agent_loop_trace
            return NodeExecutionRecord(
                ok=True,
                node_id=self.node.get("id"),
                op=self.node.get("op"),
                output_name=self.node.get("output"),
                output=output,
                trace=trace,
            )
        except Exception as exc:  # noqa: BLE001 - node failures become execution records.
            elapsed_ms = _elapsed_ms(started)
            error = error_payload(exc)
            trace.update(
                {
                    "status": "failed",
                    "elapsed_ms": elapsed_ms,
                    "error": error,
                }
            )
            if self.agent_loop_trace is not None:
                trace["agent_loop"] = self.agent_loop_trace
            return NodeExecutionRecord(
                ok=False,
                node_id=self.node.get("id"),
                op=self.node.get("op"),
                output_name=self.node.get("output"),
                trace=trace,
                error=error,
            )

    def try_fast_path(self) -> FastPathDecision:
        """Return whether this node can be handled by deterministic tools."""

        raise NotImplementedError

    def execute_fast_path(self, decision: FastPathDecision) -> Any:
        """Execute a previously accepted Fast Path decision."""

        raise NotImplementedError

    def should_run_agent_loop(
        self,
        decision: FastPathDecision,
        *,
        trigger: str,
        error: Exception | None = None,
    ) -> bool:
        """Return whether an Agent Loop should attempt this recovery."""

        return not decision.hit and error is None

    def run_agent_loop(
        self,
        decision: FastPathDecision,
        *,
        trigger: str = "fast_path_miss",
        error: Exception | None = None,
    ) -> Any:
        """Run the task-specific Agent Loop after a Fast Path miss."""

        reason = decision.miss_reason or "fast_path_miss"
        raise AgentLoopNotImplementedError(
            f"Agent Loop is not implemented for node {self.node.get('id')} "
            f"after Fast Path miss: {reason}"
        )

    def _base_trace(self, decision: FastPathDecision) -> dict[str, Any]:
        execution_path = "fast_path" if decision.hit else "agent_loop"
        trace = {
            "node_id": self.node.get("id"),
            "op": self.node.get("op"),
            "output": self.node.get("output"),
            "dependency": list(self.node.get("dependency", [])),
            "input": list(self.node.get("input", [])),
            "execution_path": execution_path,
            "fast_path_hit": decision.hit,
            "tool": decision.tool,
            "backend": decision.backend,
        }
        if not decision.hit:
            trace["fast_path_miss_reason"] = decision.miss_reason
            trace["fast_path_miss_detail"] = decision.miss_detail
        return trace


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
