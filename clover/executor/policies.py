"""Execution policies shared by scheduler and executor runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from clover.executor.result import NodeExecutionRecord
from clover.executor.scheduler import ExecutionUnit


class NodeFailurePolicy(Protocol):
    """Decide how a failed execution unit should affect plan execution."""

    def should_soft_fail(
        self,
        unit: ExecutionUnit,
        record: NodeExecutionRecord,
    ) -> bool:
        """Return whether this failed unit should become a placeholder output."""

    def soft_failure_output(
        self,
        unit: ExecutionUnit,
        record: NodeExecutionRecord,
    ) -> Any:
        """Return the artifact value written when a failed unit is soft-failed."""


@dataclass(frozen=True)
class DefaultNodeFailurePolicy:
    """Default failure semantics for executor-ready units.

    Map-group units are independent evidence workers, so a failed worker can be
    represented as empty evidence and collected with the rest of the group.
    Ordinary physical nodes remain fail-fast because downstream operators depend
    on their exact value.
    """

    soft_source_kinds: frozenset[str] = frozenset({"map_group"})

    def should_soft_fail(
        self,
        unit: ExecutionUnit,
        record: NodeExecutionRecord,
    ) -> bool:
        mode = _failure_mode(unit)
        if mode == "hard":
            return False
        if mode == "soft":
            return True
        return str(unit.metadata.get("source_kind") or "") in self.soft_source_kinds

    def soft_failure_output(
        self,
        unit: ExecutionUnit,
        record: NodeExecutionRecord,
    ) -> Any:
        if str(unit.metadata.get("source_kind") or "") == "map_group":
            return _map_group_failure_placeholder(unit, record)
        return {
            "ok": False,
            "node_id": unit.id,
            "output": unit.output,
            "error": _record_error(record),
        }


def node_failure_policy_for_plan(
    physical_plan: dict[str, Any],
) -> NodeFailurePolicy:
    """Return the failure policy for an executor physical plan."""

    policy = physical_plan.get("failure_policy")
    if not isinstance(policy, dict):
        return DefaultNodeFailurePolicy()
    soft_source_kinds = policy.get("soft_source_kinds")
    if isinstance(soft_source_kinds, list) and all(
        isinstance(item, str) and item for item in soft_source_kinds
    ):
        return DefaultNodeFailurePolicy(soft_source_kinds=frozenset(soft_source_kinds))
    return DefaultNodeFailurePolicy()


def _failure_mode(unit: ExecutionUnit) -> str | None:
    mode = unit.metadata.get("failure_mode")
    if mode is None:
        mode = unit.node.get("failure_mode")
    if mode is None:
        return None
    value = str(mode).strip().lower()
    return value if value in {"soft", "hard"} else None


def _map_group_failure_placeholder(
    unit: ExecutionUnit,
    record: NodeExecutionRecord,
) -> dict[str, Any]:
    error = _record_error(record)
    message = error.get("message") if isinstance(error, dict) else str(error)
    chunk: dict[str, Any] = {}
    for key in ("chunk_resource_id", "chunk_index", "replica_index"):
        value = unit.metadata.get(key)
        if value is not None:
            chunk[key] = value
    return {
        "answer": None,
        "citation": None,
        "explanation": f"worker failed: {message}",
        "chunk": chunk,
        "sample": "",
        "error": error,
    }


def _record_error(record: NodeExecutionRecord) -> dict[str, Any]:
    if isinstance(record.error, dict):
        return record.error
    return {"type": "NodeExecutionError", "message": "node failed"}
