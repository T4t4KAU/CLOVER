"""Result and trace helpers for physical DAG execution."""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any


@dataclass
class NodeExecutionRecord:
    """Result of running one physical plan node."""

    ok: bool
    node_id: str | None
    op: str | None
    output_name: str | None
    output: Any = None
    trace: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None


@dataclass
class ExecutionResult:
    """Final result of executing a physical plan."""

    ok: bool
    answer: Any
    outputs: dict[str, Any]
    traces: list[dict[str, Any]]
    output_summaries: dict[str, dict[str, Any]]
    collector_outputs: dict[str, Any] = field(default_factory=dict)
    failing_node: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    elapsed_ms: float = 0.0
    fast_path_hits: int = 0
    fast_path_misses: int = 0

    def to_dict(self, *, include_outputs: bool = False) -> dict[str, Any]:
        """Return a JSON-friendly representation of the execution result."""

        payload = {
            "ok": self.ok,
            "answer": json_ready(self.answer),
            "collector_outputs": json_ready(self.collector_outputs),
            "traces": json_ready(self.traces),
            "output_summaries": json_ready(self.output_summaries),
            "failing_node": json_ready(self.failing_node),
            "error": json_ready(self.error),
            "elapsed_ms": self.elapsed_ms,
            "fast_path_hits": self.fast_path_hits,
            "fast_path_misses": self.fast_path_misses,
        }
        if include_outputs:
            payload["outputs"] = json_ready(self.outputs)
        return payload


def slice_execution_result_by_namespace(
    result: ExecutionResult,
    namespace: str,
) -> ExecutionResult:
    """Return a single-plan view from a namespaced mixed execution result."""

    prefix = f"{namespace}__"
    traces = [
        _strip_trace_namespace(trace, prefix=prefix)
        for trace in result.traces
        if _trace_belongs_to_namespace(trace, prefix=prefix)
    ]
    collector_outputs = _strip_namespaced_keys(
        result.collector_outputs,
        prefix=prefix,
    )
    failing_node = _strip_failing_node(result.failing_node, prefix=prefix)
    error = result.error if failing_node is not None else _namespace_error(result, traces)
    ok = result.ok and failing_node is None and not _has_failed_trace(traces)
    return ExecutionResult(
        ok=ok,
        answer=_slice_answer(result.answer, prefix=prefix),
        outputs=_strip_namespaced_keys(result.outputs, prefix=prefix),
        traces=traces,
        output_summaries=_strip_namespaced_keys(
            result.output_summaries,
            prefix=prefix,
        ),
        collector_outputs=collector_outputs,
        failing_node=failing_node,
        error=error if not ok else None,
        elapsed_ms=result.elapsed_ms,
        fast_path_hits=sum(1 for trace in traces if trace.get("fast_path_hit") is True),
        fast_path_misses=sum(1 for trace in traces if trace.get("fast_path_hit") is False),
    )


def error_payload(exc: Exception) -> dict[str, Any]:
    """Build a compact, serializable error payload."""

    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exception(
            type(exc),
            exc,
            exc.__traceback__,
        )[-6:],
    }


def _slice_answer(answer: Any, *, prefix: str) -> Any:
    if not isinstance(answer, dict):
        return None
    return _strip_namespaced_keys(answer, prefix=prefix)


def _strip_namespaced_keys(payload: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    return {
        key[len(prefix) :]: value
        for key, value in payload.items()
        if isinstance(key, str) and key.startswith(prefix)
    }


def _trace_belongs_to_namespace(trace: Any, *, prefix: str) -> bool:
    if not isinstance(trace, dict):
        return False
    for key in ("node_id", "output", "job_id", "sequence_id"):
        value = trace.get(key)
        if isinstance(value, str) and value.startswith(prefix):
            return True
    agent_loop = trace.get("agent_loop")
    if isinstance(agent_loop, dict) and _contains_namespaced_string(
        agent_loop,
        prefix=prefix,
    ):
        return True
    return False


def _strip_trace_namespace(trace: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    return _strip_namespace_from_trace_value(trace, prefix=prefix)


def _strip_namespace_from_trace_value(value: Any, *, prefix: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_namespace_from_trace_value(item, prefix=prefix)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_strip_namespace_from_trace_value(item, prefix=prefix) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_namespace_from_trace_value(item, prefix=prefix) for item in value)
    if isinstance(value, str) and value.startswith(prefix):
        return value[len(prefix) :]
    return value


def _contains_namespaced_string(value: Any, *, prefix: str) -> bool:
    if isinstance(value, str):
        return value.startswith(prefix)
    if isinstance(value, dict):
        return any(_contains_namespaced_string(item, prefix=prefix) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_namespaced_string(item, prefix=prefix) for item in value)
    return False


def _strip_failing_node(
    failing_node: dict[str, Any] | None,
    *,
    prefix: str,
) -> dict[str, Any] | None:
    if not isinstance(failing_node, dict):
        return None
    belongs = False
    stripped = {}
    for key, value in failing_node.items():
        if isinstance(value, str) and value.startswith(prefix):
            belongs = True
            stripped[key] = value[len(prefix) :]
        else:
            stripped[key] = value
    return stripped if belongs else None


def _namespace_error(
    result: ExecutionResult,
    traces: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for trace in traces:
        error = trace.get("error")
        if isinstance(error, dict):
            return error
    return result.error if not result.ok else None


def _has_failed_trace(traces: list[dict[str, Any]]) -> bool:
    for trace in traces:
        if trace.get("soft_failure") is True:
            continue
        status = str(trace.get("status", "")).lower()
        if status in {"failed", "error", "timeout"}:
            return True
    return False


def summarize_output(value: Any, *, max_rows: int = 3) -> dict[str, Any]:
    """Summarize an intermediate output without serializing whole tables."""

    frame = getattr(value, "frame", None)
    if frame is not None:
        summary = {
            "type": "table",
            "rows": int(len(frame)),
            "columns": [str(column) for column in frame.columns],
            "preview": json_ready(frame.head(max_rows).to_dict(orient="records")),
        }
        group_keys = getattr(value, "group_keys", None)
        if group_keys:
            summary["group_keys"] = json_ready(group_keys)
        return summary
    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "preview": json_ready(value[:max_rows]),
        }
    return {
        "type": "value",
        "preview": json_ready(value),
    }


def json_ready(value: Any) -> Any:
    """Convert common runtime objects into JSON-compatible values."""

    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    frame = getattr(value, "frame", None)
    if frame is not None:
        return summarize_output(value)
    if hasattr(value, "item"):
        try:
            return json_ready(value.item())
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, type):
        return value.__name__
    return str(value)
