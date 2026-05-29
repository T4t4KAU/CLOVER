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
    return value
