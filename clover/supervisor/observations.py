"""Compact observation builders for Supervisor prompts."""

from __future__ import annotations

from typing import Any

from clover.executor.result import ExecutionResult, json_ready
from clover.supervisor.prompt_safety import strip_sensitive_prompt_fields


DEFAULT_DOCUMENT_EVIDENCE_CHARS = 6000
DOCUMENT_COLLECTOR_KINDS = frozenset(
    {
        "map_group_evidence",
        "minions_transform_outputs",
    }
)


def build_compact_document_observation(
    observation: ExecutionResult,
    *,
    round_index: int | None = None,
    feedback: str | None = None,
    scratchpad: str | None = None,
    max_evidence_chars: int = DEFAULT_DOCUMENT_EVIDENCE_CHARS,
) -> dict[str, Any]:
    """Return the small document observation payload visible to Supervisor."""

    payload = _compact_execution_observation(
        observation,
        max_evidence_chars=max_evidence_chars,
    )
    if round_index is not None:
        payload["round_index"] = round_index
    else:
        payload.setdefault("round_index", None)
    payload["feedback"] = feedback if feedback is not None else payload.get("feedback")
    payload["scratchpad"] = (
        scratchpad if scratchpad is not None else payload.get("scratchpad")
    )
    return strip_sensitive_prompt_fields(json_ready(payload))


def _compact_execution_observation(
    observation: ExecutionResult,
    *,
    max_evidence_chars: int,
) -> dict[str, Any]:
    collector_summary = _summarize_document_collectors(observation.collector_outputs)
    evidence_summary, evidence_truncated = _truncate_text(
        collector_summary["evidence_summary"],
        max_chars=max_evidence_chars,
    )
    return {
        "ok": observation.ok,
        "worker_count": collector_summary["worker_count"],
        "included_count": collector_summary["included_count"],
        "failed_count": _failed_trace_count(observation.traces),
        "evidence_summary": evidence_summary,
        "evidence_truncated": evidence_truncated,
        "prior_evidence_summary": "",
        "prior_evidence_round_count": 0,
        "prior_evidence_truncated": False,
        "fallback_used": collector_summary["fallback_used"],
        "transform_error": collector_summary["transform_error"],
        "error": _compact_error(observation.error) if observation.ok is False else None,
        "round_index": None,
        "feedback": None,
        "scratchpad": None,
    }


def _summarize_document_collectors(collector_outputs: Any) -> dict[str, Any]:
    worker_count = 0
    included_count = 0
    fallback_used = False
    transform_errors: list[str] = []
    evidence_parts: list[str] = []

    if isinstance(collector_outputs, dict):
        for value in collector_outputs.values():
            if not isinstance(value, dict):
                continue
            if value.get("kind") not in DOCUMENT_COLLECTOR_KINDS:
                continue
            worker_count += _non_negative_int(value.get("worker_count"))
            included_count += _non_negative_int(value.get("included_count"))
            fallback_used = fallback_used or bool(value.get("fallback_used", False))
            transform_error = _optional_string(value.get("transform_error"))
            if transform_error:
                transform_errors.append(transform_error)
            summary = _optional_string(value.get("evidence_summary"))
            if summary:
                evidence_parts.append(summary.strip())

    return {
        "worker_count": worker_count,
        "included_count": included_count,
        "fallback_used": fallback_used,
        "transform_error": "; ".join(transform_errors) if transform_errors else None,
        "evidence_summary": "\n\n".join(evidence_parts),
    }


def _failed_trace_count(traces: list[Any]) -> int:
    failed = 0
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        if trace.get("soft_failure") is True:
            failed += 1
            continue
        if str(trace.get("status", "")).lower() in {"failed", "error", "timeout"}:
            failed += 1
    return failed


def _truncate_text(text: str, *, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", bool(text)
    if len(text) <= max_chars:
        return text, False
    suffix = "\n...[truncated]"
    kept = max(0, max_chars - len(suffix))
    return text[:kept].rstrip() + suffix, True


def _compact_error(error: Any) -> dict[str, Any] | None:
    if error is None:
        return None
    if isinstance(error, dict):
        payload = {
            key: error.get(key)
            for key in ("type", "message")
            if error.get(key) is not None
        }
        return payload or None
    return {"message": str(error)}


def _non_negative_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def _optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _string_or_empty(value: Any) -> str:
    return _optional_string(value) or ""
