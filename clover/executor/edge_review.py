"""Bounded local evidence review for table answers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from clover.config import ENABLE_EDGE_AGENT, runtime_feature_enabled
from clover.executor.agents.template_tree import (
    TABLE_LOCAL_REVIEW_LEAF_KEY,
    render_table_local_review_prompt,
)
from clover.executor.result import json_ready
from clover.executor.slm_dispatcher import (
    LocalSlmSequenceDispatcher,
    LocalSlmSequenceRequest,
)
from clover.supervisor.client import extract_token_usage
from clover.supervisor.decision import extract_supervisor_json

EDGE_REVIEW_OFF = "off"
EDGE_REVIEW_SHADOW = "shadow"
EDGE_REVIEW_SAFE = "safe"
EDGE_REVIEW_MODES = frozenset(
    {
        EDGE_REVIEW_OFF,
        EDGE_REVIEW_SHADOW,
        EDGE_REVIEW_SAFE,
    }
)

_NUMBER_TYPES = frozenset({"number", "float", "integer", "int"})
_BOOLEAN_TYPES = frozenset({"boolean", "bool"})
_TEXT_TYPES = frozenset({"string", "entity", "category"})
_IGNORED_EVIDENCE_KEYS = frozenset(
    {
        "_clover_full_res",
        "execution_traces",
        "logic_dag",
        "q",
        "sql",
    }
)
_NON_VALUE_FACT_KEYS = frozenset(
    {
        "cols",
        "columns",
        "i",
        "kind",
        "n",
        "ok",
        "op",
        "reason",
        "route",
    }
)


@dataclass(frozen=True)
class EdgeReviewFact:
    """One scalar fact that the Edge reviewer may cite."""

    fact_id: str
    path: str
    value: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.fact_id,
            "path": self.path,
            "value": json_ready(self.value),
        }


@dataclass(frozen=True)
class EdgeReviewResult:
    """Validated result from one local Edge review call."""

    mode: str
    route: str
    accepted: bool
    answer: Any
    reason: str | None
    operation: str | None
    support: tuple[str, ...]
    prompt: str
    response: str
    raw: dict[str, Any]
    token_usage: dict[str, int]
    sequence_trace: dict[str, Any]
    validation_error: str | None = None

    def to_trace(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "route": self.route,
            "accepted": self.accepted,
            "answer": json_ready(self.answer),
            "reason": self.reason,
            "operation": self.operation,
            "support": list(self.support),
            "raw": json_ready(self.raw),
            "token_usage": dict(self.token_usage),
            "sequence": dict(self.sequence_trace),
            "validation_error": self.validation_error,
        }


def edge_review_mode(slm_config: dict[str, Any] | None) -> str:
    """Return the configured local review mode."""

    if not isinstance(slm_config, dict):
        return EDGE_REVIEW_OFF
    selected = str(slm_config.get("edge_review_mode") or EDGE_REVIEW_OFF).strip().lower()
    if selected not in EDGE_REVIEW_MODES:
        available = ", ".join(sorted(EDGE_REVIEW_MODES))
        raise ValueError(
            f"Unsupported edge_review_mode: {selected!r}. Available: {available}"
        )
    return selected


def run_edge_local_review(
    *,
    question: str,
    answer_type: str,
    evidence: Any,
    scope: str,
    slm_config: dict[str, Any] | None,
    dispatcher: LocalSlmSequenceDispatcher | None,
    job_id: str,
) -> EdgeReviewResult | None:
    """Review small, clean evidence and return a statically validated answer."""

    if not runtime_feature_enabled(slm_config, ENABLE_EDGE_AGENT):
        return None
    mode = edge_review_mode(slm_config)
    if mode == EDGE_REVIEW_OFF or dispatcher is None:
        return None
    prepared = _prepare_review_payload(
        question=question,
        answer_type=answer_type,
        evidence=evidence,
        scope=scope,
        slm_config=slm_config,
    )
    if prepared is None:
        return None
    facts, payload = prepared
    prompt = render_table_local_review_prompt(payload=payload)
    sequence_result = dispatcher.generate(
        LocalSlmSequenceRequest(
            prompt=prompt,
            leaf_key=TABLE_LOCAL_REVIEW_LEAF_KEY,
            prompt_kind="table_reasoning_local_review",
            node_id=job_id,
            job_id=job_id,
            iteration=0,
            slm_config=slm_config,
        )
    )
    response = sequence_result.text
    token_usage = extract_token_usage(sequence_result.response_payload)
    sequence_trace = sequence_result.trace_metadata()
    try:
        raw = extract_supervisor_json(response)
    except Exception as exc:  # noqa: BLE001 - invalid local output falls back to Cloud.
        return EdgeReviewResult(
            mode=mode,
            route="escalate",
            accepted=False,
            answer=None,
            reason="invalid_edge_review_json",
            operation=None,
            support=(),
            prompt=prompt,
            response=response,
            raw={},
            token_usage=token_usage,
            sequence_trace=sequence_trace,
            validation_error=str(exc),
        )
    return _validate_review_response(
        raw=raw,
        mode=mode,
        answer_type=answer_type,
        facts=facts,
        prompt=prompt,
        response=response,
        token_usage=token_usage,
        sequence_trace=sequence_trace,
    )


def _prepare_review_payload(
    *,
    question: str,
    answer_type: str,
    evidence: Any,
    scope: str,
    slm_config: dict[str, Any] | None,
) -> tuple[tuple[EdgeReviewFact, ...], dict[str, Any]] | None:
    normalized_type = str(answer_type or "").strip().lower()
    if not _supported_answer_type(normalized_type):
        return None
    limits = {
        "max_actions": _positive_config_int(slm_config, "edge_review_max_actions", 4),
        "max_rows": _positive_config_int(slm_config, "edge_review_max_rows", 5),
        "max_columns": _positive_config_int(slm_config, "edge_review_max_columns", 5),
        "max_facts": _positive_config_int(slm_config, "edge_review_max_facts", 40),
    }
    ready = _bounded_evidence(evidence, **limits)
    if ready is None or not _clean_evidence(ready):
        return None
    facts = _evidence_facts(ready, max_facts=limits["max_facts"])
    if not facts:
        return None
    payload = {
        "scope": scope,
        "question": str(question or ""),
        "answer_type": normalized_type,
        "facts": [fact.to_dict() for fact in facts],
    }
    return facts, payload


def _validate_review_response(
    *,
    raw: dict[str, Any],
    mode: str,
    answer_type: str,
    facts: tuple[EdgeReviewFact, ...],
    prompt: str,
    response: str,
    token_usage: dict[str, int],
    sequence_trace: dict[str, Any],
) -> EdgeReviewResult:
    route = str(raw.get("route") or "").strip().lower()
    reason = _optional_text(raw.get("reason"))
    operation = _optional_text(raw.get("operation"))
    support_value = raw.get("support")
    support = tuple(
        str(item)
        for item in support_value
        if isinstance(item, str) and item.strip()
    ) if isinstance(support_value, list) else ()
    answer = raw.get("a")
    validation_error: str | None = None
    accepted = False

    if route not in {"accept", "normalize", "escalate"}:
        validation_error = "route must be accept, normalize, or escalate"
        route = "escalate"
    elif route != "escalate":
        validation_error = _validate_supported_answer(
            answer=answer,
            answer_type=answer_type,
            operation=operation,
            support=support,
            facts=facts,
        )
        accepted = validation_error is None
        if not accepted:
            route = "escalate"

    return EdgeReviewResult(
        mode=mode,
        route=route,
        accepted=accepted,
        answer=answer if accepted else None,
        reason=reason,
        operation=operation,
        support=support,
        prompt=prompt,
        response=response,
        raw=raw,
        token_usage=token_usage,
        sequence_trace=sequence_trace,
        validation_error=validation_error,
    )


def _validate_supported_answer(
    *,
    answer: Any,
    answer_type: str,
    operation: str | None,
    support: tuple[str, ...],
    facts: tuple[EdgeReviewFact, ...],
) -> str | None:
    facts_by_id = {fact.fact_id: fact for fact in facts}
    if not support:
        return "support must cite at least one evidence fact"
    if len(support) > 12 or len(set(support)) != len(support):
        return "support must contain unique bounded fact ids"
    if any(fact_id not in facts_by_id for fact_id in support):
        return "support references an unknown evidence fact"
    values = [facts_by_id[fact_id].value for fact_id in support]
    normalized_type = str(answer_type or "").strip().lower()

    if normalized_type in _BOOLEAN_TYPES:
        expected = _boolean_operation_value(operation, values)
        actual = _coerce_boolean(answer)
        if expected is None or actual is None or actual is not expected:
            return "boolean answer is not reproducible from cited evidence"
        return None

    if normalized_type in _NUMBER_TYPES:
        actual_number = _coerce_number(answer)
        supported_numbers = [_coerce_number(value) for value in values]
        if actual_number is None or not any(
            number is not None and math.isclose(actual_number, number, rel_tol=1e-9, abs_tol=1e-9)
            for number in supported_numbers
        ):
            return "number answer is not present in cited evidence"
        return None

    if normalized_type.startswith("list"):
        if not isinstance(answer, list) or not answer:
            return "list answer must be a non-empty array"
        if len(answer) > 12:
            return "list answer exceeds the local review limit"
        unmatched = list(values)
        for item in answer:
            index = _matching_value_index(unmatched, item)
            if index is None:
                return "list answer contains a value absent from cited evidence"
            unmatched.pop(index)
        return None

    if normalized_type in _TEXT_TYPES:
        if not isinstance(answer, str) or not answer.strip():
            return "text answer must be a non-empty string"
        if not any(_same_text(answer, value) for value in values):
            return "text answer is not present in cited evidence"
        return None

    return "unsupported answer type"


def _boolean_operation_value(operation: str | None, values: list[Any]) -> bool | None:
    selected = str(operation or "identity").strip().lower()
    booleans = [_coerce_boolean(value) for value in values]
    if selected == "identity" and len(booleans) == 1:
        return booleans[0]
    if selected == "not" and len(booleans) == 1 and booleans[0] is not None:
        return not booleans[0]
    if selected in {"and", "or"} and booleans and all(value is not None for value in booleans):
        return all(booleans) if selected == "and" else any(booleans)
    if selected in {"eq", "ne", "gt", "ge", "lt", "le"} and len(values) == 2:
        return _compare_values(selected, values[0], values[1])
    return None


def _compare_values(operation: str, left: Any, right: Any) -> bool | None:
    left_number = _coerce_number(left)
    right_number = _coerce_number(right)
    if left_number is not None and right_number is not None:
        left_value: Any = left_number
        right_value: Any = right_number
    else:
        if left is None or right is None:
            return None
        left_value = str(left).strip().casefold()
        right_value = str(right).strip().casefold()
    if operation == "eq":
        return left_value == right_value
    if operation == "ne":
        return left_value != right_value
    if operation == "gt":
        return left_value > right_value
    if operation == "ge":
        return left_value >= right_value
    if operation == "lt":
        return left_value < right_value
    if operation == "le":
        return left_value <= right_value
    return None


def _bounded_evidence(
    evidence: Any,
    *,
    max_actions: int,
    max_rows: int,
    max_columns: int,
    max_facts: int,
) -> Any | None:
    del max_facts
    ready = _json_evidence(evidence)
    if isinstance(ready, dict) and isinstance(ready.get("obs"), list):
        if len(ready["obs"]) > max_actions:
            return None
    if not _tables_within_limits(
        ready,
        max_rows=max_rows,
        max_columns=max_columns,
    ):
        return None
    return ready


def _json_evidence(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return {
            "n": len(value.index),
            "cols": [str(column) for column in value.columns],
            "rows": json_ready(value.to_dict(orient="records")),
        }
    frame = getattr(value, "frame", None)
    if isinstance(frame, pd.DataFrame):
        return _json_evidence(frame)
    if isinstance(value, dict):
        return {
            str(key): _json_evidence(item)
            for key, item in value.items()
            if key not in _IGNORED_EVIDENCE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_json_evidence(item) for item in value]
    return json_ready(value)


def _tables_within_limits(
    value: Any,
    *,
    max_rows: int,
    max_columns: int,
) -> bool:
    if isinstance(value, dict):
        rows = value.get("rows")
        columns = value.get("cols", value.get("columns"))
        if isinstance(rows, list):
            declared_rows = value.get("n")
            if isinstance(declared_rows, int) and declared_rows != len(rows):
                return False
            if len(rows) > max_rows:
                return False
            if isinstance(columns, list) and len(columns) > max_columns:
                return False
            if rows and isinstance(rows[0], dict) and len(rows[0]) > max_columns:
                return False
        return all(
            _tables_within_limits(item, max_rows=max_rows, max_columns=max_columns)
            for item in value.values()
        )
    if isinstance(value, list):
        return all(
            _tables_within_limits(item, max_rows=max_rows, max_columns=max_columns)
            for item in value
        )
    return True


def _clean_evidence(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("ok") is False:
            return False
        if value.get("err") not in (None, "", {}, []):
            return False
        route = str(value.get("route") or "").strip().lower()
        if route in {"edge_repair", "cloud_replan"}:
            return False
        return all(_clean_evidence(item) for item in value.values())
    if isinstance(value, list):
        return all(_clean_evidence(item) for item in value)
    return True


def _evidence_facts(
    evidence: Any,
    *,
    max_facts: int,
) -> tuple[EdgeReviewFact, ...]:
    facts: list[EdgeReviewFact] = []

    def visit(value: Any, path: str) -> None:
        if len(facts) > max_facts:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if key in _NON_VALUE_FACT_KEYS:
                    continue
                visit(item, f"{path}/{_escape_path(str(key))}")
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}/{index}")
            return
        if value is None or isinstance(value, (str, int, float, bool)):
            facts.append(
                EdgeReviewFact(
                    fact_id=f"e{len(facts)}",
                    path=path or "/",
                    value=value,
                )
            )

    visit(evidence, "")
    return tuple(facts) if len(facts) <= max_facts else ()


def _escape_path(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _supported_answer_type(answer_type: str) -> bool:
    return (
        answer_type in _NUMBER_TYPES
        or answer_type in _BOOLEAN_TYPES
        or answer_type in _TEXT_TYPES
        or answer_type.startswith("list")
    )


def _positive_config_int(
    config: dict[str, Any] | None,
    key: str,
    default: int,
) -> int:
    value = config.get(key) if isinstance(config, dict) else None
    if value is None:
        return default
    try:
        selected = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a positive integer: {value!r}") from exc
    if selected <= 0:
        raise ValueError(f"{key} must be positive")
    return selected


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().replace(",", "")
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _coerce_boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        selected = value.strip().lower()
        if selected in {"true", "yes", "y", "1", "support", "supports"}:
            return True
        if selected in {"false", "no", "n", "0", "refute", "refutes"}:
            return False
    return None


def _matching_value_index(values: list[Any], target: Any) -> int | None:
    for index, value in enumerate(values):
        if _same_scalar(target, value):
            return index
    return None


def _same_scalar(left: Any, right: Any) -> bool:
    left_number = _coerce_number(left)
    right_number = _coerce_number(right)
    if left_number is not None and right_number is not None:
        return math.isclose(left_number, right_number, rel_tol=1e-9, abs_tol=1e-9)
    return _same_text(left, right)


def _same_text(left: Any, right: Any) -> bool:
    if right is None:
        return False
    return _normalized_text(left) == _normalized_text(right)


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    selected = value.strip()
    return selected or None
