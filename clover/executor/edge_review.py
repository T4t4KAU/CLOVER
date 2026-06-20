"""Bounded local evidence review for table answers."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import pandas as pd

from clover.config import ENABLE_TERMINAL_EDGE_REVIEW, runtime_feature_enabled
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
class EdgeReviewOpportunity:
    """A statically detected bounded-semantic answer opportunity."""

    kind: str
    reason: str
    proactive: bool
    row_count: int | None = None
    column_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "reason": self.reason,
            "proactive": self.proactive,
        }
        if self.row_count is not None:
            payload["row_count"] = self.row_count
        if self.column_count is not None:
            payload["column_count"] = self.column_count
        return payload


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
    opportunity: EdgeReviewOpportunity | None = None
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
            "opportunity": (self.opportunity.to_dict() if self.opportunity is not None else None),
            "validation_error": self.validation_error,
        }


def edge_review_mode(slm_config: dict[str, Any] | None) -> str:
    """Return the configured local review mode."""

    if not isinstance(slm_config, dict):
        return EDGE_REVIEW_OFF
    selected = str(slm_config.get("edge_review_mode") or EDGE_REVIEW_OFF).strip().lower()
    if selected not in EDGE_REVIEW_MODES:
        available = ", ".join(sorted(EDGE_REVIEW_MODES))
        raise ValueError(f"Unsupported edge_review_mode: {selected!r}. Available: {available}")
    return selected


def proactive_edge_review_enabled(slm_config: dict[str, Any] | None) -> bool:
    """Return whether bounded semantic opportunities may preempt static finalization."""

    if not isinstance(slm_config, dict):
        return True
    value = slm_config.get("edge_review_proactive", True)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"edge_review_proactive must be a boolean, got {value!r}")


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

    if not runtime_feature_enabled(slm_config, ENABLE_TERMINAL_EDGE_REVIEW):
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
    facts, payload, opportunity = prepared
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
            opportunity=opportunity,
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
        opportunity=opportunity,
    )


def detect_edge_review_opportunity(
    *,
    question: str = "",
    answer_type: str,
    evidence: Any,
    slm_config: dict[str, Any] | None,
) -> EdgeReviewOpportunity | None:
    """Detect a closed local ambiguity worth reviewing before static finalization."""

    prepared = _prepare_review_payload(
        question=question,
        answer_type=answer_type,
        evidence=evidence,
        scope="opportunity_detection",
        slm_config=slm_config,
    )
    if prepared is None:
        return None
    return prepared[2]


def _prepare_review_payload(
    *,
    question: str,
    answer_type: str,
    evidence: Any,
    scope: str,
    slm_config: dict[str, Any] | None,
) -> (
    tuple[
        tuple[EdgeReviewFact, ...],
        dict[str, Any],
        EdgeReviewOpportunity,
    ]
    | None
):
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
    opportunity = _classify_review_opportunity(
        question=question,
        answer_type=normalized_type,
        evidence=ready,
        facts=facts,
    )
    payload = {
        "scope": scope,
        "question": str(question or ""),
        "answer_type": normalized_type,
        "opportunity": opportunity.to_dict(),
        "facts": [fact.to_dict() for fact in facts],
    }
    return facts, payload, opportunity


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
    opportunity: EdgeReviewOpportunity | None,
) -> EdgeReviewResult:
    route = str(raw.get("route") or "").strip().lower()
    reason = _optional_text(raw.get("reason"))
    operation = _optional_text(raw.get("operation"))
    support_value = raw.get("support")
    support = (
        tuple(str(item) for item in support_value if isinstance(item, str) and item.strip())
        if isinstance(support_value, list)
        else ()
    )
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
        opportunity=opportunity,
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
        selected_operation = str(operation or "identity").strip().lower()
        if selected_operation not in {"identity", "extract_number", "percent_value"}:
            return "unsupported numeric normalization operation"
        supported_numbers = [
            _coerce_number(
                value,
                allow_embedded=selected_operation != "identity",
            )
            for value in values
        ]
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
        selected_operation = str(operation or "identity").strip().lower()
        if selected_operation not in {
            "identity",
            "strip_quotes",
            "strip_parenthetical",
            "strip_label",
        }:
            return "unsupported text normalization operation"
        candidates = [_normalized_text_candidate(value, selected_operation) for value in values]
        if not any(
            candidate is not None and _same_text(answer, candidate) for candidate in candidates
        ):
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


def _classify_review_opportunity(
    *,
    question: str,
    answer_type: str,
    evidence: Any,
    facts: tuple[EdgeReviewFact, ...],
) -> EdgeReviewOpportunity:
    tables = _evidence_tables(evidence)
    table = tables[0] if tables else None
    row_count = table[0] if table is not None else None
    column_count = table[1] if table is not None else None

    if answer_type in _BOOLEAN_TYPES and len(facts) >= 2:
        return EdgeReviewOpportunity(
            kind="boolean_composition",
            reason="multiple bounded facts require a simple boolean operation",
            proactive=True,
            row_count=row_count,
            column_count=column_count,
        )
    if answer_type.startswith("list") and len(facts) >= 2:
        return EdgeReviewOpportunity(
            kind="list_assembly",
            reason="bounded evidence contains multiple answer candidates",
            proactive=True,
            row_count=row_count,
            column_count=column_count,
        )
    if row_count == 1 and column_count is not None and column_count > 1:
        return EdgeReviewOpportunity(
            kind="field_selection",
            reason="one result row exposes multiple possible answer fields",
            proactive=True,
            row_count=row_count,
            column_count=column_count,
        )
    if row_count is not None and 1 < row_count <= 5:
        if _question_requires_deterministic_table_operation(question):
            return EdgeReviewOpportunity(
                kind="deterministic_result_review",
                reason="multi-row evidence still requires a deterministic table operation",
                proactive=False,
                row_count=row_count,
                column_count=column_count,
            )
        return EdgeReviewOpportunity(
            kind="candidate_selection",
            reason="a small closed result contains multiple candidate rows",
            proactive=True,
            row_count=row_count,
            column_count=column_count,
        )
    if len(facts) == 1 and _value_needs_normalization(facts[0].value, answer_type):
        return EdgeReviewOpportunity(
            kind="value_normalization",
            reason="the sole answer fact contains a locally normalizable representation",
            proactive=False,
            row_count=row_count,
            column_count=column_count,
        )
    return EdgeReviewOpportunity(
        kind="bounded_answer_review",
        reason="bounded evidence can be cited and statically validated",
        proactive=False,
        row_count=row_count,
        column_count=column_count,
    )


def _question_requires_deterministic_table_operation(question: str) -> bool:
    text = str(question or "").casefold()
    return bool(
        re.search(
            r"\b(how many|number of|count|total|sum|average|mean|difference|"
            r"most|least|highest|lowest|maximum|minimum|top|bottom|rank|"
            r"more|less|greater|fewer|before|after|earliest|latest|"
            r"first|last|sort|order|ratio|percent(?:age)?)\b",
            text,
        )
    )


def _evidence_tables(value: Any) -> list[tuple[int, int]]:
    tables: list[tuple[int, int]] = []
    if isinstance(value, dict):
        rows = value.get("rows")
        columns = value.get("cols", value.get("columns"))
        if isinstance(rows, list):
            if isinstance(columns, list):
                column_count = len(columns)
            elif rows and isinstance(rows[0], dict):
                column_count = len(rows[0])
            else:
                column_count = 1 if rows else 0
            tables.append((len(rows), column_count))
        for item in value.values():
            tables.extend(_evidence_tables(item))
    elif isinstance(value, list):
        for item in value:
            tables.extend(_evidence_tables(item))
    return sorted(tables, key=lambda item: (item[0] * item[1], item[0]), reverse=True)


def _value_needs_normalization(value: Any, answer_type: str) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if answer_type in _NUMBER_TYPES:
        return (
            _coerce_number(text) is None
            and _coerce_number(
                text,
                allow_embedded=True,
            )
            is not None
        )
    if answer_type in _TEXT_TYPES:
        return bool(
            re.fullmatch(r"""["'“”‘’].+["'“”‘’]""", text)
            or re.search(r"\s+\([^()]+\)\s*$", text)
            or re.match(
                r"^(?=[^:=]{1,40}\s*[:=])[^:=]*[^\W\d_][^:=]*\s*[:=]\s*\S",
                text,
            )
        )
    return False


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


def _coerce_number(
    value: Any,
    *,
    allow_embedded: bool = False,
) -> float | None:
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
            if not allow_embedded:
                return None
            matches = re.findall(
                r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?![\w.])",
                stripped,
            )
            if len(matches) != 1:
                return None
            try:
                return float(matches[0])
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
    normalized = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", normalized.strip()).casefold()


def _normalized_text_candidate(value: Any, operation: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if operation == "identity":
        return text
    if operation == "strip_quotes":
        pairs = {
            '"': '"',
            "'": "'",
            "“": "”",
            "‘": "’",
        }
        if len(text) >= 2 and pairs.get(text[0]) == text[-1]:
            return text[1:-1].strip()
        return None
    if operation == "strip_parenthetical":
        match = re.fullmatch(r"(.+?)\s*\([^()]+\)\s*", text)
        return match.group(1).strip() if match else None
    if operation == "strip_label":
        match = re.fullmatch(
            r"(?=[^:=]{1,40}\s*[:=])([^:=]*[^\W\d_][^:=]*)\s*[:=]\s*(.+)",
            text,
        )
        return match.group(2).strip() if match else None
    return None


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    selected = value.strip()
    return selected or None
