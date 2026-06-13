"""Supervisor decision parsing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


class SupervisorParseError(ValueError):
    """Raised when a Supervisor response cannot be parsed."""


@dataclass(frozen=True)
class SupervisorAction:
    """One action requested by the Supervisor for the next local round."""

    op: str
    q: str | None = None
    seed: str | None = None
    kind: str | None = None


@dataclass(frozen=True)
class SupervisorDecision:
    """Normalized Supervisor synthesis decision."""

    answer: Any
    retry: bool
    raw: dict[str, Any]
    sufficient: bool | None = None
    explanation: str | None = None
    feedback: str | None = None
    scratchpad: str | None = None
    next_python_code: str | None = None
    done: bool | None = None
    sqls: tuple[str, ...] = ()
    action_op: str = "sql"
    actions: tuple[SupervisorAction, ...] = ()


def parse_supervisor_decision(remote_response: str) -> SupervisorDecision:
    """Parse a Supervisor JSON response."""

    payload = extract_supervisor_json(remote_response)
    payload = _unwrap_react_payload(payload)
    if "final" in payload:
        raise SupervisorParseError("Supervisor response must not include final")
    if "retry" in payload or "new_sql" in payload or "done" in payload:
        raise SupervisorParseError(
            "Legacy retry/new_sql/done protocol is not supported"
        )
    action_op = _action_op(payload)
    if action_op and action_op not in {"sql", "inspect", "analyze", "answer"}:
        raise SupervisorParseError(f"Unsupported Supervisor action op: {action_op}")
    if action_op == "answer":
        if "a" not in payload:
            raise SupervisorParseError("Supervisor answer action requires a")
        return _parse_react_answer_decision(payload)
    if "a" in payload and "retry" not in payload:
        return _parse_react_answer_decision(payload)
    actions = _normalize_actions(payload)
    if actions:
        sqls = tuple(action.q for action in actions if action.op == "sql" and action.q)
        return SupervisorDecision(
            answer=None,
            retry=True,
            raw=payload,
            explanation=_optional_string(payload.get("e")),
            feedback=_optional_string(payload.get("fb")),
            scratchpad=_optional_string(payload.get("scratchpad")),
            done=False,
            sqls=sqls,
            action_op=actions[0].op,
            actions=actions,
        )
    if (
        ("q" in payload or "sql" in payload or "sqls" in payload)
        and "retry" not in payload
    ):
        return _parse_react_action_decision(payload)
    if "sufficient" in payload and "retry" not in payload:
        return _parse_sufficient_decision(payload)
    raise SupervisorParseError("Supervisor response must be an answer or action")


def _unwrap_react_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept accidental next-action wrappers while preserving one-step ReAct."""

    for key in ("steps", "plan"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, dict):
                return _unwrap_react_payload(first)
    sqls = payload.get("sqls")
    if isinstance(sqls, list) and sqls and all(isinstance(item, dict) for item in sqls):
        return _unwrap_react_payload(sqls[0])
    return payload


def _parse_react_answer_decision(payload: dict[str, Any]) -> SupervisorDecision:
    return SupervisorDecision(
        answer=payload.get("a"),
        retry=False,
        raw=payload,
        explanation=_optional_string(payload.get("e")),
        feedback=_optional_string(payload.get("fb")),
        scratchpad=_optional_string(payload.get("scratchpad")),
        done=True,
        action_op="answer",
    )


def _parse_react_action_decision(payload: dict[str, Any]) -> SupervisorDecision:
    sqls = _normalize_sqls(payload.get("q", payload.get("sqls", payload.get("sql"))))
    if not sqls:
        raise SupervisorParseError("Supervisor action response requires q SQL")
    return SupervisorDecision(
        answer=None,
        retry=True,
        raw=payload,
        explanation=_optional_string(payload.get("e")),
        feedback=_optional_string(payload.get("fb")),
        scratchpad=_optional_string(payload.get("scratchpad")),
        done=False,
        sqls=sqls,
        action_op="sql",
        actions=tuple(SupervisorAction(op="sql", q=sql) for sql in sqls),
    )


def _parse_sufficient_decision(payload: dict[str, Any]) -> SupervisorDecision:
    sufficient = payload.get("sufficient")
    if not isinstance(sufficient, bool):
        raise SupervisorParseError("Supervisor response sufficient must be boolean")
    return SupervisorDecision(
        answer=payload.get("answer"),
        retry=False,
        raw=payload,
        sufficient=sufficient,
        explanation=_optional_string(payload.get("explanation")),
        feedback=_optional_string(payload.get("feedback")),
        scratchpad=_optional_string(payload.get("scratchpad")),
        next_python_code=_optional_string(payload.get("next_python_code")),
    )


def extract_supervisor_json(remote_response: str) -> dict[str, Any]:
    """Extract one JSON object from plain or fenced Supervisor output."""

    if not isinstance(remote_response, str) or not remote_response.strip():
        raise SupervisorParseError("Supervisor response is empty")

    candidates = _extract_fenced_blocks(remote_response)
    candidates.append(remote_response)
    errors = []
    for candidate in candidates:
        try:
            return _load_json_object(candidate)
        except SupervisorParseError as exc:
            errors.append(str(exc))
    detail = "; ".join(errors) if errors else "no JSON candidate found"
    raise SupervisorParseError(f"Unable to parse Supervisor JSON: {detail}")


def _extract_fenced_blocks(text: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    ]


def _load_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = _load_first_json_object(stripped)
    if not isinstance(payload, dict):
        raise SupervisorParseError("Supervisor JSON must be an object")
    return payload


def _load_first_json_object(text: str) -> Any:
    start = text.find("{")
    if start < 0:
        raise SupervisorParseError("No JSON object found")
    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(text, idx=start)
    except json.JSONDecodeError as exc:
        raise SupervisorParseError(str(exc)) from exc
    return obj


def _normalize_actions(payload: dict[str, Any]) -> tuple[SupervisorAction, ...]:
    action_list = payload.get("acts", payload.get("actions"))
    if action_list is not None:
        if not isinstance(action_list, list) or not action_list:
            raise SupervisorParseError("acts must be a non-empty list")
        actions: list[SupervisorAction] = []
        for index, item in enumerate(action_list):
            if not isinstance(item, dict):
                raise SupervisorParseError(f"acts[{index}] must be an object")
            actions.extend(_normalize_one_action(item, label=f"acts[{index}]"))
        return tuple(actions)

    action_op = _action_op(payload)
    if action_op in {"sql", "inspect", "analyze"}:
        return tuple(_normalize_one_action(payload, label="action"))

    if any(key in payload for key in ("q", "sql", "sqls")):
        return tuple(_normalize_one_action({"op": "sql", **payload}, label="action"))

    return ()


def _normalize_one_action(payload: dict[str, Any], *, label: str) -> tuple[SupervisorAction, ...]:
    action_op = _action_op(payload) or "sql"
    if action_op == "sql":
        sqls = _normalize_sqls(payload.get("q", payload.get("sqls", payload.get("sql"))))
        if not sqls:
            raise SupervisorParseError(f"{label} requires SQL q")
        return tuple(SupervisorAction(op="sql", q=sql) for sql in sqls)
    if action_op == "inspect":
        question = payload.get("q", payload.get("ask", payload.get("task")))
        if not isinstance(question, str) or not question.strip():
            raise SupervisorParseError(f"{label} inspect requires non-empty q")
        seed = payload.get("seed")
        normalized_seed = None
        if seed is not None and seed != "":
            seed_sqls = _normalize_sqls(seed)
            if len(seed_sqls) != 1:
                raise SupervisorParseError(f"{label} inspect seed requires one SQL")
            normalized_seed = seed_sqls[0]
        return (
            SupervisorAction(
                op="inspect",
                q=question.strip(),
                seed=normalized_seed,
            ),
        )
    if action_op == "analyze":
        kind = payload.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            raise SupervisorParseError(f"{label} analyze requires non-empty kind")
        seed = payload.get("seed")
        seed_sqls = _normalize_sqls(seed)
        if len(seed_sqls) != 1:
            raise SupervisorParseError(f"{label} analyze requires one seed SQL")
        return (
            SupervisorAction(
                op="analyze",
                seed=seed_sqls[0],
                kind=kind.strip().lower(),
            ),
        )
    raise SupervisorParseError(f"Unsupported Supervisor action op: {action_op}")


def _normalize_sqls(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if _looks_like_sql(stripped) else ()
    if not isinstance(value, list):
        raise SupervisorParseError("sqls must be a list of SQL strings")
    sqls: list[str] = []
    for index, item in enumerate(value):
        if isinstance(item, dict):
            if "final" in item:
                raise SupervisorParseError(f"sqls[{index}] must not include final")
            item = item.get("sql")
        if not isinstance(item, str) or not item.strip():
            raise SupervisorParseError(f"sqls[{index}] must be a non-empty SQL string")
        stripped = item.strip()
        if not _looks_like_sql(stripped):
            raise SupervisorParseError(f"sqls[{index}] must be SELECT or WITH SQL")
        sqls.append(stripped)
    return tuple(sqls)


def _looks_like_sql(value: str) -> bool:
    return value.upper().startswith(("SELECT", "WITH"))


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _action_op(payload: dict[str, Any]) -> str:
    value = payload.get("op")
    if not isinstance(value, str):
        return ""
    return value.strip().lower()
