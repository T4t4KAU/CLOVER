"""Reporter decision parsing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


class ReporterParseError(ValueError):
    """Raised when a Reporter response cannot be parsed."""


@dataclass(frozen=True)
class ReporterDecision:
    """Normalized Reporter decision."""

    answer: Any
    retry: bool
    new_sql: dict[str, Any] | None
    raw: dict[str, Any]


def parse_reporter_decision(remote_response: str) -> ReporterDecision:
    """Parse a Reporter JSON response."""

    payload = extract_reporter_json(remote_response)
    retry = payload.get("retry")
    if not isinstance(retry, bool):
        raise ReporterParseError("Reporter response must include boolean retry")

    new_sql = _normalize_new_sql(payload.get("new_sql"))
    if retry and new_sql is None:
        raise ReporterParseError("Reporter retry=true requires new_sql")

    return ReporterDecision(
        answer=payload.get("answer"),
        retry=retry,
        new_sql=new_sql,
        raw=payload,
    )


def extract_reporter_json(remote_response: str) -> dict[str, Any]:
    """Extract one JSON object from plain or fenced Reporter output."""

    if not isinstance(remote_response, str) or not remote_response.strip():
        raise ReporterParseError("Reporter response is empty")

    candidates = _extract_fenced_blocks(remote_response)
    candidates.append(remote_response)
    errors = []
    for candidate in candidates:
        try:
            return _load_json_object(candidate)
        except ReporterParseError as exc:
            errors.append(str(exc))
    detail = "; ".join(errors) if errors else "no JSON candidate found"
    raise ReporterParseError(f"Unable to parse Reporter JSON: {detail}")


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
        raise ReporterParseError("Reporter JSON must be an object")
    return payload


def _load_first_json_object(text: str) -> Any:
    start = text.find("{")
    if start < 0:
        raise ReporterParseError("No JSON object found")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : index + 1])
                except json.JSONDecodeError as exc:
                    raise ReporterParseError(str(exc)) from exc
    raise ReporterParseError("Unclosed JSON object")


def _normalize_new_sql(value: Any) -> dict[str, Any] | None:
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        if "sql" not in value:
            return _normalize_keyed_sql_map(value)
        sql = value.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ReporterParseError("new_sql.sql must be a non-empty string")
        return {**value, "sql": sql.strip()}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            # Be tolerant during migration: some models return the SQL string
            # directly even when the schema says new_sql is an object.
            if stripped.upper().startswith(("SELECT", "WITH")):
                return {"sql": stripped}
            raise ReporterParseError("new_sql string must contain JSON or SQL") from exc
        if isinstance(payload, dict):
            return _normalize_new_sql(payload)
        if isinstance(payload, str):
            return _normalize_new_sql(payload)
    raise ReporterParseError("new_sql must be an object or SQL string")


def _normalize_keyed_sql_map(value: dict[str, Any]) -> dict[str, str]:
    if not value:
        raise ReporterParseError("new_sql object must not be empty")
    normalized: dict[str, str] = {}
    for key, sql in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ReporterParseError("new_sql keys must be non-empty answer names")
        if not isinstance(sql, str) or not sql.strip():
            raise ReporterParseError(
                f"new_sql.{key} must be a non-empty SQL string"
            )
        stripped = sql.strip()
        if not stripped.upper().startswith(("SELECT", "WITH")):
            raise ReporterParseError(
                f"new_sql.{key} must be a SELECT or WITH SQL statement"
            )
        normalized[key.strip()] = stripped
    return normalized
