"""Compact prompt payloads for table repair Agent Loops."""

from __future__ import annotations

import json
from typing import Any

from clover.executor.node_views import NodeView
from clover.executor.result import json_ready


def empty_filter_repair_case_json(
    *,
    view: NodeView,
    steps: list[dict[str, Any]],
) -> str:
    """Return the dynamic tail payload for empty Filter repair prompts."""

    return json.dumps(
        json_ready(_repair_state_payload(view=view, steps=steps)),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _repair_state_payload(
    *,
    view: NodeView,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    view_payload = view.to_dict()
    world = view_payload.get("world")
    if not isinstance(world, dict):
        world = {}
    task_code = view_payload.get("task")
    task_code = task_code if isinstance(task_code, str) else ""
    observation = _last_observation(steps)
    payload: dict[str, Any] = {
        "sig": _extract_solve_signature(task_code),
        "goal": _extract_task_goal(task_code),
        "prev": _last_action_code(steps),
        "check": _visible_check_message(observation),
        "hint": _repair_state_hint(observation),
        "evidence": _visible_evidence(world=world, observation=observation),
    }
    columns = _visible_columns(world=world, observation=observation)
    if columns:
        payload["cols"] = columns
    return {
        key: value
        for key, value in payload.items()
        if value not in (None, "", [], {})
    }


def _last_observation(steps: list[dict[str, Any]]) -> dict[str, Any]:
    for step in reversed(steps):
        observation = step.get("observation")
        if isinstance(observation, dict):
            return observation
    return {}


def _last_action_code(steps: list[dict[str, Any]]) -> str | None:
    for step in reversed(steps):
        action = step.get("action")
        if not isinstance(action, dict):
            continue
        code = action.get("code")
        if isinstance(code, str) and code.strip():
            return _truncate_text(code.strip(), 900)
    return None


def _extract_solve_signature(task_code: str) -> str:
    for line in task_code.splitlines():
        stripped = line.strip()
        if stripped.startswith("def solve("):
            return stripped
    return "def solve(df):"


def _extract_task_goal(task_code: str) -> str:
    in_docstring = False
    collected: list[str] = []
    for line in task_code.splitlines():
        stripped = line.strip()
        if stripped == '"""':
            if in_docstring:
                break
            in_docstring = True
            continue
        if in_docstring and stripped:
            collected.append(stripped)
    return _truncate_text(" ".join(collected), 220) if collected else ""


def _visible_check_message(observation: dict[str, Any]) -> str:
    if not observation:
        return "The previous check returned an empty DataFrame."
    error = observation.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("msg")
        if isinstance(message, str) and message:
            return _truncate_text(message, 220)
    observation_type = observation.get("type")
    if observation_type == "invalid_action_json":
        return "The previous answer was not valid JSON."
    if observation_type == "invalid_solve_function":
        return "The previous code was not a valid solve function."
    return "The previous check failed."


def _repair_state_hint(observation: dict[str, Any]) -> list[str]:
    observation_type = observation.get("type") if observation else None
    error = observation.get("error") if observation else None
    message = ""
    if isinstance(error, dict):
        message = str(error.get("message") or error.get("msg") or "")
    message_lower = message.lower()
    if observation_type == "invalid_action_json":
        return [
            "Return one JSON object with key s.",
            "Avoid regex or backslash-heavy strings in JSON.",
        ]
    if observation_type == "invalid_solve_function":
        return [
            "Keep exactly one top-level def solve function.",
            "Use the exact signature shown in sig.",
        ]
    if observation_type == "python_error":
        return [
            "Fix the shown runtime error with the smallest code change.",
            "Use only visible arguments, columns, and libraries.",
        ]
    if not observation or "empty" in message_lower:
        return [
            "Do not repeat exact text equality.",
            "Normalize both cell text and target text with .str.casefold() and .str.replace(r'[^a-z0-9]', '', regex=True) before comparing.",
            "Use str.contains(pattern, case=False, regex=False) for substring match. Do NOT pass casefold= as a keyword argument.",
            "Return matching rows from the original DataFrame.",
        ]
    return ["Use the check field to make the smallest repair."]


def _visible_evidence(
    *,
    world: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, list[Any]]:
    values: dict[str, list[Any]] = {}
    _merge_value_evidence(values, _diag_value_evidence(world))
    _merge_value_evidence(values, _observation_value_evidence(observation))
    return {
        column: items[:8]
        for column, items in values.items()
        if items
    }


def _diag_value_evidence(world: dict[str, Any]) -> dict[str, list[Any]]:
    diag = world.get("diag")
    if not isinstance(diag, dict):
        return {}
    inputs = diag.get("inputs")
    if not isinstance(inputs, dict):
        return {}
    values: dict[str, list[Any]] = {}
    for input_diag in inputs.values():
        if not isinstance(input_diag, dict):
            continue
        input_values = input_diag.get("values")
        if not isinstance(input_values, dict):
            continue
        for column, items in input_values.items():
            if not isinstance(items, list):
                continue
            values[str(column)] = [
                item.get("v") if isinstance(item, dict) else item
                for item in items
            ]
    return values


def _observation_value_evidence(observation: dict[str, Any]) -> dict[str, list[Any]]:
    feedback = observation.get("feedback")
    if not isinstance(feedback, dict):
        return {}
    column_values = feedback.get("column_values")
    if not isinstance(column_values, dict):
        return {}
    values: dict[str, list[Any]] = {}
    for column, items in column_values.items():
        if not isinstance(items, list):
            continue
        values[str(column)] = [
            item.get("value") if isinstance(item, dict) else item
            for item in items
        ]
    return values


def _merge_value_evidence(
    target: dict[str, list[Any]],
    source: dict[str, list[Any]],
) -> None:
    for column, items in source.items():
        selected = target.setdefault(column, [])
        for item in items:
            if item is None or item in selected:
                continue
            selected.append(item)


def _visible_columns(
    *,
    world: dict[str, Any],
    observation: dict[str, Any],
) -> list[str]:
    feedback = observation.get("feedback")
    if isinstance(feedback, dict):
        columns_payload = feedback.get("columns")
        if isinstance(columns_payload, dict):
            columns: list[str] = []
            for items in columns_payload.values():
                if isinstance(items, list):
                    columns.extend(str(item) for item in items)
            if columns:
                return _unique_texts(columns)[:24]
        if isinstance(columns_payload, list):
            return _unique_texts(str(item) for item in columns_payload)[:24]
    inputs = world.get("inputs")
    if isinstance(inputs, dict) and len(inputs) == 1:
        summary = next(iter(inputs.values()))
        if isinstance(summary, dict) and isinstance(summary.get("cols"), list):
            return _unique_texts(str(item) for item in summary["cols"])[:24]
    return []


def _unique_texts(items: Any) -> list[str]:
    values: list[str] = []
    for item in items:
        text = str(item)
        if text not in values:
            values.append(text)
    return values


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."
