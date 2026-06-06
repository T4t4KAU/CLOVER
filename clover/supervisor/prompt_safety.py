"""Helpers for keeping evaluation labels out of model prompts."""

from __future__ import annotations

from typing import Any

from clover.reasoning_profiles import (
    HINTS_KEY,
    PROFILE_KEY,
)


SENSITIVE_ANSWER_KEYS = frozenset(
    {
        "answer_key",
        "answer_label",
        "correct_answer",
        "correct_answers",
        "expected_answer",
        "expected_answers",
        "gold_answer",
        "gold_answers",
        "ground_truth",
        "ground_truth_answer",
        "ground_truth_answers",
        "label_answer",
        "label_answers",
        "reference_answer",
        "reference_answers",
    }
)

LOCAL_RESOURCE_KEYS = frozenset({"file", "path"})


def strip_sensitive_prompt_fields(value: Any) -> Any:
    """Recursively remove known evaluation-label fields from prompt payloads."""

    if isinstance(value, dict):
        return {
            key: strip_sensitive_prompt_fields(item)
            for key, item in value.items()
            if str(key).lower() not in SENSITIVE_ANSWER_KEYS
        }
    if isinstance(value, list):
        return [strip_sensitive_prompt_fields(item) for item in value]
    if isinstance(value, tuple):
        return [strip_sensitive_prompt_fields(item) for item in value]
    return value


def sanitize_task_dsl_for_prompt(task_dsl: dict[str, Any]) -> dict[str, Any]:
    """Return the task fields a remote model may see."""

    payload = {"task_type": task_dsl.get("task_type")}
    if "sources" in task_dsl:
        payload["sources"] = [
            sanitize_source_for_prompt(source)
            for source in task_dsl.get("sources", [])
            if isinstance(source, dict)
        ]
    if PROFILE_KEY in task_dsl:
        payload[PROFILE_KEY] = task_dsl[PROFILE_KEY]
    if HINTS_KEY in task_dsl:
        payload[HINTS_KEY] = task_dsl[HINTS_KEY]

    for key in (
        "question",
        "questions",
        "answer",
        "answers",
        "round_state",
    ):
        if key in task_dsl:
            payload[key] = task_dsl[key]
    return strip_sensitive_prompt_fields(payload)


def sanitize_source_for_prompt(source: dict[str, Any]) -> dict[str, Any]:
    """Remove local resource locators from a source before remote prompting."""

    return {
        key: strip_sensitive_prompt_fields(item)
        for key, item in source.items()
        if str(key).lower() not in LOCAL_RESOURCE_KEYS
        and str(key).lower() not in SENSITIVE_ANSWER_KEYS
    }
