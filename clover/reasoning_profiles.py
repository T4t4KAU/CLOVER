"""Semantic workflow profiles carried by task DSLs."""

from __future__ import annotations

from typing import Any


PROFILE_KEY = "profile"
HINTS_KEY = "hints"

TABLE_REASONING_QUERY_PROFILE = "query"
TABLE_REASONING_ANALYZE_PROFILE = "analyze"

TABLE_REASONING_PROFILES = frozenset(
    {
        TABLE_REASONING_QUERY_PROFILE,
        TABLE_REASONING_ANALYZE_PROFILE,
    }
)


def normalize_table_reasoning_profile(
    value: Any,
    *,
    default: str = TABLE_REASONING_QUERY_PROFILE,
) -> str:
    """Return a supported table reasoning profile."""

    if value is None or value == "":
        value = default
    profile = str(value).strip()
    if profile not in TABLE_REASONING_PROFILES:
        available = ", ".join(sorted(TABLE_REASONING_PROFILES))
        raise ValueError(
            f"Unsupported table reasoning profile: {profile!r}. "
            f"Available profiles: {available}"
        )
    return profile


def short_table_reasoning_profile(
    value: Any,
    *,
    default: str = TABLE_REASONING_QUERY_PROFILE,
) -> str:
    """Return the compact table reasoning profile name."""

    return normalize_table_reasoning_profile(value, default=default)


def table_reasoning_profile_from_dsl(
    task_dsl: dict[str, Any] | None,
    *,
    default: str = TABLE_REASONING_QUERY_PROFILE,
) -> str:
    """Read a table reasoning profile from a task DSL."""

    if not isinstance(task_dsl, dict):
        return normalize_table_reasoning_profile(default)
    if task_dsl.get(PROFILE_KEY) is not None:
        return normalize_table_reasoning_profile(task_dsl.get(PROFILE_KEY), default=default)
    task_type = str(task_dsl.get("task_type", "")).strip()
    if task_type.endswith(".analyze"):
        return TABLE_REASONING_ANALYZE_PROFILE
    if task_type.endswith(".query"):
        return TABLE_REASONING_QUERY_PROFILE
    return normalize_table_reasoning_profile(default)


def table_reasoning_hints_from_dsl(task_dsl: dict[str, Any] | None) -> Any:
    """Read compact profile hints from a task DSL."""

    if not isinstance(task_dsl, dict):
        return None
    return task_dsl.get(HINTS_KEY)
