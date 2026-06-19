"""Model config loading with environment-variable field overrides."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any


STRING_FIELDS = {
    "provider",
    "api_type",
    "api_key",
    "api_key_env",
    "base_url",
    "model",
    "proxy",
    "reasoning_effort",
    "slm_scheduler",
    "edge_review_mode",
}
INTEGER_FIELDS = {
    "agent_loop_max_iterations",
    "agent_loop_repeat_error_early_stop",
    "max_parallel_slm_node_jobs",
    "max_parallel_slm_sequences",
    "max_pending_slm_sequences",
    "max_retries",
    "max_tokens",
    "max_output_tokens",
    "max_tptt_leaf_sequences_per_tree",
    "top_k",
    "tptt_prefix_tokens",
    "edge_review_max_actions",
    "edge_review_max_columns",
    "edge_review_max_facts",
    "edge_review_max_rows",
}
FLOAT_FIELDS = {
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "sleep_after_request_seconds",
    "tptt_coalesce_ms",
    "node_timeout_seconds",
    "timeout",
}
BOOLEAN_FIELDS = {
    "disable_agent_loop",
    "http2",
    "trust_env",
}
JSON_FIELDS = {
    "extra_body",
}
REFERENCE_ENV_FIELDS = {
    f"{field}_env"
    for field in STRING_FIELDS | INTEGER_FIELDS | FLOAT_FIELDS | BOOLEAN_FIELDS | JSON_FIELDS
    if not field.endswith("_env")
}
ENV_OVERRIDE_FIELDS = tuple(
    sorted(
        STRING_FIELDS
        | INTEGER_FIELDS
        | FLOAT_FIELDS
        | BOOLEAN_FIELDS
        | JSON_FIELDS
        | REFERENCE_ENV_FIELDS
    )
)


def load_model_config(
    path: str | Path,
    *,
    env_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Load one model config file and apply supported env overrides."""

    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Model config must be a JSON object: {path}")
    return resolve_model_config_env(config, env_prefixes=env_prefixes)


def load_optional_model_config(
    path: str | Path,
    *,
    env_prefixes: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """Load an optional model config, still allowing env-only configs."""

    try:
        return load_model_config(path, env_prefixes=env_prefixes)
    except FileNotFoundError:
        config = resolve_model_config_env({}, env_prefixes=env_prefixes)
        return config or None


def resolve_model_config_env(
    config: dict[str, Any],
    *,
    env_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Resolve `<field>_env` references and prefixed env overrides."""

    resolved = copy.deepcopy(config)
    _apply_reference_env_fields(resolved)
    _apply_prefixed_env_fields(resolved, env_prefixes)
    return resolved


def _apply_reference_env_fields(config: dict[str, Any]) -> None:
    for key, env_name in list(config.items()):
        if not key.endswith("_env") or not isinstance(env_name, str) or not env_name:
            continue
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        field = key[: -len("_env")]
        config[field] = _coerce_field_value(field, raw_value)


def _apply_prefixed_env_fields(config: dict[str, Any], prefixes: tuple[str, ...]) -> None:
    for prefix in prefixes:
        normalized_prefix = prefix.strip().upper()
        if not normalized_prefix:
            continue
        for field in ENV_OVERRIDE_FIELDS:
            env_name = f"{normalized_prefix}_{field.upper()}"
            raw_value = os.environ.get(env_name)
            if raw_value is None:
                continue
            config[field] = _coerce_field_value(field, raw_value)
            if field.endswith("_env"):
                _apply_prefixed_reference(config, prefix=normalized_prefix, field=field)


def _coerce_field_value(field: str, raw_value: str) -> Any:
    if field in INTEGER_FIELDS:
        return int(raw_value)
    if field in FLOAT_FIELDS:
        return float(raw_value)
    if field in BOOLEAN_FIELDS:
        return _parse_bool(raw_value)
    if field in JSON_FIELDS:
        return json.loads(raw_value)
    return raw_value


def _apply_prefixed_reference(
    config: dict[str, Any],
    *,
    prefix: str,
    field: str,
) -> None:
    target_field = field[: -len("_env")]
    direct_env_name = f"{prefix}_{target_field.upper()}"
    if os.environ.get(direct_env_name) is not None:
        return
    env_name = config.get(field)
    if not isinstance(env_name, str) or not env_name:
        return
    raw_value = os.environ.get(env_name)
    if raw_value is not None:
        config[target_field] = _coerce_field_value(target_field, raw_value)


def _parse_bool(raw_value: str) -> bool:
    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean model config value: {raw_value!r}")
