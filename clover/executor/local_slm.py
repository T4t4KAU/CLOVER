"""Local-repair model client facade.

The paper configuration uses a locally deployed OpenAI-compatible endpoint.
This module keeps endpoint details outside NodeAgent code by loading the local
model config and reusing the shared client implementation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from clover.config import load_model_config
from clover.supervisor.client import (
    RemoteLLMResult,
    create_remote_llm_client,
    generate_remote_text,
)


DEFAULT_SLM_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "model_config" / "local_slm_config.json"
)
LOCAL_SLM_ENV_PREFIXES = (
    "CLOVER_LOCAL_SLM",
    "CLOVER_SLM",
    "LOCAL_SLM",
    "SLM",
)


def load_slm_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the SLM config used by local NodeAgents."""

    return load_model_config(
        config_path or _default_slm_config_path(),
        env_prefixes=LOCAL_SLM_ENV_PREFIXES,
    )


def _default_slm_config_path() -> Path:
    for name in ("CLOVER_LOCAL_SLM_CONFIG", "CLOVER_SLM_CONFIG", "SLM_CONFIG"):
        value = os.environ.get(name)
        if value:
            return Path(value)
    return DEFAULT_SLM_CONFIG_PATH


def create_slm_client(slm_config: dict[str, Any] | None = None) -> Any:
    """Create an OpenAI-compatible SLM client."""

    return create_remote_llm_client(slm_config or load_slm_config())


def generate_slm_text(
    prompt: str,
    *,
    slm_config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    client: Any | None = None,
) -> RemoteLLMResult:
    """Generate text with the configured SLM endpoint."""

    selected_config = slm_config or load_slm_config(config_path)
    return generate_remote_text(
        prompt=prompt,
        remote_config=selected_config,
        client=client,
    )


def limit_slm_request_timeout(
    slm_config: dict[str, Any],
    *,
    node_timeout_seconds: float | None,
) -> dict[str, Any]:
    """Return a config whose request timeout cannot exceed the node timeout."""

    selected = dict(slm_config)
    if node_timeout_seconds is None:
        return selected
    current_timeout = selected.get("timeout")
    if current_timeout is None:
        selected["timeout"] = node_timeout_seconds
        return selected
    try:
        current_timeout_float = float(current_timeout)
    except (TypeError, ValueError):
        selected["timeout"] = node_timeout_seconds
        return selected
    selected["timeout"] = min(current_timeout_float, float(node_timeout_seconds))
    return selected
