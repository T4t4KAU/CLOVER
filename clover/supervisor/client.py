"""Provider-neutral helpers for calling Remote LLM APIs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
from openai import OpenAI

from clover.config import resolve_model_config_env


class RemoteLLMConfigError(ValueError):
    """Raised when a Remote LLM config is incomplete or unsupported."""


@dataclass(frozen=True)
class RemoteLLMResult:
    """Normalized Remote LLM response used by the planning pipeline."""

    text: str
    response_payload: dict[str, Any]
    response_id: str | None
    response_status: str | None
    api_type: str


def extract_token_usage(response_payload: dict[str, Any]) -> dict[str, int]:
    """Extract provider-neutral token usage from a Remote LLM response payload."""

    usage = response_payload.get("usage")
    if not isinstance(usage, dict):
        usage = response_payload
    input_tokens = _int_usage_value(
        usage,
        "input_tokens",
        "prompt_tokens",
        "prompt_token_count",
    )
    output_tokens = _int_usage_value(
        usage,
        "output_tokens",
        "completion_tokens",
        "completion_token_count",
        "generated_tokens",
    )
    cached_input_tokens = _nested_int_usage_value(
        usage,
        ("input_tokens_details", "cached_tokens"),
        ("prompt_tokens_details", "cached_tokens"),
    )
    reasoning_tokens = _nested_int_usage_value(
        usage,
        ("output_tokens_details", "reasoning_tokens"),
        ("completion_tokens_details", "reasoning_tokens"),
    )
    total_tokens = _int_usage_value(usage, "total_tokens")
    if total_tokens == 0 and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }


def create_remote_llm_client(remote_config: dict[str, Any]) -> OpenAI:
    """Create an OpenAI-compatible client from config."""

    remote_config = resolve_model_config_env(remote_config)
    timeout = remote_config.get("timeout", 180)
    client_kwargs: dict[str, Any] = {
        "api_key": _resolve_api_key(remote_config),
        "base_url": remote_config["base_url"],
        "timeout": timeout,
        "max_retries": remote_config.get("max_retries", 2),
    }
    if "trust_env" in remote_config or "http2" in remote_config:
        client_kwargs["http_client"] = httpx.Client(
            timeout=timeout,
            trust_env=bool(remote_config.get("trust_env", True)),
            http2=bool(remote_config.get("http2", False)),
        )
    return OpenAI(**client_kwargs)


def generate_remote_text(
    prompt: str,
    remote_config: dict[str, Any],
    client: Any | None = None,
) -> RemoteLLMResult:
    """Generate text through either Responses or Chat Completions API."""

    remote_config = resolve_model_config_env(remote_config)
    if client is None:
        client = create_remote_llm_client(remote_config)

    # Providers are normalized behind OpenAI-compatible client methods, so the
    # planning pipeline consumes one RemoteLLMResult shape regardless of vendor.
    api_type = remote_config.get("api_type", "responses")
    if api_type == "responses":
        response = _create_responses_completion(client, prompt, remote_config)
        return RemoteLLMResult(
            text=_responses_text(response),
            response_payload=_model_dump(response),
            response_id=getattr(response, "id", None),
            response_status=getattr(response, "status", None),
            api_type=api_type,
        )
    if api_type == "chat_completions":
        response = _create_chat_completion(client, prompt, remote_config)
        return RemoteLLMResult(
            text=_chat_completion_text(response),
            response_payload=_model_dump(response),
            response_id=getattr(response, "id", None),
            response_status=getattr(response, "status", "completed"),
            api_type=api_type,
        )
    raise RemoteLLMConfigError(f"Unsupported Remote LLM api_type: {api_type}")


def _create_responses_completion(
    client: Any,
    prompt: str,
    remote_config: dict[str, Any],
) -> Any:
    request: dict[str, Any] = {
        "model": remote_config["model"],
        "input": [{"role": "user", "content": prompt}],
        "temperature": remote_config.get("temperature", 0),
        "max_output_tokens": remote_config.get("max_output_tokens", 12000),
    }
    return client.responses.create(**request)


def _create_chat_completion(
    client: Any,
    prompt: str,
    remote_config: dict[str, Any],
) -> Any:
    request: dict[str, Any] = {
        "model": remote_config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": remote_config.get("temperature", 0),
    }
    max_tokens = remote_config.get("max_tokens", remote_config.get("max_output_tokens"))
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    if "stop" in remote_config:
        request["stop"] = remote_config["stop"]
    for key in ("top_p", "frequency_penalty", "presence_penalty", "reasoning_effort", "extra_body"):
        if key in remote_config:
            request[key] = remote_config[key]
    return client.chat.completions.create(**request)


def _responses_text(response: Any) -> str:
    parts = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    parts.append(text)
    return "\n".join(parts).strip()


def _chat_completion_text(response: Any) -> str:
    parts = []
    for choice in getattr(response, "choices", []) or []:
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str) and content:
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
                if text:
                    parts.append(text)
    return _strip_chat_stop_tokens("\n".join(parts).strip())


def _strip_chat_stop_tokens(text: str) -> str:
    for token in ("<|im_end|>", "<|endoftext|>"):
        while text.endswith(token):
            text = text[: -len(token)].rstrip()
    return text


def _model_dump(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if isinstance(response, dict):
        return response
    return {
        "id": getattr(response, "id", None),
        "status": getattr(response, "status", None),
    }


def _resolve_api_key(remote_config: dict[str, Any]) -> str:
    api_key_env = remote_config.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if api_key:
            return api_key
    api_key = remote_config.get("api_key")
    if api_key:
        return api_key
    if api_key_env:
        raise RemoteLLMConfigError(f"Environment variable is not set: {api_key_env}")
    raise RemoteLLMConfigError("Remote LLM config must define api_key or api_key_env")


def _int_usage_value(payload: dict[str, Any], *names: str) -> int:
    for name in names:
        value = payload.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _nested_int_usage_value(
    payload: dict[str, Any],
    *paths: tuple[str, str],
) -> int:
    for parent_name, child_name in paths:
        parent = payload.get(parent_name)
        if not isinstance(parent, dict):
            continue
        value = _int_usage_value(parent, child_name)
        if value:
            return value
    return 0
