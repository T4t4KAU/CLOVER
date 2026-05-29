"""Provider-neutral helpers for calling Remote LLM APIs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

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


@dataclass
class RemoteLLMSession:
    """Stateful Remote LLM conversation reused across CLOVER stages."""

    remote_config: dict[str, Any]
    client: Any | None = None
    previous_response_id: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.remote_config = resolve_model_config_env(self.remote_config)
        if self.client is None:
            self.client = create_remote_llm_client(self.remote_config)

    def generate(self, prompt: str) -> RemoteLLMResult:
        """Generate one assistant turn while preserving conversation state."""

        api_type = self.remote_config.get("api_type", "responses")
        if api_type == "responses":
            response = _create_responses_completion(
                self.client,
                prompt,
                self.remote_config,
                previous_response_id=self.previous_response_id,
            )
            result = RemoteLLMResult(
                text=_responses_text(response),
                response_payload=_model_dump(response),
                response_id=getattr(response, "id", None),
                response_status=getattr(response, "status", None),
                api_type=api_type,
            )
            self.previous_response_id = result.response_id
            return result

        if api_type == "chat_completions":
            # Chat-style providers do not accept previous_response_id, so the
            # session owns the message history explicitly.
            if not self.messages and self.remote_config.get("system_message"):
                self.messages.append(
                    {
                        "role": "system",
                        "content": self.remote_config["system_message"],
                    }
                )
            self.messages.append({"role": "user", "content": prompt})
            response = _create_chat_completion(
                self.client,
                prompt,
                self.remote_config,
                messages=self.messages,
            )
            result = RemoteLLMResult(
                text=_chat_completion_text(response),
                response_payload=_model_dump(response),
                response_id=getattr(response, "id", None),
                response_status=getattr(response, "status", "completed"),
                api_type=api_type,
            )
            self.messages.append({"role": "assistant", "content": result.text})
            return result

        raise RemoteLLMConfigError(f"Unsupported Remote LLM api_type: {api_type}")


def create_remote_llm_client(remote_config: dict[str, Any]) -> OpenAI:
    """Create an OpenAI-compatible client from config."""

    remote_config = resolve_model_config_env(remote_config)
    return OpenAI(
        api_key=_resolve_api_key(remote_config),
        base_url=remote_config["base_url"],
        timeout=remote_config.get("timeout", 180),
        max_retries=remote_config.get("max_retries", 2),
    )


def create_remote_llm_session(
    remote_config: dict[str, Any],
    client: Any | None = None,
) -> RemoteLLMSession:
    """Create a stateful Remote LLM session for multi-stage CLOVER runs."""

    return RemoteLLMSession(remote_config=remote_config, client=client)


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
    *,
    previous_response_id: str | None = None,
) -> Any:
    request: dict[str, Any] = {
        "model": remote_config["model"],
        "input": [{"role": "user", "content": prompt}],
        "temperature": remote_config.get("temperature", 0),
        "max_output_tokens": remote_config.get("max_output_tokens", 12000),
    }
    if previous_response_id:
        request["previous_response_id"] = previous_response_id
    return client.responses.create(**request)


def _create_chat_completion(
    client: Any,
    prompt: str,
    remote_config: dict[str, Any],
    *,
    messages: list[dict[str, str]] | None = None,
) -> Any:
    request_messages = messages
    if request_messages is None:
        request_messages = []
        system_message = remote_config.get("system_message")
        if system_message:
            request_messages.append({"role": "system", "content": system_message})
        request_messages.append({"role": "user", "content": prompt})

    request: dict[str, Any] = {
        "model": remote_config["model"],
        "messages": request_messages,
        "stream": False,
        "temperature": remote_config.get("temperature", 0),
    }
    max_tokens = remote_config.get("max_tokens", remote_config.get("max_output_tokens"))
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    for key in ("reasoning_effort", "extra_body"):
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
    return "\n".join(parts).strip()


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
