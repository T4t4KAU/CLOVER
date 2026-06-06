"""Token counting helpers used by local prompt scheduling."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from typing import Any


TOKENIZER_ENV_NAMES = (
    "CLOVER_LOCAL_SLM_TOKENIZER",
    "CLOVER_SLM_TOKENIZER",
    "LOCAL_SLM_TOKENIZER",
    "SLM_TOKENIZER",
    "CLOVER_TOKENIZER",
)


def configured_tokenizer_name(config: dict[str, Any] | None = None) -> str | None:
    """Return the tokenizer name selected by config or environment, if any."""

    selected = config if isinstance(config, dict) else {}
    for key in ("tokenizer", "tokenizer_name", "model"):
        value = selected.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for env_name in TOKENIZER_ENV_NAMES:
        value = os.environ.get(env_name)
        if value and value.strip():
            return value.strip()
    return None


def count_tokens(
    text: str,
    *,
    tokenizer_name: str | None = None,
    allow_remote: bool | None = None,
) -> int:
    """Count tokens with an internal tokenizer, falling back to a rough estimate."""

    selected_text = text if isinstance(text, str) else str(text)
    if tokenizer_name:
        tokenizer = _load_tokenizer(
            tokenizer_name,
            allow_remote=_allow_remote_tokenizer_load(allow_remote),
        )
        if tokenizer is not None:
            try:
                return max(0, len(tokenizer.encode(selected_text, add_special_tokens=False)))
            except Exception:  # noqa: BLE001 - scheduling estimates must be best effort.
                pass
    return rough_token_count(selected_text)


def token_ids(
    text: str,
    *,
    tokenizer_name: str | None = None,
    allow_remote: bool | None = None,
) -> tuple[int, ...] | None:
    """Return exact tokenizer ids when an internal tokenizer is available."""

    if not tokenizer_name:
        return None
    selected_text = text if isinstance(text, str) else str(text)
    tokenizer = _load_tokenizer(
        tokenizer_name,
        allow_remote=_allow_remote_tokenizer_load(allow_remote),
    )
    if tokenizer is None:
        return None
    try:
        encoded = tokenizer.encode(selected_text, add_special_tokens=False)
    except Exception:  # noqa: BLE001 - scheduling metadata is best effort.
        return None
    return tuple(int(item) for item in encoded)


def prefix_signature(
    text: str,
    *,
    tokenizer_name: str | None = None,
    prefix_tokens: int = 64,
    allow_remote: bool | None = None,
) -> tuple[str, int]:
    """Return a stable signature for the prompt prefix used by TPTT scheduling."""

    selected_prefix_tokens = max(1, int(prefix_tokens or 1))
    ids = token_ids(
        text,
        tokenizer_name=tokenizer_name,
        allow_remote=allow_remote,
    )
    if ids is not None:
        prefix = ids[:selected_prefix_tokens]
        return (
            hashlib.sha256(repr(prefix).encode("utf-8")).hexdigest()[:16],
            len(prefix),
        )

    selected_text = (text if isinstance(text, str) else str(text))[
        : selected_prefix_tokens * 4
    ]
    return (
        hashlib.sha256(selected_text.encode("utf-8")).hexdigest()[:16],
        rough_token_count(selected_text),
    )


def rough_token_count(text: str) -> int:
    """Return a deterministic dependency-free token estimate."""

    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, (len(stripped) + 3) // 4)


@lru_cache(maxsize=8)
def _load_tokenizer(tokenizer_name: str, *, allow_remote: bool) -> Any | None:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            tokenizer_name,
            local_files_only=not allow_remote,
            trust_remote_code=True,
        )
    except Exception:  # noqa: BLE001 - fallback keeps scheduler construction robust.
        return None


def _allow_remote_tokenizer_load(value: bool | None) -> bool:
    if value is not None:
        return bool(value)
    selected = os.environ.get("CLOVER_TOKENIZER_ALLOW_REMOTE", "")
    return selected.strip().lower() in {"1", "true", "yes", "on"}
