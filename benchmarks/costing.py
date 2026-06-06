"""Remote token cost estimates for benchmark summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


OPENAI_PRICING_SOURCE = "https://platform.openai.com/docs/pricing"
OPENAI_PRICING_SNAPSHOT_DATE = "2026-05-29"
OPENAI_PRICING_TIER = "standard_text_tokens"
DEFAULT_OPENAI_REFERENCE_MODEL = "gpt-5.2"
DEEPSEEK_PRICING_SOURCE = "https://api-docs.deepseek.com/quick_start/pricing"
DEEPSEEK_PRICING_SNAPSHOT_DATE = "2026-05-31"
DEEPSEEK_PRICING_TIER = "standard_text_tokens"


@dataclass(frozen=True)
class OpenAITextTokenPricing:
    """USD rates per 1M text tokens."""

    input_per_1m: float
    cached_input_per_1m: float | None
    output_per_1m: float


OPENAI_TEXT_TOKEN_PRICES_USD_PER_1M: dict[str, OpenAITextTokenPricing] = {
    "gpt-5.2": OpenAITextTokenPricing(1.75, 0.175, 14.00),
    "gpt-5.1": OpenAITextTokenPricing(1.25, 0.125, 10.00),
    "gpt-5": OpenAITextTokenPricing(1.25, 0.125, 10.00),
    "gpt-5-mini": OpenAITextTokenPricing(0.25, 0.025, 2.00),
    "gpt-5-nano": OpenAITextTokenPricing(0.05, 0.005, 0.40),
    "gpt-5.2-chat-latest": OpenAITextTokenPricing(1.75, 0.175, 14.00),
    "gpt-5.1-chat-latest": OpenAITextTokenPricing(1.25, 0.125, 10.00),
    "gpt-5-chat-latest": OpenAITextTokenPricing(1.25, 0.125, 10.00),
    "gpt-5.2-codex": OpenAITextTokenPricing(1.75, 0.175, 14.00),
    "gpt-5.1-codex-max": OpenAITextTokenPricing(1.25, 0.125, 10.00),
    "gpt-5.1-codex": OpenAITextTokenPricing(1.25, 0.125, 10.00),
    "gpt-5.1-codex-mini": OpenAITextTokenPricing(0.25, 0.025, 2.00),
    "gpt-5-codex": OpenAITextTokenPricing(1.25, 0.125, 10.00),
    "gpt-5.2-pro": OpenAITextTokenPricing(21.00, None, 168.00),
    "gpt-5-pro": OpenAITextTokenPricing(15.00, None, 120.00),
    "gpt-4.1": OpenAITextTokenPricing(2.00, 0.50, 8.00),
    "gpt-4.1-mini": OpenAITextTokenPricing(0.40, 0.10, 1.60),
    "gpt-4.1-nano": OpenAITextTokenPricing(0.10, 0.025, 0.40),
    "gpt-4o": OpenAITextTokenPricing(2.50, 1.25, 10.00),
    "gpt-4o-2024-05-13": OpenAITextTokenPricing(5.00, None, 15.00),
    "gpt-4o-mini": OpenAITextTokenPricing(0.15, 0.075, 0.60),
    "o1": OpenAITextTokenPricing(15.00, 7.50, 60.00),
    "o1-pro": OpenAITextTokenPricing(150.00, None, 600.00),
    "o3": OpenAITextTokenPricing(2.00, 0.50, 8.00),
    "o3-pro": OpenAITextTokenPricing(20.00, None, 80.00),
    "o3-deep-research": OpenAITextTokenPricing(10.00, 2.50, 40.00),
    "o4-mini": OpenAITextTokenPricing(1.10, 0.275, 4.40),
    "o4-mini-deep-research": OpenAITextTokenPricing(2.00, 0.50, 8.00),
    "o3-mini": OpenAITextTokenPricing(1.10, 0.55, 4.40),
    "o1-mini": OpenAITextTokenPricing(1.10, 0.55, 4.40),
    "codex-mini-latest": OpenAITextTokenPricing(1.50, 0.375, 6.00),
}

DEEPSEEK_TEXT_TOKEN_PRICES_USD_PER_1M: dict[str, OpenAITextTokenPricing] = {
    "deepseek-v4-pro": OpenAITextTokenPricing(0.435, 0.003625, 0.87),
    "deepseek-v4-flash": OpenAITextTokenPricing(0.14, 0.0028, 0.28),
}


def estimate_openai_text_cost(
    token_usage: dict[str, Any] | None,
    *,
    remote_config: dict[str, Any] | None = None,
    pricing_model: str | None = None,
) -> dict[str, Any]:
    """Estimate USD cost for remote text token usage with OpenAI pricing.

    The estimate is intentionally metadata-rich because the price table is a
    market snapshot and should be easy to update or audit later.
    """

    usage = normalize_remote_token_usage(token_usage)
    remote_config = remote_config or {}
    remote_provider = str(remote_config.get("provider") or "").strip().lower()
    remote_model = str(remote_config.get("model") or "").strip()
    if (
        pricing_model is None
        and remote_provider == "deepseek"
        and remote_model in DEEPSEEK_TEXT_TOKEN_PRICES_USD_PER_1M
    ):
        return _estimate_text_cost(
            usage=usage,
            pricing=DEEPSEEK_TEXT_TOKEN_PRICES_USD_PER_1M[remote_model],
            provider="deepseek",
            pricing_tier=DEEPSEEK_PRICING_TIER,
            pricing_source=DEEPSEEK_PRICING_SOURCE,
            pricing_snapshot_date=DEEPSEEK_PRICING_SNAPSHOT_DATE,
            pricing_model=remote_model,
            pricing_model_source="remote_config",
            remote_config=remote_config,
            assumptions=[
                "DeepSeek API text-token prices are used for the configured remote model.",
                "cached_input_tokens are charged at the cache-hit rate when available.",
                "reasoning_tokens are diagnostics and are not added again beyond output_tokens.",
            ],
        )
    selected_model, model_source = _select_openai_pricing_model(
        remote_config=remote_config,
        requested_model=pricing_model,
    )
    pricing = OPENAI_TEXT_TOKEN_PRICES_USD_PER_1M[selected_model]
    return _estimate_text_cost(
        usage=usage,
        pricing=pricing,
        provider="openai",
        pricing_tier=OPENAI_PRICING_TIER,
        pricing_source=OPENAI_PRICING_SOURCE,
        pricing_snapshot_date=OPENAI_PRICING_SNAPSHOT_DATE,
        pricing_model=selected_model,
        pricing_model_source=model_source,
        remote_config=remote_config,
        assumptions=[
            "Standard OpenAI API text-token prices are used as a reference.",
            "cached_input_tokens are charged at the cached-input rate when available.",
            "reasoning_tokens are diagnostics and are not added again beyond output_tokens.",
        ],
    )


def _estimate_text_cost(
    *,
    usage: dict[str, int],
    pricing: OpenAITextTokenPricing,
    provider: str,
    pricing_tier: str,
    pricing_source: str,
    pricing_snapshot_date: str,
    pricing_model: str,
    pricing_model_source: str,
    remote_config: dict[str, Any],
    assumptions: list[str],
) -> dict[str, Any]:
    cached_rate = (
        pricing.cached_input_per_1m
        if pricing.cached_input_per_1m is not None
        else pricing.input_per_1m
    )
    input_tokens = usage["input_tokens"]
    cached_input_tokens = min(usage["cached_input_tokens"], input_tokens)
    billable_input_tokens = max(0, input_tokens - cached_input_tokens)
    output_tokens = usage["output_tokens"]
    input_usd = _token_cost(billable_input_tokens, pricing.input_per_1m)
    cached_input_usd = _token_cost(cached_input_tokens, cached_rate)
    output_usd = _token_cost(output_tokens, pricing.output_per_1m)
    total_usd = input_usd + cached_input_usd + output_usd
    return {
        "provider": provider,
        "currency": "USD",
        "pricing_tier": pricing_tier,
        "pricing_source": pricing_source,
        "pricing_snapshot_date": pricing_snapshot_date,
        "pricing_model": pricing_model,
        "pricing_model_source": pricing_model_source,
        "remote_provider": remote_config.get("provider"),
        "remote_model": remote_config.get("model"),
        "unit": "per_1m_text_tokens",
        "rates_per_1m_tokens": {
            "input": pricing.input_per_1m,
            "cached_input": cached_rate,
            "output": pricing.output_per_1m,
        },
        "usage": {
            **usage,
            "billable_input_tokens": billable_input_tokens,
        },
        "cost_usd": {
            "input": _round_usd(input_usd),
            "cached_input": _round_usd(cached_input_usd),
            "output": _round_usd(output_usd),
            "total": _round_usd(total_usd),
        },
        "assumptions": assumptions,
    }


def normalize_remote_token_usage(token_usage: dict[str, Any] | None) -> dict[str, int]:
    """Return the canonical token usage counters expected by eval summaries."""

    token_usage = token_usage or {}
    return {
        "input_tokens": _int_token_count(token_usage.get("input_tokens")),
        "cached_input_tokens": _int_token_count(token_usage.get("cached_input_tokens")),
        "output_tokens": _int_token_count(token_usage.get("output_tokens")),
        "reasoning_tokens": _int_token_count(token_usage.get("reasoning_tokens")),
        "total_tokens": _int_token_count(token_usage.get("total_tokens")),
    }


def _select_openai_pricing_model(
    *,
    remote_config: dict[str, Any] | None,
    requested_model: str | None,
) -> tuple[str, str]:
    if requested_model:
        model = requested_model.strip()
        if model not in OPENAI_TEXT_TOKEN_PRICES_USD_PER_1M:
            available = ", ".join(sorted(OPENAI_TEXT_TOKEN_PRICES_USD_PER_1M))
            raise ValueError(
                f"Unknown OpenAI pricing model: {model!r}. "
                f"Update benchmarks/costing.py or choose one of: {available}"
            )
        return model, "requested"
    remote_config = remote_config or {}
    remote_model = str(remote_config.get("model") or "").strip()
    remote_provider = str(remote_config.get("provider") or "").strip().lower()
    if remote_provider == "openai" and remote_model in OPENAI_TEXT_TOKEN_PRICES_USD_PER_1M:
        return remote_model, "remote_config"
    return DEFAULT_OPENAI_REFERENCE_MODEL, "default_reference"


def _token_cost(tokens: int, usd_per_1m: float) -> float:
    return tokens * usd_per_1m / 1_000_000


def _round_usd(value: float) -> float:
    return round(value, 8)


def _int_token_count(value: Any) -> int:
    if value is None:
        return 0
    return max(0, int(value))
