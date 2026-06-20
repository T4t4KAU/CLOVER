"""Small shared helpers used by the table benchmark evaluators."""

from __future__ import annotations

import json
import traceback
from datetime import date, datetime, time
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(json_ready(record), ensure_ascii=False) + "\n")


def json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_ready(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def format_error(exc: Exception) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def preview(value: Any, max_length: int = 240) -> str:
    text = str(value)
    if len(text) <= max_length:
        return text
    return text[: max(0, max_length - 3)] + "..."


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _model_ref(config: dict[str, Any] | None) -> str:
    if not isinstance(config, dict):
        return "n/a"
    model = config.get("model")
    if not model:
        return "n/a"
    return str(model)


def _token_total(usage: dict[str, Any] | None) -> int:
    if not isinstance(usage, dict):
        return 0
    try:
        return int(usage.get("total_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _cost_total(cost_estimate: dict[str, Any] | None) -> float:
    if not isinstance(cost_estimate, dict):
        return 0.0
    cost_usd = cost_estimate.get("cost_usd")
    if not isinstance(cost_usd, dict):
        return 0.0
    try:
        return float(cost_usd.get("total", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def build_brief_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Extract the 11 key metrics from a run summary.

    Fields: benchmark, cloud_model, edge_model, accuracy_pct, cloud_tokens,
    edge_tokens, api_cost_usd, cloud_tokens_per_q, edge_tokens_per_q,
    calls_per_q, api_cost_per_q_usd.
    """

    total_cases = int(summary.get("total_cases", 0) or 0)
    accuracy = summary.get("accuracy_on_all_cases")
    accuracy_pct = round(float(accuracy) * 100.0, 2) if accuracy is not None else None

    cloud_tokens = _token_total(summary.get("remote_token_usage"))
    edge_tokens = _token_total(summary.get("local_slm_token_usage"))
    api_cost_usd = round(_cost_total(summary.get("remote_cost_estimate")), 6)

    cloud_calls = int(summary.get("remote_calls", 0) or 0)
    edge_calls = int(summary.get("local_slm_calls", 0) or 0)
    total_calls = cloud_calls + edge_calls

    def _per_q(value: float | int) -> float | None:
        if total_cases <= 0:
            return None
        return round(float(value) / total_cases, 4)

    return {
        "benchmark": summary.get("stage", "unknown"),
        "cloud_model": _model_ref(summary.get("remote_llm")),
        "edge_model": _model_ref(summary.get("local_slm")),
        "acc_pct": accuracy_pct,
        "cloud_tokens": cloud_tokens,
        "edge_tokens": edge_tokens,
        "api_cost_usd": api_cost_usd,
        "cloud_tokens_per_q": _per_q(cloud_tokens),
        "edge_tokens_per_q": _per_q(edge_tokens),
        "calls_per_q": _per_q(total_calls),
        "api_cost_per_q_usd": _per_q(api_cost_usd),
    }


_BRIEF_HEADERS = (
    "Benchmark",
    "Cloud Model",
    "Edge Model",
    "Acc. (%)",
    "Cloud Tokens",
    "Edge Tokens",
    "API Cost (USD)",
    "Cloud Tok/Q",
    "Edge Tok/Q",
    "Calls/Q",
    "API Cost/Q (USD)",
)


def format_brief_summary(brief: dict[str, Any]) -> str:
    """Render the brief summary as a fixed-width table for stdout."""

    def _fmt(value: Any, *, is_cost: bool = False) -> str:
        if value is None:
            return "n/a"
        if is_cost and isinstance(value, (int, float)):
            return f"{value:.6f}"
        if isinstance(value, float):
            return f"{value:.4f}" if abs(value) < 1000 else f"{value:.2f}"
        return str(value)

    row = (
        brief.get("benchmark", "unknown"),
        brief.get("cloud_model", "n/a"),
        brief.get("edge_model", "n/a"),
        _fmt(brief.get("acc_pct")),
        _fmt(brief.get("cloud_tokens")),
        _fmt(brief.get("edge_tokens")),
        _fmt(brief.get("api_cost_usd"), is_cost=True),
        _fmt(brief.get("cloud_tokens_per_q")),
        _fmt(brief.get("edge_tokens_per_q")),
        _fmt(brief.get("calls_per_q")),
        _fmt(brief.get("api_cost_per_q_usd"), is_cost=True),
    )
    widths = [max(len(str(header)), len(str(cell))) for header, cell in zip(_BRIEF_HEADERS, row)]
    sep = "+".join("-" * (w + 2) for w in widths)
    header_line = " | ".join(str(h).ljust(w) for h, w in zip(_BRIEF_HEADERS, widths))
    value_line = " | ".join(str(c).ljust(w) for c, w in zip(row, widths))
    return f"{sep}\n{header_line}\n{sep}\n{value_line}\n{sep}"
