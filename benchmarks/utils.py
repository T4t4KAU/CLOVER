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


def _token_value(usage: dict[str, Any] | None, key: str) -> int:
    if not isinstance(usage, dict):
        return 0
    try:
        return int(usage.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _token_total(usage: dict[str, Any] | None) -> int:
    return _token_value(usage, "total_tokens")


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


def _combined_token_value(summary: dict[str, Any], key: str) -> int:
    root_value = summary.get(key)
    if root_value is not None:
        try:
            return int(root_value or 0)
        except (TypeError, ValueError):
            pass
    return _token_value(summary.get("remote_token_usage"), key) + _token_value(
        summary.get("local_slm_token_usage"),
        key,
    )


def build_brief_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Extract the compact metrics shown in benchmark summaries."""

    total_cases = int(summary.get("total_cases", 0) or 0)
    accuracy = summary.get("accuracy_on_all_cases")
    accuracy_pct = round(float(accuracy) * 100.0, 2) if accuracy is not None else None

    def _per_q(value: float | int) -> float | None:
        if total_cases <= 0:
            return None
        return round(float(value) / total_cases, 4)

    input_tokens = _combined_token_value(summary, "input_tokens")
    output_tokens = _combined_token_value(summary, "output_tokens")
    total_tokens = _combined_token_value(summary, "total_tokens")
    if total_tokens == 0 and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens
    remote_tokens = _token_total(summary.get("remote_token_usage"))
    local_tokens = _token_total(summary.get("local_slm_token_usage"))
    remote_calls = int(summary.get("remote_calls", 0) or 0)
    local_calls = int(summary.get("local_slm_calls", 0) or 0)
    api_cost_usd = round(_cost_total(summary.get("remote_cost_estimate")), 6)

    # Keep a few legacy aliases so older CSV aggregators keep working, but the
    # formatted stdout only prints the compact fields in _BRIEF_LABELS below.
    brief = {
        "benchmark": summary.get("stage", "unknown"),
        "total_cases": total_cases,
        "correct": summary.get("correct"),
        "acc_pct": accuracy_pct,
        "accuracy_percent": accuracy_pct,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "avg_input_tokens_per_query": _per_q(input_tokens),
        "avg_output_tokens_per_query": _per_q(output_tokens),
        "avg_total_tokens_per_query": _per_q(total_tokens),
        "cloud_model": _model_ref(summary.get("remote_llm")),
        "edge_model": _model_ref(summary.get("local_slm")),
        "cloud_tokens": remote_tokens,
        "edge_tokens": local_tokens,
        "api_cost_usd": api_cost_usd,
        "cloud_tokens_per_q": _per_q(remote_tokens),
        "edge_tokens_per_q": _per_q(local_tokens),
        "calls_per_q": _per_q(remote_calls + local_calls),
        "api_cost_per_q_usd": _per_q(api_cost_usd),
    }
    brief["Acc. (%)"] = accuracy_pct
    brief["Input Tokens"] = input_tokens
    brief["Output Tokens"] = output_tokens
    brief["Total Tokens"] = total_tokens
    brief["Avg Input Tok / Query"] = brief["avg_input_tokens_per_query"]
    brief["Avg Output Tok / Query"] = brief["avg_output_tokens_per_query"]
    brief["Avg Total Tok / Query"] = brief["avg_total_tokens_per_query"]
    return brief


def compact_run_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Return a compact run summary while keeping ablation-critical counters."""

    brief = build_brief_summary(summary)
    total_cases = int(summary.get("total_cases", 0) or 0)
    elapsed_seconds = float(summary.get("elapsed_seconds", 0.0) or 0.0)
    system_profile = summary.get("system_profile") if isinstance(summary, dict) else {}
    counters = {}
    if isinstance(system_profile, dict):
        counters = dict(system_profile.get("counters") or {})
    existing_counters = summary.get("system_counters")
    if isinstance(existing_counters, dict):
        counters.update(existing_counters)

    compact_keys = (
        "run_name",
        "stage",
        "created_at",
        "workflow",
        "tablebench_standard",
        "wikitq_standard",
        "tablefact_standard",
        "mmqa_standard",
        "sample_size",
        "requested_sample_size",
        "seed",
        "parallel_workers",
        "max_retries",
        "validation_mode",
        "eval_batch_size",
        "profile_baseline",
        "remote_llm",
        "synthesize_llm",
        "local_slm",
        "remote_calls",
        "local_slm_calls",
        "edge_local_review_calls",
        "edge_local_review_hits",
        "edge_local_review_escalations",
        "total_cases",
        "parse_successes",
        "parse_failures",
        "runtime_successes",
        "runtime_failures",
        "correct",
        "mismatches",
        "failures",
        "accuracy_on_all_cases",
        "accuracy_on_successes",
        "retry_cases",
        "total_retry_rounds",
        "remote_token_usage",
        "local_slm_token_usage",
        "elapsed_seconds",
        "run_dir",
        "cases_index",
        "answer_mismatch_cases",
        "failure_cases",
        "avg_max_context_tokens_per_query",
        "max_context_tokens_stats",
    )
    compact = {key: summary[key] for key in compact_keys if key in summary}
    compact.update(
        {
            "acc_pct": brief["acc_pct"],
            "input_tokens": brief["input_tokens"],
            "output_tokens": brief["output_tokens"],
            "total_tokens": brief["total_tokens"],
            "avg_input_tokens_per_query": brief["avg_input_tokens_per_query"],
            "avg_output_tokens_per_query": brief["avg_output_tokens_per_query"],
            "avg_total_tokens_per_query": brief["avg_total_tokens_per_query"],
            "avg_seconds_per_query": (
                round(elapsed_seconds / total_cases, 4) if total_cases else None
            ),
            "system_counters": counters,
            "brief_summary": brief,
        }
    )
    return compact


_BRIEF_LABELS = (
    ("Benchmark", "benchmark"),
    ("Total Cases", "total_cases"),
    ("Correct", "correct"),
    ("Acc. (%)", "acc_pct"),
    ("Input Tokens", "input_tokens"),
    ("Output Tokens", "output_tokens"),
    ("Total Tokens", "total_tokens"),
    ("Avg Input Tok / Query", "avg_input_tokens_per_query"),
    ("Avg Output Tok / Query", "avg_output_tokens_per_query"),
    ("Avg Total Tok / Query", "avg_total_tokens_per_query"),
)


def format_brief_summary(brief: dict[str, Any]) -> str:
    """Render the brief summary as a vertical key-value list for stdout."""

    def _fmt(value: Any, *, is_cost: bool = False) -> str:
        if value is None:
            return "n/a"
        if is_cost and isinstance(value, (int, float)):
            return f"{value:.6f}"
        if isinstance(value, float):
            return f"{value:.4f}" if abs(value) < 1000 else f"{value:.2f}"
        return str(value)

    label_width = max(len(label) for label, _ in _BRIEF_LABELS)
    lines = []
    for label, key in _BRIEF_LABELS:
        is_cost = key in ("api_cost_usd", "api_cost_per_q_usd")
        lines.append(f"{label.ljust(label_width)} : {_fmt(brief.get(key), is_cost=is_cost)}")
    return "\n".join(lines)
