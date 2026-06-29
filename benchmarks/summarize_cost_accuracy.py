"""Aggregate cost-accuracy pairs across runs for Pareto analysis.

Collects (accuracy, cost, calls) tuples from:
  - CLOVER full eval runs (run_summary.json)
  - Pure CoT baseline runs (run_summary.json)
  - Ablation suite runs (one row per variant)
  - External baseline numbers (manual JSON)

Output:
  cost_accuracy_pareto.csv
  cost_accuracy_pareto.md
  cost_accuracy_pareto.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

VARIANT_DISPLAY_NAMES = {
    "full": "CLOVER (full)",
    "all_edge": "w/o Static Fast Path",
    "no_edge": "w/o Edge Agent",
    "static": "w/o Edge Repair",
    "no_contract": "w/o Contract Gate",
    "end_review": "End-only Review",
    "one_shot": "w/o Global Replan",
    "cloud_finalize": "Global Finalize",
    "static_only": "Static-only",
    "no_static": "w/o Static",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_run_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    total = int(summary.get("total_cases", 0) or 0)
    correct = int(summary.get("correct_cases", 0) or 0)
    if total == 0:
        # Fallback: derive from cases_index if present
        correct = int(summary.get("runtime_successes", 0) or 0)
        total = int(summary.get("total_cases", 0) or 0)
    accuracy = correct / total if total else 0.0
    remote_usage = summary.get("remote_token_usage") or {}
    local_usage = summary.get("local_slm_token_usage") or {}
    cost = (
        summary.get("remote_cost_estimate", {}).get("cost_usd", {}).get("total")
    )
    remote_tokens = int(remote_usage.get("total_tokens", 0) or 0)
    local_tokens = int(local_usage.get("total_tokens", 0) or 0)
    remote_calls = int(summary.get("remote_calls", 0) or 0)
    local_calls = int(summary.get("local_slm_calls", 0) or 0)
    return {
        "total_cases": total,
        "correct": correct,
        "accuracy": accuracy,
        "remote_calls": remote_calls,
        "local_slm_calls": local_calls,
        "remote_tokens": remote_tokens,
        "local_slm_tokens": local_tokens,
        "total_tokens": remote_tokens + local_tokens,
        "estimated_cost_usd": float(cost) if cost is not None else None,
    }


def _per_query(value: float | int | None, total: int) -> float | None:
    if value is None or total == 0:
        return None
    return float(value) / total


def _per_1k_queries(value: float | int | None, total: int) -> float | None:
    per_query = _per_query(value, total)
    if per_query is None:
        return None
    return per_query * 1000.0


def _collect_ablation_suite(
    suite_root: Path,
    dataset: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, display_name in VARIANT_DISPLAY_NAMES.items():
        summary_path = suite_root / f"{dataset}_{variant}" / "run_summary.json"
        if not summary_path.is_file():
            continue
        summary = _read_json(summary_path)
        metrics = _extract_run_metrics(summary)
        rows.append(_build_row(
            method=display_name,
            method_key=variant,
            dataset=dataset,
            source="ablation_suite",
            run_path=str(summary_path.parent),
            metrics=metrics,
        ))
    return rows


def _collect_single_run(
    run_root: Path,
    method: str,
    method_key: str,
    dataset: str,
    source: str,
) -> dict[str, Any] | None:
    summary_path = run_root / "run_summary.json"
    if not summary_path.is_file():
        return None
    summary = _read_json(summary_path)
    metrics = _extract_run_metrics(summary)
    return _build_row(
        method=method,
        method_key=method_key,
        dataset=dataset,
        source=source,
        run_path=str(run_root),
        metrics=metrics,
    )


def _collect_external_baselines(
    external_json: Path | None,
    dataset: str,
) -> list[dict[str, Any]]:
    if external_json is None or not external_json.is_file():
        return []
    payload = _read_json(external_json)
    entries = payload.get("baselines", [])
    if not isinstance(entries, list):
        return []
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if str(entry.get("dataset", "")).lower() != dataset.lower():
            continue
        total = int(entry.get("total_cases", 0) or 0)
        correct = int(entry.get("correct_cases", 0) or 0)
        accuracy = float(entry.get("accuracy", correct / total if total else 0.0))
        cost = entry.get("estimated_cost_usd")
        rows.append(_build_row(
            method=str(entry.get("method", "unknown")),
            method_key=str(entry.get("method_key", entry.get("method", "unknown"))),
            dataset=dataset,
            source="external",
            run_path="",
            metrics={
                "total_cases": total,
                "correct": correct,
                "accuracy": accuracy,
                "remote_calls": int(entry.get("remote_calls", 0) or 0),
                "local_slm_calls": int(entry.get("local_slm_calls", 0) or 0),
                "remote_tokens": int(entry.get("remote_tokens", 0) or 0),
                "local_slm_tokens": int(entry.get("local_slm_tokens", 0) or 0),
                "total_tokens": int(entry.get("remote_tokens", 0) or 0)
                + int(entry.get("local_slm_tokens", 0) or 0),
                "estimated_cost_usd": float(cost) if cost is not None else None,
            },
        ))
    return rows


def _build_row(
    *,
    method: str,
    method_key: str,
    dataset: str,
    source: str,
    run_path: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    total = metrics["total_cases"]
    return {
        "method": method,
        "method_key": method_key,
        "dataset": dataset,
        "source": source,
        "run_path": run_path,
        "total_cases": total,
        "correct": metrics["correct"],
        "accuracy": metrics["accuracy"],
        "remote_calls": metrics["remote_calls"],
        "local_slm_calls": metrics["local_slm_calls"],
        "remote_tokens": metrics["remote_tokens"],
        "local_slm_tokens": metrics["local_slm_tokens"],
        "total_tokens": metrics["total_tokens"],
        "estimated_cost_usd": metrics["estimated_cost_usd"],
        "global_calls_per_query": _per_query(metrics["remote_calls"], total),
        "cloud_calls_per_query": _per_query(metrics["remote_calls"], total),
        "edge_calls_per_query": _per_query(metrics["local_slm_calls"], total),
        "total_tokens_per_query": _per_query(metrics["total_tokens"], total),
        "cost_usd_per_query": _per_query(metrics["estimated_cost_usd"], total),
        "cost_usd_per_1k_queries": _per_1k_queries(
            metrics["estimated_cost_usd"], total
        ),
    }


def _format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _format_cost(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"${value:.4f}"


def _format_float(value: float | None, precision: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def write_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "cost_accuracy_pareto.csv"
    fieldnames = [
        "method",
        "method_key",
        "dataset",
        "source",
        "total_cases",
        "correct",
        "accuracy",
        "remote_calls",
        "local_slm_calls",
        "remote_tokens",
        "local_slm_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "global_calls_per_query",
        "cloud_calls_per_query",
        "edge_calls_per_query",
        "total_tokens_per_query",
        "cost_usd_per_query",
        "cost_usd_per_1k_queries",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    json_path = output_dir / "cost_accuracy_pareto.json"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    md_path = output_dir / "cost_accuracy_pareto.md"
    lines: list[str] = [
        "# Cost-Accuracy Pareto",
        "",
        "| Method | Dataset | Accuracy | Cost / 1K queries | Global calls / query | Edge calls / query | Total tokens / query |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: (r["dataset"], -(r["accuracy"] or 0))):
        lines.append(
            "| {method} | {dataset} | {acc} | {cost} | {cloud} | {edge} | {tokens} |".format(
                method=row["method"],
                dataset=row["dataset"],
                acc=_format_percent(row["accuracy"]),
                cost=_format_cost(row["cost_usd_per_1k_queries"]),
                cloud=_format_float(row["global_calls_per_query"], 3),
                edge=_format_float(row["edge_calls_per_query"], 3),
                tokens=_format_float(row["total_tokens_per_query"], 1),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate cost-accuracy pairs for Pareto analysis."
    )
    parser.add_argument(
        "--dataset",
        choices=("tablebench", "wikitq", "tablefact"),
        required=True,
    )
    parser.add_argument(
        "--ablation-suite",
        type=Path,
        default=None,
        help="Ablation suite root (contains <dataset>_<variant>/run_summary.json).",
    )
    parser.add_argument(
        "--clover-run",
        type=Path,
        default=None,
        help="Standalone CLOVER full run directory.",
    )
    parser.add_argument(
        "--pure-cot-run",
        type=Path,
        default=None,
        help="Pure CoT baseline run directory.",
    )
    parser.add_argument(
        "--external-baselines",
        type=Path,
        default=None,
        help="JSON file with external baseline numbers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows: list[dict[str, Any]] = []

    if args.ablation_suite is not None:
        rows.extend(_collect_ablation_suite(args.ablation_suite, args.dataset))
    if args.clover_run is not None:
        row = _collect_single_run(
            args.clover_run,
            method="CLOVER (full)",
            method_key="full",
            dataset=args.dataset,
            source="clover_run",
        )
        if row is not None:
            rows.append(row)
    if args.pure_cot_run is not None:
        row = _collect_single_run(
            args.pure_cot_run,
            method="Pure CoT",
            method_key="pure_cot",
            dataset=args.dataset,
            source="pure_cot",
        )
        if row is not None:
            rows.append(row)
    rows.extend(_collect_external_baselines(args.external_baselines, args.dataset))

    if not rows:
        print("No runs found. Provide at least one of --ablation-suite/--clover-run/--pure-cot-run/--external-baselines.")
        return 1

    write_outputs(rows, args.output_dir)
    print(f"Wrote {len(rows)} rows to {args.output_dir}/cost_accuracy_pareto.{{csv,md,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
