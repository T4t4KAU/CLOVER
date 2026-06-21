"""Summarize Edge model scale sweep results.

Reads run_summary.json from each <dataset>_<edge_model>_<variant>/ directory
under the sweep root and produces:
  edge_model_sweep.csv
  edge_model_sweep.md
  edge_model_sweep.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    total = int(summary.get("total_cases", 0) or 0)
    correct = int(summary.get("correct_cases", 0) or 0)
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
    elapsed = float(summary.get("elapsed_seconds", 0) or 0)
    model_calls = remote_calls + local_calls
    return {
        "total_cases": total,
        "correct": correct,
        "accuracy": accuracy,
        "remote_calls": remote_calls,
        "local_slm_calls": local_calls,
        "model_calls": model_calls,
        "remote_tokens": remote_tokens,
        "local_slm_tokens": local_tokens,
        "total_tokens": remote_tokens + local_tokens,
        "estimated_cost_usd": float(cost) if cost is not None else None,
        "elapsed_seconds": elapsed,
    }


def _per_query(value: float | int | None, total: int) -> float | None:
    if value is None or total == 0:
        return None
    return float(value) / total


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


def collect_sweep_rows(sweep_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    known_variants = ("no_static", "full")
    for run_dir in sorted(sweep_root.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "run_summary.json"
        if not summary_path.is_file():
            continue
        # Parse run name: <dataset>_<edge_model>_<variant>
        # variant may contain underscores (e.g., no_static), so match from the end.
        name = run_dir.name
        variant = None
        prefix = name
        for candidate in known_variants:
            suffix = "_" + candidate
            if name.endswith(suffix):
                variant = candidate
                prefix = name[: -len(suffix)]
                break
        if variant is None:
            continue
        # Split dataset and edge model. Dataset is one of the known names.
        for dataset in ("tablebench", "wikitq", "tablefact"):
            if prefix.startswith(dataset + "_"):
                edge_model = prefix[len(dataset) + 1:]
                break
        else:
            continue
        summary = _read_json(summary_path)
        metrics = _extract_metrics(summary)
        total = metrics["total_cases"]
        rows.append({
            "dataset": dataset,
            "edge_model": edge_model,
            "variant": variant,
            "run_name": name,
            **metrics,
            "cloud_calls_per_query": _per_query(metrics["remote_calls"], total),
            "edge_calls_per_query": _per_query(metrics["local_slm_calls"], total),
            "model_calls_per_query": _per_query(metrics["model_calls"], total),
            "total_tokens_per_query": _per_query(metrics["total_tokens"], total),
            "cost_usd_per_query": _per_query(metrics["estimated_cost_usd"], total),
            "cost_usd_per_1k_queries": (
                _per_query(metrics["estimated_cost_usd"], total) or 0
            ) * 1000.0 if metrics["estimated_cost_usd"] is not None else None,
        })
    return rows


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "edge_model_sweep.csv"
    fieldnames = [
        "dataset",
        "edge_model",
        "variant",
        "run_name",
        "total_cases",
        "correct",
        "accuracy",
        "remote_calls",
        "local_slm_calls",
        "model_calls",
        "remote_tokens",
        "local_slm_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "elapsed_seconds",
        "cloud_calls_per_query",
        "edge_calls_per_query",
        "model_calls_per_query",
        "total_tokens_per_query",
        "cost_usd_per_query",
        "cost_usd_per_1k_queries",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    json_path = output_dir / "edge_model_sweep.json"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    md_path = output_dir / "edge_model_sweep.md"
    lines: list[str] = [
        "# Edge Model Scale Sweep",
        "",
        "## Full CLOVER",
        "",
        "| Dataset | Edge model | Accuracy | Cost / 1K queries | Cloud calls / query | Edge calls / query | Model calls / query |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in sorted(
        rows,
        key=lambda r: (r["dataset"], r["edge_model"], r["variant"]),
    ):
        if row["variant"] != "full":
            continue
        lines.append(
            "| {ds} | {em} | {acc} | {cost} | {cloud} | {edge} | {mc} |".format(
                ds=row["dataset"],
                em=row["edge_model"],
                acc=_format_percent(row["accuracy"]),
                cost=_format_cost(row["cost_usd_per_1k_queries"]),
                cloud=_format_float(row["cloud_calls_per_query"], 3),
                edge=_format_float(row["edge_calls_per_query"], 3),
                mc=_format_float(row["model_calls_per_query"], 3),
            )
        )
    lines.extend([
        "",
        "## w/o Static (Edge-only finalization)",
        "",
        "| Dataset | Edge model | Accuracy | Cost / 1K queries | Cloud calls / query | Edge calls / query | Model calls / query |",
        "|---|---|---|---|---|---|---|",
    ])
    for row in sorted(
        rows,
        key=lambda r: (r["dataset"], r["edge_model"], r["variant"]),
    ):
        if row["variant"] != "no_static":
            continue
        lines.append(
            "| {ds} | {em} | {acc} | {cost} | {cloud} | {edge} | {mc} |".format(
                ds=row["dataset"],
                em=row["edge_model"],
                acc=_format_percent(row["accuracy"]),
                cost=_format_cost(row["cost_usd_per_1k_queries"]),
                cloud=_format_float(row["cloud_calls_per_query"], 3),
                edge=_format_float(row["edge_calls_per_query"], 3),
                mc=_format_float(row["model_calls_per_query"], 3),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize Edge model scale sweep results."
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        required=True,
        help="Sweep output root (contains <dataset>_<edge_model>_<variant>/ dirs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = collect_sweep_rows(args.sweep_root)
    if not rows:
        print(f"No runs found under: {args.sweep_root}")
        return 1
    write_outputs(rows, args.output_dir)
    print(f"Wrote {len(rows)} rows to {args.output_dir}/edge_model_sweep.{{csv,md,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
