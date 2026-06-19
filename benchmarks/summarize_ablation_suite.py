"""Build human-readable tables for a completed CLOVER ablation suite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

VARIANTS = (
    ("full", "Full CLOVER"),
    ("static", "w/o Edge Repair/Review"),
    ("no_contract", "w/o Contract Verification"),
    ("end_review", "End-only Review"),
    ("one_shot", "w/o Cloud Replan"),
    ("cloud_finalize", "Cloud Finalization"),
)


def summarize_ablation_suite(*, suite_root: Path, dataset: str) -> dict[str, Any]:
    """Read all variant outputs and write JSON, CSV, and Markdown summaries."""

    variant_data: dict[str, dict[str, Any]] = {}
    missing: list[Path] = []
    for variant, display_name in VARIANTS:
        run_dir = suite_root / f"{dataset}_{variant}"
        summary_path = run_dir / "run_summary.json"
        cases_path = run_dir / "cases_index.jsonl"
        if not summary_path.is_file():
            missing.append(summary_path)
        if not cases_path.is_file():
            missing.append(cases_path)
        if summary_path.is_file() and cases_path.is_file():
            variant_data[variant] = {
                "display_name": display_name,
                "run_dir": run_dir,
                "summary": _read_json(summary_path),
                "cases": _case_results(cases_path),
            }
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Ablation suite is incomplete:\n{missing_text}")

    full_cases = variant_data["full"]["cases"]
    rows = [
        _build_row(
            variant=variant,
            display_name=display_name,
            summary=variant_data[variant]["summary"],
            cases=variant_data[variant]["cases"],
            full_cases=full_cases,
        )
        for variant, display_name in VARIANTS
    ]
    full_accuracy = rows[0]["accuracy"]
    for row in rows:
        row["delta_vs_full_pp"] = (row["accuracy"] - full_accuracy) * 100.0

    report = {
        "dataset": dataset,
        "suite_root": suite_root.as_posix(),
        "reference_variant": "full",
        "total_cases": rows[0]["total_cases"],
        "variants": rows,
    }
    markdown = render_markdown(report)

    (suite_root / "ablation_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(suite_root / "ablation_summary.csv", rows)
    (suite_root / "ablation_summary.md").write_text(markdown, encoding="utf-8")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Render the two compact tables printed after an ablation suite."""

    lines = [
        f"# {report['dataset']} ablation summary",
        "",
        "## Effectiveness",
        "",
        "| Experiment | Correct | Accuracy | Δ vs Full | Runtime failures "
        "| Regressions | Recoveries |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["variants"]:
        values = {
            **row,
            "accuracy_text": _format_percent(row["accuracy"]),
            "delta_text": _format_pp(row["delta_vs_full_pp"]),
        }
        lines.append(
            "| {display_name} | {correct}/{total_cases} | {accuracy_text} | {delta_text} "
            "| {runtime_failures} | {regressions_vs_full} | {recoveries_vs_full} |".format_map(
                values
            )
        )

    lines.extend(
        [
            "",
            "## Calls and cost",
            "",
            "| Experiment | Cloud calls | Cloud synthesis | Edge calls | Edge hits "
            "| Cloud tokens | Edge tokens | Est. cost | Time |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["variants"]:
        values = {
            **row,
            "remote_tokens_text": _format_integer(row["remote_tokens"]),
            "local_slm_tokens_text": _format_integer(row["local_slm_tokens"]),
            "cost_text": _format_cost(row["estimated_remote_cost_usd"]),
            "time_text": _format_duration(row["elapsed_seconds"]),
        }
        lines.append(
            "| {display_name} | {remote_calls} | {cloud_synthesis_calls} "
            "| {edge_review_calls} | {edge_review_hits} | {remote_tokens_text} "
            "| {local_slm_tokens_text} | {cost_text} | {time_text} |".format_map(
                values
            )
        )

    lines.extend(
        [
            "",
            "Δ vs Full uses percentage points. Regressions are cases that Full answered "
            "correctly but the variant did not; recoveries are the reverse.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_row(
    *,
    variant: str,
    display_name: str,
    summary: dict[str, Any],
    cases: dict[tuple[str, str], bool],
    full_cases: dict[tuple[str, str], bool],
) -> dict[str, Any]:
    if set(cases) != set(full_cases):
        raise ValueError(f"{variant} does not contain the same case set as Full CLOVER")

    total = len(cases)
    correct = sum(cases.values())
    regressions = sum(
        full_correct and not cases[key]
        for key, full_correct in full_cases.items()
    )
    recoveries = sum(
        not full_correct and cases[key]
        for key, full_correct in full_cases.items()
    )
    counters = summary.get("system_profile", {}).get("counters", {})
    remote_usage = summary.get("remote_token_usage") or {}
    local_usage = summary.get("local_slm_token_usage") or {}
    cost = (
        summary.get("remote_cost_estimate", {})
        .get("cost_usd", {})
        .get("total")
    )
    return {
        "variant": variant,
        "display_name": display_name,
        "total_cases": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "delta_vs_full_pp": 0.0,
        "runtime_successes": int(summary.get("runtime_successes", 0) or 0),
        "runtime_failures": int(summary.get("runtime_failures", 0) or 0),
        "mismatches": int(summary.get("mismatches", 0) or 0),
        "regressions_vs_full": regressions,
        "recoveries_vs_full": recoveries,
        "remote_calls": int(summary.get("remote_calls", 0) or 0),
        "cloud_synthesis_calls": int(
            counters.get("supervisor_synthesis_calls", 0) or 0
        ),
        "edge_review_calls": int(summary.get("edge_local_review_calls", 0) or 0),
        "edge_review_hits": int(summary.get("edge_local_review_hits", 0) or 0),
        "remote_tokens": int(remote_usage.get("total_tokens", 0) or 0),
        "local_slm_tokens": int(local_usage.get("total_tokens", 0) or 0),
        "estimated_remote_cost_usd": float(cost) if cost is not None else None,
        "elapsed_seconds": float(summary.get("elapsed_seconds", 0.0) or 0.0),
    }


def _case_results(path: Path) -> dict[tuple[str, str], bool]:
    results: dict[tuple[str, str], bool] = {}
    for record in _read_jsonl(path):
        key = (str(record["dataset_id"]), str(record["case_id"]))
        if key in results:
            raise ValueError(f"Duplicate case in {path}: {key}")
        results[key] = bool(record.get("answer_correct"))
    return results


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_pp(value: float) -> str:
    return f"{value:+.1f} pp"


def _format_integer(value: int) -> str:
    return f"{value:,}"


def _format_cost(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("tablebench", "wikitq"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = summarize_ablation_suite(
        suite_root=args.suite_root,
        dataset=args.dataset,
    )
    print(render_markdown(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
