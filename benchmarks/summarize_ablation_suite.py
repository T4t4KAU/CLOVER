"""Build human-readable tables for a completed CLOVER ablation suite."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

VARIANTS = (
    ("full", "Full CLOVER"),
    ("no_edge", "w/o Edge Agent"),
    ("static", "w/o Edge Repair"),
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
    full_row = rows[0]
    for row in rows:
        row["delta_vs_full_pp"] = (row["accuracy"] - full_accuracy) * 100.0
        row["remote_calls_delta_vs_full"] = row["remote_calls"] - full_row["remote_calls"]
        row["cloud_synthesis_delta_vs_full"] = (
            row["cloud_synthesis_calls"] - full_row["cloud_synthesis_calls"]
        )
        row["cloud_replan_delta_vs_full"] = (
            row["cloud_replan_calls"] - full_row["cloud_replan_calls"]
        )
        row["remote_tokens_delta_vs_full"] = row["remote_tokens"] - full_row["remote_tokens"]
        row["local_slm_calls_delta_vs_full"] = row["local_slm_calls"] - full_row["local_slm_calls"]
        row["estimated_remote_cost_delta_vs_full"] = _optional_delta(
            row["estimated_remote_cost_usd"],
            full_row["estimated_remote_cost_usd"],
        )

    row_by_variant = {row["variant"]: row for row in rows}
    edge_substitution = _edge_substitution_summary(
        full=full_row,
        no_edge=row_by_variant["no_edge"],
    )

    report = {
        "dataset": dataset,
        "suite_root": suite_root.as_posix(),
        "reference_variant": "full",
        "total_cases": rows[0]["total_cases"],
        "variants": rows,
        "edge_substitution": edge_substitution,
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
    """Render the compact tables printed after an ablation suite."""

    lines = [
        f"# {report['dataset']} ablation summary",
        "",
        "## Effectiveness",
        "",
        "| Experiment | Correct | Accuracy | Δ vs Full | Runtime failures "
        "| Regressions | Recoveries | McNemar p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["variants"]:
        values = {
            **row,
            "accuracy_text": _format_percent(row["accuracy"]),
            "delta_text": _format_pp(row["delta_vs_full_pp"]),
        }
        lines.append(
            "| {display_name} | {correct}/{total_cases} | {accuracy_text} | {delta_text} "
            "| {runtime_failures} | {regressions_vs_full} | {recoveries_vs_full} "
            "| {mcnemar_text} |".format_map(
                {
                    **values,
                    "mcnemar_text": _format_p_value(row["mcnemar_exact_p"]),
                }
            )
        )

    lines.extend(
        [
            "",
            "## Mechanism activity",
            "",
            "| Experiment | Node Edge runs | Node Edge successes | Node Edge steps "
            "| Node reviews | Contract rejects | Terminal Edge calls "
            "| Terminal hits | Terminal escalations | Cloud replans |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["variants"]:
        lines.append(
            "| {display_name} | {node_edge_calls} | {node_edge_successes} "
            "| {node_edge_steps} | {node_review_calls} | {contract_rejections} "
            "| {terminal_edge_calls} | {terminal_edge_hits} "
            "| {terminal_edge_escalations} | {cloud_replan_calls} |".format_map(row)
        )

    lines.extend(
        [
            "",
            "## Edge-to-Cloud substitution",
            "",
            "| Metric | Full CLOVER | w/o Edge Agent | Change after disabling Edge |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    edge = report["edge_substitution"]
    edge_rows = (
        (
            "Accuracy",
            _format_percent(edge["full_accuracy"]),
            _format_percent(edge["no_edge_accuracy"]),
            _format_pp(edge["accuracy_delta_pp"]),
        ),
        (
            "Cloud calls",
            str(edge["full_cloud_calls"]),
            str(edge["no_edge_cloud_calls"]),
            _format_signed_integer(edge["cloud_call_increase"]),
        ),
        (
            "Cloud calls/query",
            f"{edge['full_cloud_calls_per_query']:.3f}",
            f"{edge['no_edge_cloud_calls_per_query']:.3f}",
            _format_signed_float(edge["cloud_call_increase_per_query"], digits=3),
        ),
        (
            "Cloud-call reduction from Edge",
            _format_percent(edge["cloud_call_reduction_vs_no_edge"]),
            "0.0%",
            "Full vs w/o Edge",
        ),
        (
            "Cloud synthesis calls",
            str(edge["full_cloud_synthesis_calls"]),
            str(edge["no_edge_cloud_synthesis_calls"]),
            _format_signed_integer(edge["cloud_synthesis_increase"]),
        ),
        (
            "Cloud replan calls",
            str(edge["full_cloud_replan_calls"]),
            str(edge["no_edge_cloud_replan_calls"]),
            _format_signed_integer(edge["cloud_replan_increase"]),
        ),
        (
            "Cloud tokens",
            _format_integer(edge["full_remote_tokens"]),
            _format_integer(edge["no_edge_remote_tokens"]),
            _format_signed_integer(edge["remote_token_increase"]),
        ),
        (
            "Estimated API cost",
            _format_cost(edge["full_remote_cost_usd"]),
            _format_cost(edge["no_edge_remote_cost_usd"]),
            _format_signed_cost(edge["remote_cost_increase_usd"]),
        ),
        (
            "Local SLM calls",
            str(edge["full_local_slm_calls"]),
            str(edge["no_edge_local_slm_calls"]),
            _format_signed_integer(edge["local_slm_call_change"]),
        ),
    )
    for metric, full_value, no_edge_value, change in edge_rows:
        lines.append(f"| {metric} | {full_value} | {no_edge_value} | {change} |")
    lines.extend(
        [
            "",
            edge["interpretation"],
            "",
            "Full CLOVER recorded "
            f"{edge['full_terminal_edge_hits']} terminal Edge hits and "
            f"{edge['full_node_edge_successes']} successful node-level Edge runs. "
            "These counts need not equal the Cloud-call increase exactly because a "
            "node repair can change later execution and replanning paths.",
            "",
            "## Calls and cost",
            "",
            "| Experiment | Cloud calls | Δ Cloud calls | Cloud synthesis | Local SLM calls "
            "| Cloud tokens | Local tokens | Est. cost | Time |",
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
            "| {display_name} | {remote_calls} | {remote_calls_delta_text} "
            "| {cloud_synthesis_calls} "
            "| {local_slm_calls} | {remote_tokens_text} "
            "| {local_slm_tokens_text} | {cost_text} | {time_text} |".format_map(
                {
                    **values,
                    "remote_calls_delta_text": _format_signed_integer(
                        row["remote_calls_delta_vs_full"]
                    ),
                }
            )
        )

    lines.extend(
        [
            "",
            "## Final answer sources",
            "",
            "| Experiment | Static | Terminal Edge | Cloud synthesis | Other/failed |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["variants"]:
        lines.append(
            "| {display_name} | {static_final_answers} | {terminal_final_answers} "
            "| {cloud_final_answers} | {other_final_answers} |".format_map(row)
        )

    lines.extend(
        [
            "",
            "Δ vs Full uses percentage points. Regressions are cases that Full answered "
            "correctly but the variant did not; recoveries are the reverse. McNemar p is "
            "the two-sided exact paired-test value. Contract rejects refer to the local "
            "Edge Agent output contract; terminal Edge calls are reported separately "
            "from node-level Edge runs.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_row(
    *,
    variant: str,
    display_name: str,
    summary: dict[str, Any],
    cases: dict[tuple[str, str], dict[str, Any]],
    full_cases: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    if set(cases) != set(full_cases):
        raise ValueError(f"{variant} does not contain the same case set as Full CLOVER")

    total = len(cases)
    correct = sum(bool(record.get("answer_correct")) for record in cases.values())
    regressions = sum(
        bool(full_record.get("answer_correct")) and not bool(cases[key].get("answer_correct"))
        for key, full_record in full_cases.items()
    )
    recoveries = sum(
        not bool(full_record.get("answer_correct")) and bool(cases[key].get("answer_correct"))
        for key, full_record in full_cases.items()
    )
    source_counts = Counter(
        str(record.get("final_answer_source") or "other") for record in cases.values()
    )
    static_sources = {
        "action_static",
        "edge_static_relative_row",
        "format_answer",
    }
    static_answers = sum(source_counts[source] for source in static_sources)
    terminal_answers = source_counts["edge_local_review"]
    cloud_answers = source_counts["synthesis"]
    counters = summary.get("system_profile", {}).get("counters", {})
    remote_usage = summary.get("remote_token_usage") or {}
    local_usage = summary.get("local_slm_token_usage") or {}
    cost = summary.get("remote_cost_estimate", {}).get("cost_usd", {}).get("total")
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
        "mcnemar_exact_p": _mcnemar_exact_p(regressions, recoveries),
        "remote_calls": int(summary.get("remote_calls", 0) or 0),
        "cloud_synthesis_calls": int(counters.get("supervisor_synthesis_calls", 0) or 0),
        "cloud_replan_calls": int(counters.get("cloud_replan_calls", 0) or 0),
        "cloud_replan_blocked": int(counters.get("cloud_replan_blocked", 0) or 0),
        "local_slm_calls": int(summary.get("local_slm_calls", 0) or 0),
        "node_edge_calls": int(counters.get("executor_edge_agent_calls", 0) or 0),
        "node_edge_successes": int(counters.get("executor_edge_agent_successes", 0) or 0),
        "node_edge_failures": int(counters.get("executor_edge_agent_failures", 0) or 0),
        "node_edge_fallbacks": int(counters.get("executor_edge_agent_fallbacks", 0) or 0),
        "node_edge_steps": int(counters.get("executor_local_slm_steps", 0) or 0),
        "node_review_calls": int(counters.get("executor_edge_local_reviews", 0) or 0),
        "contract_rejections": int(counters.get("executor_contract_rejections", 0) or 0),
        "terminal_edge_calls": int(summary.get("edge_local_review_calls", 0) or 0),
        "terminal_edge_hits": int(summary.get("edge_local_review_hits", 0) or 0),
        "terminal_edge_escalations": int(summary.get("edge_local_review_escalations", 0) or 0),
        "terminal_edge_validation_failures": int(
            counters.get("edge_local_review_validation_failures", 0) or 0
        ),
        "remote_tokens": int(remote_usage.get("total_tokens", 0) or 0),
        "local_slm_tokens": int(local_usage.get("total_tokens", 0) or 0),
        "estimated_remote_cost_usd": float(cost) if cost is not None else None,
        "elapsed_seconds": float(summary.get("elapsed_seconds", 0.0) or 0.0),
        "static_final_answers": static_answers,
        "terminal_final_answers": terminal_answers,
        "cloud_final_answers": cloud_answers,
        "other_final_answers": total - static_answers - terminal_answers - cloud_answers,
        "final_answer_sources": dict(sorted(source_counts.items())),
    }


def _edge_substitution_summary(
    *,
    full: dict[str, Any],
    no_edge: dict[str, Any],
) -> dict[str, Any]:
    total = int(full["total_cases"])
    cloud_call_increase = no_edge["remote_calls"] - full["remote_calls"]
    cloud_synthesis_increase = no_edge["cloud_synthesis_calls"] - full["cloud_synthesis_calls"]
    cloud_replan_increase = no_edge["cloud_replan_calls"] - full["cloud_replan_calls"]
    if cloud_call_increase > 0:
        reduction = _safe_divide(cloud_call_increase, no_edge["remote_calls"])
        interpretation = (
            "**Observed direction: supported.** Disabling Edge increased Cloud "
            f"calls by {cloud_call_increase} "
            f"({_safe_divide(cloud_call_increase, total):.3f} per query) in this run. "
            f"Equivalently, Full CLOVER avoided {reduction * 100:.1f}% of the "
            "Cloud calls required by the no-Edge variant."
        )
        status = "supported"
    elif cloud_call_increase == 0:
        interpretation = (
            "**Observed direction: inconclusive.** Disabling Edge did not change "
            "the number of Cloud calls in this run; inspect Edge trigger and hit rates."
        )
        status = "inconclusive"
    else:
        interpretation = (
            "**Observed direction: contradicted in this run.** Disabling Edge reduced "
            f"Cloud calls by {abs(cloud_call_increase)}; inspect routing and run variance."
        )
        status = "contradicted"
    return {
        "status": status,
        "interpretation": interpretation,
        "total_cases": total,
        "full_accuracy": full["accuracy"],
        "no_edge_accuracy": no_edge["accuracy"],
        "accuracy_delta_pp": no_edge["delta_vs_full_pp"],
        "full_cloud_calls": full["remote_calls"],
        "no_edge_cloud_calls": no_edge["remote_calls"],
        "cloud_call_increase": cloud_call_increase,
        "full_cloud_calls_per_query": _safe_divide(full["remote_calls"], total),
        "no_edge_cloud_calls_per_query": _safe_divide(
            no_edge["remote_calls"],
            total,
        ),
        "cloud_call_increase_per_query": _safe_divide(
            cloud_call_increase,
            total,
        ),
        "cloud_call_reduction_vs_no_edge": _safe_divide(
            cloud_call_increase,
            no_edge["remote_calls"],
        ),
        "full_cloud_synthesis_calls": full["cloud_synthesis_calls"],
        "no_edge_cloud_synthesis_calls": no_edge["cloud_synthesis_calls"],
        "cloud_synthesis_increase": cloud_synthesis_increase,
        "full_cloud_replan_calls": full["cloud_replan_calls"],
        "no_edge_cloud_replan_calls": no_edge["cloud_replan_calls"],
        "cloud_replan_increase": cloud_replan_increase,
        "full_remote_tokens": full["remote_tokens"],
        "no_edge_remote_tokens": no_edge["remote_tokens"],
        "remote_token_increase": no_edge["remote_tokens"] - full["remote_tokens"],
        "full_remote_cost_usd": full["estimated_remote_cost_usd"],
        "no_edge_remote_cost_usd": no_edge["estimated_remote_cost_usd"],
        "remote_cost_increase_usd": _optional_delta(
            no_edge["estimated_remote_cost_usd"],
            full["estimated_remote_cost_usd"],
        ),
        "full_local_slm_calls": full["local_slm_calls"],
        "no_edge_local_slm_calls": no_edge["local_slm_calls"],
        "local_slm_call_change": (no_edge["local_slm_calls"] - full["local_slm_calls"]),
        "full_terminal_edge_hits": full["terminal_edge_hits"],
        "full_node_edge_successes": full["node_edge_successes"],
    }


def _case_results(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for record in _read_jsonl(path):
        key = (str(record["dataset_id"]), str(record["case_id"]))
        if key in results:
            raise ValueError(f"Duplicate case in {path}: {key}")
        results[key] = record
    return results


def _mcnemar_exact_p(regressions: int, recoveries: int) -> float:
    discordant = regressions + recoveries
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(regressions, recoveries) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


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


def _format_signed_integer(value: int) -> str:
    return f"{value:+,}"


def _format_signed_float(value: float, *, digits: int) -> str:
    return f"{value:+.{digits}f}"


def _format_cost(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _format_signed_cost(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.4f} USD"


def _format_p_value(value: float) -> str:
    return f"{value:.4f}"


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


def _optional_delta(
    value: float | None,
    reference: float | None,
) -> float | None:
    if value is None or reference is None:
        return None
    return value - reference


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


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
