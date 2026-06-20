"""Sanity-check completed CLOVER ablation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

VARIANT_FLAGS = {
    "full": {
        "enable_edge_agent": True,
        "enable_edge_repair": True,
        "enable_terminal_edge_review": True,
        "enable_contract_gate": True,
        "enable_node_review": True,
        "enable_cloud_recovery": True,
        "enable_cloud_replan": True,
        "enable_cloud_synthesis": True,
        "enable_static_finalization": True,
    },
    "static": {
        "enable_edge_agent": True,
        "enable_edge_repair": False,
        "enable_terminal_edge_review": True,
        "enable_contract_gate": True,
        "enable_node_review": True,
        "enable_cloud_recovery": True,
        "enable_cloud_replan": True,
        "enable_cloud_synthesis": True,
        "enable_static_finalization": True,
    },
    "no_contract": {
        "enable_edge_agent": True,
        "enable_edge_repair": True,
        "enable_terminal_edge_review": True,
        "enable_contract_gate": False,
        "enable_node_review": True,
        "enable_cloud_recovery": True,
        "enable_cloud_replan": True,
        "enable_cloud_synthesis": True,
        "enable_static_finalization": True,
    },
    "end_review": {
        "enable_edge_agent": True,
        "enable_edge_repair": False,
        "enable_terminal_edge_review": True,
        "enable_contract_gate": True,
        "enable_node_review": False,
        "enable_cloud_recovery": True,
        "enable_cloud_replan": True,
        "enable_cloud_synthesis": True,
        "enable_static_finalization": True,
    },
    "one_shot": {
        "enable_edge_agent": True,
        "enable_edge_repair": True,
        "enable_terminal_edge_review": True,
        "enable_contract_gate": True,
        "enable_node_review": True,
        "enable_cloud_recovery": True,
        "enable_cloud_replan": False,
        "enable_cloud_synthesis": True,
        "enable_static_finalization": True,
    },
    "cloud_finalize": {
        "enable_edge_agent": True,
        "enable_edge_repair": True,
        "enable_terminal_edge_review": False,
        "enable_contract_gate": True,
        "enable_node_review": True,
        "enable_cloud_recovery": True,
        "enable_cloud_replan": True,
        "enable_cloud_synthesis": True,
        "enable_static_finalization": False,
    },
}


def check_ablation_suite(*, suite_root: Path, dataset: str) -> dict[str, Any]:
    """Validate variant flags, fixed cases, and key mechanism counters."""

    expected_cases = _manifest_case_keys(suite_root / "cases.jsonl")
    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    for variant, expected_flags in VARIANT_FLAGS.items():
        run_dir = suite_root / f"{dataset}_{variant}"
        summary_path = run_dir / "run_summary.json"
        if not summary_path.is_file():
            failures.append(f"missing summary: {summary_path}")
            continue
        summary = _read_json(summary_path)
        actual_cases = _run_case_keys(run_dir / "cases_index.jsonl")
        _record_check(
            checks,
            failures,
            variant,
            "fixed_case_set",
            actual_cases == expected_cases,
            {"expected": len(expected_cases), "actual": len(actual_cases)},
        )
        features = (
            summary.get("local_slm", {})
            .get("runtime_features", {})
        )
        _record_check(
            checks,
            failures,
            variant,
            "feature_flags",
            all(features.get(key) is value for key, value in expected_flags.items()),
            {"expected": expected_flags, "actual": features},
        )
        counters = summary.get("system_profile", {}).get("counters", {})
        if variant in {"static", "end_review"}:
            edge_steps = int(counters.get("executor_local_slm_steps", 0) or 0)
            _record_check(
                checks,
                failures,
                variant,
                "edge_repair_disabled",
                edge_steps == 0,
                {"executor_local_slm_steps": edge_steps},
            )
        if variant == "end_review":
            node_reviews = int(counters.get("executor_edge_local_reviews", 0) or 0)
            _record_check(
                checks,
                failures,
                variant,
                "node_review_disabled",
                node_reviews == 0,
                {"executor_edge_local_reviews": node_reviews},
            )
        if variant == "one_shot":
            replan_calls = int(counters.get("cloud_replan_calls", 0) or 0)
            _record_check(
                checks,
                failures,
                variant,
                "cloud_replan_disabled",
                replan_calls == 0,
                {
                    "cloud_replan_calls": replan_calls,
                    "supervisor_synthesis_calls": int(
                        counters.get("supervisor_synthesis_calls", 0) or 0
                    ),
                },
            )
        if variant == "cloud_finalize":
            static_hits = int(counters.get("static_final_answer_hits", 0) or 0)
            action_hits = int(counters.get("action_group_static_answer_hits", 0) or 0)
            terminal_edge_calls = int(
                counters.get("edge_local_review_calls", 0) or 0
            )
            _record_check(
                checks,
                failures,
                variant,
                "static_finalization_disabled",
                static_hits == 0
                and action_hits == 0
                and terminal_edge_calls == 0,
                {
                    "static_final_answer_hits": static_hits,
                    "action_group_static_answer_hits": action_hits,
                    "edge_local_review_calls": terminal_edge_calls,
                },
            )

    report = {
        "dataset": dataset,
        "suite_root": suite_root.as_posix(),
        "ok": not failures,
        "case_count": len(expected_cases),
        "checks": checks,
        "failures": failures,
    }
    (suite_root / "sanity_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _record_check(
    checks: list[dict[str, Any]],
    failures: list[str],
    variant: str,
    name: str,
    ok: bool,
    detail: dict[str, Any],
) -> None:
    checks.append(
        {
            "variant": variant,
            "check": name,
            "ok": ok,
            "detail": detail,
        }
    )
    if not ok:
        failures.append(f"{variant}: {name}")


def _manifest_case_keys(path: Path) -> set[tuple[str, str]]:
    return {
        (str(record["dataset_id"]), str(record["case_id"]))
        for record in _read_jsonl(path)
    }


def _run_case_keys(path: Path) -> set[tuple[str, str]]:
    return {
        (str(record["dataset_id"]), str(record["case_id"]))
        for record in _read_jsonl(path)
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("tablebench", "wikitq"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = check_ablation_suite(
        suite_root=args.suite_root,
        dataset=args.dataset,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
