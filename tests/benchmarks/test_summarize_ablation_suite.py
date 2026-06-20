from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.summarize_ablation_suite import (
    VARIANTS,
    summarize_ablation_suite,
)


class SummarizeAblationSuiteTest(unittest.TestCase):
    def test_writes_tables_and_compares_each_variant_with_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_root = Path(tmpdir)
            full_results = [True, True, False, False]
            variant_results = {
                "full": full_results,
                "static": [True, False, False, False],
                "no_contract": [True, True, True, False],
                "end_review": [True, False, True, False],
                "one_shot": [True, True, False, False],
                "cloud_finalize": [False, True, False, True],
            }
            for index, (variant, _) in enumerate(VARIANTS):
                _write_variant(
                    suite_root=suite_root,
                    dataset="wikitq",
                    variant=variant,
                    results=variant_results[variant],
                    index=index,
                )

            report = summarize_ablation_suite(
                suite_root=suite_root,
                dataset="wikitq",
            )

            rows = {row["variant"]: row for row in report["variants"]}
            self.assertEqual(rows["static"]["regressions_vs_full"], 1)
            self.assertEqual(rows["static"]["recoveries_vs_full"], 0)
            self.assertEqual(rows["static"]["delta_vs_full_pp"], -25.0)
            self.assertEqual(rows["static"]["mcnemar_exact_p"], 1.0)
            self.assertEqual(rows["no_contract"]["recoveries_vs_full"], 1)
            self.assertEqual(rows["no_contract"]["delta_vs_full_pp"], 25.0)
            self.assertTrue((suite_root / "ablation_summary.json").is_file())
            self.assertTrue((suite_root / "ablation_summary.csv").is_file())
            markdown = (suite_root / "ablation_summary.md").read_text(
                encoding="utf-8"
            )

        self.assertIn("| Full CLOVER | 2/4 | 50.0% | +0.0 pp |", markdown)
        self.assertIn("| w/o Edge Repair | 1/4 | 25.0% | -25.0 pp |", markdown)
        self.assertIn("Cloud calls", markdown)
        self.assertIn("Mechanism activity", markdown)


def _write_variant(
    *,
    suite_root: Path,
    dataset: str,
    variant: str,
    results: list[bool],
    index: int,
) -> None:
    run_dir = suite_root / f"{dataset}_{variant}"
    run_dir.mkdir()
    records = [
        {
            "dataset_id": f"table-{case_index}",
            "case_id": f"case-{case_index}",
            "answer_correct": correct,
            "final_answer_source": "synthesis" if case_index % 2 else "format_answer",
        }
        for case_index, correct in enumerate(results)
    ]
    (run_dir / "cases_index.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "runtime_successes": len(results),
        "runtime_failures": 0,
        "mismatches": len(results) - sum(results),
        "remote_calls": 10 + index,
        "edge_local_review_calls": index,
        "edge_local_review_hits": max(0, index - 1),
        "edge_local_review_escalations": 1,
        "local_slm_calls": index + 2,
        "remote_token_usage": {"total_tokens": 1000 + index},
        "local_slm_token_usage": {"total_tokens": 100 + index},
        "remote_cost_estimate": {"cost_usd": {"total": 0.01 + index / 100}},
        "elapsed_seconds": 60 + index,
        "system_profile": {
            "counters": {
                "supervisor_synthesis_calls": 2 + index,
                "cloud_replan_calls": index,
                "executor_edge_agent_calls": index + 1,
                "executor_edge_agent_successes": index,
                "executor_local_slm_steps": index + 3,
                "executor_edge_local_reviews": index + 4,
                "executor_contract_rejections": index + 5,
            }
        },
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
