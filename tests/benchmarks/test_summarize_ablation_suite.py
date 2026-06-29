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
            (suite_root / "cases.summary.json").write_text(
                json.dumps(
                    {
                        "selection_policy": "edge_opportunity_outcome_blind",
                        "size": 4,
                        "uses_model_predictions": False,
                        "uses_answer_correctness": False,
                    }
                ),
                encoding="utf-8",
            )
            full_results = [True, True, False, False]
            variant_results = {
                "full": full_results,
                "all_edge": [True, True, False, False],
                "no_edge": [True, False, False, False],
                "static": [True, False, False, False],
                "no_contract": [True, True, True, False],
                "end_review": [True, False, True, False],
                "one_shot": [True, True, False, False],
                "no_retry": [True, False, True, False],
                "cloud_finalize": [False, True, False, True],
                "static_only": [True, False, False, False],
                "no_static": [True, True, False, False],
                "no_closure_checker": [True, True, False, True],
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
            self.assertEqual(
                report["case_selection"]["selection_policy"],
                "edge_opportunity_outcome_blind",
            )
            self.assertEqual(rows["static"]["regressions_vs_full"], 1)
            self.assertEqual(rows["static"]["recoveries_vs_full"], 0)
            self.assertEqual(rows["static"]["delta_vs_full_pp"], -25.0)
            self.assertEqual(rows["static"]["mcnemar_exact_p"], 1.0)
            self.assertEqual(rows["no_contract"]["recoveries_vs_full"], 1)
            self.assertEqual(rows["no_contract"]["delta_vs_full_pp"], 25.0)
            self.assertEqual(rows["no_edge"]["remote_calls_delta_vs_full"], 6)
            self.assertEqual(
                report["edge_substitution"]["cloud_synthesis_increase"],
                6,
            )
            self.assertEqual(report["edge_substitution"]["status"], "supported")
            self.assertEqual(
                report["edge_substitution"]["cloud_call_reduction_vs_no_edge"],
                0.375,
            )
            self.assertEqual(
                report["edge_substitution"]["no_edge_local_slm_calls"],
                0,
            )
            self.assertEqual(rows["full"]["proactive_edge_opportunities"], 3)
            self.assertEqual(rows["full"]["proactive_edge_hits"], 2)
            self.assertEqual(rows["full"]["retry_cases"], 1)
            self.assertEqual(rows["full"]["retry_correct"], 1)
            self.assertEqual(rows["no_edge"]["regression_case_ids"], ["case-1"])
            self.assertEqual(rows["all_edge"]["all_edge_routed_nodes"], 8)
            self.assertEqual(
                report["static_fast_path_ablation"]["agreement_rate"],
                6 / 7,
            )
            self.assertEqual(
                report["static_fast_path_ablation"]["edge_call_delta"],
                7,
            )
            roles = {row["role"]: row for row in report["edge_role_decomposition"]}
            self.assertEqual(
                roles["Terminal/proactive semantic review"]["accuracy_contribution_pp"],
                25.0,
            )
            self.assertEqual(
                roles["Node-local repair and review"]["accuracy_contribution_pp"],
                0.0,
            )
            self.assertTrue((suite_root / "ablation_summary.json").is_file())
            self.assertTrue((suite_root / "ablation_summary.csv").is_file())
            self.assertTrue(
                (suite_root / "ablation_case_diagnostics.jsonl").is_file()
            )
            self.assertTrue(
                (suite_root / "ablation_discordant_cases.csv").is_file()
            )
            markdown = (suite_root / "ablation_summary.md").read_text(encoding="utf-8")

        self.assertIn("| Full CLOVER | 2/4 | 50.0% | +0.0 pp |", markdown)
        self.assertIn("edge_opportunity_outcome_blind", markdown)
        self.assertIn("| w/o Edge Agent | 1/4 | 25.0% | -25.0 pp |", markdown)
        self.assertIn("| w/o Edge Repair | 1/4 | 25.0% | -25.0 pp |", markdown)
        self.assertIn("Global calls", markdown)
        self.assertIn("Mechanism activity", markdown)
        self.assertIn("Edge-to-Global substitution", markdown)
        self.assertIn("Edge role decomposition", markdown)
        self.assertIn("Static fast-path ablation", markdown)
        self.assertIn("| Edge/static agreement | --- | 85.7% | 6 agree / 1 disagree |", markdown)
        self.assertIn("Paired case diagnostics", markdown)
        self.assertIn("`case-1`", markdown)
        self.assertIn("Observed direction: supported", markdown)


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
            "question": f"Question {case_index}?",
            "answer_correct": correct,
            "runtime_ok": True,
            "retry_count": 1 if variant == "full" and case_index == 0 else 0,
            "error": None,
            "final_answer_source": "synthesis" if case_index % 2 else "format_answer",
        }
        for case_index, correct in enumerate(results)
    ]
    (run_dir / "cases_index.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    if variant == "full":
        remote_calls = 10
        synthesis_calls = 2
        replan_calls = 0
        local_slm_calls = 5
        terminal_calls = 3
        terminal_hits = 2
        node_calls = 4
        node_successes = 3
        node_steps = 6
        node_reviews = 4
        contract_rejections = 2
        terminal_escalations = 1
        local_tokens = 100
        proactive_opportunities = 3
        proactive_calls = 3
        proactive_hits = 2
        all_edge_routed = 0
        all_edge_edge_successes = 0
        all_edge_edge_failures = 0
        all_edge_reference_successes = 0
        all_edge_reference_failures = 0
        all_edge_comparisons = 0
        all_edge_agreements = 0
        all_edge_disagreements = 0
    elif variant == "all_edge":
        remote_calls = 10
        synthesis_calls = 2
        replan_calls = 0
        local_slm_calls = 12
        terminal_calls = 3
        terminal_hits = 2
        node_calls = 8
        node_successes = 7
        node_steps = 12
        node_reviews = 4
        contract_rejections = 1
        terminal_escalations = 1
        local_tokens = 500
        proactive_opportunities = 3
        proactive_calls = 3
        proactive_hits = 2
        all_edge_routed = 8
        all_edge_edge_successes = 7
        all_edge_edge_failures = 1
        all_edge_reference_successes = 8
        all_edge_reference_failures = 0
        all_edge_comparisons = 7
        all_edge_agreements = 6
        all_edge_disagreements = 1
    elif variant == "no_edge":
        remote_calls = 16
        synthesis_calls = 8
        replan_calls = 2
        local_slm_calls = 0
        terminal_calls = 0
        terminal_hits = 0
        node_calls = 0
        node_successes = 0
        node_steps = 0
        node_reviews = 0
        contract_rejections = 0
        terminal_escalations = 0
        local_tokens = 0
        proactive_opportunities = 0
        proactive_calls = 0
        proactive_hits = 0
        all_edge_routed = 0
        all_edge_edge_successes = 0
        all_edge_edge_failures = 0
        all_edge_reference_successes = 0
        all_edge_reference_failures = 0
        all_edge_comparisons = 0
        all_edge_agreements = 0
        all_edge_disagreements = 0
    else:
        remote_calls = 10 + index
        synthesis_calls = 2 + index
        replan_calls = index
        local_slm_calls = index + 2
        terminal_calls = index
        terminal_hits = max(0, index - 1)
        node_calls = index + 1
        node_successes = index
        node_steps = index + 3
        node_reviews = index + 4
        contract_rejections = index + 5
        terminal_escalations = 1
        local_tokens = 100 + index
        proactive_opportunities = index
        proactive_calls = index
        proactive_hits = max(0, index - 1)
        all_edge_routed = 0
        all_edge_edge_successes = 0
        all_edge_edge_failures = 0
        all_edge_reference_successes = 0
        all_edge_reference_failures = 0
        all_edge_comparisons = 0
        all_edge_agreements = 0
        all_edge_disagreements = 0

    summary = {
        "runtime_successes": len(results),
        "runtime_failures": 0,
        "mismatches": len(results) - sum(results),
        "remote_calls": remote_calls,
        "edge_local_review_calls": terminal_calls,
        "edge_local_review_hits": terminal_hits,
        "edge_local_review_escalations": terminal_escalations,
        "local_slm_calls": local_slm_calls,
        "remote_token_usage": {"total_tokens": 1000 + index},
        "local_slm_token_usage": {"total_tokens": local_tokens},
        "remote_cost_estimate": {"cost_usd": {"total": 0.01 + index / 100}},
        "elapsed_seconds": 60 + index,
        "system_profile": {
            "counters": {
                "supervisor_synthesis_calls": synthesis_calls,
                "cloud_replan_calls": replan_calls,
                "executor_edge_agent_calls": node_calls,
                "executor_edge_agent_successes": node_successes,
                "executor_local_slm_steps": node_steps,
                "executor_edge_local_reviews": node_reviews,
                "executor_contract_rejections": contract_rejections,
                "edge_review_proactive_opportunities": proactive_opportunities,
                "edge_review_proactive_calls": proactive_calls,
                "edge_review_proactive_hits": proactive_hits,
                "executor_all_edge_routed_nodes": all_edge_routed,
                "executor_all_edge_edge_successes": all_edge_edge_successes,
                "executor_all_edge_edge_failures": all_edge_edge_failures,
                "executor_all_edge_static_reference_successes": (
                    all_edge_reference_successes
                ),
                "executor_all_edge_static_reference_failures": (
                    all_edge_reference_failures
                ),
                "executor_all_edge_comparisons": all_edge_comparisons,
                "executor_all_edge_agreements": all_edge_agreements,
                "executor_all_edge_disagreements": all_edge_disagreements,
            }
        },
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
