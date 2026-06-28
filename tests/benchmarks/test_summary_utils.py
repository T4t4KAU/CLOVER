from __future__ import annotations

import unittest

from benchmarks.utils import build_brief_summary, compact_run_summary, format_brief_summary


class SummaryUtilsTest(unittest.TestCase):
    def test_brief_summary_reports_compact_token_totals(self) -> None:
        summary = {
            "stage": "tablebench_eval",
            "total_cases": 4,
            "correct": 3,
            "accuracy_on_all_cases": 0.75,
            "remote_token_usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            },
            "local_slm_token_usage": {
                "input_tokens": 100,
                "output_tokens": 200,
                "total_tokens": 300,
            },
        }

        brief = build_brief_summary(summary)

        self.assertEqual(brief["acc_pct"], 75.0)
        self.assertEqual(brief["input_tokens"], 110)
        self.assertEqual(brief["output_tokens"], 220)
        self.assertEqual(brief["total_tokens"], 330)
        self.assertEqual(brief["avg_total_tokens_per_query"], 82.5)
        rendered = format_brief_summary(brief)
        self.assertIn("Input Tokens", rendered)
        self.assertNotIn("Cloud Tokens", rendered)

    def test_compact_run_summary_drops_full_profile_but_keeps_counters(self) -> None:
        summary = {
            "run_name": "demo",
            "stage": "mmqa_eval",
            "total_cases": 2,
            "correct": 1,
            "accuracy_on_all_cases": 0.5,
            "remote_token_usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            "local_slm_token_usage": {"input_tokens": 4, "output_tokens": 5, "total_tokens": 9},
            "elapsed_seconds": 10,
            "system_profile": {
                "counters": {"executor_local_slm_steps": 7},
                "large_debug_payload": {"x": "y"},
            },
        }

        compact = compact_run_summary(summary)

        self.assertNotIn("system_profile", compact)
        self.assertEqual(compact["system_counters"]["executor_local_slm_steps"], 7)
        self.assertEqual(compact["input_tokens"], 5)
        self.assertEqual(compact["output_tokens"], 7)
        self.assertEqual(compact["avg_seconds_per_query"], 5.0)


if __name__ == "__main__":
    unittest.main()
