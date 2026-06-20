from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.check_ablation_suite import (
    VARIANT_FLAGS,
    check_ablation_suite,
)


class CheckAblationSuiteTest(unittest.TestCase):
    def test_no_edge_variant_has_no_local_edge_activity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_root = Path(tmpdir)
            _write_suite(suite_root, dataset="wikitq")

            report = check_ablation_suite(
                suite_root=suite_root,
                dataset="wikitq",
            )

        self.assertTrue(report["ok"])
        no_edge_checks = {
            check["check"]: check for check in report["checks"] if check["variant"] == "no_edge"
        }
        self.assertTrue(no_edge_checks["all_edge_paths_disabled"]["ok"])

    def test_detects_hidden_edge_activity_in_no_edge_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_root = Path(tmpdir)
            _write_suite(suite_root, dataset="wikitq")
            summary_path = suite_root / "wikitq_no_edge" / "run_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["local_slm_calls"] = 1
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            report = check_ablation_suite(
                suite_root=suite_root,
                dataset="wikitq",
            )

        self.assertFalse(report["ok"])
        self.assertIn("no_edge: all_edge_paths_disabled", report["failures"])


def _write_suite(suite_root: Path, *, dataset: str) -> None:
    case = {"dataset_id": "table-1", "case_id": "case-1"}
    (suite_root / "cases.jsonl").write_text(
        json.dumps(case) + "\n",
        encoding="utf-8",
    )
    for variant, flags in VARIANT_FLAGS.items():
        run_dir = suite_root / f"{dataset}_{variant}"
        run_dir.mkdir()
        (run_dir / "cases_index.jsonl").write_text(
            json.dumps(case) + "\n",
            encoding="utf-8",
        )
        counters: dict[str, int] = {}
        summary = {
            "local_slm": {"runtime_features": flags},
            "local_slm_calls": 0,
            "local_slm_token_usage": {"total_tokens": 0},
            "edge_local_review_calls": 0,
            "edge_local_review_hits": 0,
            "edge_local_review_escalations": 0,
            "system_profile": {"counters": counters},
        }
        (run_dir / "run_summary.json").write_text(
            json.dumps(summary),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
