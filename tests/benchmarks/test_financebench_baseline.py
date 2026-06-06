from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.financebench.adapter import load_financebench_task, select_cases
from benchmarks.financebench.remote_baseline import (
    EVAL_MODE_ORACLE,
    build_context_payload,
    financebench_answer_correct,
    render_financebench_prompt,
)


class FinanceBenchBaselineTest(unittest.TestCase):
    def test_numerical_scoring_requires_expected_numbers(self) -> None:
        self.assertTrue(
            financebench_answer_correct("0.8", "The ratio is 80%.")["correct"]
        )
        self.assertTrue(
            financebench_answer_correct(
                "Best Buy generated the most cash flow from operating activities in FY 2023 ($1.8 bn)",
                "Operating activities generated $1,800 million.",
            )["correct"]
        )
        self.assertTrue(
            financebench_answer_correct(
                "Best Buy generated the most cash flow from operating activities in FY 2023 ($1.8 bn)",
                "Operating activities generated $1,824 million.",
            )["correct"]
        )
        self.assertTrue(
            financebench_answer_correct(
                "No. Verizon's debt decreased by $229 million.",
                "No, debt decreased by $229 million.",
            )["correct"]
        )
        self.assertTrue(
            financebench_answer_correct(
                "No. Verizon's debt decreased by $229 million.",
                "No, total debt moved from $150,868 million to $150,639 million.",
            )["correct"]
        )
        self.assertFalse(
            financebench_answer_correct(
                "No. Verizon's debt decreased by $229 million.",
                "No.",
            )["correct"]
        )

    def test_select_cases_and_oracle_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_fixture(root)

            selected = select_cases(
                financebench_root=root,
                sample_size=1,
                seed=7,
                question_reasoning="Numerical reasoning",
            )
            task = load_financebench_task(root, case_id=selected[0]["case_id"])
            context = build_context_payload(
                task,
                eval_mode=EVAL_MODE_ORACLE,
                max_context_chars=100,
            )
            prompt = render_financebench_prompt(
                question=task.task_dsl["question"],
                context=context["context"],
                eval_mode=EVAL_MODE_ORACLE,
            )

        self.assertEqual(selected[0]["case_id"], "financebench_id_1")
        self.assertEqual(task.task_dsl["task_type"], "document_reasoning")
        self.assertIn("evidence page text", context["context"])
        self.assertFalse(context["stats"]["truncated"])
        self.assertIn("Here is the relevant evidence", prompt)


def _write_fixture(root: Path) -> None:
    (root / "data").mkdir(parents=True)
    (root / "pdfs").mkdir(parents=True)
    rows = [
        {
            "financebench_id": "financebench_id_1",
            "company": "ExampleCo",
            "doc_name": "EXAMPLE_2023_10K",
            "question_type": "metrics-generated",
            "question_reasoning": "Numerical reasoning",
            "question": "What is revenue?",
            "answer": "$10.00",
            "evidence": [{"evidence_text_full_page": "evidence page text"}],
        }
    ]
    infos = [
        {
            "doc_name": "EXAMPLE_2023_10K",
            "company": "ExampleCo",
            "doc_type": "10k",
            "doc_period": 2023,
            "doc_link": "https://example.com/doc.pdf",
        }
    ]
    _write_jsonl(root / "data" / "financebench_open_source.jsonl", rows)
    _write_jsonl(root / "data" / "financebench_document_information.jsonl", infos)
    (root / "pdfs" / "EXAMPLE_2023_10K.pdf").write_bytes(b"%PDF-1.4\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
