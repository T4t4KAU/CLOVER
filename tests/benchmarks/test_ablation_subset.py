from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.ablation_subset import build_ablation_subset


class AblationSubsetTest(unittest.TestCase):
    def test_tablebench_subset_keeps_only_two_reasoning_qtypes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "tablebench"
            _write_tablebench_cases(root)
            manifest = Path(tmpdir) / "subset.jsonl"

            summary = build_ablation_subset(
                dataset="tablebench",
                dataset_root=root,
                output_path=manifest,
                size=10,
                seed=7,
            )
            records = _read_jsonl(manifest)

        self.assertEqual(summary["size"], 10)
        self.assertEqual(len({record["case_id"] for record in records}), 10)
        self.assertEqual(len({record["dataset_id"] for record in records}), 10)
        self.assertEqual(
            {record["qtype"] for record in records},
            {"FactChecking", "NumericalReasoning"},
        )

    def test_wikitq_subset_is_deterministic_and_stratified(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "wikitq"
            _write_wikitq_cases(root)
            first = Path(tmpdir) / "first.jsonl"
            second = Path(tmpdir) / "second.jsonl"

            build_ablation_subset(
                dataset="wikitq",
                dataset_root=root,
                output_path=first,
                size=14,
                seed=11,
            )
            build_ablation_subset(
                dataset="wikitq",
                dataset_root=root,
                output_path=second,
                size=14,
                seed=11,
            )
            first_text = first.read_text(encoding="utf-8")
            second_text = second.read_text(encoding="utf-8")
            records = _read_jsonl(first)

        self.assertEqual(first_text, second_text)
        strata = {record["stratum"] for record in records}
        self.assertIn("aggregation", strata)
        self.assertIn("multi_answer", strata)
        self.assertIn("direct_lookup", strata)


def _write_tablebench_cases(root: Path) -> None:
    strata = [
        ("FactChecking", "MatchBased"),
        ("FactChecking", "Multi-hop FactChecking"),
        ("NumericalReasoning", "Aggregation"),
        ("NumericalReasoning", "ArithmeticCalculation"),
        ("NumericalReasoning", "Comparison"),
        ("NumericalReasoning", "Counting"),
        ("NumericalReasoning", "Multi-hop NumericalReasoing"),
        ("NumericalReasoning", "Ranking"),
        ("NumericalReasoning", "Domain-Specific"),
        ("NumericalReasoning", "Time-basedCalculation"),
    ]
    for index, (qtype, qsubtype) in enumerate(strata):
        dataset_dir = root / f"table_{index:02d}"
        dataset_dir.mkdir(parents=True)
        record = {
            "case_id": f"case-{index}",
            "dataset_id": dataset_dir.name,
            "question": f"Question {index}?",
            "answer": str(index),
            "type": "number" if qtype == "NumericalReasoning" else "string",
            "qtype": qtype,
            "qsubtype": qsubtype,
        }
        (dataset_dir / "cases.jsonl").write_text(
            json.dumps(record) + "\n",
            encoding="utf-8",
        )
    excluded = root / "table_analysis"
    excluded.mkdir(parents=True)
    (excluded / "cases.jsonl").write_text(
        json.dumps(
            {
                "case_id": "analysis",
                "dataset_id": excluded.name,
                "question": "Describe a trend.",
                "answer": "up",
                "type": "string",
                "qtype": "DataAnalysis",
                "qsubtype": "TrendForecasting",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_wikitq_cases(root: Path) -> None:
    templates = [
        ("How many entries are there?", ["3"], "number"),
        ("Which player scored more points?", ["A"], "string"),
        ("Who scored the most points?", ["A"], "string"),
        ("Who came after Alice?", ["Bob"], "string"),
        ("Is Alice listed?", ["yes"], "string"),
        ("Which city is listed?", ["Paris"], "string"),
        ("Which teams qualified?", ["A", "B"], "list[string]"),
    ]
    for repeat in range(2):
        for index, (question, answer, answer_type) in enumerate(templates):
            table_index = repeat * len(templates) + index
            dataset_dir = root / f"wikitq_{table_index:02d}"
            dataset_dir.mkdir(parents=True)
            record = {
                "case_id": f"nu-{table_index}",
                "dataset_id": dataset_dir.name,
                "question": question,
                "answer": answer,
                "type": answer_type,
                "split": "pristine-unseen-tables",
            }
            (dataset_dir / "cases.jsonl").write_text(
                json.dumps(record) + "\n",
                encoding="utf-8",
            )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


if __name__ == "__main__":
    unittest.main()
