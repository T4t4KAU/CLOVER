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

    def test_edge_opportunity_policy_is_outcome_blind_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "wikitq"
            _write_edge_opportunity_cases(root)
            manifest = Path(tmpdir) / "edge.jsonl"

            summary = build_ablation_subset(
                dataset="wikitq",
                dataset_root=root,
                output_path=manifest,
                size=10,
                seed=13,
                selection_policy="edge_opportunity",
            )
            records = _read_jsonl(manifest)

        self.assertEqual(summary["selection_policy"], "edge_opportunity_outcome_blind")
        self.assertFalse(summary["uses_model_predictions"])
        self.assertFalse(summary["uses_answer_correctness"])
        self.assertEqual(len(records), 10)
        self.assertTrue(
            {
                "field_selection",
                "value_normalization",
                "list_assembly",
                "candidate_selection",
                "deterministic_control",
            }.issubset({record["stratum"] for record in records})
        )
        self.assertTrue(
            all(record["selection_policy"] == "edge_opportunity" for record in records)
        )
        self.assertTrue(
            all("selection_features" in record for record in records)
        )

    def test_tablefact_subset_keeps_only_factchecking_qtype(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "tablefact"
            _write_tablefact_cases(root)
            manifest = Path(tmpdir) / "subset.jsonl"

            summary = build_ablation_subset(
                dataset="tablefact",
                dataset_root=root,
                output_path=manifest,
                size=9,
                seed=7,
            )
            records = _read_jsonl(manifest)

        self.assertEqual(summary["size"], 9)
        self.assertEqual(summary["dataset"], "tablefact")
        self.assertEqual(len({record["case_id"] for record in records}), 9)
        self.assertEqual(
            {record["qtype"] for record in records},
            {"FactChecking"},
        )
        self.assertTrue(
            {
                "FactChecking/simple",
                "FactChecking/complex",
            }.issubset({record["stratum"] for record in records})
        )

    def test_tablefact_edge_opportunity_policy_is_outcome_blind(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "tablefact"
            _write_tablefact_edge_opportunity_cases(root)
            manifest = Path(tmpdir) / "edge.jsonl"

            summary = build_ablation_subset(
                dataset="tablefact",
                dataset_root=root,
                output_path=manifest,
                size=10,
                seed=13,
                selection_policy="edge_opportunity",
            )
            records = _read_jsonl(manifest)

        self.assertEqual(summary["selection_policy"], "edge_opportunity_outcome_blind")
        self.assertFalse(summary["uses_model_predictions"])
        self.assertFalse(summary["uses_answer_correctness"])
        self.assertEqual(len(records), 10)
        self.assertTrue(
            {
                "field_selection",
                "value_normalization",
                "candidate_selection",
                "deterministic_control",
            }.issubset({record["stratum"] for record in records})
        )
        self.assertTrue(
            all(record["selection_policy"] == "edge_opportunity" for record in records)
        )

    def test_full_eval_policy_keeps_all_eligible_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "tablebench"
            _write_tablebench_cases(root)
            manifest = Path(tmpdir) / "full.jsonl"

            summary = build_ablation_subset(
                dataset="tablebench",
                dataset_root=root,
                output_path=manifest,
                size=100,
                seed=7,
                selection_policy="full_eval",
            )
            records = _read_jsonl(manifest)

        self.assertEqual(summary["selection_policy"], "full_eval_outcome_blind")
        self.assertFalse(summary["uses_model_predictions"])
        self.assertFalse(summary["uses_answer_correctness"])
        self.assertEqual(summary["quotas"], {})
        # full_eval ignores --size and keeps every eligible case.
        self.assertEqual(len(records), summary["eligible_cases"])
        self.assertEqual(summary["size"], len(records))
        self.assertEqual(summary["requested_size"], len(records))
        self.assertEqual(
            {record["qtype"] for record in records},
            {"FactChecking", "NumericalReasoning"},
        )
        self.assertTrue(
            all(record["selection_policy"] == "full_eval" for record in records)
        )


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
        (dataset_dir / "table.csv").write_text(
            "entity,value\n"
            f"item-{index},{index}\n",
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
    (excluded / "table.csv").write_text(
        "entity,value\ntrend,up\n",
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
            (dataset_dir / "table.csv").write_text(
                "name,value\nAlice,10\nBob,20\n",
                encoding="utf-8",
            )


def _write_edge_opportunity_cases(root: Path) -> None:
    templates = [
        (
            "What city is listed for Alice?",
            "string",
            "name,city,score\nAlice,Paris,10\nBob,Rome,12\n",
        ),
        (
            "What result is listed for Team (A)?",
            "string",
            "team,result\nTeam (A),Winner\nTeam B,Runner-up\n",
        ),
        (
            "Which teams qualified?",
            "list[string]",
            "team,status\nA,qualified\nB,qualified\nC,out\n",
        ),
        (
            "Which is the domestic route?",
            "string",
            "route,airline\nHouston,Delta\nAustin,United\n",
        ),
        (
            "How many teams qualified?",
            "number",
            "team,status\nA,qualified\nB,qualified\nC,out\n",
        ),
    ]
    case_index = 0
    for repeat in range(2):
        for question, answer_type, table in templates:
            dataset_dir = root / f"edge_{case_index:02d}"
            dataset_dir.mkdir(parents=True)
            record = {
                "case_id": f"edge-{case_index}",
                "dataset_id": dataset_dir.name,
                "question": question,
                "answer": ["placeholder"],
                "type": answer_type,
                "split": "test",
            }
            (dataset_dir / "cases.jsonl").write_text(
                json.dumps(record) + "\n",
                encoding="utf-8",
            )
            (dataset_dir / "table.csv").write_text(table, encoding="utf-8")
            case_index += 1


def _write_tablefact_cases(root: Path) -> None:
    """Write TableFact cases covering the three representative strata.

    Strata come from ``qtype``/``qsubtype`` pairs since TableFact reuses the
    TableBench stratum formatter. Non-FactChecking qtypes are also written to
    confirm they are filtered out.
    """
    strata = [
        ("FactChecking", "simple"),
        ("FactChecking", "complex"),
        ("FactChecking", "small_test"),
    ]
    for index, (qtype, qsubtype) in enumerate(strata * 3):
        dataset_dir = root / f"tablefact_{index:02d}"
        dataset_dir.mkdir(parents=True)
        record = {
            "case_id": f"tf-{index}",
            "dataset_id": dataset_dir.name,
            "question": f"Is statement {index} true?",
            "answer": "yes" if index % 2 == 0 else "no",
            "type": "boolean",
            "qtype": qtype,
            "qsubtype": qsubtype,
        }
        (dataset_dir / "cases.jsonl").write_text(
            json.dumps(record) + "\n",
            encoding="utf-8",
        )
        (dataset_dir / "table.csv").write_text(
            "entity,value\n"
            f"item-{index},{index}\n",
            encoding="utf-8",
        )
    excluded = root / "tablefact_excluded"
    excluded.mkdir(parents=True)
    (excluded / "cases.jsonl").write_text(
        json.dumps(
            {
                "case_id": "tf-excluded",
                "dataset_id": excluded.name,
                "question": "Describe the trend.",
                "answer": "up",
                "type": "string",
                "qtype": "DataAnalysis",
                "qsubtype": "TrendForecasting",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (excluded / "table.csv").write_text(
        "entity,value\ntrend,up\n",
        encoding="utf-8",
    )


def _write_tablefact_edge_opportunity_cases(root: Path) -> None:
    """Write TableFact cases that exercise each edge-opportunity stratum.

    TableFact edge-opportunity strata are driven by cell mentions, format risk
    and table shape rather than qsubtype.
    """
    templates = [
        # value_normalization: question mentions a cell with format risk (parens)
        (
            "Is Team (A) the winner?",
            "team,result\nTeam (A),Winner\nTeam B,Runner-up\n",
        ),
        # field_selection: question mentions a plain body cell
        (
            "Is Alice listed?",
            "name,score\nAlice,10\nBob,20\n",
        ),
        # candidate_selection: small table, no cell mention, no format risk
        (
            "Is the first row valid?",
            "id,status\n1,ok\n2,bad\n",
        ),
        # deterministic_control: larger table, no cell mention
        (
            "Is the statement true?",
            "id,value\n"
            + "\n".join(f"{i},{i}" for i in range(20))
            + "\n",
        ),
    ]
    case_index = 0
    for repeat in range(3):
        for question, table in templates:
            dataset_dir = root / f"tf_edge_{case_index:02d}"
            dataset_dir.mkdir(parents=True)
            record = {
                "case_id": f"tf-edge-{case_index}",
                "dataset_id": dataset_dir.name,
                "question": question,
                "answer": "yes",
                "type": "boolean",
                "qtype": "FactChecking",
                "qsubtype": "simple",
            }
            (dataset_dir / "cases.jsonl").write_text(
                json.dumps(record) + "\n",
                encoding="utf-8",
            )
            (dataset_dir / "table.csv").write_text(table, encoding="utf-8")
            case_index += 1


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


if __name__ == "__main__":
    unittest.main()
