from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.tablefact.adapter import load_tablefact_task
from benchmarks.tablefact.download import convert_tablefact_release
from benchmarks.tablefact.eval import select_tablefact_cases


class TableFactDownloadConversionTest(unittest.TestCase):
    def test_converts_official_test_split_and_preserves_caption_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "Table-Fact-Checking"
            output = root / "tablefact"
            self._write_source(source)

            summary = convert_tablefact_release(
                source_root=source,
                output_root=output,
                splits=("test",),
            )
            dataset_dir = output / "tablefact_2-1570274-4"
            cases = [
                json.loads(line)
                for line in (dataset_dir / "cases.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            task = load_tablefact_task(
                output,
                "tablefact_2-1570274-4",
                case_id="tablefact_2-1570274-4_0000",
            )

        self.assertEqual(summary["case_count"], 3)
        self.assertEqual(summary["label_counts"], {"entailed": 2, "refuted": 1})
        self.assertEqual(cases[0]["answer"], True)
        self.assertEqual(cases[1]["answer"], False)
        self.assertEqual(cases[0]["qsubtype"], "simple")
        self.assertEqual(cases[0]["caption"], "tony lema")
        self.assertEqual(cases[0]["type"], "boolean")
        self.assertIn("Answer true", cases[0]["question"])
        self.assertEqual(task.metadata["dataset"], "tablefact")
        self.assertEqual(task.task_dsl["answer"]["type"], "boolean")
        self.assertEqual(task.task_dsl["hints"]["source_context"], "tony lema")
        self.assertEqual(task.task_dsl["hints"]["category"], "FactChecking")
        self.assertEqual(task.task_dsl["hints"]["subcategory"], "simple")

    def test_selects_simple_complex_and_small_subsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "Table-Fact-Checking"
            output = root / "tablefact"
            self._write_source(source)
            convert_tablefact_release(
                source_root=source,
                output_root=output,
                splits=("test",),
            )

            simple = select_tablefact_cases(
                tablefact_root=output,
                max_cases=None,
                case_ids=set(),
                dataset_id=None,
                split="test",
                subset="simple",
                sample_size=None,
                seed=1,
            )
            complex_cases = select_tablefact_cases(
                tablefact_root=output,
                max_cases=None,
                case_ids=set(),
                dataset_id=None,
                split="test",
                subset="complex",
                sample_size=None,
                seed=1,
            )
            small = select_tablefact_cases(
                tablefact_root=output,
                max_cases=None,
                case_ids=set(),
                dataset_id=None,
                split="test",
                subset="small",
                sample_size=None,
                seed=1,
            )

        self.assertEqual(len(simple), 2)
        self.assertEqual(len(complex_cases), 1)
        self.assertEqual(len(small), 2)
        self.assertEqual(small[0]["subset"], "simple")

    @staticmethod
    def _write_source(source: Path) -> None:
        tokenized = source / "tokenized_data"
        tables = source / "data" / "all_csv"
        tokenized.mkdir(parents=True)
        tables.mkdir(parents=True)
        payload = {
            "2-1570274-4.html.csv": [
                [
                    "tony lema won the open championship",
                    "tony lema won the pga championship",
                ],
                [1, 0],
                "tony lema",
            ],
            "2-1859269-1.html.csv": [
                ["there were 59 clubs in the third round"],
                [1],
                "turkish cup",
            ],
        }
        (tokenized / "test_examples.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        (tables / "2-1570274-4.html.csv").write_text(
            "tournament#wins\nopen championship#1\npga championship#0\n",
            encoding="utf-8",
        )
        (tables / "2-1859269-1.html.csv").write_text(
            "round#clubs\nthird round#59\n",
            encoding="utf-8",
        )
        data_root = source / "data"
        (data_root / "simple_test_id.json").write_text(
            json.dumps(["2-1570274-4.html.csv"]),
            encoding="utf-8",
        )
        (data_root / "complex_test_id.json").write_text(
            json.dumps(["2-1859269-1.html.csv"]),
            encoding="utf-8",
        )
        (data_root / "small_test_id.json").write_text(
            json.dumps(["2-1570274-4.html.csv"]),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
