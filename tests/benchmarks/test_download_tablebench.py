from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.tablebench.adapter import load_tablebench_task
from benchmarks.tablebench.download import (
    convert_tablebench_rows,
    download_and_convert_tablebench,
    write_source_rows,
)


class TablebenchDownloadConversionTest(unittest.TestCase):
    def test_converts_rows_to_databench_style_layout(self) -> None:
        rows = [
            {
                "id": "case-a",
                "qtype": "NumericalReasoning",
                "qsubtype": "Aggregation",
                "question": "What is the sum of sales?",
                "answer": "15",
                "table": {
                    "columns": ["region", "sales"],
                    "data": [["east", "10"], ["west", "5"]],
                },
                "split": "TQA_test",
            },
            {
                "id": "case-b",
                "qtype": "FactChecking",
                "qsubtype": "Comparison",
                "question": "Is east greater than west?",
                "answer": "yes",
                "table": {
                    "columns": ["region", "sales"],
                    "data": [["east", "10"], ["west", "5"]],
                },
                "split": "TQA_test",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "tablebench"
            summary = convert_tablebench_rows(rows=rows, output_root=root)
            dataset_id = summary["datasets"][0]["dataset_id"]
            dataset_dir = root / dataset_id
            cases = [
                json.loads(line)
                for line in (dataset_dir / "cases.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            task_spec = json.loads(
                (dataset_dir / "task_specs" / "case-a.json").read_text(
                    encoding="utf-8"
                )
            )
            table_text = (dataset_dir / "table.csv").read_text(encoding="utf-8")
            task = load_tablebench_task(root, dataset_id, case_id="case-a")

        self.assertEqual(summary["dataset_count"], 1)
        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(cases[0]["type"], "number")
        self.assertEqual(cases[1]["type"], "boolean")
        self.assertEqual(task_spec["task_type"], "table_reasoning.analyze")
        self.assertEqual(task_spec["sources"][0]["file"], "table.csv")
        self.assertEqual(task.metadata["qtype"], "NumericalReasoning")
        self.assertIn("region,sales", table_text)

    def test_filters_by_qtype_and_limit(self) -> None:
        rows = [
            {
                "id": "a",
                "qtype": "Visualization",
                "qsubtype": "BarChart",
                "question": "Draw a bar chart",
                "answer": "bar",
                "table": {"columns": ["x"], "data": [["a"]]},
            },
            {
                "id": "b",
                "qtype": "NumericalReasoning",
                "qsubtype": "Aggregation",
                "question": "How many rows?",
                "answer": "1",
                "table": {"columns": ["x"], "data": [["a"]]},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = convert_tablebench_rows(
                rows=rows,
                output_root=root,
                qtypes=["NumericalReasoning"],
                limit_cases=1,
            )
            dataset_id = summary["datasets"][0]["dataset_id"]
            case = json.loads((root / dataset_id / "cases.jsonl").read_text())

        self.assertEqual(summary["case_count"], 1)
        self.assertEqual(case["case_id"], "b")
        self.assertEqual(case["qtype"], "NumericalReasoning")

    def test_excludes_visualization_cases_by_default(self) -> None:
        rows = [
            {
                "id": "chart",
                "qtype": "Visualization",
                "qsubtype": "ChartGeneration",
                "chart_type": "bar",
                "question": "Draw a bar chart.",
                "answer": "bar",
                "table": {"columns": ["x"], "data": [["a"]]},
            },
            {
                "id": "number",
                "qtype": "NumericalReasoning",
                "qsubtype": "Counting",
                "question": "How many rows?",
                "answer": "1",
                "table": {"columns": ["x"], "data": [["a"]]},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = convert_tablebench_rows(rows=rows, output_root=root)
            dataset_id = summary["datasets"][0]["dataset_id"]
            cases = [
                json.loads(line)
                for line in (root / dataset_id / "cases.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

        self.assertEqual(summary["source_case_count"], 2)
        self.assertEqual(summary["visualization_case_count"], 1)
        self.assertTrue(summary["visualization_excluded"])
        self.assertEqual(summary["case_count"], 1)
        self.assertEqual(cases[0]["case_id"], "number")
        self.assertNotEqual(cases[0]["qtype"], "Visualization")

    def test_can_include_visualization_cases_explicitly(self) -> None:
        rows = [
            {
                "id": "chart",
                "qtype": "Visualization",
                "qsubtype": "ChartGeneration",
                "question": "Draw a bar chart.",
                "answer": "bar",
                "table": {"columns": ["x"], "data": [["a"]]},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = convert_tablebench_rows(
                rows=rows,
                output_root=root,
                include_visualization=True,
            )

        self.assertFalse(summary["visualization_excluded"])
        self.assertEqual(summary["case_count"], 1)

    def test_refuses_to_overwrite_existing_dataset_by_default(self) -> None:
        rows = [
            {
                "id": "a",
                "qtype": "NumericalReasoning",
                "qsubtype": "Aggregation",
                "question": "How many rows?",
                "answer": "1",
                "table": {"columns": ["x"], "data": [["a"]]},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            convert_tablebench_rows(rows=rows, output_root=root)

            with self.assertRaises(FileExistsError):
                convert_tablebench_rows(rows=rows, output_root=root)

    def test_source_rows_are_written_without_absolute_paths_in_summary(self) -> None:
        rows = [
            {
                "id": "a",
                "split": "TQA_test",
                "qtype": "NumericalReasoning",
                "qsubtype": "Aggregation",
                "question": "How many rows?",
                "answer": "1",
                "table": {"columns": ["x"], "data": [["a"]]},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "source"
            summary = write_source_rows(
                rows=rows,
                source_root=root,
                repo_id="Multilingual-Multimodal-NLP/TableBench",
                config_name="table_bench",
                splits=("TQA_test",),
            )
            saved_summary = json.loads(
                (root / "download_summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(summary["row_count"], 1)
        self.assertEqual(saved_summary["source_root"], "<external>/source")
        self.assertNotIn(str(root), json.dumps(saved_summary))

    def test_modelscope_source_uses_modelscope_loader(self) -> None:
        rows = [
            {
                "id": "a",
                "qtype": "NumericalReasoning",
                "qsubtype": "Aggregation",
                "question": "How many rows?",
                "answer": "1",
                "table": {"columns": ["x"], "data": [["a"]]},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "tablebench"
            source_root = Path(tmpdir) / "source"
            with patch(
                "benchmarks.tablebench.download.load_modelscope_rows",
                return_value=rows,
            ) as load_rows:
                summary = download_and_convert_tablebench(
                    output_root=root,
                    source_root=source_root,
                    repo_id="owner/tablebench",
                    dataset_source="modelscope",
                )
            saved_source_summary = json.loads(
                (source_root / "download_summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(summary["source"], "modelscope")
        self.assertEqual(summary["stage"], "tablebench_modelscope_download")
        self.assertEqual(saved_source_summary["source"], "modelscope")
        load_rows.assert_called_once()


if __name__ == "__main__":
    unittest.main()
