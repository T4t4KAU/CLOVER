from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from benchmarks.databench.download import (
    convert_databench_rows,
    download_and_convert_databench,
)


class DatabenchDownloadConversionTest(unittest.TestCase):
    def test_converts_huggingface_rows_to_clover_layout(self) -> None:
        rows = [
            {
                "dataset": "001_Forbes",
                "question": "Is the richest person self-made?",
                "answer": "True",
                "type": "boolean",
                "columns_used": "['finalWorth', 'selfMade']",
                "column_types": "['number', 'boolean']",
                "split": "train",
            },
            {
                "dataset": "001_Forbes",
                "question": "What country is the richest person from?",
                "answer": "United States",
                "type": "category",
                "split": "train",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "databench"
            summary = convert_databench_rows(
                rows=rows,
                output_root=root,
                table_loader=lambda dataset_id: pd.DataFrame(
                    [
                        {
                            "finalWorth": 100,
                            "selfMade": True,
                            "country": "United States",
                        }
                    ]
                ),
            )

            dataset_dir = root / "001_Forbes"
            case_lines = (dataset_dir / "cases.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            first_case = json.loads(case_lines[0])
            task_spec = json.loads(
                (dataset_dir / "task_specs" / "001_Forbes_0000.json").read_text(
                    encoding="utf-8"
                )
            )
            table_text = (dataset_dir / "table.csv").read_text(encoding="utf-8")

        self.assertEqual(summary["dataset_count"], 1)
        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(first_case["case_id"], "001_Forbes_0000")
        self.assertEqual(first_case["dataset_id"], "001_Forbes")
        self.assertEqual(first_case["type"], "boolean")
        self.assertEqual(task_spec["task_type"], "table_reasoning.query")
        self.assertEqual(task_spec["sources"][0]["file"], "table.csv")
        self.assertIn("finalWorth,selfMade,country", table_text)

    def test_refuses_to_overwrite_existing_dataset_by_default(self) -> None:
        rows = [
            {
                "dataset": "toy",
                "question": "How many rows?",
                "answer": "1",
                "type": "number",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            convert_databench_rows(
                rows=rows,
                output_root=root,
                table_loader=lambda dataset_id: pd.DataFrame([{"value": 1}]),
            )

            with self.assertRaises(FileExistsError):
                convert_databench_rows(
                    rows=rows,
                    output_root=root,
                    table_loader=lambda dataset_id: pd.DataFrame([{"value": 1}]),
                )

    def test_sample_tables_use_sample_answer(self) -> None:
        rows = [
            {
                "dataset": "toy",
                "question": "How many rows are in the sample?",
                "answer": "2668",
                "sample_answer": "20",
                "type": "number",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = convert_databench_rows(
                rows=rows,
                output_root=root,
                table_kind="sample",
                table_loader=lambda dataset_id: pd.DataFrame([{"value": 1}]),
            )
            case = json.loads(
                (root / "toy" / "cases.jsonl").read_text(encoding="utf-8")
            )

        self.assertEqual(summary["answer_field"], "sample_answer")
        self.assertEqual(case["answer"], "20")
        self.assertEqual(case["full_answer"], "2668")

    def test_summary_uses_portable_paths(self) -> None:
        rows = [
            {
                "dataset": "toy",
                "question": "How many rows?",
                "answer": "1",
                "type": "number",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "databench"
            summary = convert_databench_rows(
                rows=rows,
                output_root=root,
                table_loader=lambda dataset_id: pd.DataFrame([{"value": 1}]),
            )
            saved_summary = json.loads(
                (root / "conversion_summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(summary["output_root"], "<external>/databench")
        self.assertEqual(saved_summary["output_root"], "<external>/databench")
        self.assertEqual(summary["datasets"][0]["table_csv"], "toy/table.csv")
        self.assertNotIn(str(root), json.dumps(saved_summary))

    def test_modelscope_source_uses_modelscope_loaders(self) -> None:
        rows = [
            {
                "dataset": "toy",
                "question": "How many rows?",
                "answer": "1",
                "type": "number",
            }
        ]
        frame = pd.DataFrame([{"value": 1}])

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "databench"
            with (
                patch(
                    "benchmarks.databench.download.load_modelscope_qa_rows",
                    return_value=rows,
                ) as load_rows,
                patch(
                    "benchmarks.databench.download.download_modelscope_dataset_snapshot",
                    return_value=Path(tmpdir) / "snapshot",
                ) as download_snapshot,
                patch(
                    "benchmarks.databench.download.load_modelscope_table",
                    return_value=frame,
                ) as load_table,
            ):
                summary = download_and_convert_databench(
                    output_root=root,
                    repo_id="owner/databench",
                    dataset_source="modelscope",
                )

        self.assertEqual(summary["source"], "modelscope")
        self.assertEqual(summary["stage"], "databench_modelscope_download")
        load_rows.assert_called_once()
        download_snapshot.assert_called_once()
        load_table.assert_called_once()


if __name__ == "__main__":
    unittest.main()
