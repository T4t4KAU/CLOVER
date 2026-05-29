from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from threading import Lock
from unittest.mock import patch

from benchmarks.databench.adapter import load_databench_task, run_databench_case
from benchmarks.databench.eval import (
    _load_v2_cases_parallel,
    _preprocess_v2_cases_parallel,
    base_case_record,
)
import benchmarks.databench.eval as databench_eval
from clover.preprocess import preprocess_task_dsl
from clover.preprocess.csv_schema import extract_csv_schema


class DatabenchPreprocessTest(unittest.TestCase):
    def test_preprocesses_databench_task_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_databench_dataset(
                Path(tmpdir),
                dataset_id="toy",
                case_id="toy_0000",
                question="Did any children below the age of 18 survive?",
                answer="True",
                answer_type="boolean",
                csv_rows=[
                    ["Survived", "Pclass", "Name", "Age"],
                    ["False", "3", "Alice", "22.0"],
                    ["True", "1", "Bob", "17.0"],
                    ["True", "2", "Cara", ""],
                ],
            )
            result = _load_and_preprocess_task(tmpdir, dataset_id="toy", case_index=0)

        local_source = result["local_dsl"]["sources"][0]
        remote_source = result["remote_dsl"]["sources"][0]
        context = result["context"]

        self.assertEqual(local_source["id"], "table_1")
        self.assertEqual(local_source["schema"]["shape"], {"rows": 3, "columns": 4})
        self.assertEqual(
            local_source["schema"]["columns"],
            ["Survived", "Pclass", "Name", "Age"],
        )
        self.assertEqual(remote_source["schema"]["columns"][0], "Survived")
        self.assertEqual(remote_source["schema"]["columns"][3], "Age")
        self.assertNotIn("path", remote_source)
        self.assertEqual(set(context), {"task_type", "base_dir", "source_map"})
        source_path = Path(context["source_map"]["table_1"]["path"])
        self.assertTrue(source_path.is_absolute())
        self.assertEqual(source_path.name, "table.csv")

    def test_databench_adapter_falls_back_to_cases_jsonl_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "toy"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "cases.jsonl").write_text(
                json.dumps(
                    {
                        "case_id": "toy_0000",
                        "dataset_id": "toy",
                        "question": "How many rows are present?",
                        "answer": "2",
                        "type": "number",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (dataset_dir / "table.csv").write_text("value\n1\n2\n", encoding="utf-8")

            result = _load_and_preprocess_task(tmpdir, "toy", case_index=0)

            self.assertEqual(result["metadata"]["task_spec_path"], None)

        self.assertEqual(result["local_dsl"]["sources"][0]["schema"]["shape"]["rows"], 2)
        self.assertEqual(result["local_dsl"]["answer"]["type"], "number")

    def test_run_databench_case_writes_only_system_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_databench_dataset(
                root,
                dataset_id="toy",
                case_id="toy_0000",
                question="Did any children below the age of 18 survive?",
                answer="True",
                answer_type="boolean",
                csv_rows=[
                    ["Survived", "Pclass", "Name", "Age"],
                    ["False", "3", "Alice", "22.0"],
                    ["True", "1", "Bob", "17.0"],
                ],
            )

            summary = run_databench_case(
                databench_root=root,
                dataset_id="toy",
                case_id="toy_0000",
                case_index=0,
                output_root=root / "benchmark",
                run_name="toy_run",
            )

            case_dir = Path(summary["case_dir"])
            artifact_names = {path.name for path in case_dir.iterdir()}
            has_preprocess = (case_dir / "preprocess.json").exists()
            has_metadata = (case_dir / "metadata.json").exists()
            has_case_summary = (case_dir / "case_summary.json").exists()

        self.assertEqual(
            artifact_names,
            {"task_dsl.json", "local_dsl.json", "remote_dsl.json", "context.json"},
        )
        self.assertEqual(
            set(summary["files"]),
            {"task_dsl", "local_dsl", "remote_dsl", "context"},
        )
        self.assertFalse(has_preprocess)
        self.assertFalse(has_metadata)
        self.assertFalse(has_case_summary)

    def test_csv_schema_handles_multiline_quoted_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "rank,name,finalWorth,bio\n"
                "1,Alice,1000,\"first line\nsecond line with \"\"quoted\"\" text\"\n"
                "2,Bob,500,\"plain text\"\n",
                encoding="utf-8",
            )

            schema = extract_csv_schema(table_path)

        self.assertEqual(schema["shape"], {"rows": 2, "columns": 4})
        self.assertEqual(schema["columns"], ["rank", "name", "finalWorth", "bio"])

    def test_v2_startup_preprocess_reuses_same_table_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_databench_cases_dataset(
                root,
                dataset_id="toy",
                cases=[
                    {
                        "case_id": "toy_0000",
                        "question": "How many rows are present?",
                        "answer": "2",
                        "type": "number",
                    },
                    {
                        "case_id": "toy_0001",
                        "question": "Is any value above 1?",
                        "answer": "True",
                        "type": "boolean",
                    },
                ],
                csv_rows=[["value"], ["1"], ["2"]],
            )
            selected_cases = [
                {
                    "dataset_id": "toy",
                    "case_id": "toy_0000",
                    "case_index": 0,
                    "answer_type": "number",
                },
                {
                    "dataset_id": "toy",
                    "case_id": "toy_0001",
                    "case_index": 1,
                    "answer_type": "boolean",
                },
            ]
            output_dir = root / "run"
            records_by_case = {}
            started_by_case = {}
            for sample_index, sampled_case in enumerate(selected_cases):
                case_dir = output_dir / "cases" / sampled_case["case_id"]
                case_dir.mkdir(parents=True)
                started_by_case[sampled_case["case_id"]] = 0.0
                records_by_case[sampled_case["case_id"]] = base_case_record(
                    sampled_case=sampled_case,
                    sample_index=sample_index,
                    case_dir=case_dir,
                )
            completed_records = []
            progress_lock = Lock()

            loaded_cases = _load_v2_cases_parallel(
                databench_root=root,
                selected_cases=selected_cases,
                output_dir=output_dir,
                worker_count=2,
                records_by_case=records_by_case,
                started_by_case=started_by_case,
                completed_records=completed_records,
                progress_bar=None,
                progress_lock=progress_lock,
            )
            call_count = 0
            original_preprocess = databench_eval.preprocess_task_dsl

            def wrapped_preprocess(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return original_preprocess(*args, **kwargs)

            profile = {
                "preprocess_cache_hits": 0,
                "preprocess_cache_misses": 0,
                "preprocess_failed_cases": 0,
            }
            with patch(
                "benchmarks.databench.eval.preprocess_task_dsl",
                side_effect=wrapped_preprocess,
            ):
                prepared_cases = _preprocess_v2_cases_parallel(
                    loaded_cases=loaded_cases,
                    worker_count=2,
                    records_by_case=records_by_case,
                    started_by_case=started_by_case,
                    completed_records=completed_records,
                    progress_bar=None,
                    progress_lock=progress_lock,
                    startup_profile=profile,
                )

        self.assertEqual(call_count, 1)
        self.assertEqual(len(prepared_cases), 2)
        self.assertEqual(profile["preprocess_cache_hits"], 1)
        self.assertEqual(profile["preprocess_cache_misses"], 1)
        prepared_cases.sort(key=lambda item: item.sample_index)
        self.assertEqual(
            [item.preprocess_result["remote_dsl"]["question"] for item in prepared_cases],
            ["How many rows are present?", "Is any value above 1?"],
        )


def _write_databench_dataset(
    root: Path,
    dataset_id: str,
    case_id: str,
    question: str,
    answer: str,
    answer_type: str,
    csv_rows: list[list[str]],
) -> None:
    dataset_dir = root / dataset_id
    task_specs_dir = dataset_dir / "task_specs"
    task_specs_dir.mkdir(parents=True)

    (dataset_dir / "cases.jsonl").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "dataset_id": dataset_id,
                "question": question,
                "answer": answer,
                "type": answer_type,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (task_specs_dir / f"{case_id}.json").write_text(
        json.dumps(
            {
                "task_type": "table_reasoning",
                "question": question,
                "sources": [{"id": 0, "type": "table", "file": "table.csv"}],
                "answer": {"name": "answer", "type": answer_type},
            }
        ),
        encoding="utf-8",
    )

    with (dataset_dir / "table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(csv_rows)


def _write_databench_cases_dataset(
    root: Path,
    dataset_id: str,
    cases: list[dict[str, str]],
    csv_rows: list[list[str]],
) -> None:
    dataset_dir = root / dataset_id
    task_specs_dir = dataset_dir / "task_specs"
    task_specs_dir.mkdir(parents=True)
    with (dataset_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps({"dataset_id": dataset_id, **case}) + "\n")
            (task_specs_dir / f"{case['case_id']}.json").write_text(
                json.dumps(
                    {
                        "task_type": "table_reasoning",
                        "question": case["question"],
                        "sources": [{"id": 0, "type": "table", "file": "table.csv"}],
                        "answer": {"name": "answer", "type": case["type"]},
                    }
                ),
                encoding="utf-8",
            )
    with (dataset_dir / "table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(csv_rows)


def _load_and_preprocess_task(
    databench_root: str | Path,
    dataset_id: str,
    case_index: int,
) -> dict:
    task = load_databench_task(databench_root, dataset_id, case_index=case_index)
    result = preprocess_task_dsl(task.task_dsl, base_dir=task.base_dir)
    result["metadata"] = task.metadata
    return result


if __name__ == "__main__":
    unittest.main()
