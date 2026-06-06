from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmarks.databench.remote_sql import run_remote_sql_case


class DatabenchRemoteSqlTest(unittest.TestCase):
    def test_remote_sql_case_does_not_write_dataset_bound_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_databench_dataset(root)
            case_dir = root / "run" / "cases" / "toy_0000"

            summary = run_remote_sql_case(
                databench_root=root,
                sampled_case={
                    "dataset_id": "toy",
                    "case_id": "toy_0000",
                    "answer_type": "number",
                },
                case_dir=case_dir,
                client=_FakeClient('SELECT COUNT(*) AS answer FROM "table_1";'),
                remote_config={"model": "fake-model"},
                sample_index=0,
            )

            artifact_names = {path.name for path in case_dir.iterdir()}
            has_preprocess = (case_dir / "preprocess.json").exists()
            has_metadata = (case_dir / "metadata.json").exists()
            has_case_summary = (case_dir / "case_summary.json").exists()

        self.assertTrue(summary["parse_ok"])
        self.assertNotIn("preprocess", summary["files"])
        self.assertTrue(
            {
                "task_dsl.json",
                "local_dsl.json",
                "remote_dsl.json",
                "context.json",
                "remote_response.json",
                "remote_output.txt",
                "parsed_sql.json",
                "logic_dag.json",
                "physical_plan.json",
            }.issubset(artifact_names)
        )
        self.assertTrue(summary["files"]["physical_plan"].endswith("physical_plan.json"))
        self.assertFalse(has_preprocess)
        self.assertFalse(has_metadata)
        self.assertFalse(has_case_summary)


class _FakeClient:
    def __init__(self, output_text: str) -> None:
        self.responses = _FakeResponses(output_text)


class _FakeResponses:
    def __init__(self, output_text: str) -> None:
        self._output_text = output_text

    def create(self, **_: object) -> "_FakeResponse":
        return _FakeResponse(self._output_text)


class _FakeResponse:
    id = "resp_fake"
    status = "completed"

    def __init__(self, output_text: str) -> None:
        self.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(text=output_text)],
            )
        ]

    def model_dump(self, mode: str) -> dict[str, object]:
        return {
            "id": self.id,
            "status": self.status,
            "mode": mode,
        }


def _write_databench_dataset(root: Path) -> None:
    dataset_dir = root / "toy"
    task_specs_dir = dataset_dir / "task_specs"
    task_specs_dir.mkdir(parents=True)

    case = {
        "case_id": "toy_0000",
        "dataset_id": "toy",
        "question": "How many rows are present?",
        "answer": "2",
        "type": "number",
    }
    (dataset_dir / "cases.jsonl").write_text(
        json.dumps(case) + "\n",
        encoding="utf-8",
    )
    (task_specs_dir / "toy_0000.json").write_text(
        json.dumps(
            {
                "task_type": "table_query_reasoning",
                "question": case["question"],
                "sources": [{"id": 0, "type": "table", "file": "table.csv"}],
                "answer": {"name": "answer", "type": "number"},
            }
        ),
        encoding="utf-8",
    )
    with (dataset_dir / "table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows([["value"], ["1"], ["2"]])


if __name__ == "__main__":
    unittest.main()
