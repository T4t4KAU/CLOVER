from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from clover.executor import execute_physical_plan
from clover.executor.resources import ResourceLimits, ResourceStore
from clover.tools.table_reasoning.pandas_backend import PandasTable


class ExecutorResourceTest(unittest.TestCase):
    def test_large_output_spills_to_tmp_and_materializes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ResourceStore(
                limits=ResourceLimits(
                    memory_budget_bytes=1024,
                    spill_threshold_bytes=1,
                    spill_root=Path(tmpdir),
                )
            )
            resource = store.put_output(
                "T0",
                PandasTable(pd.DataFrame({"value": [1, 2, 3]})),
                producer_node="N0",
            )

            self.assertEqual(resource.storage, "file_spilled")
            self.assertTrue(resource.path.is_file())
            materialized = resource.materialize()
            self.assertEqual(materialized.frame["value"].tolist(), [1, 2, 3])
            spilled_path = resource.path

            store.close_all()

            self.assertFalse(spilled_path.exists())

    def test_external_file_resource_close_does_not_delete_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n1\n2\n", encoding="utf-8")
            store = ResourceStore(
                external_resources={
                    "table_1": _resource("table_1", table_path),
                },
                limits=ResourceLimits(spill_root=Path(tmpdir) / "spill"),
            )

            source = store.get_source("table_1")
            materialized = source.materialize(target="pandas")
            self.assertEqual(materialized.frame["value"].tolist(), [1, 2])

            store.close_all()

            self.assertTrue(table_path.exists())

    def test_executor_reads_spilled_intermediate_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n1\n2\n3\n", encoding="utf-8")
            spill_root = Path(tmpdir) / "spill"

            result = execute_physical_plan(
                _count_plan(table_path),
                resource_memory_budget_bytes=1024,
                resource_spill_threshold_bytes=1,
                resource_spill_root=str(spill_root),
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.answer, 3)
            self.assertEqual(result.output_summaries["T0"]["rows"], 3)
            self.assertFalse(any(spill_root.glob("**/*.pkl")))


def _resource(source_id: str, table_path: Path) -> dict:
    return {
        "id": source_id,
        "type": "table",
        "path": str(table_path),
        "format": "csv",
        "schema": {"format": "csv", "columns": ["value"]},
    }


def _count_plan(table_path: Path) -> dict:
    return {
        "task_type": "table_reasoning_v1",
        "resources": [_resource("table_1", table_path)],
        "nodes": [
            {
                "id": "N0",
                "op": "Scan",
                "dependency": [],
                "input": ["table_1"],
                "params": {"source": "table_1"},
                "output": "T0",
            },
            {
                "id": "N1",
                "op": "Aggregate",
                "dependency": ["T0"],
                "input": [],
                "params": {
                    "aggregations": [
                        {
                            "function": "COUNT",
                            "argument": {"type": "wildcard"},
                            "distinct": False,
                            "alias": "answer",
                        }
                    ],
                    "grouped": False,
                },
                "output": "T1",
            },
            {
                "id": "N2",
                "op": "FormatAnswer",
                "dependency": ["T1"],
                "input": [],
                "params": {"answer": {"name": "answer", "type": "number"}},
                "output": "answer",
            },
        ],
        "edges": [{"from": "N0", "to": "N1"}, {"from": "N1", "to": "N2"}],
    }


if __name__ == "__main__":
    unittest.main()
