from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from clover.executor import ExecutionPlanBuilder, execute_execution_plan
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
            resource = store.put_artifact(
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

    def test_document_chunk_resource_materializes_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_store = Path(tmpdir) / "chunks.jsonl"
            chunk_store.write_text(
                json.dumps(
                    {
                        "chunk_id": "chunk_3",
                        "text": "Revenue was 140. Operating income was 35.",
                        "page_start": 1,
                        "page_end": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            store = ResourceStore(
                external_resources={
                    "document_1:chunk_3": {
                        "id": "document_1:chunk_3",
                        "type": "document_chunk",
                        "source": "document_1",
                        "source_type": "pdf",
                        "path": str(chunk_store),
                        "format": "text",
                        "item_id": "chunk_3",
                        "chunk_id": "chunk_3",
                    }
                }
            )

            source = store.get_source("document_1:chunk_3")
            self.assertEqual(
                source.materialize(target="text"),
                "Revenue was 140. Operating income was 35.",
            )
            self.assertEqual(
                source.materialize(target="chunk_record")["page_start"],
                1,
            )
            self.assertEqual(
                source.materialize(target="resource_spec")["chunk_id"],
                "chunk_3",
            )

            store.close_all()

    def test_executor_reads_spilled_intermediate_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n1\n2\n3\n", encoding="utf-8")
            spill_root = Path(tmpdir) / "spill"

            result = _execute_plan(
                _count_plan(table_path),
                resource_memory_budget_bytes=1024,
                resource_spill_threshold_bytes=1,
                resource_spill_root=str(spill_root),
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.answer, 3)
            self.assertEqual(result.collector_outputs["answer"], 3)
            self.assertEqual(result.to_dict()["collector_outputs"]["answer"], 3)
            self.assertEqual(result.output_summaries["T0"]["rows"], 3)
            self.assertFalse(any(spill_root.glob("**/*.pkl")))

    def test_resource_store_lifecycle_releases_unused_artifacts(self) -> None:
        store = ResourceStore()
        try:
            store.configure_artifact_lifecycle(
                consumers_by_artifact={"T0": 1},
                retained_artifacts={"answer"},
            )

            store.put_artifact("unused", 1, producer_node="N0")
            self.assertFalse(store.has_artifact("unused"))
            self.assertIn("unused", store.summaries())

            store.put_artifact("T0", 2, producer_node="N1")
            self.assertTrue(store.has_artifact("T0"))
            store.mark_dependencies_consumed(["T0"])
            self.assertFalse(store.has_artifact("T0"))
            self.assertIn("T0", store.summaries())

            store.put_artifact("answer", 3, producer_node="N2")
            self.assertTrue(store.has_artifact("answer"))
            self.assertEqual(store.to_dict()["answer"], 3)
        finally:
            store.close_all()

    def test_memory_output_can_materialize_none_value(self) -> None:
        store = ResourceStore()
        try:
            store.configure_artifact_lifecycle(
                consumers_by_artifact={},
                retained_artifacts={"answer"},
            )

            resource = store.put_artifact("answer", None, producer_node="N0")

            self.assertIsNone(resource.materialize())
            self.assertTrue(store.has_artifact("answer"))
            self.assertIsNone(store.to_dict()["answer"])
        finally:
            store.close_all()


def _resource(source_id: str, table_path: Path) -> dict:
    return {
        "id": source_id,
        "type": "table",
        "path": str(table_path),
        "format": "csv",
        "schema": {"format": "csv", "columns": ["value"]},
    }


def _execute_plan(plan: dict, **kwargs):
    return execute_execution_plan(
        ExecutionPlanBuilder.default().build(plan),
        collector_context=plan,
        **kwargs,
    )


def _count_plan(table_path: Path) -> dict:
    return {
        "task_type": "table_reasoning.query",
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
