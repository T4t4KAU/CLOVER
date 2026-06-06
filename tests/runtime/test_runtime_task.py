from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clover.runtime import (
    DocumentReasoningCaseSpec,
    DocumentTaskItem,
    RuntimeCommandItem,
    RuntimeWorkItem,
    RuntimeTaskItem,
    TableReasoningCaseSpec,
    TableTaskItem,
    build_document_task_items,
)
from clover.runtime.document_reasoning import DocumentLogicDagItem, PythonCodeItem
from clover.runtime.table_reasoning.pipeline import LogicDagItem, SqlItem
from clover.runtime.table_reasoning.pipeline import _build_task_items as build_table_task_items


class RuntimeTaskTest(unittest.TestCase):
    def test_table_reasoning_uses_runtime_task_item(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n1\n", encoding="utf-8")
            spec = TableReasoningCaseSpec(
                case_id="table_case",
                task_dsl={
                    "task_type": "table_reasoning.query",
                    "question": "How many rows?",
                    "sources": [{"type": "table", "file": "table.csv"}],
                    "answer": {"type": "number"},
                },
                base_dir=tmpdir,
                preprocess_result=_table_preprocess_result(table_path),
            )

            task = build_table_task_items([spec])["answer_1"]

        self.assertIsInstance(task, RuntimeTaskItem)
        self.assertIsInstance(task, TableTaskItem)
        self.assertEqual(task.task_type, "table_reasoning.query")
        self.assertEqual(task.question, "How many rows?")
        self.assertEqual(task.answer_key, "answer_1")
        self.assertEqual(task.local_dsl["answer"]["name"], "answer_1")
        self.assertEqual(task.remote_dsl["answer"]["name"], "answer_1")
        self.assertEqual(task.group_key, task.source_file)
        self.assertIsNone(task.current_command)

    def test_document_reasoning_uses_runtime_task_item(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            document_path = Path(tmpdir) / "document.pdf"
            document_path.write_bytes(b"%PDF-1.4\n")
            spec = DocumentReasoningCaseSpec(
                case_id="document_case",
                task_dsl={
                    "task_type": "document_reasoning",
                    "question": "What was revenue?",
                    "sources": [{"type": "pdf", "file": "document.pdf"}],
                    "answer": {"type": "string"},
                },
                base_dir=tmpdir,
                preprocess_result=_document_preprocess_result(document_path),
            )

            task = build_document_task_items([spec])["answer_1"]

        self.assertIsInstance(task, RuntimeTaskItem)
        self.assertIsInstance(task, DocumentTaskItem)
        self.assertEqual(task.task_type, "document_reasoning")
        self.assertEqual(task.question, "What was revenue?")
        self.assertEqual(task.answer_key, "answer_1")
        self.assertEqual(task.source_id, "document_1")
        self.assertEqual(task.local_dsl["answer"]["name"], "answer_1")
        self.assertEqual(task.remote_dsl["answer"]["name"], "answer_1")
        self.assertEqual(task.group_key, task.source_file)
        self.assertIsNone(task.current_command)

    def test_task_specific_artifacts_share_runtime_artifact_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n1\n", encoding="utf-8")
            document_path = Path(tmpdir) / "document.pdf"
            document_path.write_bytes(b"%PDF-1.4\n")
            table_task = build_table_task_items(
                [
                    TableReasoningCaseSpec(
                        case_id="table_case",
                        task_dsl={
                            "task_type": "table_reasoning.query",
                            "question": "How many rows?",
                            "sources": [{"type": "table", "file": "table.csv"}],
                            "answer": {"type": "number"},
                        },
                        base_dir=tmpdir,
                        preprocess_result=_table_preprocess_result(table_path),
                    )
                ]
            )["answer_1"]
            document_task = build_document_task_items(
                [
                    DocumentReasoningCaseSpec(
                        case_id="document_case",
                        task_dsl={
                            "task_type": "document_reasoning",
                            "question": "What was revenue?",
                            "sources": [{"type": "pdf", "file": "document.pdf"}],
                            "answer": {"type": "string"},
                        },
                        base_dir=tmpdir,
                        preprocess_result=_document_preprocess_result(document_path),
                    )
                ]
            )["answer_1"]

        sql_item = SqlItem(
            task=table_task,
            content="SELECT COUNT(*) AS answer_1 FROM table_1",
            content_type="sql",
        )
        code_item = PythonCodeItem(
            task=document_task,
            content="def prepare_jobs(task, context):\n    return []",
            content_type="python",
        )
        table_dag_item = LogicDagItem(
            task=table_task,
            command_output=sql_item.sql,
            output_type="sql",
            logic_dag={"task_type": "table_reasoning.query"},
        )
        document_dag_item = DocumentLogicDagItem(
            task=document_task,
            command_output=code_item.code,
            output_type="python",
            logic_dag={"task_type": "document_reasoning"},
        )

        self.assertIsInstance(sql_item, RuntimeCommandItem)
        self.assertIsInstance(code_item, RuntimeCommandItem)
        self.assertIsInstance(table_dag_item, RuntimeWorkItem)
        self.assertIsInstance(document_dag_item, RuntimeWorkItem)
        self.assertEqual(sql_item.sql, sql_item.content)
        self.assertEqual(code_item.code, code_item.content)
        self.assertEqual(table_dag_item.sql, table_dag_item.command_output)
        self.assertEqual(document_dag_item.code, document_dag_item.command_output)


def _table_preprocess_result(table_path: Path) -> dict:
    return {
        "local_dsl": {
            "task_type": "table_reasoning.query",
            "question": "How many rows?",
            "sources": [
                {
                    "id": "table_1",
                    "type": "table",
                    "path": str(table_path),
                    "format": "csv",
                    "schema": {"columns": ["value"]},
                }
            ],
            "answer": {"type": "number"},
        },
        "remote_dsl": {
            "task_type": "table_reasoning.query",
            "question": "How many rows?",
            "sources": [
                {
                    "id": "table_1",
                    "type": "table",
                    "format": "csv",
                    "schema": {"columns": ["value"]},
                }
            ],
            "answer": {"type": "number"},
        },
        "context": {
            "task_type": "table_reasoning.query",
            "source_map": {},
        },
    }


def _document_preprocess_result(document_path: Path) -> dict:
    return {
        "local_dsl": {
            "task_type": "document_reasoning",
            "question": "What was revenue?",
            "sources": [
                {
                    "id": "document_1",
                    "type": "document",
                    "source_type": "pdf",
                    "path": str(document_path),
                    "format": "pdf",
                    "schema": {
                        "format": "pdf",
                        "chunking": {"chunk_count": 1},
                    },
                }
            ],
            "answer": {"type": "string"},
        },
        "remote_dsl": {
            "task_type": "document_reasoning",
            "question": "What was revenue?",
            "sources": [
                {
                    "id": "document_1",
                    "type": "document",
                    "source_type": "pdf",
                    "format": "pdf",
                    "schema": {
                        "format": "pdf",
                        "chunking": {"chunk_count": 1},
                    },
                }
            ],
            "answer": {"type": "string"},
        },
        "context": {
            "task_type": "document_reasoning",
            "source_map": {},
        },
    }


if __name__ == "__main__":
    unittest.main()
