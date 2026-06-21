from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from clover.executor import ExecutionPlanBuilder, execute_execution_plan


class MixedExecutionPlanTest(unittest.TestCase):
    def test_executor_routes_mixed_units_by_execution_unit_task_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table_path = root / "table.csv"
            table_path.write_text("value\n1\n2\n3\n", encoding="utf-8")
            chunk_store = root / "chunks.jsonl"
            _write_chunks(
                chunk_store,
                [
                    {
                        "chunk_id": "chunk_0",
                        "text": "Revenue was $140 million in fiscal 2023.",
                        "page_start": 4,
                        "page_end": 4,
                    }
                ],
            )

            execution_plan = ExecutionPlanBuilder.default().build_many(
                [
                    _table_count_plan(table_path),
                    _document_map_plan(chunk_store),
                ],
                namespaces=("table_case", "document_case"),
            )
            result = execute_execution_plan(
                execution_plan,
                slm_config={
                    "api_type": "chat_completions",
                    "model": "fake-slm",
                    "temperature": 0,
                },
                slm_client=_FakeChatClient(
                    json.dumps(
                        {
                            "answer": "$140 million",
                            "citation": "Revenue was $140 million in fiscal 2023.",
                            "explanation": "The chunk states the value.",
                        }
                    )
                ),
                max_parallel_execution_units=2,
            )

        self.assertTrue(result.ok)
        self.assertEqual(
            [trace["node_id"] for trace in result.traces],
            [
                "table_case__N0",
                "table_case__N1",
                "table_case__N2",
                "document_case__G0__0__document_1_chunk_0",
            ],
        )
        self.assertEqual(
            [trace["task_type"] for trace in result.traces],
            [
                "table_reasoning.query",
                "table_reasoning.query",
                "table_reasoning.query",
                "document_reasoning",
            ],
        )
        self.assertEqual(result.collector_outputs["table_case__answer"], 3)
        self.assertEqual(result.answer["table_case__answer"], 3)
        document_evidence = result.collector_outputs["document_case__G0"]
        self.assertEqual(document_evidence["included_count"], 1)
        self.assertIn("$140 million", document_evidence["evidence_summary"])
        self.assertEqual(
            sorted(resource["id"] for resource in execution_plan.resources),
            ["document_case__document_1:chunk_0", "table_case__table_1"],
        )

    def test_namespaces_join_resource_sources_in_mixed_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            party_path = root / "party.csv"
            host_path = root / "host.csv"
            party_host_path = root / "party_host.csv"
            other_path = root / "other.csv"
            party_path.write_text(
                "Party_ID,Location\n"
                "1,Heineken Music Hall Amsterdam\n"
                "2,Rotterdam\n",
                encoding="utf-8",
            )
            host_path.write_text(
                "Host_ID,Name\n"
                "10,Lloyd Daniels\n"
                "11,Kornel David\n"
                "12,Other Host\n",
                encoding="utf-8",
            )
            party_host_path.write_text(
                "Party_ID,Host_ID,Is_Main_in_Charge\n"
                "1,10,T\n"
                "1,11,T\n"
                "2,12,T\n",
                encoding="utf-8",
            )
            other_path.write_text("value\n1\n2\n", encoding="utf-8")

            execution_plan = ExecutionPlanBuilder.default().build_many(
                [
                    _table_bridge_join_plan(
                        party_path=party_path,
                        host_path=host_path,
                        party_host_path=party_host_path,
                    ),
                    _table_count_plan(other_path),
                ],
                namespaces=("mmqa_case", "other_case"),
            )
            join_unit = next(unit for unit in execution_plan.units if unit.op == "Join")

            result = execute_execution_plan(
                execution_plan,
                max_parallel_execution_units=2,
            )

        self.assertEqual(join_unit.node["input"], ["mmqa_case__party_host"])
        self.assertEqual(
            join_unit.node["params"]["joins"][0]["source"],
            "mmqa_case__party_host",
        )
        self.assertTrue(result.ok, result.error)
        self.assertEqual(
            result.collector_outputs["mmqa_case__answer"],
            ["Lloyd Daniels", "Kornel David"],
        )
        self.assertEqual(result.collector_outputs["other_case__answer"], 2)


def _table_count_plan(table_path: Path) -> dict:
    return {
        "task_type": "table_reasoning.query",
        "resources": [
            {
                "id": "table_1",
                "type": "table",
                "path": str(table_path),
                "format": "csv",
                "schema": {"columns": ["value"]},
            }
        ],
        "resource_processing": [],
        "nodes": [
            {
                "id": "N0",
                "op": "Scan",
                "dependency": [],
                "input": ["table_1"],
                "params": {"source": "table_1"},
                "output": "T0",
                "output_type": "table",
                "instruction": "",
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
                "output_type": "table",
                "instruction": "",
            },
            {
                "id": "N2",
                "op": "FormatAnswer",
                "dependency": ["T1"],
                "input": [],
                "params": {"answer": {"name": "answer", "type": "number"}},
                "output": "answer",
                "output_type": "value",
                "instruction": "",
            },
        ],
        "edges": [],
        "answer": {"name": "answer", "type": "number"},
    }


def _table_bridge_join_plan(
    *,
    party_path: Path,
    host_path: Path,
    party_host_path: Path,
) -> dict:
    return {
        "task_type": "table_reasoning.query",
        "resources": [
            {
                "id": "party",
                "type": "table",
                "path": str(party_path),
                "format": "csv",
                "schema": {"columns": ["Party_ID", "Location"]},
            },
            {
                "id": "host",
                "type": "table",
                "path": str(host_path),
                "format": "csv",
                "schema": {"columns": ["Host_ID", "Name"]},
            },
            {
                "id": "party_host",
                "type": "table",
                "path": str(party_host_path),
                "format": "csv",
                "schema": {"columns": ["Party_ID", "Host_ID", "Is_Main_in_Charge"]},
            },
        ],
        "resource_processing": [],
        "nodes": [
            {
                "id": "N0",
                "op": "Scan",
                "dependency": [],
                "input": ["party"],
                "params": {"source": "party"},
                "output": "T0",
                "output_type": "table",
                "instruction": "",
            },
            {
                "id": "N1",
                "op": "Join",
                "dependency": ["T0"],
                "input": ["party_host"],
                "params": {
                    "joins": [
                        {
                            "kind": "JOIN",
                            "source": "party_host",
                            "on": {
                                "type": "binary_op",
                                "op": "=",
                                "left": {
                                    "type": "column",
                                    "table": "party",
                                    "name": "Party_ID",
                                },
                                "right": {
                                    "type": "column",
                                    "table": "party_host",
                                    "name": "Party_ID",
                                },
                            },
                        }
                    ]
                },
                "output": "T1",
                "output_type": "table",
                "instruction": "",
            },
            {
                "id": "N2",
                "op": "Join",
                "dependency": ["T1"],
                "input": ["host"],
                "params": {
                    "joins": [
                        {
                            "kind": "JOIN",
                            "source": "host",
                            "on": {
                                "type": "binary_op",
                                "op": "=",
                                "left": {
                                    "type": "column",
                                    "table": "party_host",
                                    "name": "Host_ID",
                                },
                                "right": {
                                    "type": "column",
                                    "table": "host",
                                    "name": "Host_ID",
                                },
                            },
                        }
                    ]
                },
                "output": "T2",
                "output_type": "table",
                "instruction": "",
            },
            {
                "id": "N3",
                "op": "Filter",
                "dependency": ["T2"],
                "input": [],
                "params": {
                    "predicate": {
                        "type": "logical_op",
                        "op": "AND",
                        "operands": [
                            {
                                "type": "like",
                                "expr": {
                                    "type": "column",
                                    "table": "party",
                                    "name": "Location",
                                },
                                "pattern": {
                                    "type": "literal",
                                    "value": "%Heineken Music Hall Amsterdam%",
                                    "value_type": "string",
                                },
                            },
                            {
                                "type": "binary_op",
                                "op": "=",
                                "left": {
                                    "type": "column",
                                    "table": "party_host",
                                    "name": "Is_Main_in_Charge",
                                },
                                "right": {
                                    "type": "literal",
                                    "value": "T",
                                    "value_type": "string",
                                },
                            },
                        ],
                    },
                },
                "output": "T3",
                "output_type": "table",
                "instruction": "",
            },
            {
                "id": "N4",
                "op": "Project",
                "dependency": ["T3"],
                "input": [],
                "params": {
                    "expressions": [
                        {
                            "expr": {
                                "type": "column",
                                "table": "host",
                                "name": "Name",
                            },
                            "alias": "answer",
                        }
                    ]
                },
                "output": "T4",
                "output_type": "table",
                "instruction": "",
            },
            {
                "id": "N5",
                "op": "FormatAnswer",
                "dependency": ["T4"],
                "input": [],
                "params": {"answer": {"name": "answer", "type": "list[string]"}},
                "output": "answer",
                "output_type": "value",
                "instruction": "",
            },
        ],
        "edges": [],
        "answer": {"name": "answer", "type": "list[string]"},
    }


def _document_map_plan(chunk_store: Path) -> dict:
    return {
        "task_type": "document_reasoning",
        "question": "What was revenue in fiscal 2023?",
        "resources": [
            {
                "id": "document_1:chunk_0",
                "type": "document_chunk",
                "source": "document_1",
                "source_type": "pdf",
                "path": str(chunk_store),
                "format": "text",
                "item_id": "chunk_0",
                "chunk_id": "chunk_0",
            }
        ],
        "resource_processing": [],
        "map_groups": [
            {
                "id": "G0",
                "op": "map",
                "input": {"chunks": ["document_1:chunk_0"]},
                "params": {
                    "local_instruction": "Extract fiscal 2023 revenue.",
                    "local_guidance": "Return explicitly stated values only.",
                },
                "output": "G0",
                "output_type": "jsonl",
            }
        ],
        "edges": [],
    }


def _write_chunks(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


class _FakeChatClient:
    def __init__(self, output_text: str) -> None:
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(output_text))


class _FakeChatCompletions:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text

    def create(self, **kwargs: object) -> "_FakeChatResponse":
        return _FakeChatResponse(self.output_text)


class _FakeChatResponse:
    id = "mixed_execution_fake"

    def __init__(self, output_text: str) -> None:
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(content=output_text)),
        ]

    def model_dump(self, mode: str) -> dict[str, object]:
        return {"id": self.id, "mode": mode}


if __name__ == "__main__":
    unittest.main()
