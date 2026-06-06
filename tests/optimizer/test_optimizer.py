from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from typing import Any

from clover.optimizer import (
    OptimizationError,
    Optimizer,
    infer_output_type,
    optimize_logic_dag_to_physical_plan,
)


class OptimizerTest(unittest.TestCase):
    def test_builds_physical_plan_with_resources_and_node_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n1\n2\n", encoding="utf-8")
            local_dsl = _local_dsl(table_path)
            context = _context(Path(tmpdir), table_path)
            logic_dag = _logic_dag()
            original_logic_dag = copy.deepcopy(logic_dag)

            physical_plan = optimize_logic_dag_to_physical_plan(
                logic_dag=logic_dag,
                context=context,
                local_dsl=local_dsl,
            )

        self.assertEqual(logic_dag, original_logic_dag)
        self.assertEqual(
            set(physical_plan),
            {
                "task_type",
                "answers",
                "resources",
                "resource_processing",
                "nodes",
                "edges",
                "query_outputs",
                "merge_stats",
            },
        )
        self.assertEqual(physical_plan["task_type"], "table_reasoning.query")
        self.assertEqual(physical_plan["resource_processing"], [])
        self.assertEqual(
            physical_plan["resources"],
            [
                {
                    "id": "table_1",
                    "type": "table",
                    "path": str(table_path.resolve()),
                    "format": "csv",
                    "schema": {
                        "format": "csv",
                        "shape": {"rows": 2, "columns": 1},
                        "columns": ["value"],
                    },
                }
            ],
        )
        self.assertEqual(
            [node["output_type"] for node in physical_plan["nodes"]],
            ["table", "table", "number"],
        )
        self.assertEqual(
            [node["instruction"] for node in physical_plan["nodes"]],
            ["", "", ""],
        )
        self.assertEqual(physical_plan["nodes"][0]["input"], ["table_1"])
        self.assertEqual(physical_plan["nodes"][1]["input"], [])
        self.assertEqual(physical_plan["nodes"][1]["dependency"], ["T0"])

    def test_infers_output_type_from_op(self) -> None:
        self.assertEqual(infer_output_type({"op": "Filter"}, {}), "table")
        self.assertEqual(infer_output_type({"op": "Aggregate"}, {}), "table")
        self.assertEqual(infer_output_type({"op": "RepeatUnion"}, {}), "table")
        self.assertEqual(
            infer_output_type(
                {"op": "FormatAnswer", "params": {"answer": {"type": "boolean"}}},
                {},
            ),
            "boolean",
        )
        self.assertEqual(infer_output_type({"op": "SLM"}, {}), "json")

    def test_rejects_unknown_resource_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n1\n", encoding="utf-8")
            logic_dag = _logic_dag()
            logic_dag["query_plans"][0]["nodes"][0]["input"] = ["missing_table"]

            with self.assertRaisesRegex(OptimizationError, "unknown resources"):
                optimize_logic_dag_to_physical_plan(
                    logic_dag=logic_dag,
                    context=_context(Path(tmpdir), table_path),
                    local_dsl=_local_dsl(table_path),
                )

    def test_optimizer_accepts_additional_strategies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n1\n", encoding="utf-8")
            optimizer = Optimizer.default()
            optimizer.strategies.append(_PlanMarkerStrategy())

            physical_plan = optimize_logic_dag_to_physical_plan(
                logic_dag=_logic_dag(),
                context=_context(Path(tmpdir), table_path),
                local_dsl=_local_dsl(table_path),
                optimizer=optimizer,
            )

        self.assertEqual(physical_plan["optimizer_marker"], "custom")

    def test_merges_equivalent_batch_physical_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "finalWorth,selfMade,country\n10,true,US\n20,false,CN\n",
                encoding="utf-8",
            )

            physical_plan = optimize_logic_dag_to_physical_plan(
                logic_dag=_batch_shared_prefix_logic_dag(),
                context=_batch_context(Path(tmpdir), {"table_1": table_path}),
                local_dsl=_batch_local_dsl({"table_1": table_path}),
            )

        self.assertEqual(physical_plan["task_type"], "table_reasoning.query")
        self.assertEqual(
            [node["op"] for node in physical_plan["nodes"]],
            [
                "Scan",
                "Sort",
                "Limit",
                "Project",
                "FormatAnswer",
                "Project",
                "FormatAnswer",
            ],
        )
        self.assertEqual(physical_plan["merge_stats"]["reused_nodes"], 3)
        self.assertEqual(
            [node["output"] for node in physical_plan["nodes"]],
            ["T0", "T1", "T2", "T3", "answer_1", "T4", "answer_2"],
        )
        self.assertEqual(physical_plan["nodes"][5]["dependency"], ["T2"])
        self.assertEqual(
            [item["output"] for item in physical_plan["query_outputs"]],
            ["answer_1", "answer_2"],
        )
        self.assertEqual(
            physical_plan["edges"],
            [
                {"from": "N0", "to": "N1"},
                {"from": "N1", "to": "N2"},
                {"from": "N2", "to": "N3"},
                {"from": "N3", "to": "N4"},
                {"from": "N2", "to": "N5"},
                {"from": "N5", "to": "N6"},
            ],
        )

    def test_keeps_non_equivalent_batch_branches_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            left_path = Path(tmpdir) / "left.csv"
            right_path = Path(tmpdir) / "right.csv"
            left_path.write_text("value\n1\n", encoding="utf-8")
            right_path.write_text("value\n2\n", encoding="utf-8")

            physical_plan = optimize_logic_dag_to_physical_plan(
                logic_dag=_batch_independent_logic_dag(),
                context=_batch_context(
                    Path(tmpdir),
                    {"table_1": left_path, "table_2": right_path},
                ),
                local_dsl=_batch_local_dsl(
                    {"table_1": left_path, "table_2": right_path},
                    columns=["value"],
                ),
            )

        self.assertEqual(physical_plan["merge_stats"]["reused_nodes"], 0)
        self.assertEqual(
            [node["op"] for node in physical_plan["nodes"]],
            ["Scan", "FormatAnswer", "Scan", "FormatAnswer"],
        )
        self.assertEqual(
            [node["output"] for node in physical_plan["nodes"]],
            ["T0", "answer_1", "T1", "answer_2"],
        )
        self.assertEqual(
            physical_plan["edges"],
            [{"from": "N0", "to": "N1"}, {"from": "N2", "to": "N3"}],
        )

    def test_builds_document_plan_with_resource_processing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            document_path = Path(tmpdir) / "report.pdf"
            document_path.write_bytes(b"%PDF-1.4\n")
            physical_plan = optimize_logic_dag_to_physical_plan(
                logic_dag=_document_logic_dag(),
                context=_document_context(Path(tmpdir), document_path),
                local_dsl=_document_local_dsl(document_path),
            )

        self.assertEqual(physical_plan["task_type"], "document_reasoning")
        self.assertEqual(
            [step["op"] for step in physical_plan["resource_processing"]],
            ["extract_text", "chunk_text"],
        )
        self.assertEqual(
            physical_plan["resource_processing"][1]["params"],
            {
                "strategy": "sliding_window",
                "unit": "char",
                "size": 3000,
                "overlap": 20,
                "preserve_page_spans": True,
            },
        )
        self.assertEqual(physical_plan["resources"][0]["source_type"], "pdf")
        self.assertEqual(
            physical_plan["map_groups"],
            [
                {
                    "id": "G0",
                    "op": "map",
                    "input": {
                        "resource_view": "CV0",
                        "chunks": "all",
                    },
                    "params": {
                        "local_instruction": "Extract revenue evidence.",
                        "local_guidance": "Return answer, explanation, and citation.",
                        "output_contract": {
                            "format": "json",
                            "fields": ["answer", "explanation", "citation"],
                        },
                    },
                    "output": "G0",
                    "output_type": "jsonl",
                }
            ],
        )


class _PlanMarkerStrategy:
    def apply(
        self,
        physical_plan: dict[str, Any],
        logic_dag: dict[str, Any],
        context: dict[str, Any],
        local_dsl: dict[str, Any],
    ) -> None:
        physical_plan["optimizer_marker"] = "custom"


def _logic_dag() -> dict[str, Any]:
    return {
        "task_type": "table_reasoning.query",
        "query_plans": [
            {
                "id": "Q0",
                "index": 0,
                "answer": {"name": "answer", "type": "number"},
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
        ],
    }


def _local_dsl(table_path: Path) -> dict[str, Any]:
    return {
        "task_type": "table_reasoning.query",
        "question": "How many rows are present?",
        "sources": [
            {
                "id": "table_1",
                "type": "table",
                "path": str(table_path),
                "format": "csv",
                "schema": {
                    "format": "csv",
                    "shape": {"rows": 2, "columns": 1},
                    "columns": ["value"],
                },
            }
        ],
        "answer": {"name": "answer", "type": "number"},
    }


def _context(base_dir: Path, table_path: Path) -> dict[str, Any]:
    return {
        "task_type": "table_reasoning.query",
        "base_dir": str(base_dir),
        "source_map": {
            "table_1": {
                "type": "table",
                "path": str(table_path),
                "format": "csv",
            }
        },
    }


def _batch_shared_prefix_logic_dag() -> dict[str, Any]:
    return {
        "task_type": "table_reasoning.query",
        "query_plans": [
            {
                "id": "Q0",
                "index": 0,
                "answer": {"name": "answer_1", "type": "boolean"},
                **_query_fragment(
                    _highest_worth_dag(
                        project_column="selfMade",
                        answer={"name": "answer_1", "type": "boolean"},
                    )
                ),
            },
            {
                "id": "Q1",
                "index": 1,
                "answer": {"name": "answer_2", "type": "string"},
                **_query_fragment(
                    _highest_worth_dag(
                        project_column="country",
                        answer={"name": "answer_2", "type": "string"},
                    )
                ),
            },
        ],
    }


def _highest_worth_dag(project_column: str, answer: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_type": "table_reasoning.query",
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
                "op": "Sort",
                "dependency": ["T0"],
                "input": [],
                "params": {
                    "keys": [
                        {
                            "expr": {"type": "column", "name": "finalWorth"},
                            "direction": "DESC",
                            "nulls": "LAST",
                        }
                    ]
                },
                "output": "T1",
            },
            {
                "id": "N2",
                "op": "Limit",
                "dependency": ["T1"],
                "input": [],
                "params": {"count": 1},
                "output": "T2",
            },
            {
                "id": "N3",
                "op": "Project",
                "dependency": ["T2"],
                "input": [],
                "params": {
                    "expressions": [
                        {"expr": {"type": "column", "name": project_column}}
                    ]
                },
                "output": "T3",
            },
            {
                "id": "N4",
                "op": "FormatAnswer",
                "dependency": ["T3"],
                "input": [],
                "params": {"answer": answer},
                "output": "answer",
            },
        ],
        "edges": [
            {"from": "N0", "to": "N1"},
            {"from": "N1", "to": "N2"},
            {"from": "N2", "to": "N3"},
            {"from": "N3", "to": "N4"},
        ],
    }


def _batch_independent_logic_dag() -> dict[str, Any]:
    return {
        "task_type": "table_reasoning.query",
        "query_plans": [
            {
                "id": "Q0",
                "index": 0,
                "answer": {"name": "answer_1", "type": "number"},
                **_query_fragment(
                    _scan_answer_dag(
                        source_id="table_1",
                        answer={"name": "answer_1", "type": "number"},
                    )
                ),
            },
            {
                "id": "Q1",
                "index": 1,
                "answer": {"name": "answer_2", "type": "number"},
                **_query_fragment(
                    _scan_answer_dag(
                        source_id="table_2",
                        answer={"name": "answer_2", "type": "number"},
                    )
                ),
            },
        ],
    }


def _scan_answer_dag(source_id: str, answer: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_type": "table_reasoning.query",
        "nodes": [
            {
                "id": "N0",
                "op": "Scan",
                "dependency": [],
                "input": [source_id],
                "params": {"source": source_id},
                "output": "T0",
            },
            {
                "id": "N1",
                "op": "FormatAnswer",
                "dependency": ["T0"],
                "input": [],
                "params": {"answer": answer},
                "output": "answer",
            },
        ],
        "edges": [{"from": "N0", "to": "N1"}],
    }


def _query_fragment(logic_dag: dict[str, Any]) -> dict[str, Any]:
    return {"nodes": logic_dag["nodes"], "edges": logic_dag["edges"]}


def _batch_local_dsl(
    sources: dict[str, Path],
    *,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    schema_columns = columns or ["finalWorth", "selfMade", "country"]
    return {
        "task_type": "table_reasoning.query",
        "questions": ["question 1", "question 2"],
        "sources": [
            {
                "id": source_id,
                "type": "table",
                "path": str(path),
                "format": "csv",
                "schema": {
                    "format": "csv",
                    "shape": {"rows": 2, "columns": len(schema_columns)},
                    "columns": schema_columns,
                },
            }
            for source_id, path in sources.items()
        ],
        "answers": [
            {"name": "answer_1", "type": "json"},
            {"name": "answer_2", "type": "json"},
        ],
    }


def _batch_context(base_dir: Path, sources: dict[str, Path]) -> dict[str, Any]:
    return {
        "task_type": "table_reasoning.query",
        "base_dir": str(base_dir),
        "source_map": {
            source_id: {
                "type": "table",
                "path": str(path),
                "format": "csv",
            }
            for source_id, path in sources.items()
        },
    }


def _document_logic_dag() -> dict[str, Any]:
    return {
        "task_type": "document_reasoning",
        "resource_processing": [
            {
                "id": "RP0",
                "op": "chunk_by_section",
                "source": "document_1",
                "output": "CV0",
                "params": {"max_chunk_size": 3000, "overlap": 20},
            }
        ],
        "map_groups": [
            {
                "id": "G0",
                "op": "map",
                "inputs": {"resource_view": "CV0", "chunks": "all"},
                "params": {
                    "local_instruction": "Extract revenue evidence.",
                    "local_guidance": "Return answer, explanation, and citation.",
                    "output_contract": {
                        "format": "json",
                        "fields": ["answer", "explanation", "citation"],
                    },
                },
            }
        ],
        "edges": [],
    }


def _document_local_dsl(document_path: Path) -> dict[str, Any]:
    return {
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
                    "page_count": 3,
                    "page_indexing": "zero_based",
                    "text_extraction": {"extractor": "pymupdf"},
                    "chunking": {
                        "strategy": "sliding_window",
                        "unit": "char",
                        "size": 3000,
                        "overlap": 20,
                        "preserve_page_spans": True,
                        "chunk_count": 2,
                    },
                },
            }
        ],
        "answer": {"name": "answer_1", "type": "string"},
    }


def _document_context(base_dir: Path, document_path: Path) -> dict[str, Any]:
    return {
        "task_type": "document_reasoning",
        "base_dir": str(base_dir),
        "source_map": {
            "document_1": {
                "type": "document",
                "source_type": "pdf",
                "path": str(document_path),
                "format": "pdf",
            }
        },
    }


if __name__ == "__main__":
    unittest.main()
