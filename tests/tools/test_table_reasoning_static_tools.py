from __future__ import annotations

import unittest

from clover.tools import (
    StaticToolError,
    build_static_tool_call,
    build_static_tool_calls,
    get_static_tool,
    static_tool_registry_for_task,
)


class TableReasoningStaticToolsTest(unittest.TestCase):
    def test_registry_is_scoped_by_task_type(self) -> None:
        registry = static_tool_registry_for_task("table_reasoning.query")

        self.assertIn("Scan", registry)
        self.assertEqual(
            get_static_tool("table_reasoning.query", "Scan").tool_name,
            "table_reasoning.scan",
        )
        with self.assertRaisesRegex(StaticToolError, "Unsupported task_type"):
            static_tool_registry_for_task("document_reasoning")

    def test_all_table_reasoning_ops_have_static_tools(self) -> None:
        registry = static_tool_registry_for_task("table_reasoning.query")

        self.assertEqual(
            set(registry),
            {
                "Scan",
                "Filter",
                "Project",
                "Derive",
                "Aggregate",
                "Group",
                "Sort",
                "Limit",
                "Distinct",
                "Join",
                "SetOp",
                "RepeatUnion",
                "FormatAnswer",
            },
        )

    def test_builds_tool_call_with_external_params(self) -> None:
        node = {
            "id": "N0",
            "op": "Scan",
            "dependency": [],
            "input": ["table_1"],
            "params": {"source": "table_1"},
            "output": "T0",
        }
        resources = {"table_1": {"id": "table_1", "path": "/tmp/table.csv"}}
        external_params = {"batch_size": 1024, "runtime": {"engine": "duckdb"}}

        call = build_static_tool_call(
            "table_reasoning.query",
            node,
            resources=resources,
            external_params=external_params,
        )

        self.assertEqual(call["task_type"], "table_reasoning.query")
        self.assertEqual(call["tool"], "table_reasoning.scan")
        self.assertEqual(call["resources"], resources)
        self.assertEqual(call["external_params"], external_params)
        self.assertEqual(call["output"], "T0")

    def test_builds_calls_for_logic_dag(self) -> None:
        logic_dag = {
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
                    "op": "Filter",
                    "dependency": ["T0"],
                    "input": [],
                    "params": {
                        "predicate": {
                            "type": "binary_op",
                            "op": ">",
                            "left": {"type": "column", "name": "value"},
                            "right": {"type": "literal", "value": 0},
                        }
                    },
                    "output": "T1",
                },
            ],
        }

        calls = build_static_tool_calls(
            logic_dag,
            resources={"table_1": {"id": "table_1", "path": "/tmp/table.csv"}},
            upstream_outputs={"T0": {"handle": "table://T0"}},
            external_params={"engine": "duckdb"},
        )

        self.assertEqual([call["tool"] for call in calls], ["table_reasoning.scan", "table_reasoning.filter"])
        self.assertEqual(calls[1]["upstream_outputs"], {"T0": {"handle": "table://T0"}})

    def test_repeat_union_tool_accepts_recursive_params(self) -> None:
        node = {
            "id": "N0",
            "op": "RepeatUnion",
            "dependency": [],
            "input": ["table_1"],
            "params": {
                "name": "descendants",
                "transient_table": "descendants",
                "termination": "until_empty_delta",
                "seed_plan": {"nodes": [], "edges": [], "output": "S0"},
                "recursive_plan": {"nodes": [], "edges": [], "output": "R0"},
            },
            "output": "T0",
        }

        call = build_static_tool_call(
            "table_reasoning.query",
            node,
            resources={"table_1": {"id": "table_1", "path": "/tmp/table.csv"}},
            external_params={"max_iterations": 100},
        )

        self.assertEqual(call["tool"], "table_reasoning.repeat_union")
        self.assertEqual(call["params"]["transient_table"], "descendants")
        self.assertEqual(call["external_params"], {"max_iterations": 100})

    def test_rejects_missing_required_params(self) -> None:
        node = {
            "id": "N1",
            "op": "Filter",
            "dependency": ["T0"],
            "input": [],
            "params": {},
            "output": "T1",
        }

        with self.assertRaisesRegex(StaticToolError, "missing params"):
            build_static_tool_call("table_reasoning.query", node)


if __name__ == "__main__":
    unittest.main()
