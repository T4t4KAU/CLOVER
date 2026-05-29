from __future__ import annotations

from datetime import date
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from clover.executor import execute_physical_plan
from clover.executor.agents.base import FastPathDecision
from clover.executor.agents.table_reasoning import TableReasoningNodeAgent
from clover.executor.result import json_ready


class ExecutorTest(unittest.TestCase):
    def test_executes_table_reasoning_plan_with_fast_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n1\n2\n3\n", encoding="utf-8")

            result = execute_physical_plan(_count_plan(table_path))

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, 3)
        self.assertEqual(result.fast_path_hits, 3)
        self.assertEqual(result.fast_path_misses, 0)
        self.assertEqual([trace["execution_path"] for trace in result.traces], ["fast_path"] * 3)
        self.assertNotIn("T0", result.outputs)
        self.assertEqual(result.output_summaries["T0"]["rows"], 3)

    def test_schedules_independent_ready_nodes_before_join_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            left_path = Path(tmpdir) / "left.csv"
            right_path = Path(tmpdir) / "right.csv"
            left_path.write_text("value\n1\n2\n", encoding="utf-8")
            right_path.write_text("value\n3\n4\n", encoding="utf-8")

            result = execute_physical_plan(_union_count_plan(left_path, right_path))

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, 4)
        self.assertEqual(result.fast_path_hits, 5)
        self.assertEqual([trace["node_id"] for trace in result.traces], ["N0", "N1", "N2", "N3", "N4"])

    def test_executes_v2_merged_plan_with_shared_nodes_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "selfMade,country,finalWorth\n"
                "False,France,100\n"
                "True,United States,200\n",
                encoding="utf-8",
            )

            result = execute_physical_plan(_v2_shared_prefix_plan(table_path))

        self.assertTrue(result.ok)
        self.assertEqual(
            result.answer,
            {"answer_1": True, "answer_2": "United States"},
        )
        self.assertEqual(result.fast_path_hits, 7)
        self.assertEqual(
            [trace["node_id"] for trace in result.traces],
            ["N0", "N1", "N2", "N3", "N4", "N5", "N6"],
        )
        self.assertEqual(result.output_summaries["T2"]["rows"], 1)

    def test_fast_path_miss_runs_agent_loop(self) -> None:
        plan = {
            "task_type": "table_reasoning_v1",
            "resources": [],
            "nodes": [
                {
                    "id": "N0",
                    "op": "CustomReason",
                    "dependency": [],
                    "input": [],
                    "params": {},
                    "output": "answer",
                }
            ],
            "edges": [],
        }

        result = execute_physical_plan(
            plan,
            slm_config={
                "api_type": "chat_completions",
                "model": "fake-slm",
                "temperature": 0,
            },
            slm_client=_FakeChatClient(
                [
                    '{"action":"run_python","code":"result = {\'status\': \'complete\', \'output\': \'ok\'}"}',
                    '{"action":"submit_result","name":"result"}',
                ]
            ),
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.fast_path_hits, 0)
        self.assertEqual(result.fast_path_misses, 1)
        self.assertEqual(result.traces[0]["fast_path_miss_reason"], "unsupported_op")
        self.assertEqual(result.traces[0]["execution_path"], "agent_loop")
        self.assertEqual(result.answer, {"status": "complete", "output": "ok"})
        self.assertEqual(result.traces[0]["agent_loop"]["iterations"], 1)

    def test_fast_path_execution_error_recovers_with_agent_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n7\n8\n", encoding="utf-8")

            result = execute_physical_plan(
                _first_value_plan(table_path),
                slm_config={
                    "api_type": "chat_completions",
                    "model": "fake-slm",
                    "temperature": 0,
                },
                slm_client=_FakeChatClient(
                    [
                        (
                            '{"action":"run_python","code":"'
                            "df = inputs['dep_0'].copy()\\n"
                            "result = pd.DataFrame({'answer': [df['value'].iloc[0]]})"
                            '"}'
                        ),
                        '{"action":"submit_result","name":"result"}',
                    ]
                ),
                agent_loop_max_iterations=3,
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, 7)
        aggregate_trace = result.traces[1]
        self.assertEqual(aggregate_trace["execution_path"], "agent_loop")
        self.assertEqual(aggregate_trace["agent_loop_trigger"], "fast_path_execution_error")
        self.assertEqual(aggregate_trace["agent_loop"]["iterations"], 1)

    def test_agent_loop_accepts_result_created_by_run_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n7\n8\n", encoding="utf-8")

            result = execute_physical_plan(
                _first_value_plan(table_path),
                slm_config={
                    "api_type": "chat_completions",
                    "model": "fake-slm",
                    "temperature": 0,
                },
                slm_client=_FakeChatClient(
                    (
                        '{"action":"run_python","code":"'
                        "df = dep_0\\n"
                        "result = pd.DataFrame({'answer': [df['value'].iloc[0]]})"
                        '"}'
                    )
                ),
                agent_loop_max_iterations=3,
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, 7)
        aggregate_trace = result.traces[1]
        self.assertEqual(aggregate_trace["execution_path"], "agent_loop")
        self.assertEqual(aggregate_trace["agent_loop"]["iterations"], 1)

    def test_agent_loop_can_optionally_call_tool_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n7\n8\n", encoding="utf-8")

            result = execute_physical_plan(
                _first_value_plan(table_path),
                slm_config={
                    "api_type": "chat_completions",
                    "model": "fake-slm",
                    "temperature": 0,
                },
                slm_client=_FakeChatClient(
                    [
                        (
                            '{"action":"run_python","code":"'
                            "result = tool.run({'aggregations': ["
                            "{'function': 'MAX', 'argument': {'type': 'column', 'name': 'value'}, "
                            "'distinct': False, 'alias': 'answer'}], 'grouped': False})"
                            '"}'
                        ),
                        '{"action":"submit_result","name":"result"}',
                    ]
                ),
                agent_loop_max_iterations=3,
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, 8)

    def test_agent_loop_tool_reference_preserves_group_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\nA\nA\nB\n", encoding="utf-8")
            original_try_fast_path = TableReasoningNodeAgent.try_fast_path

            def force_aggregate_agent_loop(
                agent: TableReasoningNodeAgent,
            ) -> FastPathDecision:
                if agent.node.get("id") == "N2":
                    return FastPathDecision(
                        hit=False,
                        tool="table_reasoning.aggregate",
                        backend=agent.backend_name,
                        miss_reason="forced_test",
                    )
                return original_try_fast_path(agent)

            with patch.object(
                TableReasoningNodeAgent,
                "try_fast_path",
                force_aggregate_agent_loop,
            ):
                result = execute_physical_plan(
                    _group_count_top_plan(table_path),
                    slm_config={
                        "api_type": "chat_completions",
                        "model": "fake-slm",
                        "temperature": 0,
                    },
                    slm_client=_FakeChatClient(
                        '{"action":"run_python","code":"result = tool.run()"}'
                    ),
                    agent_loop_max_iterations=2,
                )

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, "A")
        self.assertEqual(result.traces[2]["execution_path"], "agent_loop")
        self.assertIn("group_keys", result.output_summaries["T1"])

    def test_agent_loop_group_dataframe_output_keeps_group_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\nA\nA\nB\n", encoding="utf-8")
            original_try_fast_path = TableReasoningNodeAgent.try_fast_path

            def force_group_agent_loop(
                agent: TableReasoningNodeAgent,
            ) -> FastPathDecision:
                if agent.node.get("id") == "N1":
                    return FastPathDecision(
                        hit=False,
                        tool="table_reasoning.group",
                        backend=agent.backend_name,
                        miss_reason="forced_test",
                    )
                return original_try_fast_path(agent)

            with patch.object(
                TableReasoningNodeAgent,
                "try_fast_path",
                force_group_agent_loop,
            ):
                result = execute_physical_plan(
                    _group_count_top_plan(table_path),
                    slm_config={
                        "api_type": "chat_completions",
                        "model": "fake-slm",
                        "temperature": 0,
                    },
                    slm_client=_FakeChatClient(
                        '{"action":"run_python","code":"result = dep_0.copy()"}'
                    ),
                    agent_loop_max_iterations=2,
                )

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, "A")
        self.assertEqual(result.traces[1]["execution_path"], "agent_loop")
        self.assertIn("group_keys", result.output_summaries["T1"])

    def test_agent_loop_format_answer_unwraps_single_nested_list(self) -> None:
        original_try_fast_path = TableReasoningNodeAgent.try_fast_path

        def force_format_answer_agent_loop(
            agent: TableReasoningNodeAgent,
        ) -> FastPathDecision:
            if agent.node.get("id") == "N0":
                return FastPathDecision(
                    hit=False,
                    tool="table_reasoning.format_answer",
                    backend=agent.backend_name,
                    miss_reason="forced_test",
                )
            return original_try_fast_path(agent)

        with patch.object(
            TableReasoningNodeAgent,
            "try_fast_path",
            force_format_answer_agent_loop,
        ):
            result = execute_physical_plan(
                {
                    "task_type": "table_reasoning_v1",
                    "resources": [],
                    "nodes": [
                        {
                            "id": "N0",
                            "op": "FormatAnswer",
                            "dependency": [],
                            "input": [],
                            "params": {"answer": {"name": "answer", "type": "list[number]"}},
                            "output": "answer",
                        }
                    ],
                    "edges": [],
                },
                slm_config={
                    "api_type": "chat_completions",
                    "model": "fake-slm",
                    "temperature": 0,
                },
                slm_client=_FakeChatClient(
                    '{"action":"run_python","code":"result = [[1, 2, 3]]"}'
                ),
                agent_loop_max_iterations=2,
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, [1, 2, 3])

    def test_agent_loop_workspace_uses_dependency_copies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n7\n8\n", encoding="utf-8")

            result = execute_physical_plan(
                _unused_agent_loop_then_count_plan(table_path),
                slm_config={
                    "api_type": "chat_completions",
                    "model": "fake-slm",
                    "temperature": 0,
                },
                slm_client=_FakeChatClient(
                    [
                        (
                            '{"action":"run_python","code":"'
                            "df = inputs['dep_0']\\n"
                            "df.drop(df.index, inplace=True)\\n"
                            "result = pd.DataFrame({'answer': [999]})"
                            '"}'
                        ),
                        '{"action":"submit_result","name":"result"}',
                    ]
                ),
                agent_loop_max_iterations=3,
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, 2)
        self.assertEqual(result.output_summaries["T0"]["rows"], 2)
        self.assertNotIn("T0", result.outputs)

    def test_agent_loop_prompt_does_not_expose_resource_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n7\n8\n", encoding="utf-8")
            client = _FakeChatClient('{"action":"abort","reason":"stop"}')

            execute_physical_plan(
                {
                    "task_type": "table_reasoning_v1",
                    "resources": [_resource("table_1", table_path)],
                    "nodes": [
                        {
                            "id": "N0",
                            "op": "CustomReason",
                            "dependency": [],
                            "input": ["table_1"],
                            "params": {"instruction": "Inspect the resource."},
                            "output": "answer",
                        }
                    ],
                    "edges": [],
                },
                slm_config={
                    "api_type": "chat_completions",
                    "model": "fake-slm",
                    "temperature": 0,
                },
                slm_client=client,
            )

        prompt = client.chat.completions.last_request["messages"][-1]["content"]
        self.assertNotIn(str(table_path), prompt)
        self.assertNotIn(str(table_path.parent), prompt)
        self.assertIn("source_0", prompt)

    def test_agent_loop_prompt_exposes_only_node_operation_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n7\n8\n", encoding="utf-8")
            client = _FakeChatClient('{"action":"abort","reason":"stop"}')

            execute_physical_plan(
                {
                    "task_type": "table_reasoning_v1",
                    "resources": [_resource("table_1", table_path)],
                    "nodes": [
                        {
                            "id": "N0",
                            "op": "CustomReason",
                            "dependency": [],
                            "input": ["table_1"],
                            "params": {"local_mode": "mechanical_node_execution"},
                            "output": "answer",
                            "original_question": "SECRET_USER_QUESTION",
                            "remote_sql": "SECRET_REMOTE_SQL",
                            "global_instruction": "SECRET_GLOBAL_INSTRUCTION",
                            "expected_answer": "SECRET_EXPECTED_ANSWER",
                        }
                    ],
                    "edges": [],
                },
                slm_config={
                    "api_type": "chat_completions",
                    "model": "fake-slm",
                    "temperature": 0,
                },
                slm_client=client,
            )

        prompt = client.chat.completions.last_request["messages"][-1]["content"]
        self.assertIn("CustomReason", prompt)
        self.assertIn("mechanical_node_execution", prompt)
        self.assertNotIn("SECRET_USER_QUESTION", prompt)
        self.assertNotIn("SECRET_REMOTE_SQL", prompt)
        self.assertNotIn("SECRET_GLOBAL_INSTRUCTION", prompt)
        self.assertNotIn("SECRET_EXPECTED_ANSWER", prompt)

    def test_fast_path_execution_error_reports_failing_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n1\n", encoding="utf-8")
            plan = _count_plan(table_path)
            plan["nodes"][1]["params"]["aggregations"][0]["argument"] = {
                "type": "column",
                "name": "missing",
            }

            result = execute_physical_plan(plan)

        self.assertFalse(result.ok)
        self.assertEqual(result.failing_node, {"id": "N1", "op": "Aggregate", "output": "T1"})
        self.assertEqual(result.traces[1]["fast_path_hit"], True)
        self.assertEqual(result.traces[1]["status"], "failed")
        self.assertIn("Unknown column", result.error["message"])

    def test_rejects_invalid_physical_plan_dependencies(self) -> None:
        plan = {
            "task_type": "table_reasoning_v1",
            "resources": [],
            "nodes": [
                {
                    "id": "N0",
                    "op": "FormatAnswer",
                    "dependency": ["missing"],
                    "input": [],
                    "params": {"answer": {"name": "answer", "type": "number"}},
                    "output": "answer",
                }
            ],
            "edges": [],
        }

        result = execute_physical_plan(plan)

        self.assertFalse(result.ok)
        self.assertEqual(result.error["type"], "PlanValidationError")
        self.assertIn("unknown dependencies", result.error["message"])

    def test_json_ready_serializes_date_values(self) -> None:
        self.assertEqual(
            json_ready({"answer": date(2020, 1, 2)}),
            {"answer": "2020-01-02"},
        )


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


def _first_value_plan(table_path: Path) -> dict:
    plan = _count_plan(table_path)
    plan["nodes"][1]["params"]["aggregations"] = [
        {
            "function": "FIRST",
            "argument": {"type": "column", "name": "value"},
            "distinct": False,
            "alias": "answer",
        }
    ]
    return plan


def _unused_agent_loop_then_count_plan(table_path: Path) -> dict:
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
                            "function": "FIRST",
                            "argument": {"type": "column", "name": "value"},
                            "distinct": False,
                            "alias": "unused",
                        }
                    ],
                    "grouped": False,
                },
                "output": "T1",
            },
            {
                "id": "N2",
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
                "output": "T2",
            },
            {
                "id": "N3",
                "op": "FormatAnswer",
                "dependency": ["T2"],
                "input": [],
                "params": {"answer": {"name": "answer", "type": "number"}},
                "output": "answer",
            },
        ],
        "edges": [
            {"from": "N0", "to": "N1"},
            {"from": "N0", "to": "N2"},
            {"from": "N2", "to": "N3"},
        ],
    }


def _group_count_top_plan(table_path: Path) -> dict:
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
                "op": "Group",
                "dependency": ["T0"],
                "input": [],
                "params": {"keys": [{"type": "column", "name": "value"}]},
                "output": "T1",
            },
            {
                "id": "N2",
                "op": "Aggregate",
                "dependency": ["T1"],
                "input": [],
                "params": {
                    "aggregations": [
                        {
                            "function": "COUNT",
                            "argument": {"type": "wildcard"},
                            "distinct": False,
                            "alias": "cnt",
                        }
                    ],
                    "grouped": True,
                },
                "output": "T2",
            },
            {
                "id": "N3",
                "op": "Sort",
                "dependency": ["T2"],
                "input": [],
                "params": {
                    "keys": [
                        {
                            "expr": {"type": "column", "name": "cnt"},
                            "direction": "DESC",
                            "nulls": "LAST",
                        }
                    ]
                },
                "output": "T3",
            },
            {
                "id": "N4",
                "op": "Limit",
                "dependency": ["T3"],
                "input": [],
                "params": {"count": 1},
                "output": "T4",
            },
            {
                "id": "N5",
                "op": "Project",
                "dependency": ["T4"],
                "input": [],
                "params": {
                    "expressions": [
                        {
                            "expr": {"type": "column", "name": "value"},
                            "alias": "answer",
                        }
                    ]
                },
                "output": "T5",
            },
            {
                "id": "N6",
                "op": "FormatAnswer",
                "dependency": ["T5"],
                "input": [],
                "params": {"answer": {"name": "answer", "type": "category"}},
                "output": "answer",
            },
        ],
        "edges": [
            {"from": "N0", "to": "N1"},
            {"from": "N1", "to": "N2"},
            {"from": "N2", "to": "N3"},
            {"from": "N3", "to": "N4"},
            {"from": "N4", "to": "N5"},
            {"from": "N5", "to": "N6"},
        ],
    }


def _union_count_plan(left_path: Path, right_path: Path) -> dict:
    return {
        "task_type": "table_reasoning_v1",
        "resources": [
            _resource("left", left_path),
            _resource("right", right_path),
        ],
        "nodes": [
            {
                "id": "N0",
                "op": "Scan",
                "dependency": [],
                "input": ["left"],
                "params": {"source": "left"},
                "output": "T0",
            },
            {
                "id": "N1",
                "op": "Scan",
                "dependency": [],
                "input": ["right"],
                "params": {"source": "right"},
                "output": "T1",
            },
            {
                "id": "N2",
                "op": "SetOp",
                "dependency": ["T0", "T1"],
                "input": [],
                "params": {"operator": "UNION ALL"},
                "output": "T2",
            },
            {
                "id": "N3",
                "op": "Aggregate",
                "dependency": ["T2"],
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
                "output": "T3",
            },
            {
                "id": "N4",
                "op": "FormatAnswer",
                "dependency": ["T3"],
                "input": [],
                "params": {"answer": {"name": "answer", "type": "number"}},
                "output": "answer",
            },
        ],
        "edges": [
            {"from": "N0", "to": "N2"},
            {"from": "N1", "to": "N2"},
            {"from": "N2", "to": "N3"},
            {"from": "N3", "to": "N4"},
        ],
    }


def _resource(source_id: str, table_path: Path) -> dict:
    return {
        "id": source_id,
        "type": "table",
        "path": str(table_path),
        "format": "csv",
        "schema": {"format": "csv", "columns": ["value"]},
    }


def _table_resource(source_id: str, table_path: Path, columns: list[str]) -> dict:
    return {
        "id": source_id,
        "type": "table",
        "path": str(table_path),
        "format": "csv",
        "schema": {"format": "csv", "columns": columns},
    }


def _v2_shared_prefix_plan(table_path: Path) -> dict:
    return {
        "task_type": "table_reasoning_v2",
        "resources": [
            _table_resource(
                "table_1",
                table_path,
                ["selfMade", "country", "finalWorth"],
            )
        ],
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
                        {
                            "alias": "answer_1",
                            "expr": {"type": "column", "name": "selfMade"},
                        }
                    ]
                },
                "output": "T3",
            },
            {
                "id": "N4",
                "op": "FormatAnswer",
                "dependency": ["T3"],
                "input": [],
                "params": {"answer": {"name": "answer_1", "type": "boolean"}},
                "output": "answer_1",
            },
            {
                "id": "N5",
                "op": "Project",
                "dependency": ["T2"],
                "input": [],
                "params": {
                    "expressions": [
                        {
                            "alias": "answer_2",
                            "expr": {"type": "column", "name": "country"},
                        }
                    ]
                },
                "output": "T4",
            },
            {
                "id": "N6",
                "op": "FormatAnswer",
                "dependency": ["T4"],
                "input": [],
                "params": {"answer": {"name": "answer_2", "type": "string"}},
                "output": "answer_2",
            },
        ],
        "edges": [
            {"from": "N0", "to": "N1"},
            {"from": "N1", "to": "N2"},
            {"from": "N2", "to": "N3"},
            {"from": "N3", "to": "N4"},
            {"from": "N2", "to": "N5"},
            {"from": "N5", "to": "N6"},
        ],
        "answers": [
            {"name": "answer_1", "type": "boolean"},
            {"name": "answer_2", "type": "string"},
        ],
        "subtask_outputs": [
            {
                "id": "Q0",
                "index": 0,
                "answer": {"name": "answer_1", "type": "boolean"},
                "output": "answer_1",
            },
            {
                "id": "Q1",
                "index": 1,
                "answer": {"name": "answer_2", "type": "string"},
                "output": "answer_2",
            },
        ],
    }


class _FakeChatClient:
    def __init__(self, output_text: str | list[str]) -> None:
        self.chat = SimpleNamespace(
            completions=_FakeChatCompletions(output_text),
        )


class _FakeChatCompletions:
    def __init__(self, output_text: str | list[str]) -> None:
        self._outputs = output_text if isinstance(output_text, list) else [output_text]
        self._index = 0
        self.last_request: dict[str, object] = {}

    def create(self, **kwargs: object) -> "_FakeChatResponse":
        self.last_request = kwargs
        output = self._outputs[min(self._index, len(self._outputs) - 1)]
        self._index += 1
        return _FakeChatResponse(output)


class _FakeChatResponse:
    id = "agent_loop_fake"

    def __init__(self, output_text: str) -> None:
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(content=output_text)),
        ]

    def model_dump(self, mode: str) -> dict[str, object]:
        return {"id": self.id, "mode": mode}


if __name__ == "__main__":
    unittest.main()
