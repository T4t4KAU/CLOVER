from __future__ import annotations

from datetime import date
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from clover.executor import ExecutionPlanBuilder, execute_execution_plan
from clover.executor.agents.base import FastPathDecision
from clover.executor.agents.table_reasoning import TableReasoningNodeAgent
from clover.executor.result import json_ready


class ExecutorTest(unittest.TestCase):
    def test_executes_table_reasoning_plan_with_fast_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n1\n2\n3\n", encoding="utf-8")

            result = _execute_plan(_count_plan(table_path))

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

            result = _execute_plan(
                _union_count_plan(left_path, right_path),
                max_parallel_execution_units=2,
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, 4)
        self.assertEqual(result.fast_path_hits, 5)
        self.assertEqual([trace["node_id"] for trace in result.traces], ["N0", "N1", "N2", "N3", "N4"])

    def test_executes_merged_batch_plan_with_shared_nodes_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "selfMade,country,finalWorth\n"
                "False,France,100\n"
                "True,United States,200\n",
                encoding="utf-8",
            )

            result = _execute_plan(_shared_prefix_batch_plan(table_path))

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
            "task_type": "table_reasoning.query",
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

        result = _execute_plan(
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

            result = _execute_plan(
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

    def test_analyze_empty_filter_can_be_completed_by_agent_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "city,height,floors\n"
                "winnipeg,44,11\n"
                "winnipeg,50,13\n"
                "toronto,60,20\n",
                encoding="utf-8",
            )

            client = _FakeChatClient(
                [
                    (
                        '{"s":"'
                        "def solve(df):\\n"
                        "    return df[(df['city'] == 'Winnipeg') & (df['floors'] > 10)]"
                        '"}'
                    ),
                    (
                        '{"s":"'
                        "def solve(df):\\n"
                        "    print(df['city'].value_counts().head())\\n"
                        "    mask = df['city'].str.lower().eq('winnipeg') & (df['floors'] > 10)\\n"
                        "    return df.loc[mask].copy()"
                        '"}'
                    ),
                ]
            )

            result = _execute_plan(
                _city_average_plan(table_path, task_type="table_reasoning.analyze"),
                slm_config={
                    "api_type": "chat_completions",
                    "model": "fake-slm",
                    "temperature": 0,
                },
                slm_client=client,
                agent_loop_max_iterations=3,
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, 47)
        filter_trace = result.traces[1]
        self.assertEqual(filter_trace["execution_path"], "agent_loop")
        self.assertEqual(filter_trace["agent_loop_trigger"], "fast_path_empty_output")
        self.assertEqual(filter_trace["agent_loop"]["iterations"], 2)
        self.assertEqual(
            filter_trace["agent_loop"]["steps"][0]["observation_type"],
            "contract_error",
        )
        first_request = client.chat.completions.requests[0]
        first_prompt = first_request["messages"][-1]["content"]
        self.assertIn("Case:", first_prompt)
        self.assertIn('"sig":"def solve(df):"', first_prompt)
        self.assertIn('"goal"', first_prompt)
        self.assertIn('"cols"', first_prompt)
        self.assertIn('"evidence"', first_prompt)
        self.assertNotIn('"task"', first_prompt)
        self.assertNotIn('"diag"', first_prompt)
        self.assertNotIn('"head"', first_prompt)
        self.assertNotIn('"preview"', first_prompt)
        self.assertNotIn("dep_0", first_prompt)
        self.assertNotIn("T0", first_prompt)
        self.assertNotIn("T1", first_prompt)
        self.assertIn("winnipeg", first_prompt)

    def test_query_empty_filter_can_be_completed_by_agent_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "city,height,floors\n"
                "winnipeg,44,11\n"
                "winnipeg,50,13\n"
                "toronto,60,20\n",
                encoding="utf-8",
            )
            client = _FakeChatClient(
                [
                    (
                        '{"s":"'
                        "def solve(df):\\n"
                        "    return df[(df['city'] == 'Winnipeg') & (df['floors'] > 10)]"
                        '"}'
                    ),
                    (
                        '{"s":"'
                        "def solve(df):\\n"
                        "    mask = df['city'].str.lower().eq('winnipeg') & (df['floors'] > 10)\\n"
                        "    return df.loc[mask].copy()"
                        '"}'
                    ),
                ]
            )

            result = _execute_plan(
                _city_average_plan(table_path, task_type="table_reasoning.query"),
                slm_config={
                    "api_type": "chat_completions",
                    "model": "fake-slm",
                    "temperature": 0,
                },
                slm_client=client,
                agent_loop_max_iterations=3,
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, 47)
        filter_trace = result.traces[1]
        self.assertEqual(filter_trace["execution_path"], "agent_loop")
        self.assertEqual(filter_trace["agent_loop_trigger"], "fast_path_empty_output")
        first_prompt = client.chat.completions.requests[0]["messages"][-1]["content"]
        self.assertIn("Case:", first_prompt)
        self.assertIn('"sig":"def solve(df):"', first_prompt)
        self.assertIn("winnipeg", first_prompt)

    def test_analyze_unknown_column_error_recovers_with_agent_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n7\n8\n", encoding="utf-8")
            plan = _first_value_plan(table_path)
            plan["task_type"] = "table_reasoning.analyze"
            plan["nodes"][1]["params"]["aggregations"][0]["argument"] = {
                "type": "column",
                "name": "missing",
            }

            result = _execute_plan(
                plan,
                slm_config={
                    "api_type": "chat_completions",
                    "model": "fake-slm",
                    "temperature": 0,
                },
                slm_client=_FakeChatClient(
                    (
                        '{"s":"'
                        "def solve(df):\\n"
                        "    return pd.DataFrame({'answer': [df['value'].iloc[0]]})"
                        '"}'
                    )
                ),
                agent_loop_max_iterations=2,
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, 7)
        aggregate_trace = result.traces[1]
        self.assertEqual(aggregate_trace["execution_path"], "agent_loop")
        self.assertEqual(
            aggregate_trace["agent_loop_trigger"],
            "fast_path_execution_error",
        )

    def test_analyze_empty_filter_keeps_fast_path_output_when_agent_loop_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                "city,height,floors\n"
                "winnipeg,44,11\n"
                "winnipeg,50,13\n"
                "toronto,60,20\n",
                encoding="utf-8",
            )

            result = _execute_plan(
                _city_average_plan(table_path, task_type="table_reasoning.analyze"),
                slm_config={
                    "api_type": "chat_completions",
                    "model": "fake-slm",
                    "temperature": 0,
                },
                slm_client=_FakeChatClient("not json"),
                agent_loop_max_iterations=1,
        )

        self.assertTrue(result.ok)
        self.assertIsNone(result.answer)
        filter_trace = result.traces[1]
        self.assertEqual(filter_trace["execution_path"], "fast_path")
        self.assertEqual(filter_trace["agent_loop_trigger"], "fast_path_empty_output")
        self.assertEqual(filter_trace["agent_loop_fallback"], "fast_path_output")
        self.assertIn("agent_loop_error", filter_trace)
        self.assertEqual(filter_trace["output_summary"]["rows"], 0)

    def test_empty_filter_does_not_retry_downstream_empty_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("city\nwinnipeg\n", encoding="utf-8")
            client = _FakeChatClient("not json")

            result = _execute_plan(
                _empty_filter_project_plan(table_path),
                slm_config={
                    "api_type": "chat_completions",
                    "model": "fake-slm",
                    "temperature": 0,
                },
                slm_client=client,
                agent_loop_max_iterations=1,
            )

        self.assertTrue(result.ok)
        self.assertIsNone(result.answer)
        self.assertEqual(len(client.chat.completions.requests), 1)
        self.assertIn("agent_loop", result.traces[1])
        self.assertNotIn("agent_loop", result.traces[2])

    def test_wrong_column_mismatch_routes_directly_to_cloud_replan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("city\n1\n2\n", encoding="utf-8")
            plan = _empty_filter_project_plan(table_path)
            plan["nodes"][1]["params"]["predicate"]["right"]["value"] = "April 8"
            client = _FakeChatClient("not json")

            result = _execute_plan(
                plan,
                slm_config={
                    "api_type": "chat_completions",
                    "model": "fake-slm",
                    "temperature": 0,
                },
                slm_client=client,
                agent_loop_max_iterations=2,
            )

        self.assertTrue(result.ok)
        self.assertIsNone(result.answer)
        self.assertEqual(len(client.chat.completions.requests), 0)
        verdict = result.traces[1]["verification_verdict"]
        self.assertEqual(verdict["route"], "cloud_replan")
        self.assertEqual(verdict["reason"], "predicate_wrong_column")
        self.assertEqual(
            verdict["evidence"]["mismatch"]["roots"][0]["mismatch"],
            "wrong_column",
        )

    def test_rewrite_predicate_executes_with_original_dependency_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                'title\n"""Keep Hustlin"""\n"Other"\n',
                encoding="utf-8",
            )
            client = _FakeChatClient(
                json.dumps(
                    {
                        "action": "rewrite_predicate",
                        "predicate": '"title" = \'"Keep Hustlin"\'',
                    }
                )
            )

            result = _execute_plan(
                _quoted_title_plan(table_path),
                slm_config={
                    "api_type": "chat_completions",
                    "model": "fake-slm",
                    "temperature": 0,
                },
                slm_client=client,
                agent_loop_max_iterations=2,
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, '"Keep Hustlin"')
        filter_trace = result.traces[1]
        self.assertEqual(filter_trace["agent_loop"]["iterations"], 1)
        self.assertTrue(filter_trace["agent_loop"]["steps"][0]["accepted"])

    def test_failed_predicate_rewrite_falls_back_to_python_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                'title\n"""Keep Hustlin"""\n"Other"\n',
                encoding="utf-8",
            )
            client = _FakeChatClient(
                [
                    json.dumps(
                        {
                            "action": "rewrite_predicate",
                            "predicate": '"title" = \'Nope\'',
                        }
                    ),
                    (
                        '{"s":"def solve(df):\\n'
                        "    return df[df['title'].str.contains("
                        "'Keep Hustlin', regex=False)]"
                        '"}'
                    ),
                ]
            )

            result = _execute_plan(
                _quoted_title_plan(table_path),
                slm_config={
                    "api_type": "chat_completions",
                    "model": "fake-slm",
                    "temperature": 0,
                },
                slm_client=client,
                agent_loop_max_iterations=2,
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, '"Keep Hustlin"')
        steps = result.traces[1]["agent_loop"]["steps"]
        self.assertEqual(
            [step["prompt_kind"] for step in steps],
            [
                "table_reasoning_rewrite_predicate",
                "table_reasoning_empty_filter_repair",
            ],
        )
        self.assertTrue(steps[1]["accepted"])

    def test_predicate_rewrite_cannot_expand_condition_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text(
                'title\n"""Keep Hustlin"""\n"Other"\n',
                encoding="utf-8",
            )
            client = _FakeChatClient(
                [
                    json.dumps(
                        {
                            "action": "rewrite_predicate",
                            "predicate": (
                                '"title" = \'"Keep Hustlin"\' '
                                'OR "title" = \'Other\''
                            ),
                        }
                    ),
                    (
                        '{"s":"def solve(df):\\n'
                        "    return df[df['title'].str.contains("
                        "'Keep Hustlin', regex=False)]"
                        '"}'
                    ),
                ]
            )

            result = _execute_plan(
                _quoted_title_plan(table_path),
                slm_config={
                    "api_type": "chat_completions",
                    "model": "fake-slm",
                    "temperature": 0,
                },
                slm_client=client,
                agent_loop_max_iterations=2,
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, '"Keep Hustlin"')
        steps = result.traces[1]["agent_loop"]["steps"]
        self.assertEqual(steps[0]["observation_type"], "invalid_predicate_patch")
        self.assertEqual(
            steps[1]["prompt_kind"],
            "table_reasoning_empty_filter_repair",
        )
        self.assertTrue(steps[1]["accepted"])

    def test_agent_loop_accepts_result_created_by_run_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n7\n8\n", encoding="utf-8")

            result = _execute_plan(
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

            result = _execute_plan(
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
                result = _execute_plan(
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
                result = _execute_plan(
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
            result = _execute_plan(
                {
                    "task_type": "table_reasoning.query",
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

            result = _execute_plan(
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

    def test_agent_loop_early_stops_on_repeated_error(self) -> None:
        """Repeated identical errors should trigger early-stop before max_iterations."""

        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n7\n8\n", encoding="utf-8")

            result = _execute_plan(
                _first_value_plan(table_path),
                slm_config={
                    "api_type": "chat_completions",
                    "model": "fake-slm",
                    "temperature": 0,
                    "agent_loop_repeat_error_early_stop": 2,
                },
                slm_client=_FakeChatClient("not json"),
                agent_loop_max_iterations=5,
            )

        self.assertFalse(result.ok)
        agent_loop_trace = result.traces[1]["agent_loop"]
        # Early-stop threshold is 2, so the loop should stop at 2 iterations,
        # well below the max_iterations of 5.
        self.assertLess(agent_loop_trace["iterations"], 5)
        self.assertIn("early-stopped", result.error["message"].lower())

    def test_fast_path_execution_error_reports_failing_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("value\n1\n", encoding="utf-8")
            plan = _count_plan(table_path)
            plan["nodes"][1]["params"]["aggregations"][0]["argument"] = {
                "type": "column",
                "name": "missing",
            }

            result = _execute_plan(plan)

        self.assertFalse(result.ok)
        self.assertEqual(result.failing_node, {"id": "N1", "op": "Aggregate", "output": "T1"})
        self.assertEqual(result.traces[1]["fast_path_hit"], True)
        self.assertEqual(result.traces[1]["status"], "failed")
        self.assertIn("Unknown column", result.error["message"])

    def test_rejects_invalid_physical_plan_dependencies(self) -> None:
        plan = {
            "task_type": "table_reasoning.query",
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

        result = _execute_plan(plan)

        self.assertFalse(result.ok)
        self.assertEqual(result.error["type"], "PlanValidationError")
        self.assertIn("unknown dependencies", result.error["message"])

    def test_json_ready_serializes_date_values(self) -> None:
        self.assertEqual(
            json_ready({"answer": date(2020, 1, 2)}),
            {"answer": "2020-01-02"},
        )


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
        "task_type": "table_reasoning.query",
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


def _city_average_plan(table_path: Path, *, task_type: str) -> dict:
    return {
        "task_type": task_type,
        "resources": [
            _table_resource("table_1", table_path, ["city", "height", "floors"])
        ],
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
                "op": "Filter",
                "dependency": ["T0"],
                "input": [],
                "params": {
                    "predicate": {
                        "type": "logical_op",
                        "op": "AND",
                        "operands": [
                            {
                                "type": "binary_op",
                                "op": "=",
                                "left": {"type": "column", "name": "city"},
                                "right": {
                                    "type": "literal",
                                    # Misspelled so static literal binding cannot repair it;
                                    # these tests need to exercise the agent-loop fallback.
                                    "value": "Winnippeg",
                                    "value_type": "string",
                                },
                            },
                            {
                                "type": "binary_op",
                                "op": ">",
                                "left": {"type": "column", "name": "floors"},
                                "right": {
                                    "type": "literal",
                                    "value": 10,
                                    "value_type": "number",
                                },
                            },
                        ],
                    }
                },
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
                            "function": "AVG",
                            "argument": {"type": "column", "name": "height"},
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
            {"from": "N1", "to": "N2"},
            {"from": "N2", "to": "N3"},
        ],
    }


def _empty_filter_project_plan(table_path: Path) -> dict:
    return {
        "task_type": "table_reasoning.analyze",
        "resources": [_table_resource("table_1", table_path, ["city"])],
        "answer": {"name": "answer", "type": "string"},
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
                        "op": "=",
                        "left": {"type": "column", "name": "city"},
                        "right": {
                            "type": "literal",
                            "value": "missing",
                            "value_type": "string",
                        },
                    }
                },
                "output": "T1",
            },
            {
                "id": "N2",
                "op": "Project",
                "dependency": ["T1"],
                "input": [],
                "params": {
                    "expressions": [
                        {"expr": {"type": "column", "name": "city"}}
                    ]
                },
                "output": "T2",
            },
            {
                "id": "N3",
                "op": "FormatAnswer",
                "dependency": ["T2"],
                "input": [],
                "params": {"answer": {"name": "answer", "type": "string"}},
                "output": "answer",
            },
        ],
        "edges": [
            {"from": "N0", "to": "N1"},
            {"from": "N1", "to": "N2"},
            {"from": "N2", "to": "N3"},
        ],
    }


def _quoted_title_plan(table_path: Path) -> dict:
    plan = _empty_filter_project_plan(table_path)
    plan["resources"] = [_table_resource("table_1", table_path, ["title"])]
    plan["nodes"][1]["params"]["predicate"] = {
        "type": "binary_op",
        "op": "=",
        "left": {"type": "column", "name": "title"},
        "right": {
            "type": "literal",
            "value": "Keep Hustlin",
            "value_type": "string",
        },
    }
    plan["nodes"][2]["params"]["expressions"] = [
        {"expr": {"type": "column", "name": "title"}}
    ]
    return plan


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


def _shared_prefix_batch_plan(table_path: Path) -> dict:
    return {
        "task_type": "table_reasoning.query",
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
        "query_outputs": [
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
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> "_FakeChatResponse":
        self.last_request = kwargs
        self.requests.append(kwargs)
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
