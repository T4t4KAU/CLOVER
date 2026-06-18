from __future__ import annotations

import json
import threading
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from clover.executor import ExecutionResult
from clover.runtime import TableReasoningCaseSpec, run_table_reasoning_system
from clover.runtime.table_reasoning import pipeline as table_pipeline
from clover.runtime.table_reasoning.pipeline import _load_json_object


class TableReasoningRuntimeTest(unittest.TestCase):
    def test_table_command_parser_uses_first_json_object(self) -> None:
        payload = _load_json_object(
            '{"op":"sql","q":"SELECT COUNT(*) FROM table_1"}\n'
            '{"op":"answer","a":0}'
        )

        self.assertEqual(
            payload,
            {"op": "sql", "q": "SELECT COUNT(*) FROM table_1"},
        )

    def test_table_action_parser_accepts_wrapped_seed_sql(self) -> None:
        actions = table_pipeline._normalize_table_actions(  # noqa: SLF001
            {
                "acts": [
                    {
                        "op": "analyze",
                        "kind": "statistical",
                        "seed": {"sql": 'SELECT "year" FROM "table_1";'},
                    }
                ]
            }
        )

        self.assertEqual(actions[0].op, "analyze")
        self.assertEqual(actions[0].seed, 'SELECT "year" FROM "table_1"')

    def test_query_with_explicit_local_sql_skips_remote_decompose(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient([])

            result = run_table_reasoning_system(
                case_specs=[
                    TableReasoningCaseSpec(
                        case_id="case_1",
                        base_dir=Path(tmpdir),
                        task_dsl={
                            "task_type": "table_reasoning.query",
                            "question": "Which country appears first?",
                            "sources": [{"type": "table", "file": str(table_path)}],
                            "answer": {"name": "answer", "type": "string"},
                            "sql": 'SELECT "country" AS "answer_1" FROM "table_1" LIMIT 1;',
                        },
                    )
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                client=client,
            )

        self.assertEqual(result.case_results[0].answer, "France")
        self.assertEqual(len(client.chat.completions.requests), 0)
        self.assertEqual(result.profile["counters"]["local_entry_commands"], 1)

    def test_batches_same_table_questions_and_merges_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps(
                        [
                            _self_made_sql("answer_1"),
                            _country_sql("answer_2"),
                        ]
                    ),
                ],
                usages=[
                    {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
                ],
            )

            result = run_table_reasoning_system(
                case_specs=[
                    _case_spec(
                        "case_1",
                        Path(tmpdir),
                        table_path,
                        "Is the person with the highest net worth self-made?",
                        "boolean",
                    ),
                    _case_spec(
                        "case_2",
                        Path(tmpdir),
                        table_path,
                        "What is the country of the person with the highest net worth?",
                        "string",
                    ),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=8,
                client=client,
                profile_baseline=True,
            )

        self.assertEqual(
            {item.case_id: item.answer for item in result.case_results},
            {"case_1": True, "case_2": "United States"},
        )
        self.assertEqual(result.profile["counters"]["supervisor_decompose_calls"], 1)
        self.assertNotIn("supervisor_synthesis_calls", result.profile["counters"])
        self.assertEqual(result.profile["counters"]["static_synthesis_calls"], 1)
        self.assertEqual(result.profile["counters"]["remote_input_tokens"], 100)
        self.assertEqual(result.profile["counters"]["remote_output_tokens"], 10)
        self.assertEqual(result.profile["counters"]["remote_total_tokens"], 110)
        self.assertEqual(
            result.profile["summary"]["remote_token_usage"],
            {
                "input_tokens": 100,
                "cached_input_tokens": 0,
                "output_tokens": 10,
                "reasoning_tokens": 0,
                "total_tokens": 110,
            },
        )
        self.assertEqual(result.profile["summary"]["validation_mode"], "none")
        self.assertGreaterEqual(result.profile["counters"]["reused_nodes"], 3)
        self.assertIn("baseline_executor", result.profile["stages"])
        self.assertIn("local_executor_speedup", result.profile["summary"])

    def test_remote_validation_bypasses_synthesis_for_valid_format_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps(
                        [
                            _country_sql("answer_1"),
                        ]
                    ),
                ],
            )

            result = run_table_reasoning_system(
                case_specs=[
                    _case_spec(
                        "case_1",
                        Path(tmpdir),
                        table_path,
                        "What is the country of the person with the highest net worth?",
                        "string",
                    ),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=8,
                validation_mode="remote_supervisor",
                client=client,
            )

        case_result = result.case_results[0]
        self.assertEqual(case_result.answer, "United States")
        self.assertEqual(len(client.chat.completions.requests), 1)
        self.assertNotIn("supervisor_synthesis_calls", result.profile["counters"])
        self.assertEqual(case_result.metadata["final_answer_source"], "format_answer")
        diagnostics = case_result.metadata["table_diagnostics"]
        self.assertEqual(diagnostics[-1]["stage"], "sql_execution")
        self.assertIs(diagnostics[-1]["finalization"]["bypass_synthesis"], True)

    def test_batch_supervisor_report_accepts_keyed_answers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps(
                        [
                            'SELECT "selfMade" AS "answer_1" FROM "table_1" '
                            'WHERE "country" = \'missing\';',
                            'SELECT "country" AS "answer_2" FROM "table_1" '
                            'WHERE "country" = \'missing\';',
                        ]
                    ),
                    json.dumps(
                        {
                            "op": "answer",
                            "a": {
                                "answer_1": True,
                                "answer_2": "United States",
                            },
                        }
                    ),
                ]
            )

            result = run_table_reasoning_system(
                case_specs=[
                    _case_spec("case_1", Path(tmpdir), table_path, "Self-made?", "boolean"),
                    _case_spec("case_2", Path(tmpdir), table_path, "Country?", "string"),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=2,
                max_retries=1,
                validation_mode="remote_supervisor",
                client=client,
            )

        results = {item.case_id: item for item in result.case_results}
        self.assertTrue(results["case_1"].ok)
        self.assertEqual(results["case_1"].retry_count, 0)
        self.assertTrue(results["case_2"].ok)
        self.assertEqual(results["case_2"].retry_count, 0)
        self.assertEqual(results["case_2"].answer, "United States")
        self.assertEqual(result.profile["counters"]["supervisor_synthesis_calls"], 1)

    def test_prefetches_supervisor_decompose_batches_before_local_finish(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_table = _write_people_table(root / "first")
            second_table = _write_people_table(root / "second")
            client = _BlockingPrefetchChatClient(
                [
                    json.dumps([_country_sql("answer_1")]),
                    json.dumps([_self_made_sql("answer_2")]),
                ]
            )

            result = run_table_reasoning_system(
                case_specs=[
                    _case_spec("case_1", root, first_table, "Country?", "string"),
                    _case_spec("case_2", root, second_table, "Self-made?", "boolean"),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=1,
                client=client,
            )

        self.assertEqual([item.case_id for item in result.case_results], ["case_1", "case_2"])
        self.assertTrue(client.chat.completions.first_wait_saw_second)

    def test_table_runtime_uses_stateless_remote_calls_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps([_self_made_sql("answer_1")]),
                    json.dumps([_country_sql("answer_2")]),
                ]
            )

            run_table_reasoning_system(
                case_specs=[
                    _case_spec("case_1", Path(tmpdir), table_path, "Self-made?", "boolean"),
                    _case_spec("case_2", Path(tmpdir), table_path, "Country?", "string"),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=1,
                client=client,
            )

        for request in client.chat.completions.requests:
            self.assertEqual(
                [message["role"] for message in request["messages"]],
                ["user"],
            )
        self.assertEqual(len(client.chat.completions.requests), 2)
        second_supervisor_decompose_prompt = client.chat.completions.requests[1]["messages"][0]["content"]
        self.assertIn('"sources"', second_supervisor_decompose_prompt)

    def test_passes_local_parallel_limits_to_executor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps(
                        [
                            'SELECT COUNT(*) AS "answer_1" FROM "table_1";',
                        ]
                    ),
                ]
            )
            executor_kwargs: list[dict] = []

            def fake_execute_execution_plan(*args: object, **kwargs: object) -> ExecutionResult:
                del args
                executor_kwargs.append(dict(kwargs))
                return ExecutionResult(
                    ok=True,
                    answer={"answer_1": 2},
                    outputs={"answer_1": 2},
                    collector_outputs={"answer": {"answer_1": 2}},
                    traces=[],
                    output_summaries={},
                )

            with patch(
                "clover.runtime.table_reasoning.pipeline.execute_execution_plan",
                side_effect=fake_execute_execution_plan,
            ):
                result = run_table_reasoning_system(
                    case_specs=[
                        _case_spec(
                            "case_1",
                            Path(tmpdir),
                            table_path,
                            "How many rows?",
                            "number",
                        ),
                    ],
                    remote_config={"api_type": "chat_completions", "model": "fake-model"},
                    remote_batch_size=1,
                    max_parallel_execution_units=7,
                    max_parallel_slm_node_jobs=3,
                    max_parallel_slm_sequences=2,
                    max_pending_slm_sequences=19,
                    client=client,
                )

        self.assertEqual(result.case_results[0].answer, 2)
        self.assertEqual(executor_kwargs[0]["max_parallel_execution_units"], 7)
        self.assertEqual(executor_kwargs[0]["max_parallel_slm_node_jobs"], 3)
        self.assertEqual(executor_kwargs[0]["max_parallel_slm_sequences"], 2)
        self.assertEqual(executor_kwargs[0]["max_pending_slm_sequences"], 19)

    def test_analyze_action_path_passes_local_parallel_limits_to_executor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "q": (
                                'SELECT "country" AS "answer_1__a1" '
                                'FROM "table_1" LIMIT 1;'
                            )
                        }
                    ),
                    json.dumps({"a": "France"}),
                ]
            )
            executor_kwargs: list[dict] = []

            def fake_execute_execution_plan(*args: object, **kwargs: object) -> ExecutionResult:
                del args
                executor_kwargs.append(dict(kwargs))
                return ExecutionResult(
                    ok=True,
                    answer={"answer_1__a1": "France"},
                    outputs={"answer_1__a1": "France"},
                    collector_outputs={"answer": {"answer_1__a1": "France"}},
                    traces=[],
                    output_summaries={},
                )

            with patch(
                "clover.runtime.table_reasoning.pipeline.execute_execution_plan",
                side_effect=fake_execute_execution_plan,
            ):
                result = run_table_reasoning_system(
                    case_specs=[
                        _case_spec(
                            "case_1",
                            Path(tmpdir),
                            table_path,
                            "Which country appears first?",
                            "string",
                            profile="analyze",
                        ),
                    ],
                    remote_config={"api_type": "chat_completions", "model": "fake-model"},
                    remote_batch_size=1,
                    max_parallel_execution_units=7,
                    max_parallel_slm_node_jobs=3,
                    max_parallel_slm_sequences=2,
                    max_pending_slm_sequences=19,
                    client=client,
                )

        self.assertEqual(result.case_results[0].answer, "France")
        self.assertEqual(executor_kwargs[0]["max_parallel_execution_units"], 7)
        self.assertEqual(executor_kwargs[0]["max_parallel_slm_node_jobs"], 3)
        self.assertEqual(executor_kwargs[0]["max_parallel_slm_sequences"], 2)
        self.assertEqual(executor_kwargs[0]["max_pending_slm_sequences"], 19)

    def test_table_profile_records_local_slm_token_usage_from_executor_trace(self) -> None:
        profiler = table_pipeline.PipelineProfiler()
        execution_result = ExecutionResult(
            ok=True,
            answer="ok",
            outputs={"answer": "ok"},
            collector_outputs={"answer": "ok"},
            traces=[
                {
                    "fast_path_hit": False,
                    "fast_path_miss_reason": "unsupported_op",
                    "agent_loop_trigger": "fast_path_miss",
                    "agent_loop": {
                        "steps": [
                            {
                                "token_usage": {
                                    "input_tokens": 5,
                                    "cached_input_tokens": 2,
                                    "output_tokens": 3,
                                    "reasoning_tokens": 1,
                                    "total_tokens": 8,
                                }
                            },
                            {
                                "token_usage": {
                                    "input_tokens": 7,
                                    "cached_input_tokens": 0,
                                    "output_tokens": 2,
                                    "reasoning_tokens": 0,
                                    "total_tokens": 9,
                                }
                            },
                        ]
                    },
                }
            ],
            output_summaries={},
        )

        table_pipeline._record_executor_trace_counters(profiler, execution_result)
        profile = table_pipeline._profile_with_summary(profiler, validation_mode="none")

        counters = profile["counters"]
        self.assertEqual(counters["executor_local_slm_steps"], 2)
        self.assertEqual(counters["local_slm_calls"], 2)
        self.assertEqual(counters["local_slm_input_tokens"], 12)
        self.assertEqual(counters["local_slm_cached_input_tokens"], 2)
        self.assertEqual(counters["local_slm_output_tokens"], 5)
        self.assertEqual(counters["local_slm_reasoning_tokens"], 1)
        self.assertEqual(counters["local_slm_total_tokens"], 17)
        self.assertEqual(profile["summary"]["local_slm_calls"], 2)
        self.assertEqual(
            profile["summary"]["local_slm_token_usage"],
            {
                "input_tokens": 12,
                "cached_input_tokens": 2,
                "output_tokens": 5,
                "reasoning_tokens": 1,
                "total_tokens": 17,
            },
        )

    def test_execution_error_reports_observation_then_runs_next_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps(
                        [
                            'SELECT "missing" AS "answer_1" FROM "table_1";',
                        ]
                    ),
                    json.dumps(
                        {
                            "op": "sql",
                            "q": 'SELECT COUNT(*) AS "answer_1" FROM "table_1";',
                        }
                    ),
                ]
            )

            result = run_table_reasoning_system(
                case_specs=[
                    _case_spec("case_1", Path(tmpdir), table_path, "How many rows?", "number"),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=1,
                max_retries=1,
                validation_mode="remote_supervisor",
                client=client,
            )

        self.assertEqual(len(result.case_results), 1)
        self.assertTrue(result.case_results[0].ok)
        self.assertEqual(result.case_results[0].answer, 2)
        self.assertEqual(result.case_results[0].retry_count, 1)
        self.assertEqual(result.profile["counters"]["supervisor_synthesis_calls"], 1)
        self.assertEqual(result.profile["counters"]["static_final_answer_hits"], 1)
        self.assertNotIn("supervisor_repair_calls", result.profile["counters"])

    def test_default_validation_mode_does_not_retry_execution_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps(
                        [
                            'SELECT "missing" AS "answer_1" FROM "table_1";',
                        ]
                    ),
                ]
            )

            result = run_table_reasoning_system(
                case_specs=[
                    _case_spec("case_1", Path(tmpdir), table_path, "How many rows?", "number"),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=1,
                client=client,
            )

        self.assertEqual(len(client.chat.completions.requests), 1)
        self.assertEqual(len(result.case_results), 1)
        self.assertFalse(result.case_results[0].ok)
        self.assertEqual(result.case_results[0].retry_count, 0)
        self.assertNotIn("supervisor_repair_calls", result.profile["counters"])

    def test_analyze_profile_runs_evidence_sql_then_supervisor_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "q": [
                                'SELECT "country", "finalWorth" FROM "table_1" '
                                'ORDER BY "finalWorth" DESC LIMIT 2;'
                            ],
                        }
                    ),
                    json.dumps(
                        {
                            "a": "United States has the highest finalWorth.",
                        }
                    ),
                ]
            )

            result = run_table_reasoning_system(
                case_specs=[
                    _case_spec(
                        "case_1",
                        Path(tmpdir),
                        table_path,
                        "Which country has the highest finalWorth, and why?",
                        "string",
                        profile="analyze",
                    ),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=8,
                client=client,
            )

        self.assertEqual(
            result.case_results[0].answer,
            "United States has the highest finalWorth.",
        )
        self.assertEqual(result.profile["counters"]["supervisor_decompose_calls"], 1)
        self.assertEqual(result.profile["counters"]["supervisor_synthesis_calls"], 1)
        self.assertNotIn("static_synthesis_calls", result.profile["counters"])
        self.assertEqual(len(client.chat.completions.requests), 2)
        decompose_prompt = client.chat.completions.requests[0]["messages"][0]["content"]
        self.assertNotIn('"v"', decompose_prompt)
        review_prompt = client.chat.completions.requests[1]["messages"][0]["content"]
        self.assertNotIn('"profile"', review_prompt)
        self.assertNotIn('"task_type"', review_prompt)
        self.assertIn('"q":"Which country has the highest finalWorth, and why?"', review_prompt)
        self.assertIn('"ty":"string"', review_prompt)
        self.assertNotIn('"t":{"table_1"', review_prompt)
        self.assertNotIn('"v"', review_prompt)
        self.assertNotIn('"act"', review_prompt)
        self.assertNotIn('"obs"', review_prompt)
        self.assertIn('"ev"', review_prompt)
        self.assertIn('"op":"sql"', review_prompt)
        self.assertIn("United States", review_prompt)

    def test_analyze_numeric_scalar_action_can_finalize_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "q": (
                                'SELECT COUNT(*) AS "answer_1__a1" '
                                'FROM "table_1";'
                            )
                        }
                    ),
                ]
            )
            scalar_table = {
                "n": 1,
                "cols": ["answer_1__a1"],
                "rows": [{"answer_1__a1": 2}],
            }

            with patch(
                "clover.runtime.table_reasoning.pipeline.execute_execution_plan",
                return_value=ExecutionResult(
                    ok=True,
                    answer={"answer_1__a1": scalar_table},
                    outputs={"answer_1__a1": scalar_table},
                    collector_outputs={"answer": {"answer_1__a1": scalar_table}},
                    traces=[],
                    output_summaries={},
                ),
            ):
                result = run_table_reasoning_system(
                    case_specs=[
                        _case_spec(
                            "case_1",
                            Path(tmpdir),
                            table_path,
                            "How many rows are present?",
                            "number",
                            profile="analyze",
                        ),
                    ],
                    remote_config={"api_type": "chat_completions", "model": "fake-model"},
                    remote_batch_size=8,
                    client=client,
                )

        self.assertEqual(result.case_results[0].answer, 2)
        self.assertEqual(result.profile["counters"]["supervisor_decompose_calls"], 1)
        self.assertNotIn("supervisor_synthesis_calls", result.profile["counters"])
        self.assertEqual(result.profile["counters"]["action_group_static_answer_hits"], 1)
        self.assertEqual(
            result.profile["counters"]["action_group_static_answer_number_hits"],
            1,
        )
        self.assertEqual(len(client.chat.completions.requests), 1)

    def test_analyze_string_scalar_action_can_finalize_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "q": (
                                'SELECT "country" AS "answer_1__a1" '
                                'FROM "table_1" LIMIT 1;'
                            )
                        }
                    ),
                ]
            )
            scalar_table = {
                "n": 1,
                "cols": ["answer_1__a1"],
                "rows": [{"answer_1__a1": "France"}],
            }

            with patch(
                "clover.runtime.table_reasoning.pipeline.execute_execution_plan",
                return_value=ExecutionResult(
                    ok=True,
                    answer={"answer_1__a1": scalar_table},
                    outputs={"answer_1__a1": scalar_table},
                    collector_outputs={"answer": {"answer_1__a1": scalar_table}},
                    traces=[],
                    output_summaries={},
                ),
            ):
                result = run_table_reasoning_system(
                    case_specs=[
                        _case_spec(
                            "case_1",
                            Path(tmpdir),
                            table_path,
                            "Which country appears first?",
                            "string",
                            profile="analyze",
                        ),
                    ],
                    remote_config={"api_type": "chat_completions", "model": "fake-model"},
                    remote_batch_size=8,
                    client=client,
                )

        self.assertEqual(result.case_results[0].answer, "France")
        self.assertEqual(result.profile["counters"]["supervisor_decompose_calls"], 1)
        self.assertNotIn("supervisor_synthesis_calls", result.profile["counters"])
        self.assertEqual(result.profile["counters"]["action_group_static_answer_hits"], 1)
        self.assertEqual(
            result.profile["counters"]["action_group_static_answer_text_hits"],
            1,
        )
        self.assertEqual(len(client.chat.completions.requests), 1)

    def test_analyze_average_row_aggregate_action_can_finalize_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table_path = root / "table.csv"
            table_path.write_text(
                "commodity,2002 - 03,2003 - 04,2004 - 05,2005 - 06\n"
                "wheat,2692,5636,4320,5905\n",
                encoding="utf-8",
            )
            client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "q": (
                                'SELECT AVG(CAST("2002 - 03" AS REAL)) AS avg_2002_03, '
                                'AVG(CAST("2003 - 04" AS REAL)) AS avg_2003_04, '
                                'AVG(CAST("2004 - 05" AS REAL)) AS avg_2004_05, '
                                'AVG(CAST("2005 - 06" AS REAL)) AS avg_2005_06 '
                                'FROM "table_1" WHERE "commodity" = \'Wheat\';'
                            )
                        }
                    ),
                ]
            )

            result = run_table_reasoning_system(
                case_specs=[
                    _case_spec(
                        "case_1",
                        root,
                        table_path,
                        "What is the average value of wheat production from 2002-03 to 2005-06?",
                        "number",
                        profile="analyze",
                    ),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=8,
                validation_mode="remote_supervisor",
                client=client,
            )

        self.assertEqual(result.case_results[0].answer, 4638.25)
        self.assertNotIn("supervisor_synthesis_calls", result.profile["counters"])
        self.assertEqual(result.profile["counters"]["action_group_static_answer_hits"], 1)
        self.assertEqual(
            result.profile["counters"]["action_group_static_answer_number_hits"],
            1,
        )
        self.assertEqual(len(client.chat.completions.requests), 1)

    def test_analyze_average_summed_range_action_can_finalize_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table_path = root / "table.csv"
            table_path.write_text(
                "commodity,2002 - 03,2003 - 04,2004 - 05,2005 - 06\n"
                "wheat,2692,5636,4320,5905\n",
                encoding="utf-8",
            )
            client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "q": (
                                'SELECT AVG(("2002 - 03" + "2003 - 04" + '
                                '"2004 - 05" + "2005 - 06") * 1.0) '
                                'AS avg_wheat_production FROM "table_1" '
                                'WHERE "commodity" = \'Wheat\';'
                            )
                        }
                    ),
                ]
            )

            result = run_table_reasoning_system(
                case_specs=[
                    _case_spec(
                        "case_1",
                        root,
                        table_path,
                        "What is the average value of wheat production from 2002-03 to 2005-06?",
                        "number",
                        profile="analyze",
                    ),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=8,
                validation_mode="remote_supervisor",
                client=client,
            )

        self.assertEqual(result.case_results[0].answer, 4638.25)
        self.assertNotIn("supervisor_synthesis_calls", result.profile["counters"])
        self.assertEqual(result.profile["counters"]["action_group_static_answer_hits"], 1)
        self.assertEqual(
            result.profile["counters"]["action_group_static_answer_number_hits"],
            1,
        )
        self.assertEqual(len(client.chat.completions.requests), 1)

    def test_analyze_target_column_action_can_finalize_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table_path = root / "table.csv"
            rows = [
                "driver,points,laps",
                "kasey kahne,185,334",
            ]
            rows.extend(f"driver {index},1,334" for index in range(2, 25))
            rows.append("brian vickers,34,24")
            table_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "q": (
                                'SELECT "driver", "points", "laps", '
                                'CAST("points" AS REAL) / NULLIF("laps", 0) '
                                'AS "points_per_lap" FROM "table_1" '
                                'ORDER BY "points_per_lap" DESC LIMIT 1;'
                            )
                        }
                    ),
                ]
            )
            spec = _case_spec(
                "case_1",
                root,
                table_path,
                "Which driver has the highest Points Per Lap?",
                "string",
                profile="analyze",
            )
            spec.metadata["dsl_builder"] = {
                "diagnostics": {"target_column": "driver"},
            }

            result = run_table_reasoning_system(
                case_specs=[spec],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=8,
                validation_mode="remote_supervisor",
                client=client,
            )

        self.assertEqual(result.case_results[0].answer, "brian vickers")
        self.assertNotIn("supervisor_synthesis_calls", result.profile["counters"])
        self.assertEqual(result.profile["counters"]["action_group_static_answer_hits"], 1)
        self.assertEqual(len(client.chat.completions.requests), 1)

    def test_analyze_single_column_multirow_action_can_finalize_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table_path = root / "table.csv"
            table_path.write_text(
                "nation,gold,silver,bronze\n"
                "benin,1,0,0\n"
                "quebec,1,0,0\n"
                "cape verde,1,0,0\n"
                "ivory coast,1,0,0\n"
                "canada,1,1,2\n",
                encoding="utf-8",
            )
            client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "q": (
                                'SELECT "nation" FROM "table_1" '
                                'WHERE "gold" = 1 AND "silver" = 0 AND "bronze" = 0;'
                            )
                        }
                    ),
                ]
            )
            spec = _case_spec(
                "case_1",
                root,
                table_path,
                "Which nation won 1 gold medal and no silver or bronze medals?",
                "string",
                profile="analyze",
            )

            result = run_table_reasoning_system(
                case_specs=[spec],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=8,
                validation_mode="remote_supervisor",
                client=client,
            )

        self.assertEqual(
            result.case_results[0].answer,
            "benin, quebec, cape verde, ivory coast",
        )
        self.assertNotIn("supervisor_synthesis_calls", result.profile["counters"])
        self.assertEqual(result.profile["counters"]["action_group_static_answer_hits"], 1)
        self.assertEqual(len(client.chat.completions.requests), 1)

    def test_analyze_multirow_multicolumn_target_action_needs_supervisor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table_path = root / "table.csv"
            table_path.write_text(
                "driver,points,laps\n"
                "kasey kahne,185,334\n"
                "brian vickers,34,24\n",
                encoding="utf-8",
            )
            client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "q": (
                                'SELECT "driver", "points", "laps" '
                                'FROM "table_1";'
                            )
                        }
                    ),
                    json.dumps({"a": "brian vickers"}),
                ]
            )
            spec = _case_spec(
                "case_1",
                root,
                table_path,
                "Which driver has the highest Points Per Lap?",
                "string",
                profile="analyze",
            )
            spec.metadata["dsl_builder"] = {
                "diagnostics": {"target_column": "driver"},
            }

            result = run_table_reasoning_system(
                case_specs=[spec],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=8,
                validation_mode="remote_supervisor",
                client=client,
            )

        self.assertEqual(result.case_results[0].answer, "brian vickers")
        self.assertEqual(result.profile["counters"]["supervisor_synthesis_calls"], 1)
        self.assertNotIn("action_group_static_answer_hits", result.profile["counters"])
        self.assertEqual(len(client.chat.completions.requests), 2)

    def test_analyze_defined_ratio_multicolumn_target_action_can_finalize_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table_path = root / "table.csv"
            rows = [
                "driver,points,laps",
                "kasey kahne,185,334",
            ]
            rows.extend(f"driver {index},1,334" for index in range(2, 25))
            rows.append("brian vickers,34,24")
            table_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "q": (
                                'SELECT "driver", "points", "laps" '
                                'FROM "table_1";'
                            )
                        }
                    ),
                ]
            )
            spec = _case_spec(
                "case_1",
                root,
                table_path,
                (
                    "Points Per Lap is defined as the total points earned by a driver "
                    "divided by the total number of laps completed. Which driver has "
                    "the highest Points Per Lap?"
                ),
                "string",
                profile="analyze",
            )
            spec.metadata["dsl_builder"] = {
                "diagnostics": {"target_column": "driver"},
            }

            result = run_table_reasoning_system(
                case_specs=[spec],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=8,
                validation_mode="remote_supervisor",
                client=client,
            )

        self.assertEqual(result.case_results[0].answer, "brian vickers")
        self.assertNotIn("supervisor_synthesis_calls", result.profile["counters"])
        self.assertEqual(result.profile["counters"]["action_group_static_answer_hits"], 1)
        self.assertEqual(len(client.chat.completions.requests), 1)

    def test_analyze_profile_runs_static_analyze_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "acts": [
                                {
                                    "op": "analyze",
                                    "kind": "statistical",
                                    "seed": 'SELECT "finalWorth" FROM "table_1";',
                                }
                            ],
                        }
                    ),
                    json.dumps({"a": 150}),
                ]
            )

            result = run_table_reasoning_system(
                case_specs=[
                    _case_spec(
                        "case_1",
                        Path(tmpdir),
                        table_path,
                        "What is the average finalWorth?",
                        "number",
                        profile="analyze",
                    ),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=8,
                client=client,
            )

        self.assertEqual(result.case_results[0].answer, 150)
        self.assertEqual(result.profile["counters"]["action_group_analyze_calls"], 1)
        review_prompt = client.chat.completions.requests[1]["messages"][0]["content"]
        self.assertIn('"op":"analyze"', review_prompt)
        self.assertIn('"kind":"statistical"', review_prompt)
        self.assertIn('"mean":150', review_prompt)

    def test_analyze_profile_review_can_enqueue_more_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "q": [
                                'SELECT "country", "finalWorth" FROM "table_1" '
                                'ORDER BY "finalWorth" DESC LIMIT 2;'
                            ],
                        }
                    ),
                    json.dumps(
                        {
                            "q": (
                                'SELECT "country" AS "answer_1" FROM "table_1" '
                                'ORDER BY "finalWorth" DESC LIMIT 1;'
                            ),
                        }
                    ),
                    json.dumps({"a": "United States"}),
                ]
            )

            result = run_table_reasoning_system(
                case_specs=[
                    _case_spec(
                        "case_1",
                        Path(tmpdir),
                        table_path,
                        "Which country has the highest finalWorth?",
                        "string",
                        profile="analyze",
                    ),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=8,
                client=client,
            )

        self.assertEqual(result.case_results[0].answer, "United States")
        self.assertEqual(result.profile["counters"]["supervisor_decompose_calls"], 1)
        self.assertEqual(result.profile["counters"]["supervisor_synthesis_calls"], 1)
        self.assertEqual(result.profile["counters"]["action_group_static_answer_hits"], 1)
        self.assertNotIn("static_synthesis_calls", result.profile["counters"])
        self.assertEqual(len(client.chat.completions.requests), 2)

    def test_analyze_string_answer_contract_normalizes_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps({"q": 'SELECT "country" FROM "table_1" LIMIT 1;'}),
                    json.dumps({"a": {"country": "France"}}),
                ]
            )

            result = run_table_reasoning_system(
                case_specs=[
                    _case_spec(
                        "case_1",
                        Path(tmpdir),
                        table_path,
                        "Which country appears first?",
                        "string",
                        profile="analyze",
                    ),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=8,
                client=client,
            )

        self.assertEqual(result.case_results[0].answer, "France")
        self.assertNotIn("supervisor_synthesis_calls", result.profile["counters"])
        self.assertEqual(result.profile["counters"]["action_group_static_answer_hits"], 1)
        self.assertEqual(len(client.chat.completions.requests), 1)

    def test_fail_fast_interrupted_independent_answers_are_requeued(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps(
                        [
                            'SELECT "missing" AS "answer_1" FROM "table_1";',
                            _country_sql("answer_2"),
                        ]
                    ),
                    json.dumps(
                        {
                            "op": "sql",
                            "q": 'SELECT COUNT(*) AS "answer_1" FROM "table_1";',
                        }
                    ),
                    json.dumps(
                        {
                            "op": "answer",
                            "a": {"answer_2": "United States"},
                        }
                    ),
                    json.dumps(
                        {
                            "op": "answer",
                            "a": 2,
                        }
                    ),
                ]
            )

            result = run_table_reasoning_system(
                case_specs=[
                    _case_spec("case_1", Path(tmpdir), table_path, "How many rows?", "number"),
                    _case_spec("case_2", Path(tmpdir), table_path, "Country?", "string"),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=2,
                max_parallel_execution_units=1,
                max_retries=1,
                validation_mode="remote_supervisor",
                client=client,
            )

        results = {item.case_id: item for item in result.case_results}
        self.assertEqual(results["case_1"].answer, 2)
        self.assertEqual(results["case_1"].retry_count, 1)
        self.assertEqual(results["case_2"].answer, "United States")
        self.assertEqual(results["case_2"].retry_count, 0)
        self.assertEqual(result.profile["counters"]["merged_plan_count"], 3)


def _write_people_table(tmpdir: Path) -> Path:
    tmpdir.mkdir(parents=True, exist_ok=True)
    table_path = tmpdir / "table.csv"
    table_path.write_text(
        "selfMade,country,finalWorth\n"
        "False,France,100\n"
        "True,United States,200\n",
        encoding="utf-8",
    )
    return table_path


def _case_spec(
    case_id: str,
    base_dir: Path,
    table_path: Path,
    question: str,
    answer_type: str,
    profile: str = "query",
) -> TableReasoningCaseSpec:
    return TableReasoningCaseSpec(
        case_id=case_id,
        base_dir=base_dir,
        task_dsl={
            "task_type": f"table_reasoning.{profile}",
            "profile": profile,
            "question": question,
            "sources": [{"type": "table", "file": str(table_path)}],
            "answer": {"name": "answer", "type": answer_type},
        },
    )


def _self_made_sql(answer_key: str) -> str:
    return (
        f'SELECT "selfMade" AS "{answer_key}" FROM "table_1" '
        'ORDER BY "finalWorth" DESC LIMIT 1;'
    )


def _country_sql(answer_key: str) -> str:
    return (
        f'SELECT "country" AS "{answer_key}" FROM "table_1" '
        'ORDER BY "finalWorth" DESC LIMIT 1;'
    )


class _StatefulChatClient:
    def __init__(
        self,
        output_texts: list[str],
        *,
        usages: list[dict[str, int]] | None = None,
    ) -> None:
        self.chat = SimpleNamespace(
            completions=_StatefulChatCompletions(output_texts, usages=usages),
        )


class FilterEvidenceExtractionTest(unittest.TestCase):
    """Unit tests for _extract_filter_evidence_from_traces and _sql_result_is_empty."""

    def test_extract_returns_none_for_empty_traces(self) -> None:
        from clover.runtime.table_reasoning.pipeline import (
            _extract_filter_evidence_from_traces,
        )
        self.assertIsNone(_extract_filter_evidence_from_traces(None))
        self.assertIsNone(_extract_filter_evidence_from_traces([]))

    def test_extract_returns_none_when_no_agent_loop(self) -> None:
        from clover.runtime.table_reasoning.pipeline import (
            _extract_filter_evidence_from_traces,
        )
        traces = [{"op": "Scan", "output": "T0"}]
        self.assertIsNone(_extract_filter_evidence_from_traces(traces))

    def test_extract_returns_none_when_no_feedback(self) -> None:
        from clover.runtime.table_reasoning.pipeline import (
            _extract_filter_evidence_from_traces,
        )
        traces = [
            {
                "op": "Filter",
                "agent_loop": {
                    "trigger": "fast_path_empty_output",
                    "iterations": 2,
                    "steps": [
                        {"iteration": 1, "observation_type": "invalid_action_json"},
                    ],
                },
            }
        ]
        self.assertIsNone(_extract_filter_evidence_from_traces(traces))

    def test_extract_returns_column_values_from_feedback(self) -> None:
        from clover.runtime.table_reasoning.pipeline import (
            _extract_filter_evidence_from_traces,
        )
        traces = [
            {
                "op": "Filter",
                "agent_loop": {
                    "trigger": "fast_path_empty_output",
                    "iterations": 2,
                    "steps": [
                        {
                            "iteration": 1,
                            "observation_type": "python_error",
                            "feedback": {
                                "column_values": {
                                    "Film": [
                                        {"value": "Kodachrome (35mm)", "count": 3},
                                        {"value": "Kodachrome Professional (sheets)", "count": 1},
                                    ]
                                }
                            },
                        },
                    ],
                },
            }
        ]
        result = _extract_filter_evidence_from_traces(traces)
        self.assertIsNotNone(result)
        self.assertIn("column_values", result)
        self.assertIn("Film", result["column_values"])
        self.assertEqual(len(result["column_values"]["Film"]), 2)

    def test_extract_caps_to_top_8_values_per_column(self) -> None:
        from clover.runtime.table_reasoning.pipeline import (
            _extract_filter_evidence_from_traces,
        )
        many_values = [
            {"value": f"v{i}", "count": 10 - i} for i in range(15)
        ]
        traces = [
            {
                "agent_loop": {
                    "trigger": "fast_path_empty_output",
                    "iterations": 1,
                    "steps": [
                        {
                            "feedback": {"column_values": {"col": many_values}},
                        },
                    ],
                },
            }
        ]
        result = _extract_filter_evidence_from_traces(traces)
        self.assertIsNotNone(result)
        self.assertEqual(len(result["column_values"]["col"]), 8)

    def test_sql_result_is_empty_false_for_non_empty_frame(self) -> None:
        import pandas as pd
        from clover.runtime.table_reasoning.pipeline import (
            _SqlActionExecution,
            _sql_result_is_empty,
        )
        result = _SqlActionExecution(
            ok=True,
            value={"rows": [{"a": 1}], "n": 1},
            frame=pd.DataFrame({"a": [1]}),
        )
        self.assertFalse(_sql_result_is_empty(result))

    def test_sql_result_is_empty_true_for_empty_frame(self) -> None:
        import pandas as pd
        from clover.runtime.table_reasoning.pipeline import (
            _SqlActionExecution,
            _sql_result_is_empty,
        )
        result = _SqlActionExecution(
            ok=True,
            value={"rows": [], "n": 0},
            frame=pd.DataFrame({"a": []}),
        )
        self.assertTrue(_sql_result_is_empty(result))

    def test_sql_result_is_empty_false_for_failed_execution(self) -> None:
        from clover.runtime.table_reasoning.pipeline import (
            _SqlActionExecution,
            _sql_result_is_empty,
        )
        result = _SqlActionExecution(
            ok=False,
            error={"type": "ExecutionFailed", "message": "err"},
        )
        self.assertFalse(_sql_result_is_empty(result))


class TraceStepFeedbackTest(unittest.TestCase):
    """Unit tests for _trace_step preserving observation.feedback."""

    def test_trace_step_preserves_feedback_with_column_values(self) -> None:
        from clover.executor.agents.loop import _trace_step
        observation = {
            "type": "python_error",
            "ok": False,
            "error": {"message": "empty"},
            "feedback": {
                "column_values": {
                    "Award": [
                        {"value": "Golden Globe Awards, 1972", "count": 1},
                    ]
                },
                "hint": "Empty output means the predicate matched nothing.",
            },
        }
        step = _trace_step(
            iteration=0,
            action=None,
            observation=observation,
            response_id="resp_1",
            prompt_kind="table_reasoning_empty_filter_repair",
        )
        self.assertIn("feedback", step)
        self.assertIn("column_values", step["feedback"])
        self.assertIn("Award", step["feedback"]["column_values"])

    def test_trace_step_omits_feedback_when_absent(self) -> None:
        from clover.executor.agents.loop import _trace_step
        observation = {
            "type": "invalid_action_json",
            "ok": False,
            "error": {"message": "bad json"},
        }
        step = _trace_step(
            iteration=0,
            action=None,
            observation=observation,
            response_id="resp_1",
            prompt_kind="table_reasoning_empty_filter_repair",
        )
        self.assertNotIn("feedback", step)


class _StatefulChatCompletions:
    def __init__(
        self,
        output_texts: list[str],
        *,
        usages: list[dict[str, int]] | None = None,
    ) -> None:
        self._output_texts = output_texts
        self._usages = usages or []
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> "_FakeChatResponse":
        self.requests.append(
            {
                **kwargs,
                "messages": [dict(message) for message in kwargs["messages"]],
            }
        )
        index = len(self.requests) - 1
        usage = self._usages[index] if index < len(self._usages) else None
        return _FakeChatResponse(self._output_texts[index], usage=usage)


class _BlockingPrefetchChatClient:
    def __init__(self, output_texts: list[str]) -> None:
        self.chat = SimpleNamespace(
            completions=_BlockingPrefetchChatCompletions(output_texts),
        )


class _BlockingPrefetchChatCompletions:
    def __init__(self, output_texts: list[str]) -> None:
        self._output_texts = output_texts
        self.requests: list[dict[str, object]] = []
        self._lock = threading.Lock()
        self._second_started = threading.Event()
        self.first_wait_saw_second = False

    def create(self, **kwargs: object) -> "_FakeChatResponse":
        with self._lock:
            self.requests.append(
                {
                    **kwargs,
                    "messages": [dict(message) for message in kwargs["messages"]],
                }
            )
            index = len(self.requests) - 1
        if index == 0:
            self.first_wait_saw_second = self._second_started.wait(timeout=2.0)
        elif index == 1:
            self._second_started.set()
        return _FakeChatResponse(self._output_texts[index])


class _FakeChatResponse:
    id = "table_runtime_fake"

    def __init__(
        self,
        output_text: str,
        *,
        usage: dict[str, int] | None = None,
    ) -> None:
        self.usage = usage
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(content=output_text)),
        ]

    def model_dump(self, mode: str) -> dict[str, object]:
        payload: dict[str, object] = {"id": self.id, "mode": mode}
        if self.usage is not None:
            payload["usage"] = dict(self.usage)
        return payload


if __name__ == "__main__":
    unittest.main()
