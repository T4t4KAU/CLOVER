from __future__ import annotations

import json
import threading
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from clover.executor import ExecutionResult
from clover.executor.slm_dispatcher import LocalSlmSequenceDispatcher
from clover.optimizer import SqlParseError
from clover.runtime import TableReasoningCaseSpec, run_table_reasoning_system
from clover.runtime.table_reasoning import pipeline as table_pipeline
from clover.runtime.table_reasoning.pipeline import (
    _load_json_object,
    _normalize_number_answer,
)


class TableReasoningRuntimeTest(unittest.TestCase):
    def test_table_command_parser_uses_first_json_object(self) -> None:
        payload = _load_json_object(
            '{"op":"sql","q":"SELECT COUNT(*) FROM table_1"}\n' '{"op":"answer","a":0}'
        )

        self.assertEqual(
            payload,
            {"op": "sql", "q": "SELECT COUNT(*) FROM table_1"},
        )

    def test_table_command_parser_ignores_reasoning_json_before_final_answer(
        self,
    ) -> None:
        payload = _load_json_object(
            '<think>Example: {"sql": "SELECT ..."} is only a draft.</think>\n'
            '{"sql": "SELECT COUNT(*) AS answer_1 FROM table_1"}'
        )

        self.assertEqual(
            payload,
            {"sql": "SELECT COUNT(*) AS answer_1 FROM table_1"},
        )

    def test_table_command_parser_recovers_last_sql_when_reasoning_is_truncated(
        self,
    ) -> None:
        payload = _load_json_object(
            "<think>I found the query but may run out of budget.\n"
            "```sql\n"
            'SELECT "name" AS "answer_1" FROM "table_1" LIMIT 1;\n'
            "```\n"
            "Now I would output JSON"
        )

        self.assertEqual(
            payload,
            {"sql": 'SELECT "name" AS "answer_1" FROM "table_1" LIMIT 1'},
        )

    def test_table_command_parser_recovers_fenced_json_sql_without_outer_json(
        self,
    ) -> None:
        payload = _load_json_object(
            "Reasoning text.\n"
            "```json\n"
            '{"sql": "SELECT COUNT(*) AS answer_1 FROM table_1;"}\n'
            "```\n"
            "More text."
        )

        self.assertEqual(
            payload,
            {"sql": "SELECT COUNT(*) AS answer_1 FROM table_1;"},
        )

    def test_number_normalizer_rounds_approximate_counts(self) -> None:
        self.assertEqual(
            _normalize_number_answer(
                10115086.049999999,
                question=(
                    "Approximately how many passengers would the airport handle?"
                ),
            ),
            10115086,
        )

    def test_number_normalizer_preserves_non_approximate_decimals(self) -> None:
        self.assertEqual(
            _normalize_number_answer(
                48.25,
                question="What is the average annual increase in points?",
            ),
            48.25,
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

    def test_table_action_parser_rejects_removed_inspect_action(self) -> None:
        with self.assertRaises(SqlParseError):
            table_pipeline._normalize_table_actions(  # noqa: SLF001
                {
                    "op": "inspect",
                    "q": "find trend evidence",
                    "seed": 'SELECT "year" FROM "table_1";',
                }
            )

    def test_table_direct_probe_uses_generic_feature_flag_only(self) -> None:
        self.assertTrue(
            table_pipeline._table_direct_probe_enabled(  # noqa: SLF001
                {"enable_table_direct_probe": True}
            )
        )
        self.assertFalse(
            table_pipeline._table_direct_probe_enabled(  # noqa: SLF001
                {"table_direct_probe_mode": "off"}
            )
        )

    def test_table_action_parser_rejects_batch_protocol_keys(self) -> None:
        for payload in (
            {"sqls": ['SELECT "x" FROM "table_1";']},
            {
                "questions": ["one"],
                "answers": [{"name": "answer_1", "type": "string"}],
                "sql": 'SELECT "x" FROM "table_1";',
            },
            {"acts": [{"op": "sql", "sqls": ['SELECT "x" FROM "table_1";']}]},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(SqlParseError):
                    table_pipeline._normalize_table_actions(payload)  # noqa: SLF001

    def test_remote_decompose_rejects_model_batch_protocol(self) -> None:
        job = table_pipeline._RemoteDecomposeJob(  # noqa: SLF001
            batch=[],
            remote_dsl={
                "task_type": "table_reasoning.query",
                "question": "What is x?",
                "answer": {"name": "answer_1", "type": "string"},
            },
        )

        with self.assertRaises(SqlParseError):
            table_pipeline._parse_remote_decompose_output(  # noqa: SLF001
                json.dumps(
                    {
                        "questions": ["What is x?"],
                        "answers": [{"name": "answer_1", "type": "string"}],
                        "sql": 'SELECT "x" AS "answer_1" FROM "table_1";',
                    }
                ),
                job,
            )

    def test_second_repair_requires_changed_failure_signature(self) -> None:
        stalled = SimpleNamespace(memory=[{"sig": "same"}, {"sig": "same"}])
        changed = SimpleNamespace(memory=[{"sig": "first"}, {"sig": "second"}])

        self.assertFalse(
            table_pipeline._latest_repair_evidence_changed(stalled)  # noqa: SLF001
        )
        self.assertTrue(
            table_pipeline._latest_repair_evidence_changed(changed)  # noqa: SLF001
        )

    def test_difference_answers_are_normalized_to_absolute_magnitude(self) -> None:
        task = SimpleNamespace(
            answer_type="number",
            question="What is the difference between the two scores?",
        )

        self.assertEqual(
            table_pipeline._normalize_answer_for_task(task, -18),  # noqa: SLF001
            18,
        )

    def test_single_supervisor_answer_dict_can_use_generic_answer_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table_path = _write_people_table(root)
            task = next(
                iter(
                    table_pipeline._build_task_items(  # noqa: SLF001
                        [
                            _case_spec(
                                "case_1",
                                root,
                                table_path,
                                "Which country appears first?",
                                "string",
                            )
                        ]
                    ).values()
                )
            )
            item = table_pipeline.LogicDagItem(
                task=task,
                command_output="",
                output_type="logic_dag",
                logic_dag={},
            )
            final_results = []
            finalized = set()

            table_pipeline._finalize_batch_from_answer(  # noqa: SLF001
                batch=[item],
                answer={"answer": "France"},
                final_results=final_results,
                finalized=finalized,
            )

        self.assertEqual(final_results[0].answer, "France")
        self.assertEqual(final_results[0].answer_key, task.answer_key)
        self.assertEqual(final_results[0].metadata["final_answer_source"], "synthesis")

    def test_safe_edge_local_review_finalizes_grounded_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table_path = _write_people_table(root)
            task = next(
                iter(
                    table_pipeline._build_task_items(  # noqa: SLF001
                        [
                            _case_spec(
                                "case_1",
                                root,
                                table_path,
                                "Which country appears first?",
                                "string",
                            )
                        ]
                    ).values()
                )
            )
            local_config = {
                "api_type": "chat_completions",
                "model": "edge-model",
                "edge_review_mode": "safe",
                "enable_edge_repair": False,
                "enable_terminal_edge_review": True,
            }
            local_client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "route": "normalize",
                            "a": "France",
                            "support": ["e0"],
                            "operation": "identity",
                        }
                    )
                ]
            )
            dispatcher = LocalSlmSequenceDispatcher(
                slm_config=local_config,
                client=local_client,
                max_parallel_sequences=1,
                max_pending_sequences=4,
                slm_scheduler="fifo",
            )
            final_results = []
            finalized = set()
            profiler = table_pipeline.PipelineProfiler()
            try:
                hit = table_pipeline._try_finalize_edge_local_review(  # noqa: SLF001
                    task=task,
                    evidence={
                        "n": 1,
                        "cols": ["country", "year"],
                        "rows": [{"country": "France", "year": 2020}],
                    },
                    scope="format_answer",
                    local_slm_config=local_config,
                    local_slm_dispatcher=dispatcher,
                    final_results=final_results,
                    finalized=finalized,
                    profiler=profiler,
                )
            finally:
                dispatcher.close()

        self.assertTrue(hit)
        self.assertEqual(final_results[0].answer, "France")
        self.assertEqual(
            final_results[0].metadata["final_answer_source"],
            "edge_local_review",
        )
        self.assertEqual(profiler.counters["edge_local_review_hits"], 1)
        self.assertEqual(profiler.counters["local_slm_calls"], 1)

    def test_proactive_edge_review_can_select_one_field_before_static_finalization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table_path = _write_people_table(root)
            task = next(
                iter(
                    table_pipeline._build_task_items(  # noqa: SLF001
                        [
                            _case_spec(
                                "case_1",
                                root,
                                table_path,
                                "Which country appears first?",
                                "string",
                            )
                        ]
                    ).values()
                )
            )
            local_config = {
                "api_type": "chat_completions",
                "model": "edge-model",
                "edge_review_mode": "safe",
                "edge_review_proactive": True,
                "enable_terminal_edge_review": True,
            }
            local_client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "route": "normalize",
                            "a": "France",
                            "support": ["e0"],
                            "operation": "identity",
                        }
                    )
                ]
            )
            dispatcher = LocalSlmSequenceDispatcher(
                slm_config=local_config,
                client=local_client,
                max_parallel_sequences=1,
                max_pending_sequences=4,
                slm_scheduler="fifo",
            )
            final_results = []
            finalized = set()
            profiler = table_pipeline.PipelineProfiler()
            evidence = {
                "n": 1,
                "cols": ["country", "year"],
                "rows": [{"country": "France", "year": 2020}],
            }
            try:
                selected = table_pipeline._proactive_edge_review_opportunity(  # noqa: SLF001
                    question=task.question,
                    answer_type=task.answer_type,
                    evidence=evidence,
                    local_slm_config=local_config,
                    profiler=profiler,
                )
                hit = table_pipeline._try_finalize_edge_local_review(  # noqa: SLF001
                    task=task,
                    evidence=evidence,
                    scope="format_answer",
                    local_slm_config=local_config,
                    local_slm_dispatcher=dispatcher,
                    final_results=final_results,
                    finalized=finalized,
                    profiler=profiler,
                    proactive=True,
                )
            finally:
                dispatcher.close()

        self.assertTrue(selected)
        self.assertTrue(hit)
        self.assertEqual(final_results[0].answer, "France")
        self.assertEqual(profiler.counters["edge_review_proactive_calls"], 1)
        self.assertEqual(
            profiler.counters["edge_review_accepted_field_selection"],
            1,
        )

    def test_invalid_proactive_review_falls_back_to_static_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table_path = _write_people_table(root)
            task = next(
                iter(
                    table_pipeline._build_task_items(  # noqa: SLF001
                        [
                            _case_spec(
                                "case_1",
                                root,
                                table_path,
                                "Which countries are listed?",
                                "list[string]",
                            )
                        ]
                    ).values()
                )
            )
            local_config = {
                "api_type": "chat_completions",
                "model": "edge-model",
                "edge_review_mode": "safe",
                "edge_review_proactive": True,
                "enable_terminal_edge_review": True,
            }
            local_client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "route": "normalize",
                            "a": ["Germany"],
                            "support": ["e0"],
                            "operation": "identity",
                        }
                    )
                ]
            )
            dispatcher = LocalSlmSequenceDispatcher(
                slm_config=local_config,
                client=local_client,
                max_parallel_sequences=1,
                max_pending_sequences=4,
                slm_scheduler="fifo",
            )
            batch = [
                table_pipeline.LogicDagItem(
                    task=task,
                    command_output="SELECT country",
                    output_type="sql",
                    logic_dag={},
                )
            ]
            execution_result = ExecutionResult(
                ok=True,
                answer={task.answer_key: ["France", "Spain"]},
                outputs={},
                traces=[],
                output_summaries={},
            )
            final_results = []
            finalized = set()
            profiler = table_pipeline.PipelineProfiler()
            try:
                remaining = table_pipeline._run_static_synthesis_review(  # noqa: SLF001
                    batch=batch,
                    execution_result=execution_result,
                    final_results=final_results,
                    finalized=finalized,
                    profiler=profiler,
                    fail_unfinalized=False,
                    local_slm_config=local_config,
                    local_slm_dispatcher=dispatcher,
                )
            finally:
                dispatcher.close()

        self.assertEqual(remaining, [])
        self.assertEqual(final_results[0].answer, ["France", "Spain"])
        self.assertEqual(
            final_results[0].metadata["final_answer_source"],
            "format_answer",
        )
        self.assertEqual(profiler.counters["edge_review_proactive_calls"], 1)
        self.assertEqual(profiler.counters["static_final_answer_hits"], 1)

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

    def test_processes_same_table_questions_one_by_one_without_remote_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps({"sql": _self_made_sql("answer_1")}),
                    json.dumps({"sql": _country_sql("answer_2")}),
                ],
                usages=[
                    {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
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
                remote_concurrency=1,
                client=client,
                profile_baseline=True,
            )

        self.assertEqual(
            {item.case_id: item.answer for item in result.case_results},
            {"case_1": True, "case_2": "United States"},
        )
        self.assertEqual(result.profile["counters"]["supervisor_decompose_calls"], 2)
        self.assertNotIn("supervisor_synthesis_calls", result.profile["counters"])
        self.assertEqual(result.profile["counters"]["static_synthesis_calls"], 2)
        self.assertEqual(result.profile["counters"]["remote_input_tokens"], 200)
        self.assertEqual(result.profile["counters"]["remote_output_tokens"], 20)
        self.assertEqual(result.profile["counters"]["remote_total_tokens"], 220)
        self.assertEqual(
            result.profile["summary"]["remote_token_usage"],
            {
                "input_tokens": 200,
                "cached_input_tokens": 0,
                "output_tokens": 20,
                "reasoning_tokens": 0,
                "total_tokens": 220,
            },
        )
        self.assertEqual(result.profile["summary"]["validation_mode"], "none")
        self.assertIn("baseline_executor", result.profile["stages"])
        self.assertIn("local_executor_speedup", result.profile["summary"])

    def test_remote_validation_bypasses_synthesis_for_valid_format_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps({"sql": _country_sql("answer_1")}),
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

    def test_table_direct_probe_is_integrated_into_supervisor_synthesis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "answer": "United States",
                            "confidence": "high",
                            "verdict": "agree",
                            "issue": "none",
                            "evidence": "row with finalWorth 200 has country United States",
                            "repair_hint": "",
                        }
                    ),
                    json.dumps({"op": "answer", "a": "United States"}),
                ],
            )

            result = run_table_reasoning_system(
                case_specs=[
                    TableReasoningCaseSpec(
                        case_id="case_1",
                        base_dir=Path(tmpdir),
                        task_dsl={
                            "task_type": "table_reasoning.query",
                            "question": (
                                "What is the country of the person with the highest net worth?"
                            ),
                            "sources": [{"type": "table", "file": str(table_path)}],
                            "answer": {"name": "answer", "type": "string"},
                            "sql": _country_sql("answer_1"),
                        },
                    )
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                local_slm_config={
                    "enable_static_finalization": False,
                    "edge_review_mode": "off",
                    "enable_table_direct_probe": True,
                },
                validation_mode="remote_supervisor",
                client=client,
            )

        self.assertEqual(result.case_results[0].answer, "United States")
        self.assertEqual(result.profile["counters"]["table_direct_probe_calls"], 1)
        self.assertEqual(result.profile["counters"]["supervisor_synthesis_calls"], 1)
        self.assertEqual(result.profile["summary"]["remote_calls"], 2)
        self.assertEqual(len(client.chat.completions.requests), 2)
        probe_prompt = client.chat.completions.requests[0]["messages"][0]["content"]
        synthesis_prompt = client.chat.completions.requests[1]["messages"][0]["content"]
        self.assertIn("integrated semantic probe inside CLOVER for table reasoning", probe_prompt)
        self.assertIn('"direct_probe"', synthesis_prompt)
        self.assertIn('"verdict":"agree"', synthesis_prompt)

    def test_supervisor_reports_single_query_answers_after_unbatched_decompose(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))

            def respond(prompt: str, _index: int) -> dict[str, object]:
                if "Task DSL" in prompt and "Self-made?" in prompt:
                    return {
                        "sql": (
                            'SELECT "selfMade" AS "answer_1" FROM "table_1" '
                            "WHERE \"country\" = 'missing';"
                        )
                    }
                if "Task DSL" in prompt and "Country?" in prompt:
                    return {"sql": _country_sql("answer_2")}
                if "Self-made?" in prompt:
                    return {"op": "answer", "a": True}
                if "Country?" in prompt:
                    return {"op": "answer", "a": "United States"}
                return {"op": "answer", "a": None}

            client = _PromptRoutingChatClient(respond)

            result = run_table_reasoning_system(
                case_specs=[
                    _case_spec("case_1", Path(tmpdir), table_path, "Self-made?", "boolean"),
                    _case_spec("case_2", Path(tmpdir), table_path, "Country?", "string"),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=2,
                remote_concurrency=1,
                max_retries=1,
                validation_mode="remote_supervisor",
                client=client,
            )

        results = {item.case_id: item for item in result.case_results}
        self.assertTrue(results["case_1"].ok)
        self.assertTrue(results["case_2"].ok)
        self.assertEqual(results["case_2"].answer, "United States")
        self.assertGreaterEqual(result.profile["counters"]["supervisor_synthesis_calls"], 1)

    def test_prefetches_supervisor_decompose_batches_before_local_finish(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_table = _write_people_table(root / "first")
            second_table = _write_people_table(root / "second")
            client = _BlockingPrefetchChatClient(
                [
                    json.dumps({"sql": _country_sql("answer_1")}),
                    json.dumps({"sql": _self_made_sql("answer_2")}),
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
                    json.dumps({"sql": _self_made_sql("answer_1")}),
                    json.dumps({"sql": _country_sql("answer_2")}),
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
        second_supervisor_decompose_prompt = client.chat.completions.requests[1]["messages"][0][
            "content"
        ]
        self.assertIn('"sources"', second_supervisor_decompose_prompt)

    def test_passes_local_parallel_limits_to_executor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps({"sql": 'SELECT COUNT(*) AS "answer_1" FROM "table_1";'}),
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
                        {"q": ('SELECT "country" AS "answer_1__a1" ' 'FROM "table_1" LIMIT 1;')}
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
                    "all_edge_routing": {
                        "routed": True,
                        "edge_status": "ok",
                        "static_reference_status": "ok",
                        "agreement": False,
                    },
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
        self.assertEqual(counters["executor_all_edge_routed_nodes"], 1)
        self.assertEqual(counters["executor_all_edge_edge_successes"], 1)
        self.assertEqual(
            counters["executor_all_edge_static_reference_successes"],
            1,
        )
        self.assertEqual(counters["executor_all_edge_comparisons"], 1)
        self.assertEqual(counters["executor_all_edge_disagreements"], 1)
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
                    json.dumps({"sql": 'SELECT "missing" AS "answer_1" FROM "table_1";'}),
                    json.dumps(
                        {
                            "op": "sql",
                            "q": 'SELECT COUNT(*) AS "answer_1" FROM "table_1";',
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
                    json.dumps({"sql": 'SELECT "missing" AS "answer_1" FROM "table_1";'}),
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

    def test_cloud_recovery_ablation_prevents_second_cloud_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps({"sql": 'SELECT "missing" AS "answer_1" FROM "table_1";'}),
                ]
            )

            result = run_table_reasoning_system(
                case_specs=[
                    _case_spec("case_1", Path(tmpdir), table_path, "How many rows?", "number"),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                local_slm_config={
                    "enable_cloud_recovery": False,
                    "edge_review_mode": "off",
                },
                remote_batch_size=1,
                max_retries=1,
                validation_mode="remote_supervisor",
                client=client,
            )

        self.assertEqual(len(client.chat.completions.requests), 1)
        self.assertFalse(result.case_results[0].ok)
        self.assertEqual(result.case_results[0].retry_count, 0)
        self.assertNotIn("supervisor_synthesis_calls", result.profile["counters"])

    def test_cloud_replan_ablation_keeps_cloud_final_synthesis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps({"sql": 'SELECT "missing" AS "answer_1" FROM "table_1";'}),
                    json.dumps({"a": 2}),
                ]
            )

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
                local_slm_config={
                    "enable_cloud_recovery": True,
                    "enable_cloud_replan": False,
                    "enable_cloud_synthesis": True,
                    "edge_review_mode": "off",
                },
                remote_batch_size=1,
                max_retries=1,
                validation_mode="remote_supervisor",
                client=client,
            )

        self.assertEqual(len(client.chat.completions.requests), 2)
        self.assertTrue(result.case_results[0].ok)
        self.assertEqual(result.case_results[0].answer, 2)
        self.assertEqual(result.case_results[0].retry_count, 0)
        self.assertEqual(result.profile["counters"]["supervisor_synthesis_calls"], 1)
        self.assertNotIn("cloud_replan_calls", result.profile["counters"])

    def test_cloud_replan_ablation_rejects_followup_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps({"sql": 'SELECT "missing" AS "answer_1" FROM "table_1";'}),
                    json.dumps(
                        {
                            "op": "sql",
                            "q": 'SELECT COUNT(*) AS "answer_1" FROM "table_1";',
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
                    _case_spec(
                        "case_1",
                        Path(tmpdir),
                        table_path,
                        "How many rows?",
                        "number",
                    ),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                local_slm_config={
                    "enable_cloud_replan": False,
                    "enable_cloud_synthesis": True,
                    "edge_review_mode": "off",
                },
                remote_batch_size=1,
                max_retries=1,
                validation_mode="remote_supervisor",
                client=client,
            )

        self.assertEqual(len(client.chat.completions.requests), 2)
        self.assertFalse(result.case_results[0].ok)
        self.assertEqual(result.case_results[0].retry_count, 0)
        self.assertEqual(
            result.case_results[0].error["type"],
            "CloudReplanDisabled",
        )
        self.assertEqual(result.profile["counters"]["cloud_replan_blocked"], 1)
        self.assertNotIn("cloud_replan_calls", result.profile["counters"])

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
                    json.dumps({"q": ('SELECT COUNT(*) AS "answer_1__a1" ' 'FROM "table_1";')}),
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

    def test_cloud_finalize_ablation_sends_scalar_answer_to_cloud(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps({"q": ('SELECT COUNT(*) AS "answer_1__a1" ' 'FROM "table_1";')}),
                    json.dumps({"a": 2}),
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
                    local_slm_config={
                        "enable_static_finalization": False,
                        "edge_review_mode": "off",
                    },
                    remote_batch_size=8,
                    validation_mode="remote_supervisor",
                    client=client,
                )

        self.assertEqual(result.case_results[0].answer, 2)
        self.assertEqual(result.profile["counters"]["supervisor_synthesis_calls"], 1)
        self.assertNotIn("action_group_static_answer_hits", result.profile["counters"])
        self.assertEqual(len(client.chat.completions.requests), 2)

    def test_analyze_string_scalar_action_can_finalize_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = _write_people_table(Path(tmpdir))
            client = _StatefulChatClient(
                [
                    json.dumps(
                        {"q": ('SELECT "country" AS "answer_1__a1" ' 'FROM "table_1" LIMIT 1;')}
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
                "commodity,2002 - 03,2003 - 04,2004 - 05,2005 - 06\n" "wheat,2692,5636,4320,5905\n",
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
                "commodity,2002 - 03,2003 - 04,2004 - 05,2005 - 06\n" "wheat,2692,5636,4320,5905\n",
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
                                "WHERE \"commodity\" = 'Wheat';"
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
                "driver,points,laps\n" "kasey kahne,185,334\n" "brian vickers,34,24\n",
                encoding="utf-8",
            )
            client = _StatefulChatClient(
                [
                    json.dumps({"q": ('SELECT "driver", "points", "laps" ' 'FROM "table_1";')}),
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
                    json.dumps({"q": ('SELECT "driver", "points", "laps" ' 'FROM "table_1";')}),
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
            repair_calls = 0

            def respond(prompt: str, _index: int) -> dict[str, object]:
                nonlocal repair_calls
                if "Task DSL" in prompt and "How many rows?" in prompt:
                    return {"sql": 'SELECT "missing" AS "answer_1" FROM "table_1";'}
                if "Task DSL" in prompt and "Country?" in prompt:
                    return {"sql": _country_sql("answer_2")}
                if "How many rows?" in prompt:
                    if "SELECT COUNT(*)" in prompt or repair_calls:
                        return {"op": "answer", "a": 2}
                    repair_calls += 1
                    return {
                        "op": "sql",
                        "q": 'SELECT COUNT(*) AS "answer_1" FROM "table_1";',
                    }
                if "Country?" in prompt:
                    return {"op": "answer", "a": "United States"}
                return {"op": "answer", "a": None}

            client = _PromptRoutingChatClient(respond)

            result = run_table_reasoning_system(
                case_specs=[
                    _case_spec("case_1", Path(tmpdir), table_path, "How many rows?", "number"),
                    _case_spec("case_2", Path(tmpdir), table_path, "Country?", "string"),
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=2,
                remote_concurrency=1,
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

    def test_edge_relative_row_repairs_anchor_answer_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table_path = root / "table.csv"
            table_path.write_text(
                "name\nAlpha\nBeta\nGamma\n",
                encoding="utf-8",
            )
            client = _StatefulChatClient(
                [
                    json.dumps(
                        {"q": ('SELECT "name" FROM "table_1" ' "WHERE \"name\" = 'Beta' LIMIT 1")}
                    )
                ]
            )

            result = run_table_reasoning_system(
                case_specs=[
                    _case_spec(
                        "case_1",
                        root,
                        table_path,
                        "Who comes after Beta?",
                        "string",
                        profile="analyze",
                    )
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=1,
                client=client,
            )

        self.assertEqual(result.case_results[0].answer, "Gamma")
        self.assertEqual(
            result.case_results[0].metadata["final_answer_source"],
            "edge_static_relative_row",
        )
        self.assertEqual(
            result.profile["counters"]["action_group_edge_relative_row_hits"],
            1,
        )
        self.assertEqual(len(client.chat.completions.requests), 1)

    def test_edge_relative_row_overrides_wrong_lexical_neighbor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table_path = root / "table.csv"
            table_path.write_text(
                "name\nAlpha\nBeta\nGamma\n",
                encoding="utf-8",
            )
            client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "q": (
                                "SELECT name FROM table_1 "
                                "WHERE name < 'Beta' ORDER BY name ASC LIMIT 1"
                            )
                        }
                    )
                ]
            )

            result = run_table_reasoning_system(
                case_specs=[
                    _case_spec(
                        "case_1",
                        root,
                        table_path,
                        "What play is next after Beta?",
                        "string",
                        profile="analyze",
                    )
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=1,
                client=client,
            )

        self.assertEqual(result.case_results[0].answer, "Gamma")
        self.assertEqual(
            result.case_results[0].metadata["final_answer_source"],
            "edge_static_relative_row",
        )

    def test_edge_relative_row_respects_descending_dates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table_path = root / "table.csv"
            table_path.write_text(
                "Name,From\n"
                "New Coach,1 July 2012\n"
                "Christian Andersen,11 July 2009\n"
                "Anders Theil,7 November 2005\n"
                "Ebbe Skovdahl,11 October 2003\n",
                encoding="utf-8",
            )
            client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "q": (
                                'SELECT "Name" FROM "table_1" '
                                "WHERE \"Name\" = 'Anders Theil' LIMIT 1"
                            )
                        }
                    )
                ]
            )

            result = run_table_reasoning_system(
                case_specs=[
                    _case_spec(
                        "case_1",
                        root,
                        table_path,
                        "Who was the coach immediately after Anders Theil?",
                        "string",
                        profile="analyze",
                    )
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=1,
                client=client,
            )

        self.assertEqual(result.case_results[0].answer, "Christian Andersen")

    def test_static_text_answer_collapses_identical_lookup_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table_path = root / "table.csv"
            table_path.write_text(
                "team,league\nViking,Premier League\nViking,Premier League\n",
                encoding="utf-8",
            )
            client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "q": (
                                'SELECT "league" FROM "table_1" '
                                "WHERE \"team\" = 'Viking'"
                            )
                        }
                    )
                ]
            )

            result = run_table_reasoning_system(
                case_specs=[
                    _case_spec(
                        "case_1",
                        root,
                        table_path,
                        "What league does Viking belong to?",
                        "string",
                        profile="analyze",
                    )
                ],
                remote_config={"api_type": "chat_completions", "model": "fake-model"},
                remote_batch_size=1,
                client=client,
            )

        self.assertEqual(result.case_results[0].answer, "Premier League")
        self.assertEqual(
            result.case_results[0].metadata["final_answer_source"],
            "action_static",
        )

    def test_duplicate_cloud_repair_sql_is_rejected_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table_path = _write_people_table(root)
            sql = 'SELECT "country" FROM "table_1" WHERE "country" = \'Missing\''
            client = _StatefulChatClient(
                [
                    json.dumps({"q": sql}),
                    json.dumps({"acts": [{"op": "sql", "q": sql}]}),
                ]
            )
            empty_table = {
                "n": 0,
                "cols": ["country"],
                "rows": [],
            }

            with patch(
                "clover.runtime.table_reasoning.pipeline.execute_execution_plan",
                return_value=ExecutionResult(
                    ok=True,
                    answer={"answer_1__a1": empty_table},
                    outputs={"answer_1__a1": empty_table},
                    collector_outputs={"answer": {"answer_1__a1": empty_table}},
                    traces=[],
                    output_summaries={},
                ),
            ):
                result = run_table_reasoning_system(
                    case_specs=[
                        _case_spec(
                            "case_1",
                            root,
                            table_path,
                            "Which country is missing?",
                            "string",
                            profile="analyze",
                        )
                    ],
                    remote_config={
                        "api_type": "chat_completions",
                        "model": "fake-model",
                    },
                    remote_batch_size=1,
                    max_retries=2,
                    client=client,
                )

        self.assertFalse(result.case_results[0].ok)
        self.assertEqual(
            result.case_results[0].error["type"],
            "DuplicateRepairSQL",
        )
        self.assertEqual(result.case_results[0].retry_count, 0)
        self.assertEqual(len(client.chat.completions.requests), 2)

    def test_second_cloud_repair_requires_new_compact_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table_path = _write_people_table(root)
            client = _StatefulChatClient(
                [
                    json.dumps(
                        {"q": ('SELECT "country" FROM "table_1" ' "WHERE \"country\" = 'Missing'")}
                    ),
                    json.dumps(
                        {
                            "acts": [
                                {
                                    "op": "sql",
                                    "q": (
                                        'SELECT "country" FROM "table_1" '
                                        "WHERE \"country\" LIKE '%Missing%'"
                                    ),
                                }
                            ]
                        }
                    ),
                    json.dumps(
                        {
                            "acts": [
                                {
                                    "op": "sql",
                                    "q": (
                                        'SELECT "country" FROM "table_1" '
                                        'ORDER BY "finalWorth" DESC LIMIT 1'
                                    ),
                                }
                            ]
                        }
                    ),
                ]
            )
            empty_table = {"n": 0, "cols": ["country"], "rows": []}
            answer_table = {
                "n": 1,
                "cols": ["country"],
                "rows": [{"country": "United States"}],
            }
            first_trace = [
                {
                    "op": "Filter",
                    "verification_verdict": {
                        "route": "edge_repair",
                        "reason": "predicate_not_found",
                        "evidence": {
                            "node": {"id": "N1", "op": "Filter"},
                            "input_rows": 2,
                            "output_rows": 0,
                            "mismatch": {
                                "sql": "\"country\" = 'Missing'",
                                "roots": [
                                    {
                                        "col": "country",
                                        "sql_lit": ["Missing"],
                                        "actual": ["France", "United States"],
                                        "mismatch": "not_found",
                                    }
                                ],
                            },
                        },
                    },
                }
            ]
            second_trace = [
                {
                    "op": "Filter",
                    "verification_verdict": {
                        "route": "cloud_replan",
                        "reason": "predicate_wrong_column",
                        "evidence": {
                            "node": {"id": "N1", "op": "Filter"},
                            "input_rows": 2,
                            "output_rows": 0,
                            "mismatch": {
                                "sql": "\"country\" LIKE '%Missing%'",
                                "roots": [
                                    {
                                        "col": "country",
                                        "sql_lit": ["Missing"],
                                        "actual": ["France", "United States"],
                                        "mismatch": "wrong_column",
                                    }
                                ],
                            },
                        },
                    },
                }
            ]
            executions = [
                ExecutionResult(
                    ok=True,
                    answer={"answer_1__a1": empty_table},
                    outputs={"answer_1__a1": empty_table},
                    collector_outputs={"answer": {"answer_1__a1": empty_table}},
                    traces=first_trace,
                    output_summaries={},
                ),
                ExecutionResult(
                    ok=True,
                    answer={"answer_1__a1": empty_table},
                    outputs={"answer_1__a1": empty_table},
                    collector_outputs={"answer": {"answer_1__a1": empty_table}},
                    traces=second_trace,
                    output_summaries={},
                ),
                ExecutionResult(
                    ok=True,
                    answer={"answer_1__a1": answer_table},
                    outputs={"answer_1__a1": answer_table},
                    collector_outputs={"answer": {"answer_1__a1": answer_table}},
                    traces=[],
                    output_summaries={},
                ),
            ]

            with patch(
                "clover.runtime.table_reasoning.pipeline.execute_execution_plan",
                side_effect=executions,
            ):
                result = run_table_reasoning_system(
                    case_specs=[
                        _case_spec(
                            "case_1",
                            root,
                            table_path,
                            "Which country has the highest finalWorth?",
                            "string",
                            profile="analyze",
                        )
                    ],
                    remote_config={
                        "api_type": "chat_completions",
                        "model": "fake-model",
                    },
                    remote_batch_size=1,
                    max_retries=2,
                    client=client,
                )

        self.assertTrue(result.case_results[0].ok)
        self.assertEqual(result.case_results[0].answer, "United States")
        self.assertEqual(result.case_results[0].retry_count, 2)
        self.assertEqual(len(client.chat.completions.requests), 3)
        second_repair_prompt = client.chat.completions.requests[2]["messages"][0]["content"]
        self.assertIn('"prior"', second_repair_prompt)
        self.assertNotIn('"logic_dag"', second_repair_prompt)
        self.assertNotIn('"execution_traces"', second_repair_prompt)


def _write_people_table(tmpdir: Path) -> Path:
    tmpdir.mkdir(parents=True, exist_ok=True)
    table_path = tmpdir / "table.csv"
    table_path.write_text(
        "selfMade,country,finalWorth\n" "False,France,100\n" "True,United States,200\n",
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
        f'SELECT "selfMade" AS "{answer_key}" FROM "table_1" ' 'ORDER BY "finalWorth" DESC LIMIT 1;'
    )


def _country_sql(answer_key: str) -> str:
    return (
        f'SELECT "country" AS "{answer_key}" FROM "table_1" ' 'ORDER BY "finalWorth" DESC LIMIT 1;'
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


class _PromptRoutingChatClient:
    def __init__(self, responder: object) -> None:
        self.chat = SimpleNamespace(
            completions=_PromptRoutingChatCompletions(responder),
        )


class _PromptRoutingChatCompletions:
    def __init__(self, responder: object) -> None:
        self._responder = responder
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> "_FakeChatResponse":
        self.requests.append(
            {
                **kwargs,
                "messages": [dict(message) for message in kwargs["messages"]],
            }
        )
        prompt = str(kwargs["messages"][0]["content"])
        text = self._responder(prompt, len(self.requests) - 1)
        if not isinstance(text, str):
            text = json.dumps(text)
        return _FakeChatResponse(text)


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

    def test_extract_returns_cloud_replan_verdict_without_agent_loop(self) -> None:
        from clover.runtime.table_reasoning.pipeline import (
            _extract_filter_evidence_from_traces,
        )

        traces = [
            {
                "op": "Filter",
                "verification_verdict": {
                    "route": "cloud_replan",
                    "reason": "predicate_wrong_column",
                    "evidence": {
                        "node": {"id": "N1", "op": "Filter"},
                        "input_rows": 8,
                        "output_rows": 0,
                        "mismatch": {
                            "sql": "\"Date\" = 'April 8'",
                            "roots": [
                                {
                                    "col": "Date",
                                    "sql_lit": ["April 8"],
                                    "actual": ["1", "2", "3"],
                                    "mismatch": "wrong_column",
                                }
                            ],
                            "candidates": [{"col": "Rnd", "sample": ["April 8", "April 22"]}],
                        },
                    },
                },
            }
        ]

        result = _extract_filter_evidence_from_traces(traces)

        self.assertEqual(result["route"], "cloud_replan")
        self.assertEqual(result["reason"], "predicate_wrong_column")
        self.assertEqual(result["node"], {"id": "N1", "op": "Filter"})
        self.assertEqual(result["input_rows"], 8)
        self.assertEqual(
            result["mismatch"]["candidates"][0]["col"],
            "Rnd",
        )

    def test_extract_merges_trace_node_context_into_verdict(self) -> None:
        from clover.runtime.table_reasoning.pipeline import (
            _extract_filter_evidence_from_traces,
        )

        traces = [
            {
                "node_id": "N1",
                "op": "Filter",
                "output": "T1",
                "dependency": ["T0"],
                "input": [],
                "verification_verdict": {
                    "route": "cloud_replan",
                    "reason": "predicate_not_found",
                    "evidence": {
                        "node": {"id": "N1", "op": "Filter"},
                        "input_rows": 8,
                        "output_rows": 0,
                    },
                },
            }
        ]

        result = _extract_filter_evidence_from_traces(traces)

        self.assertEqual(result["node"]["output"], "T1")
        self.assertEqual(result["node"]["dependency"], ["T0"])

    def test_extract_records_failed_local_attempt(self) -> None:
        from clover.runtime.table_reasoning.pipeline import (
            _extract_filter_evidence_from_traces,
        )

        traces = [
            {
                "op": "Filter",
                "verification_verdict": {
                    "route": "edge_repair",
                    "reason": "predicate_format",
                },
                "agent_loop": {
                    "iterations": 2,
                    "steps": [
                        {
                            "accepted": False,
                            "error": {"message": "still empty"},
                        }
                    ],
                },
            }
        ]

        result = _extract_filter_evidence_from_traces(traces)

        self.assertEqual(result["route"], "edge_repair")
        self.assertEqual(
            result["local_attempt"],
            {
                "iterations": 2,
                "accepted": False,
                "last_error": "still empty",
            },
        )

    def test_extract_join_evidence_for_zero_row_join(self) -> None:
        from clover.runtime.table_reasoning.pipeline import (
            _extract_join_evidence_from_traces,
        )

        traces = [
            {
                "node_id": "N0",
                "op": "Scan",
                "output": "T0",
                "status": "ok",
                "output_summary": {"type": "table", "rows": 3},
            },
            {
                "node_id": "N1",
                "op": "Join",
                "dependency": ["T0"],
                "input": ["election"],
                "output": "T1",
                "status": "ok",
                "output_summary": {"type": "table", "rows": 0},
            },
        ]

        result = _extract_join_evidence_from_traces(traces)

        self.assertEqual(result["route"], "cloud_replan")
        self.assertEqual(result["reason"], "join_zero_rows")
        self.assertEqual(result["fault"], "join_semantic_error")
        self.assertEqual(result["node"]["id"], "N1")
        self.assertEqual(result["node"]["op"], "Join")
        self.assertEqual(result["node"]["output"], "T1")
        self.assertEqual(result["node"]["dependency"], ["T0"])
        self.assertEqual(result["input_rows"], 3)
        self.assertEqual(result["output_rows"], 0)
        self.assertEqual(result["join"]["right_sources"], ["election"])

    def test_extract_join_evidence_for_cross_join(self) -> None:
        from clover.runtime.table_reasoning.pipeline import (
            _extract_cross_join_evidence_from_logic_dag,
        )

        logic_dag = {
            "nodes": [
                {
                    "id": "N0",
                    "op": "Scan",
                    "output": "T0",
                    "params": {"source": "host"},
                },
                {
                    "id": "N1",
                    "op": "Join",
                    "dependency": ["T0"],
                    "output": "T1",
                    "params": {
                        "joins": [
                            {
                                "kind": "CROSS",
                                "source": "party",
                            }
                        ]
                    },
                },
            ]
        }

        result = _extract_cross_join_evidence_from_logic_dag(logic_dag)

        self.assertEqual(result["route"], "cloud_replan")
        self.assertEqual(result["reason"], "join_cross_product")
        self.assertEqual(result["fault"], "join_semantic_error")
        self.assertEqual(result["node"], {"id": "N1", "op": "Join"})
        self.assertEqual(result["join"]["right_sources"], ["party"])

    def test_cloud_replan_evidence_blocks_static_finalization(self) -> None:
        from clover.runtime.table_reasoning.pipeline import (
            _observation_requests_cloud_replan,
        )

        observation = {
            "ok": True,
            "answer": None,
            "obs": [
                {
                    "i": 0,
                    "op": "sql",
                    "ok": True,
                    "res": {"n": 1, "cols": ["answer"], "rows": [[999]]},
                    "ev": {
                        "route": "cloud_replan",
                        "reason": "join_cross_product",
                    },
                }
            ],
        }

        self.assertTrue(_observation_requests_cloud_replan(observation))

    def test_empty_execution_result_becomes_localized_repair_observation(self) -> None:
        from clover.runtime.table_reasoning.pipeline import (
            _maybe_add_empty_execution_repair_evidence,
        )

        task = SimpleNamespace(answer_key="answer_1")
        result = ExecutionResult(
            ok=True,
            answer={"answer_1": []},
            outputs={"answer_1": []},
            traces=[
                {
                    "node_id": "N0",
                    "op": "Scan",
                    "output": "T0",
                    "output_summary": {"type": "table", "rows": 5},
                },
                {
                    "node_id": "N1",
                    "op": "Filter",
                    "output": "T1",
                    "dependency": ["T0"],
                    "input": [],
                    "output_summary": {"type": "table", "rows": 0},
                },
                {
                    "node_id": "N2",
                    "op": "FormatAnswer",
                    "output": "answer_1",
                    "dependency": ["T1"],
                    "output_summary": {"type": "list", "length": 0},
                },
            ],
            output_summaries={},
        )

        observation = _maybe_add_empty_execution_repair_evidence(
            task=task,  # type: ignore[arg-type]
            observation=result,
            current_command={"answer_1": "SELECT name FROM swimmer WHERE event_id = 3"},
        )

        self.assertIsInstance(observation, dict)
        item = observation["obs"][0]
        self.assertEqual(item["q"], "SELECT name FROM swimmer WHERE event_id = 3")
        self.assertEqual(item["ev"]["route"], "cloud_replan")
        self.assertEqual(item["ev"]["reason"], "predicate_not_found")
        self.assertEqual(item["ev"]["node"]["id"], "N1")
        self.assertEqual(item["ev"]["node"]["dependency"], ["T0"])
        self.assertEqual(item["ev"]["input_rows"], 5)
        self.assertEqual(item["ev"]["output_rows"], 0)

    def test_extract_caps_to_top_8_values_per_column(self) -> None:
        from clover.runtime.table_reasoning.pipeline import (
            _extract_filter_evidence_from_traces,
        )

        many_values = [{"value": f"v{i}", "count": 10 - i} for i in range(15)]
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
