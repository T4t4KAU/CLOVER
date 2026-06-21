from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from clover.executor import ExecutionResult
from clover.supervisor import (
    SupervisorAgent,
    SupervisorParseError,
    parse_supervisor_decision,
    render_initial_task_prompt,
    render_synthesis_prompt,
    synthesis_payload,
)


class SupervisorTest(unittest.TestCase):
    def test_table_analyze_decompose_prompt_keeps_answer_out_of_acts(self) -> None:
        prompt = render_initial_task_prompt(
            {
                "task_type": "table_reasoning.analyze",
                "profile": "analyze",
                "question": "How many rows are present?",
                "sources": [
                    {
                        "id": "table_1",
                        "type": "table",
                        "format": "csv",
                        "schema": {"columns": ["value"]},
                    }
                ],
                "answer": {"name": "answer", "type": "number"},
                "hints": {"category": "NumericalReasoning"},
            }
        )

        self.assertIn("Evidence plan:", prompt)
        self.assertIn("Terminal answer:", prompt)
        self.assertIn("Choose one mutually exclusive action form", prompt)
        self.assertIn("`acts` may contain only `sql` or `analyze` actions", prompt)
        self.assertIn('Never put `{"op":"answer"}` inside `acts`', prompt)

    def test_synthesis_payload_strips_evaluation_labels(self) -> None:
        local_dsl = {
            **_local_dsl(),
            "expected_answer": "SECRET_EXPECTED_ANSWER",
            "metadata": {
                "case": {
                    "answer": "SECRET_EXPECTED_ANSWER",
                }
            },
        }
        local_result = {
            "ok": True,
            "answer": 2,
            "expected_answer": "SECRET_EXPECTED_ANSWER",
            "metadata": {
                "case": {
                    "answer": "SECRET_EXPECTED_ANSWER",
                }
            },
            "outputs": {"answer": 2},
        }

        payload = synthesis_payload(
            local_dsl=local_dsl,
            observation=local_result,
        )

        self.assertEqual(payload["ev"][0]["answer"], 2)
        self.assertNotIn("task", payload)
        self.assertNotIn("observations", payload)
        self.assertNotIn("expected_answer", json.dumps(payload))
        self.assertNotIn("metadata", json.dumps(payload))

    def test_table_synthesis_payload_uses_evidence_only(self) -> None:
        local_dsl = {
            **_local_dsl(),
            "profile": "analyze",
            "hints": {"category": "NumericalReasoning"},
        }
        observation = {
            "ok": True,
            "answer": None,
            "obs": [
                {
                    "i": 0,
                    "op": "sql",
                    "ok": True,
                    "q": 'SELECT "value" FROM "table_1"',
                    "res": {"n": 1, "cols": ["value"], "rows": [{"value": 2}]},
                }
            ],
        }

        payload = synthesis_payload(
            local_dsl=local_dsl,
            observation=observation,
            current_command={"acts": [{"op": "sql", "q": 'SELECT "value" FROM "table_1"'}]},
        )

        self.assertEqual(payload["q"], "How many rows are present?")
        self.assertEqual(payload["ty"], "number")
        self.assertEqual(payload["ev"][0]["res"]["rows"], [[2]])
        self.assertNotIn("ctx", payload)
        self.assertNotIn("task_type", json.dumps(payload))
        self.assertNotIn("profile", json.dumps(payload))
        self.assertNotIn("hints", json.dumps(payload))
        self.assertNotIn("/home/user/private/table.csv", json.dumps(payload))

    def test_table_synthesis_payload_builds_compact_repair_packet(self) -> None:
        observation = {
            "ok": True,
            "answer": None,
            "obs": [
                {
                    "i": 0,
                    "op": "sql",
                    "ok": True,
                    "q": 'SELECT "value" FROM "table_1" WHERE "value" = 99',
                    "res": {"n": 0, "cols": ["value"], "rows": []},
                }
            ],
        }

        payload = synthesis_payload(
            local_dsl=_local_dsl(),
            observation=observation,
            current_command={
                "acts": [
                    {
                        "op": "sql",
                        "q": 'SELECT "value" FROM "table_1" WHERE "value" = 99',
                    }
                ]
            },
        )

        self.assertEqual(payload["ev"][0]["res"]["n"], 0)
        self.assertEqual(payload["repair"]["schema"], {"table_1": ["value"]})
        self.assertEqual(
            payload["repair"]["sql"],
            'SELECT "value" FROM "table_1" WHERE "value" = 99',
        )
        self.assertEqual(payload["repair"]["failure"]["kind"], "zero_rows")
        self.assertNotIn("ctx", payload)

    def test_table_join_repair_packet_includes_join_candidates(self) -> None:
        sql = (
            'SELECT AVG("host"."Age") AS "answer" '
            'FROM "host" JOIN "party" ON "host"."Host_ID" = "party"."Party_ID" '
            'WHERE "party"."Location" LIKE "%Amsterdam%"'
        )
        observation = {
            "ok": True,
            "answer": None,
            "obs": [
                {
                    "i": 0,
                    "op": "sql",
                    "ok": True,
                    "q": sql,
                    "res": {"n": 0, "cols": ["answer"], "rows": []},
                    "ev": {
                        "route": "cloud_replan",
                        "reason": "join_zero_rows",
                        "fault": "join_semantic_error",
                        "input_rows": 10,
                        "output_rows": 0,
                        "node": {"id": "Join_1", "op": "Join"},
                        "join": {
                            "dependencies": ["Scan_host", "Scan_party"],
                            "right_sources": ["party"],
                        },
                    },
                }
            ],
        }
        local_dsl = {
            **_local_dsl(),
            "question": "What is the average age of hosts for parties in Amsterdam?",
            "sources": [
                {
                    "id": "party",
                    "type": "table",
                    "schema": {"columns": ["Party_ID", "Location"]},
                },
                {
                    "id": "host",
                    "type": "table",
                    "schema": {"columns": ["Host_ID", "Age"]},
                },
                {
                    "id": "party_host",
                    "type": "table",
                    "schema": {"columns": ["Party_ID", "Host_ID"]},
                },
            ],
            "hints": {
                "join_candidates": [
                    {
                        "left_table": "party",
                        "left_column": "Party_ID",
                        "right_table": "party_host",
                        "right_column": "Party_ID",
                        "score": 0.835,
                        "overlap": 5,
                        "sample_matches": ["1", "2"],
                    },
                    {
                        "left_table": "host",
                        "left_column": "Host_ID",
                        "right_table": "party_host",
                        "right_column": "Host_ID",
                        "score": 0.825,
                        "overlap": 6,
                        "sample_matches": ["1", "2"],
                    },
                ],
                "join_paths": [
                    {
                        "tables": ["host", "party_host", "party"],
                        "joins": [
                            {
                                "left_table": "host",
                                "left_column": "Host_ID",
                                "right_table": "party_host",
                                "right_column": "Host_ID",
                            },
                            {
                                "left_table": "party_host",
                                "left_column": "Party_ID",
                                "right_table": "party",
                                "right_column": "Party_ID",
                            },
                        ],
                        "length": 2,
                        "score": 0.83,
                    }
                ],
            },
        }

        payload = synthesis_payload(
            local_dsl=local_dsl,
            observation=observation,
            current_command={"acts": [{"op": "sql", "q": sql}]},
        )

        repair = payload["repair"]
        self.assertEqual(repair["fault"], "join_semantic_error")
        self.assertEqual(len(repair["join_candidates"]), 2)
        self.assertEqual(
            repair["join_candidates"][0]["right_table"],
            "party_host",
        )
        self.assertEqual(
            repair["join_paths"][0]["tables"],
            ["host", "party_host", "party"],
        )
        self.assertEqual(repair["evidence"]["join"]["right_sources"], ["party"])
        self.assertIn("bridge table", json.dumps(repair["requirements"]))

    def test_table_repair_packet_removes_unverified_candidate_column(self) -> None:
        observation = {
            "ok": True,
            "answer": None,
            "obs": [
                {
                    "i": 0,
                    "op": "sql",
                    "ok": True,
                    "q": 'SELECT "Title" FROM "table_1" WHERE "Title" = \'1905\'',
                    "res": {"n": 0, "cols": ["Title"], "rows": []},
                    "ev": {
                        "route": "cloud_replan",
                        "reason": "predicate_candidate_column",
                        "mismatch": {
                            "roots": [
                                {
                                    "col": "Title",
                                    "sql_lit": ["1905"],
                                    "actual": ["Example title"],
                                }
                            ],
                            "candidates": [
                                {"col": "Year", "sample": ["Another title"]}
                            ],
                        },
                    },
                }
            ],
        }
        local_dsl = {
            **_local_dsl(),
            "sources": [
                {
                    "id": "table_1",
                    "type": "table",
                    "schema": {"columns": ["Title", "Year"]},
                }
            ],
        }

        payload = synthesis_payload(
            local_dsl=local_dsl,
            observation=observation,
            current_command={
                "acts": [
                    {
                        "op": "sql",
                        "q": 'SELECT "Title" FROM "table_1" WHERE "Title" = \'1905\'',
                    }
                ]
            },
        )

        evidence = payload["repair"]["evidence"]
        self.assertEqual(evidence["reason"], "predicate_unclassified")
        self.assertEqual(evidence["route"], "edge_repair")
        self.assertNotIn("candidates", evidence["mismatch"])

    def test_table_repair_history_drops_execution_trace_payloads(self) -> None:
        local_dsl = {
            **_local_dsl(),
            "mem": [
                {
                    "act": [{"op": "sql", "q": "SELECT old"}],
                    "obs": [
                        {
                            "ok": False,
                            "logic_dag": {"secret": "FULL_DAG"},
                            "execution_traces": [{"prompt": "FULL_AGENT_PROMPT"}],
                            "err": {"type": "SqlParseError", "message": "bad sql"},
                        }
                    ],
                }
            ],
        }
        observation = {
            "ok": True,
            "obs": [
                {
                    "i": 0,
                    "op": "sql",
                    "ok": False,
                    "err": {"type": "SqlParseError", "message": "bad sql"},
                }
            ],
        }

        payload = synthesis_payload(
            local_dsl=local_dsl,
            observation=observation,
            current_command={"acts": [{"op": "sql", "q": "SELECT bad"}]},
        )

        serialized = json.dumps(payload)
        self.assertNotIn("FULL_DAG", serialized)
        self.assertNotIn("FULL_AGENT_PROMPT", serialized)
        self.assertEqual(
            payload["repair"]["prior"][0]["act"],
            [{"op": "sql", "q": "SELECT old"}],
        )

    def test_table_final_repair_round_forbids_more_actions(self) -> None:
        local_dsl = {
            **_local_dsl(),
            "task_type": "table_reasoning.analyze",
            "profile": "analyze",
        }
        prompt = render_synthesis_prompt(
            local_dsl=local_dsl,
            logic_dag={"task_type": "table_reasoning.analyze", "query_plans": []},
            observation={
                "ok": True,
                "obs": [
                    {
                        "i": 0,
                        "op": "sql",
                        "ok": True,
                        "res": {"n": 0, "cols": ["value"], "rows": []},
                    }
                ],
            },
            current_command={"acts": [{"op": "sql", "q": "SELECT value"}]},
            force_final_answer=True,
        )

        self.assertIn("No more execution rounds are available", prompt)
        self.assertIn("Do not return `acts`", prompt)

    def test_parses_answer_and_action_decisions(self) -> None:
        answer_decision = parse_supervisor_decision('{"op":"answer","a":2}')

        self.assertEqual(answer_decision.answer, 2)
        self.assertFalse(answer_decision.retry)
        document = parse_supervisor_decision(
            '{"answer": "2.8%", "sufficient": true, "explanation": "Calculated from worker evidence."}'
        )
        self.assertEqual(document.answer, "2.8%")
        self.assertTrue(document.sufficient)
        self.assertFalse(document.retry)
        self.assertEqual(document.explanation, "Calculated from worker evidence.")
        document_retry = parse_supervisor_decision(
            '{"answer": null, "sufficient": false, "feedback": "look again", '
            '"next_python_code": "def prepare_jobs(context):\\n    return []\\n\\ndef transform_outputs(jobs):\\n    return \\"\\""}'
        )
        self.assertFalse(document_retry.sufficient)
        self.assertEqual(
            document_retry.next_python_code.splitlines()[0],
            "def prepare_jobs(context):",
        )
        react_answer = parse_supervisor_decision('{"a": 42}')
        self.assertTrue(react_answer.done)
        self.assertFalse(react_answer.retry)
        self.assertEqual(react_answer.answer, 42)
        react_action = parse_supervisor_decision(
            '{"q": "SELECT COUNT(*) AS answer FROM \\"table_1\\";"}'
        )
        self.assertFalse(react_action.done)
        self.assertEqual(react_action.actions[0].op, "sql")
        self.assertEqual(
            react_action.sqls,
            ('SELECT COUNT(*) AS answer FROM "table_1";',),
        )
        react_observe = parse_supervisor_decision(
            '{"q": ["SELECT \\"year\\", \\"sales\\" FROM \\"table_1\\" LIMIT 5;"]}'
        )
        self.assertFalse(react_observe.done)
        self.assertEqual(
            react_observe.sqls,
            ('SELECT "year", "sales" FROM "table_1" LIMIT 5;',),
        )
        react_wrapped = parse_supervisor_decision(
            '{"steps": [{"q": ["SELECT \\"x\\" FROM \\"table_1\\" LIMIT 1;"]}, '
            '{"q": "SELECT COUNT(*) AS answer FROM \\"table_1\\";"}]}'
        )
        self.assertFalse(react_wrapped.done)
        self.assertEqual(react_wrapped.sqls, ('SELECT "x" FROM "table_1" LIMIT 1;',))
        explicit_answer = parse_supervisor_decision('{"op":"answer","a":"yes"}')
        self.assertTrue(explicit_answer.done)
        self.assertEqual(explicit_answer.action_op, "answer")
        self.assertEqual(explicit_answer.answer, "yes")
        action_group = parse_supervisor_decision(
            '{"acts":[{"op":"sql","q":"SELECT \\"year\\" FROM \\"table_1\\" LIMIT 5;"},'
            '{"op":"analyze","kind":"correlation",'
            '"seed":"SELECT \\"year\\", \\"sales\\" FROM \\"table_1\\";"}]}'
        )
        self.assertFalse(action_group.done)
        self.assertEqual([action.op for action in action_group.actions], ["sql", "analyze"])
        self.assertEqual(action_group.actions[1].kind, "correlation")
        self.assertEqual(
            action_group.actions[1].seed,
            'SELECT "year", "sales" FROM "table_1";',
        )
        wrapped_seed = parse_supervisor_decision(
            '{"acts":[{"op":"analyze","kind":"statistical",'
            '"seed":{"sql":"SELECT \\"year\\", \\"sales\\" FROM \\"table_1\\";"}}]}'
        )
        self.assertEqual(wrapped_seed.actions[0].op, "analyze")
        self.assertEqual(
            wrapped_seed.actions[0].seed,
            'SELECT "year", "sales" FROM "table_1";',
        )
        analyze_action = parse_supervisor_decision(
            '{"acts":[{"op":"analyze","kind":"correlation",'
            '"seed":"SELECT \\"year\\", \\"sales\\" FROM \\"table_1\\";"}]}'
        )
        self.assertEqual(analyze_action.actions[0].op, "analyze")
        self.assertEqual(analyze_action.actions[0].kind, "correlation")
        self.assertEqual(
            analyze_action.actions[0].seed,
            'SELECT "year", "sales" FROM "table_1";',
        )

    def test_rejects_remote_evidence_action(self) -> None:
        with self.assertRaises(SupervisorParseError):
            parse_supervisor_decision(
                '{"op":"ev","seed":"SELECT \\"year\\", \\"sales\\" FROM \\"table_1\\";"}'
            )
        with self.assertRaises(SupervisorParseError):
            parse_supervisor_decision(
                '{"op":"inspect","q":"find trend evidence",'
                '"seed":"SELECT \\"year\\", \\"sales\\" FROM \\"table_1\\";"}'
            )

    def test_rejects_final_protocol_marker(self) -> None:
        with self.assertRaises(SupervisorParseError):
            parse_supervisor_decision(
                '{"final": false, "q": "SELECT COUNT(*) AS answer FROM \\"table_1\\";"}'
            )

    def test_rejects_legacy_retry_protocol(self) -> None:
        with self.assertRaises(SupervisorParseError):
            parse_supervisor_decision(
                '{"answer": null, "retry": true, "new_sql": {"sql": "SELECT 1;"}}'
            )
        with self.assertRaises(SupervisorParseError):
            parse_supervisor_decision(
                '{"done": false, "sqls": ["SELECT 1;"]}'
            )

    def test_supervisor_synthesis_uses_stateless_remote_call(self) -> None:
        client = _StatefulChatClient([
            '{"op":"answer","a":2}',
        ])

        result = SupervisorAgent(
            remote_config={"api_type": "chat_completions", "model": "fake-model"},
            client=client,
        ).synthesize(
            local_dsl=_local_dsl(),
            logic_dag=_logic_dag(),
            observation=_execution_result(),
        )

        self.assertEqual(result.decision.answer, 2)
        self.assertFalse(result.decision.retry)
        request_messages = client.chat.completions.requests[0]["messages"]
        self.assertEqual(
            [message["role"] for message in request_messages],
            ["user"],
        )
        self.assertIn("Evidence payload:", request_messages[0]["content"])

    def test_supervisor_can_route_synthesis_to_separate_model(self) -> None:
        decompose_client = _StatefulChatClient(['{"acts":[{"op":"sql","q":"SELECT 1"}]}'])
        synthesize_client = _StatefulChatClient(['{"op":"answer","a":2}'])
        agent = SupervisorAgent(
            remote_config={"api_type": "chat_completions", "model": "remote-model"},
            client=decompose_client,
            synthesize_config={
                "api_type": "chat_completions",
                "model": "synthesize-model",
            },
            synthesize_client=synthesize_client,
        )

        agent.decompose(task_dsl=_local_dsl())
        result = agent.synthesize(
            local_dsl=_local_dsl(),
            logic_dag=_logic_dag(),
            observation=_execution_result(),
        )

        self.assertEqual(result.decision.answer, 2)
        self.assertEqual(
            decompose_client.chat.completions.requests[0]["model"],
            "remote-model",
        )
        self.assertEqual(
            synthesize_client.chat.completions.requests[0]["model"],
            "synthesize-model",
        )

    def test_document_synthesis_prompt_uses_worker_evidence_only(self) -> None:
        observation = _document_execution_result()
        prompt = render_synthesis_prompt(
            local_dsl=_document_local_dsl(),
            logic_dag=_document_logic_dag(),
            observation=observation,
        )
        payload = synthesis_payload(
            local_dsl=_document_local_dsl(),
            observation=observation,
        )

        self.assertEqual(
            payload["observations"]["evidence_summary"],
            "### Chunk chunk_67\nanswer: values\ncitation: page 55",
        )
        self.assertEqual(payload["observations"]["worker_count"], 2)
        self.assertEqual(payload["observations"]["included_count"], 1)
        self.assertEqual(payload["observations"]["failed_count"], 0)
        self.assertFalse(payload["observations"]["fallback_used"])
        self.assertIn("Answer the question using only the worker evidence below.", prompt)
        self.assertIn("- workers: 2", prompt)
        self.assertIn("### Chunk chunk_67", prompt)
        self.assertIn('"sufficient": false', prompt)
        self.assertNotIn("document_1:chunk_67", prompt)
        self.assertNotIn("SECRET_VERBOSE_WORKER_OUTPUT", prompt)
        self.assertNotIn("SECRET_VERBOSE_WORKER_OUTPUT", json.dumps(payload))
        self.assertNotIn('"outputs"', prompt)
        self.assertNotIn("traces", prompt)

    def test_document_synthesis_prompt_preserves_prior_worker_evidence(self) -> None:
        observation = {
            "ok": True,
            "worker_count": 1,
            "included_count": 1,
            "failed_count": 0,
            "evidence_summary": "current: D&A is $636 million",
            "prior_evidence_summary": "prior: operating income is $1,196 million",
            "prior_evidence_round_count": 1,
            "prior_evidence_truncated": False,
        }

        prompt = render_synthesis_prompt(
            local_dsl=_document_local_dsl(),
            logic_dag=_document_logic_dag(),
            observation=observation,
        )
        payload = synthesis_payload(
            local_dsl=_document_local_dsl(),
            observation=observation,
        )

        self.assertIn("Prior worker evidence from previous rounds:", prompt)
        self.assertIn("prior: operating income is $1,196 million", prompt)
        self.assertIn("current: D&A is $636 million", prompt)
        self.assertEqual(
            payload["observations"]["prior_evidence_summary"],
            "prior: operating income is $1,196 million",
        )

    def test_document_supervisor_synthesis_parses_sufficient_decision(self) -> None:
        client = _StatefulChatClient(
            [
                (
                    '{"answer": "2.8%", "sufficient": true, '
                    '"explanation": "Average of the three yearly margins."}'
                )
            ]
        )

        result = SupervisorAgent(
            remote_config={"api_type": "chat_completions", "model": "fake-model"},
            client=client,
        ).synthesize(
            local_dsl=_document_local_dsl(),
            logic_dag=_document_logic_dag(),
            observation=_document_execution_result(),
        )

        self.assertEqual(result.decision.answer, "2.8%")
        self.assertTrue(result.decision.sufficient)
        self.assertFalse(result.decision.retry)
        self.assertIn("Worker evidence:", result.prompt)


def _local_dsl() -> dict:
    return {
        "task_type": "table_reasoning.query",
        "question": "How many rows are present?",
        "sources": [
            {
                "id": "table_1",
                "type": "table",
                "format": "csv",
                "path": "/home/user/private/table.csv",
                "schema": {"columns": ["value"]},
            }
        ],
        "answer": {"name": "answer", "type": "number"},
    }


def _logic_dag() -> dict:
    return {
        "task_type": "table_reasoning.query",
        "nodes": [
            {"id": "N0", "op": "Scan", "dependency": [], "input": ["table_1"], "params": {"source": "table_1"}, "output": "T0"}
        ],
        "edges": [],
    }


def _execution_result() -> ExecutionResult:
    return ExecutionResult(
        ok=True,
        answer=2,
        outputs={"answer": 2},
        traces=[
            {
                "node_id": "N0",
                "op": "FormatAnswer",
                "status": "ok",
                "fast_path_hit": True,
            }
        ],
        output_summaries={"answer": {"type": "value", "preview": 2}},
        elapsed_ms=1.0,
        fast_path_hits=1,
        fast_path_misses=0,
    )


def _document_local_dsl() -> dict:
    return {
        "task_type": "document_reasoning",
        "question": "What is the three-year average net profit margin?",
        "sources": [
            {
                "id": "document_1",
                "type": "document",
                "format": "pdf",
                "path": "/home/user/private/report.pdf",
            }
        ],
        "answer": {"name": "answer", "type": "string"},
    }


def _document_logic_dag() -> dict:
    return {
        "task_type": "document_reasoning",
        "map_groups": [],
        "edges": [],
    }


def _document_execution_result() -> ExecutionResult:
    return ExecutionResult(
        ok=True,
        answer=None,
        outputs={
            "G0__0__document_1_chunk_67": {
                "answer": "values",
                "citation": "page 55",
            }
        },
        collector_outputs={
            "G0": {
                "kind": "map_group_evidence",
                "worker_count": 2,
                "included_count": 1,
                "evidence_summary": (
                    "### Chunk chunk_67\nanswer: values\ncitation: page 55"
                ),
                "job_outputs": [
                    {"explanation": "SECRET_VERBOSE_WORKER_OUTPUT"},
                ],
            }
        },
        traces=[{"node_id": "G0__0__document_1_chunk_67"}],
        output_summaries={},
    )


def _batch_local_dsl() -> dict:
    return {
        "task_type": "table_reasoning.query",
        "questions": ["Is the richest person self-made?", "What is the country?"],
        "sources": [
            {
                "id": "table_1",
                "type": "table",
                "path": "/home/user/private/table.csv",
                "format": "csv",
                "schema": {"columns": ["selfMade", "country", "finalWorth"]},
            }
        ],
        "answers": [
            {"name": "answer_1", "type": "boolean"},
            {"name": "answer_2", "type": "string"},
        ],
    }


def _batch_execution_result() -> ExecutionResult:
    return ExecutionResult(
        ok=True,
        answer={"answer_1": True, "answer_2": "France"},
        outputs={"answer_1": True, "answer_2": "France"},
        traces=[],
        output_summaries={
            "answer_1": {"type": "value", "preview": True},
            "answer_2": {"type": "value", "preview": "France"},
        },
    )


def _batch_failed_execution_result() -> ExecutionResult:
    return ExecutionResult(
        ok=False,
        answer=None,
        outputs={"answer_1": True},
        traces=[],
        output_summaries={"answer_1": {"type": "value", "preview": True}},
        failing_node={"id": "N5", "op": "Project", "output": "T4"},
        error={"type": "PandasExecutionError", "message": "Unknown column: missing"},
    )


def _failed_execution_result() -> ExecutionResult:
    return ExecutionResult(
        ok=False,
        answer=None,
        outputs={},
        traces=[
            {
                "node_id": "N1",
                "op": "Aggregate",
                "status": "error",
                "fast_path_hit": True,
            }
        ],
        output_summaries={},
        failing_node={"node_id": "N1", "op": "Aggregate"},
        error={"type": "PandasExecutionError", "message": "Unknown column: missing"},
        elapsed_ms=1.0,
        fast_path_hits=1,
        fast_path_misses=0,
    )


class _StatefulChatClient:
    def __init__(self, output_texts: list[str]) -> None:
        self.chat = SimpleNamespace(
            completions=_StatefulChatCompletions(output_texts),
        )


class _StatefulChatCompletions:
    def __init__(self, output_texts: list[str]) -> None:
        self._output_texts = output_texts
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> "_FakeChatResponse":
        self.requests.append(
            {
                **kwargs,
                "messages": [dict(message) for message in kwargs["messages"]],
            }
        )
        return _FakeChatResponse(self._output_texts[len(self.requests) - 1])


class _FakeChatResponse:
    id = "supervisor_fake"

    def __init__(self, output_text: str) -> None:
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(content=output_text)),
        ]

    def model_dump(self, mode: str) -> dict[str, object]:
        return {"id": self.id, "mode": mode}


if __name__ == "__main__":
    unittest.main()
