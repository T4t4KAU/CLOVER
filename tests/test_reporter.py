from __future__ import annotations

import unittest
from types import SimpleNamespace

from clover.executor import ExecutionResult
from clover.remote_llm import create_remote_llm_session
from clover.reporter import (
    available_task_types,
    initial_report_template_paths,
    parse_reporter_decision,
    render_initial_report_prompt,
    render_report_prompt,
    render_reporter_instruction_prompt,
    reporter_payload,
    run_reporter,
    sql_repair_template_paths,
    template_paths_for_task_type,
)


class ReporterTest(unittest.TestCase):
    def test_template_paths_match_session_reuse_model(self) -> None:
        self.assertEqual(available_task_types(), ("table_reasoning",))
        self.assertEqual(
            initial_report_template_paths("table_reasoning"),
            ("common/root.md", "table_reasoning/v1/report.md"),
        )
        self.assertEqual(
            template_paths_for_task_type("table_reasoning"),
            ("table_reasoning/v1/report.md",),
        )
        self.assertEqual(
            sql_repair_template_paths("table_reasoning"),
            ("table_reasoning/v1/sql_repair.md",),
        )
        self.assertEqual(
            initial_report_template_paths("table_reasoning_v2"),
            ("common/root.md", "table_reasoning/v2/report.md"),
        )
        self.assertEqual(
            sql_repair_template_paths("table_reasoning_v2"),
            ("table_reasoning/v2/sql_repair.md",),
        )

    def test_renders_reporter_instruction_prompt_from_user_text(self) -> None:
        prompt = render_reporter_instruction_prompt()

        self.assertIn("You are the Reporter in CLOVER", prompt)
        self.assertIn("Follow the task-specific JSON schema", prompt)
        self.assertNotIn("Report payload", prompt)

    def test_renders_report_prompt_without_repeating_instruction(self) -> None:
        prompt = render_report_prompt(
            local_dsl=_local_dsl(),
            logic_dag=_logic_dag(),
            local_result=_execution_result(),
            current_sql='SELECT COUNT(*) AS answer FROM "table_1";',
        )

        self.assertIn("Report payload", prompt)
        self.assertIn('"question": "How many rows are present?"', prompt)
        self.assertIn('SELECT COUNT(*) AS answer FROM', prompt)
        self.assertIn("reuse the SQL generation constraints", prompt)
        self.assertIn('"answer": 2', prompt)
        self.assertNotIn('"nodes"', prompt)
        self.assertNotIn("You are the Reporter in CLOVER", prompt)
        self.assertNotIn("/home/", prompt)

    def test_failed_execution_uses_sql_repair_prompt(self) -> None:
        prompt = render_report_prompt(
            local_dsl=_local_dsl(),
            logic_dag=_logic_dag(),
            local_result=_failed_execution_result(),
            current_sql='SELECT "missing" AS answer FROM "table_1";',
        )

        self.assertIn("SQL repair", prompt)
        self.assertIn("Set retry to true", prompt)
        self.assertIn("Set answer to null", prompt)
        self.assertIn("reuse the SQL generation constraints", prompt)
        self.assertIn('SELECT \\"missing\\" AS answer FROM', prompt)
        self.assertIn("Unknown column: missing", prompt)
        self.assertNotIn('"columns": [', prompt)
        self.assertNotIn('"schema"', prompt)
        self.assertNotIn('"sources"', prompt)
        self.assertNotIn("If retry is true, set new_sql", prompt)
        self.assertNotIn('"nodes"', prompt)

    def test_initial_report_prompt_includes_instruction_and_report(self) -> None:
        prompt = render_initial_report_prompt(
            local_dsl=_local_dsl(),
            logic_dag=_logic_dag(),
            local_result=_execution_result(),
        )

        self.assertIn("You are the Reporter in CLOVER", prompt)
        self.assertIn("Report payload", prompt)

    def test_report_payload_strips_evaluation_labels(self) -> None:
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

        payload = reporter_payload(
            local_dsl=local_dsl,
            local_result=local_result,
        )
        prompt = render_initial_report_prompt(
            local_dsl=local_dsl,
            logic_dag=_logic_dag(),
            local_result=local_result,
        )

        self.assertEqual(payload["local_results"]["answer"], 2)
        self.assertNotIn("expected_answer", prompt)
        self.assertNotIn("SECRET_EXPECTED_ANSWER", prompt)
        self.assertNotIn("metadata", prompt)
        self.assertNotIn("execution_result", prompt)

    def test_parses_final_and_retry_decisions(self) -> None:
        final = parse_reporter_decision(
            '{"answer": 2, "retry": false, "new_sql": null}'
        )
        retry = parse_reporter_decision(
            '```json\n{"answer": null, "retry": true, "new_sql": {"sql": "SELECT COUNT(*) AS answer FROM \\"table_1\\";"}}\n```'
        )

        self.assertEqual(final.answer, 2)
        self.assertFalse(final.retry)
        self.assertIsNone(final.new_sql)
        self.assertTrue(retry.retry)
        self.assertEqual(
            retry.new_sql,
            {"sql": 'SELECT COUNT(*) AS answer FROM "table_1";'},
        )
        direct_sql = parse_reporter_decision(
            '{"answer": null, "retry": true, "new_sql": "SELECT COUNT(*) AS answer FROM \\"table_1\\";"}'
        )
        self.assertEqual(
            direct_sql.new_sql,
            {"sql": 'SELECT COUNT(*) AS answer FROM "table_1";'},
        )
        partial_retry = parse_reporter_decision(
            '{"answer": {"answer_1": 2, "answer_2": null}, "retry": true, "new_sql": {"answer_2": "SELECT \\"x\\" AS \\"answer_2\\" FROM \\"table_1\\";"}}'
        )
        self.assertEqual(
            partial_retry.new_sql,
            {"answer_2": 'SELECT "x" AS "answer_2" FROM "table_1";'},
        )

    def test_renders_v2_partial_report_and_repair_prompts(self) -> None:
        report_prompt = render_report_prompt(
            local_dsl=_v2_local_dsl(),
            logic_dag={"task_type": "table_reasoning_v2", "subtasks": []},
            local_result=_v2_execution_result(),
            current_sql={
                "answer_1": 'SELECT "selfMade" AS "answer_1" FROM "table_1";',
                "answer_2": 'SELECT "country" AS "answer_2" FROM "table_1";',
            },
        )
        self.assertIn("Multiple answers may be reviewed together.", report_prompt)
        self.assertIn("If an answer is sufficient", report_prompt)
        self.assertIn('"questions": [', report_prompt)
        self.assertIn('"answer_2": "France"', report_prompt)
        self.assertIn('"local_results": {', report_prompt)
        self.assertNotIn('"sources"', report_prompt)
        self.assertNotIn('"schema"', report_prompt)
        self.assertNotIn('"execution_result"', report_prompt)
        self.assertNotIn('"traces"', report_prompt)
        self.assertNotIn('"output_summaries"', report_prompt)

        repair_prompt = render_report_prompt(
            local_dsl=_v2_local_dsl(),
            logic_dag={"task_type": "table_reasoning_v2", "subtasks": []},
            local_result=_v2_failed_execution_result(),
            current_sql={"answer_2": 'SELECT "missing" AS "answer_2" FROM "table_1";'},
        )
        self.assertIn("SQL repair for the failed answers only", repair_prompt)
        self.assertIn('"What is the country?"', repair_prompt)
        self.assertIn("Unknown column: missing", repair_prompt)
        self.assertNotIn('"failing_node"', repair_prompt)

    def test_run_reporter_reuses_existing_session(self) -> None:
        client = _StatefulChatClient([
            "SELECT COUNT(*) AS answer FROM table_1;",
            '{"answer": 2, "retry": false, "new_sql": null}',
        ])
        session = create_remote_llm_session(
            {"api_type": "chat_completions", "model": "fake-model"},
            client=client,
        )
        session.generate("initial task dsl")

        result = run_reporter(
            local_dsl=_local_dsl(),
            logic_dag=_logic_dag(),
            local_result=_execution_result(),
            session=session,
        )

        self.assertEqual(result.decision.answer, 2)
        self.assertFalse(result.decision.retry)
        second_request_messages = client.chat.completions.requests[1]["messages"]
        self.assertEqual(
            [message["role"] for message in second_request_messages],
            ["user", "assistant", "user"],
        )
        self.assertEqual(second_request_messages[0]["content"], "initial task dsl")
        self.assertIn("Report payload", second_request_messages[2]["content"])


def _local_dsl() -> dict:
    return {
        "task_type": "table_reasoning",
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
        "task_type": "table_reasoning_v1",
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


def _v2_local_dsl() -> dict:
    return {
        "task_type": "table_reasoning_v2",
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


def _v2_execution_result() -> ExecutionResult:
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


def _v2_failed_execution_result() -> ExecutionResult:
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
    id = "reporter_fake"

    def __init__(self, output_text: str) -> None:
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(content=output_text)),
        ]

    def model_dump(self, mode: str) -> dict[str, object]:
        return {"id": self.id, "mode": mode}


if __name__ == "__main__":
    unittest.main()
