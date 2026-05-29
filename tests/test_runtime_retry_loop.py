from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from clover.remote_llm import create_remote_llm_session
from clover.runtime import run_reporter_retry_loop


class RuntimeRetryLoopTest(unittest.TestCase):
    def test_retry_loop_executes_reporter_sql_through_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("Height\n135\n196\n", encoding="utf-8")
            local_dsl = _local_dsl(table_path)
            context = _context(Path(tmpdir), table_path)
            client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "answer": None,
                            "retry": True,
                            "new_sql": {"sql": _meter_retry_sql()},
                        }
                    ),
                    json.dumps(
                        {
                            "answer": 0.61,
                            "retry": False,
                            "new_sql": None,
                        }
                    ),
                ]
            )
            session = create_remote_llm_session(
                {"api_type": "chat_completions", "model": "fake-model"},
                client=client,
            )

            result = run_reporter_retry_loop(
                logic_dag=_centimeter_range_dag(),
                context=context,
                local_dsl=local_dsl,
                session=session,
                initial_sql=_centimeter_range_sql(),
                max_retries=1,
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, 0.61)
        self.assertEqual(len(result.rounds), 2)
        self.assertEqual(result.rounds[0].execution_result.answer, 61)
        self.assertEqual(result.rounds[1].execution_result.answer, 0.61)
        self.assertEqual(result.rounds[0].sql, _centimeter_range_sql())
        self.assertEqual(result.rounds[1].sql, _meter_retry_sql())
        self.assertIn("You are the Reporter in CLOVER", client.chat.completions.requests[0]["messages"][0]["content"])
        self.assertNotIn("You are the Reporter in CLOVER", client.chat.completions.requests[1]["messages"][-1]["content"])

    def test_retry_loop_reports_retry_exhaustion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("Height\n135\n196\n", encoding="utf-8")
            client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "answer": None,
                            "retry": True,
                            "new_sql": {"sql": _meter_retry_sql()},
                        }
                    )
                ]
            )
            session = create_remote_llm_session(
                {"api_type": "chat_completions", "model": "fake-model"},
                client=client,
            )

            result = run_reporter_retry_loop(
                logic_dag=_centimeter_range_dag(),
                context=_context(Path(tmpdir), table_path),
                local_dsl=_local_dsl(table_path),
                session=session,
                initial_sql=_centimeter_range_sql(),
                max_retries=0,
            )

        self.assertFalse(result.ok)
        self.assertTrue(result.retry_exhausted)
        self.assertEqual(result.error["type"], "RetryLimitExceeded")
        self.assertEqual(len(result.rounds), 1)

    def test_retry_loop_reports_invalid_reporter_sql(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table.csv"
            table_path.write_text("Height\n135\n196\n", encoding="utf-8")
            client = _StatefulChatClient(
                [
                    json.dumps(
                        {
                            "answer": None,
                            "retry": True,
                            "new_sql": {"sql": 'DELETE FROM "table_1";'},
                        }
                    )
                ]
            )
            session = create_remote_llm_session(
                {"api_type": "chat_completions", "model": "fake-model"},
                client=client,
            )

            result = run_reporter_retry_loop(
                logic_dag=_centimeter_range_dag(),
                context=_context(Path(tmpdir), table_path),
                local_dsl=_local_dsl(table_path),
                session=session,
                initial_sql=_centimeter_range_sql(),
                max_retries=1,
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.error["type"], "SqlParseError")
        self.assertEqual(len(result.rounds), 1)


def _local_dsl(table_path: Path) -> dict:
    return {
        "task_type": "table_reasoning_v1",
        "question": "What is the range (max-min) of the different heights in meters?",
        "sources": [
            {
                "id": "table_1",
                "type": "table",
                "path": str(table_path),
                "format": "csv",
                "schema": {"format": "csv", "columns": ["Height"]},
            }
        ],
        "answer": {"name": "answer", "type": "number"},
    }


def _context(base_dir: Path, table_path: Path) -> dict:
    return {
        "task_type": "table_reasoning_v1",
        "base_dir": str(base_dir),
        "source_map": {
            "table_1": {
                "type": "table",
                "path": str(table_path),
                "format": "csv",
            }
        },
    }


def _centimeter_range_dag() -> dict:
    return {
        "task_type": "table_reasoning_v1",
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
                            "function": "MAX",
                            "argument": {"type": "column", "name": "Height"},
                            "distinct": False,
                            "alias": "_max",
                        },
                        {
                            "function": "MIN",
                            "argument": {"type": "column", "name": "Height"},
                            "distinct": False,
                            "alias": "_min",
                        },
                    ],
                    "grouped": False,
                },
                "output": "T1",
            },
            {
                "id": "N2",
                "op": "Derive",
                "dependency": ["T1"],
                "input": [],
                "params": {
                    "expressions": [
                        {
                            "alias": "answer",
                            "expr": {
                                "type": "binary_op",
                                "op": "-",
                                "left": {"type": "column", "name": "_max"},
                                "right": {"type": "column", "name": "_min"},
                            },
                        }
                    ]
                },
                "output": "T2",
            },
            _format_node(),
        ],
        "edges": [
            {"from": "N0", "to": "N1"},
            {"from": "N1", "to": "N2"},
            {"from": "N2", "to": "N3"},
        ],
    }


def _centimeter_range_sql() -> str:
    return 'SELECT MAX("Height") - MIN("Height") AS answer FROM "table_1";'


def _meter_retry_sql() -> str:
    return 'SELECT (MAX("Height") - MIN("Height")) / 100 AS answer FROM "table_1";'


def _format_node() -> dict:
    return {
        "id": "N3",
        "op": "FormatAnswer",
        "dependency": ["T2"],
        "input": [],
        "params": {"answer": {"name": "answer", "type": "number"}},
        "output": "answer",
    }


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
    id = "retry_loop_fake"

    def __init__(self, output_text: str) -> None:
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(content=output_text)),
        ]

    def model_dump(self, mode: str) -> dict[str, object]:
        return {"id": self.id, "mode": mode}


if __name__ == "__main__":
    unittest.main()
